"""Tier 3 disk-budget LRU eviction (experimental, opt-in).

PRD §12.4.3. When ``[index].disk_budget_gb`` is set above 0 in the
service config, this module enforces a soft cap on the total cache
footprint by evicting projects in LRU order (oldest queried first).

**Experimental** for v1:
  * ``last_query_at`` is best-effort. The watcher tracks it in-memory
    (per ``_ProjectState``); on daemon restart we fall back to the
    project's persisted ``last_full_scan_at``. A persistent column on
    the projects table is the proper fix and ships in a follow-up.
  * Eviction is destructive: the project entry stays in config but
    its chunks + embeddings get dropped. Next query against the
    project re-indexes from scratch.
  * The watcher's periodic-check loop calls ``maybe_evict`` once per
    cycle; users can also trigger a one-shot run via
    ``longctx clean`` (when --disk-budget is passed).

If you want predictable disk use, set ``disk_budget_gb`` to
something concrete (e.g. ``5.0``) and accept the re-index cost on
re-warm. If you want unbounded retention, leave it at ``0.0``
(default) and the module is a no-op.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProjectFootprint:
    name: str
    last_query_at: float       # epoch seconds; falls back to last_full_scan_at
    bytes_used: int


@dataclass(frozen=True)
class EvictionPlan:
    """List of projects to evict + estimated bytes freed."""
    targets: tuple[str, ...]
    bytes_to_free: int
    pre_eviction_bytes: int
    budget_bytes: int
    budget_exceeded_by: int


def cache_size_bytes(cache_dir: Path) -> int:
    """Total bytes occupied by the longctx cache. Walks the directory
    once, summing file sizes. Symlinks counted but not followed."""
    if not cache_dir.is_dir():
        return 0
    total = 0
    for p in cache_dir.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _project_footprint(
    chunk_store, project_name: str,
    *, last_query_overrides: Optional[dict[str, float]] = None,
) -> _ProjectFootprint:
    """Estimate bytes used by one project + best-effort last_query_at.

    ``last_query_overrides`` is the watcher's in-memory timestamp map;
    when None we fall back to ``project.last_full_scan_at``. The
    watcher's loop passes its live state in; the manual CLI sweep
    can pass None and accept the slightly-staler signal.
    """
    proj = chunk_store.get_project(project_name)
    if proj is None:
        return _ProjectFootprint(
            name=project_name, last_query_at=0.0, bytes_used=0,
        )
    last = 0.0
    if last_query_overrides and project_name in last_query_overrides:
        last = float(last_query_overrides[project_name])
    if last <= 0:
        last = float(getattr(proj, "last_full_scan_at", 0) or 0)

    # Footprint: chunk text + embedding rows. Conservative 1.5 KB/chunk
    # (matches the heuristic used by ``longctx clean``).
    bytes_used = 0
    try:
        for f in chunk_store.list_files(project=project_name):
            chunks = chunk_store.get_chunks_by_file(f.id)
            bytes_used += sum(len(c.text.encode("utf-8")) for c in chunks)
            bytes_used += len(chunks) * 1536
    except Exception:
        pass
    return _ProjectFootprint(
        name=project_name, last_query_at=last, bytes_used=bytes_used,
    )


def plan_eviction(
    cache_dir: Path,
    chunk_store,
    *,
    budget_gb: float,
    last_query_overrides: Optional[dict[str, float]] = None,
    pinned_projects: Iterable[str] = (),
) -> EvictionPlan:
    """Compute which projects to evict to bring cache under budget.

    Args:
        cache_dir: where the on-disk index lives. Used to measure
            current total bytes.
        chunk_store: provides project list + per-project size estimates.
        budget_gb: target. ``0`` disables (returns empty plan).
        last_query_overrides: watcher's live ``last_query_at`` map for
            each project; None falls back to ``last_full_scan_at``.
        pinned_projects: names that are NEVER evicted (e.g. session-
            bound projects with active connections, the project
            currently being indexed).

    Returns an ``EvictionPlan``. ``targets`` empty when under budget
    or budget=0. Sorted with the LRU candidate first.
    """
    if budget_gb <= 0:
        return EvictionPlan(
            targets=(), bytes_to_free=0,
            pre_eviction_bytes=cache_size_bytes(cache_dir),
            budget_bytes=0, budget_exceeded_by=0,
        )

    budget_bytes = int(budget_gb * 1024 ** 3)
    current = cache_size_bytes(cache_dir)
    if current <= budget_bytes:
        return EvictionPlan(
            targets=(), bytes_to_free=0,
            pre_eviction_bytes=current,
            budget_bytes=budget_bytes, budget_exceeded_by=0,
        )

    pinned = set(pinned_projects)
    footprints = [
        _project_footprint(
            chunk_store, p.name,
            last_query_overrides=last_query_overrides,
        )
        for p in chunk_store.list_projects()
        if p.name not in pinned
    ]
    # Sort LRU-first (oldest query timestamp first; ties broken by
    # smallest bytes — prefer evicting a small idle project before a
    # large idle one when their query timestamps tie).
    footprints.sort(key=lambda f: (f.last_query_at, f.bytes_used))

    overflow = current - budget_bytes
    targets: list[str] = []
    freed = 0
    for fp in footprints:
        if freed >= overflow:
            break
        if fp.bytes_used <= 0:
            continue   # nothing to free; skip
        targets.append(fp.name)
        freed += fp.bytes_used

    return EvictionPlan(
        targets=tuple(targets),
        bytes_to_free=freed,
        pre_eviction_bytes=current,
        budget_bytes=budget_bytes,
        budget_exceeded_by=overflow,
    )


def execute_eviction(
    plan: EvictionPlan,
    *,
    indexer=None,
    chunk_store=None,
) -> int:
    """Apply the plan. Prefer ``indexer.delete_project`` (frees memmap
    rows + chunks atomically); fall back to chunk_store cascade if no
    indexer is provided.

    Returns the number of projects actually evicted. Errors are
    logged + counted as failures; the rest of the plan continues.
    """
    if not plan.targets:
        return 0
    n = 0
    for name in plan.targets:
        try:
            if indexer is not None:
                indexer.delete_project(name)
            elif chunk_store is not None:
                chunk_store.delete_project(name)
            else:
                raise RuntimeError(
                    "execute_eviction needs indexer or chunk_store"
                )
            logger.warning(
                "tier3_eviction: dropped project %r (LRU)",
                name,
            )
            n += 1
        except Exception:
            logger.exception(
                "tier3_eviction: failed to drop project %r", name,
            )
    return n


def maybe_evict(
    cache_dir: Path,
    chunk_store,
    *,
    budget_gb: float,
    indexer=None,
    last_query_overrides: Optional[dict[str, float]] = None,
    pinned_projects: Iterable[str] = (),
) -> EvictionPlan:
    """One-shot helper: plan + execute. Used by the watcher's periodic
    loop and by ``longctx clean --disk-budget Ng``."""
    plan = plan_eviction(
        cache_dir, chunk_store,
        budget_gb=budget_gb,
        last_query_overrides=last_query_overrides,
        pinned_projects=pinned_projects,
    )
    if plan.targets:
        execute_eviction(plan, indexer=indexer, chunk_store=chunk_store)
    return plan
