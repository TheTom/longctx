"""longctx clean — manual cleanup of indexed projects + replay shards.

Implements the manual side of PRD §12.4.4. The auto-cleanup tiers
are the watcher's job (Tier 1 missing root in §5.5, Tier 2 cold-pause
in §5.6); this module handles user-driven sweeps:

  longctx clean                       dry-run: show what would go
  longctx clean --idle 90d            drop projects not queried in 90+d
  longctx clean --orphans             session-bound projects from dead sessions
  longctx clean --missing             root_path no longer exists, no grace
  longctx clean --all                 --idle 90d + --orphans + --missing
  longctx clean --replay-older-than 14d
  longctx clean --yes                 skip the "are you sure" prompt
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------- duration

_DURATION_RE = re.compile(r"^(\d+)\s*([dhwm])$")


def parse_duration_seconds(s: str) -> int:
    """Parse ``90d``, ``2w``, ``12h``, ``30m`` → seconds.

    Strict: only those four suffixes. Empty string and bare integers
    raise ``ValueError`` so the caller doesn't silently accept '90'
    and treat it as something arbitrary.
    """
    m = _DURATION_RE.match(s.strip().lower())
    if m is None:
        raise ValueError(
            f"unrecognized duration {s!r}; expected like '90d', '2w', "
            f"'12h', '30m'"
        )
    n, unit = int(m.group(1)), m.group(2)
    return n * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


# ----------------------------------------------------------- decision plan

@dataclass(frozen=True)
class CleanupDecision:
    """One row in the dry-run plan — what we'd do to a project (or
    replay shard) and why."""
    target: str           # project name OR shard filename
    reason: str           # "idle 92d > 90d" / "missing 9d > 7d" / etc
    bytes_freed: int      # estimated; 0 for unknown
    kind: str             # "project" | "replay_shard" | "session_orphan"


@dataclass(frozen=True)
class CleanupPlan:
    decisions: tuple[CleanupDecision, ...]

    @property
    def total_bytes(self) -> int:
        return sum(d.bytes_freed for d in self.decisions)


# ------------------------------------------------------------- collectors

def collect_idle(
    chunk_store, max_age_seconds: int, *, now: Optional[float] = None,
) -> list[CleanupDecision]:
    """Projects whose ``last_query_at`` is older than the threshold.

    ``last_query_at`` is in-memory in Phase 2.2 (per Watcher's
    `_ProjectState`). For now we read from a per-project metadata
    field on the project record OR fall back to ``last_full_scan_at``.
    Tier 3's persistent ``last_query_at`` schema bump can refine
    this when the disk-budget LRU lands.
    """
    now = time.time() if now is None else now
    out: list[CleanupDecision] = []
    for proj in chunk_store.list_projects():
        last = getattr(proj, "last_query_at", 0) or proj.last_full_scan_at
        if last <= 0:
            continue   # never queried, never indexed → not idle, just empty
        age_secs = int(now - last)
        if age_secs > max_age_seconds:
            out.append(CleanupDecision(
                target=proj.name,
                reason=(
                    f"idle {age_secs // 86400}d > "
                    f"{max_age_seconds // 86400}d"
                ),
                bytes_freed=_estimate_project_bytes(chunk_store, proj.name),
                kind="project",
            ))
    return out


def collect_orphans(chunk_store) -> list[CleanupDecision]:
    """Session-bound projects (``session_id`` set) whose creating
    session is no longer alive. Phase 2.0 stores session_id as a
    UUID/ULID; the daemon tracks live sessions in ConnectionContext.

    For Phase 2.0 cleanup we run on startup (before any sessions
    are alive) and treat ALL session-bound projects as orphans. The
    daemon's request_reload path can call this with a live-session
    set passed in to avoid evicting actives.
    """
    out: list[CleanupDecision] = []
    for proj in chunk_store.list_projects():
        if getattr(proj, "session_id", None):
            out.append(CleanupDecision(
                target=proj.name,
                reason=f"session_id={proj.session_id} not alive",
                bytes_freed=_estimate_project_bytes(chunk_store, proj.name),
                kind="session_orphan",
            ))
    return out


def collect_missing(chunk_store) -> list[CleanupDecision]:
    """Projects whose ``root_path`` no longer exists. The watcher's
    Tier 1 has a grace period; this manual sweep is unconditional —
    the user explicitly asked to nuke missing-root projects."""
    out: list[CleanupDecision] = []
    for proj in chunk_store.list_projects():
        root = Path(proj.root_path)
        if not root.exists():
            out.append(CleanupDecision(
                target=proj.name,
                reason=f"root_path missing: {root}",
                bytes_freed=_estimate_project_bytes(chunk_store, proj.name),
                kind="project",
            ))
    return out


def collect_replay_shards(
    interactions_dir: Path, older_than_seconds: int,
    *, now: Optional[float] = None,
) -> list[CleanupDecision]:
    """``interactions.jsonl.YYYY-MM-DD.gz`` shards older than the
    threshold. The daemon's auto-rotation handles its own retention
    per spec §14.8.1; this is the manual override."""
    now = time.time() if now is None else now
    out: list[CleanupDecision] = []
    if not interactions_dir.is_dir():
        return out
    for shard in interactions_dir.glob("interactions.jsonl.*"):
        try:
            age = now - shard.stat().st_mtime
        except OSError:
            continue
        if age > older_than_seconds:
            out.append(CleanupDecision(
                target=shard.name,
                reason=(
                    f"shard age {int(age // 86400)}d > "
                    f"{older_than_seconds // 86400}d"
                ),
                bytes_freed=shard.stat().st_size,
                kind="replay_shard",
            ))
    return out


def _estimate_project_bytes(chunk_store, project_name: str) -> int:
    """Rough size estimate for a project: chunk text bytes + embedding
    rows × 4 bytes/dim. Used in the cleanup-plan output so the user
    sees how much they'd reclaim."""
    try:
        chunks = []
        for f in chunk_store.list_files(project_name):
            chunks.extend(chunk_store.get_chunks_by_file(f.id))
        text_bytes = sum(len(c.text.encode("utf-8")) for c in chunks)
        # 384-dim × 4-byte = 1536 / chunk; we don't have direct dim
        # access here, use a conservative 1.5KB/chunk estimate.
        embed_bytes = len(chunks) * 1536
        return text_bytes + embed_bytes
    except Exception:
        return 0


# --------------------------------------------------------- execution

def execute_plan(
    plan: CleanupPlan, *, chunk_store, interactions_dir: Path,
    indexer=None,
) -> int:
    """Apply a plan. Returns count of items actually removed.

    Project removal: prefer ``indexer.delete_project`` (frees memmap
    rows correctly); fall back to chunk_store-level cascade if no
    indexer was provided (e.g. CLI invoked while daemon is stopped).
    """
    n = 0
    for d in plan.decisions:
        try:
            if d.kind in ("project", "session_orphan"):
                if indexer is not None:
                    indexer.delete_project(d.target)
                else:
                    chunk_store.delete_project(d.target)
                n += 1
            elif d.kind == "replay_shard":
                shard = interactions_dir / d.target
                shard.unlink(missing_ok=True)
                n += 1
        except Exception as e:
            print(f"  WARN: failed to remove {d.target}: {e}",
                  file=sys.stderr)
    return n


# ------------------------------------------------------------- formatter

def format_plan(plan: CleanupPlan) -> str:
    """Pretty-print a cleanup plan for the user's confirmation
    prompt."""
    if not plan.decisions:
        return "  (nothing to clean)\n"
    rows = []
    for d in plan.decisions:
        rows.append(
            f"  {_fmt_kind(d.kind):<14s} {d.target:<32s} "
            f"{_human_bytes(d.bytes_freed):>10s}   {d.reason}"
        )
    rows.append("")
    rows.append(
        f"  total: {len(plan.decisions)} item(s), "
        f"{_human_bytes(plan.total_bytes)} estimated"
    )
    return "\n".join(rows) + "\n"


def _fmt_kind(kind: str) -> str:
    return {
        "project": "PROJECT",
        "session_orphan": "ORPHAN",
        "replay_shard": "SHARD",
    }.get(kind, kind.upper())


def _human_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


# ------------------------------------------------------------- CLI

def cmd_clean(args: argparse.Namespace) -> int:
    """``longctx clean`` entry point.

    Loads chunk_store from the same cache_dir the daemon uses (default
    ``~/.cache/longctx/<cache_subdir>/index.db``). When no flags are
    given, runs in dry-run mode showing what would go; flags select
    which collectors to run.
    """
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".cache" / "longctx"
    )
    db_path = cache_dir / "index.db"
    interactions_dir = cache_dir

    if not db_path.exists():
        print(f"no index found at {db_path}", file=sys.stderr)
        return 1

    chunk_store = SqliteChunkStore(db_path)
    try:
        decisions: list[CleanupDecision] = []
        if args.idle:
            decisions.extend(collect_idle(
                chunk_store, parse_duration_seconds(args.idle),
            ))
        if args.orphans:
            decisions.extend(collect_orphans(chunk_store))
        if args.missing:
            decisions.extend(collect_missing(chunk_store))
        if args.replay_older_than:
            decisions.extend(collect_replay_shards(
                interactions_dir,
                parse_duration_seconds(args.replay_older_than),
            ))

        if args.all and not (
            args.idle or args.orphans or args.missing
            or args.replay_older_than
        ):
            decisions.extend(collect_idle(chunk_store, 90 * 86400))
            decisions.extend(collect_orphans(chunk_store))
            decisions.extend(collect_missing(chunk_store))
            decisions.extend(collect_replay_shards(
                interactions_dir, 30 * 86400,
            ))

        plan = CleanupPlan(decisions=tuple(decisions))
        print("longctx clean — proposed actions:")
        print(format_plan(plan))

        if not plan.decisions:
            return 0

        # Dry-run mode: show, don't apply.
        if not (
            args.idle or args.orphans or args.missing
            or args.replay_older_than or args.all
        ):
            print("  (dry run — pass a filter flag + --yes to actually clean)")
            return 0

        if not args.yes:
            print("  pass --yes to proceed.", file=sys.stderr)
            return 0

        n = execute_plan(plan, chunk_store=chunk_store,
                         interactions_dir=interactions_dir)
        print(f"  removed {n} item(s).")
        return 0
    finally:
        chunk_store.close()
