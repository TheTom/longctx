"""Parent-directory watcher for live project add/remove (Phase 2.1, PRD §3.7).

Shallow watcher over ``parent_dir`` (typically ``~/dev``). Detects when
sentinel files appear/disappear in IMMEDIATE subdirectories, then fires
``on_project_added`` / ``on_project_removed`` callbacks. The DEEP watcher
(per-project, file-level, owned by Agent G) handles file events INSIDE
projects.

Why two watchers? Different lifetimes + different cost profiles:

* Parent watcher is one ``watchfiles.awatch`` over a small directory
  list — cheap, always on.
* Deep watchers are per-project, expensive (FSEvents/inotify slots),
  and pause when projects go cold (PRD §5.6, Tier 2).

Coalescing semantics (PRD §3.7 + §5.3):

* Events on the same path inside a debounce window collapse to one
  state (final ``exists/sentinel?`` check at debounce-fire time).
* "Project sentinel disappeared" is treated identically to "directory
  removed" — both fire ``on_project_removed``. The user's perspective:
  the project is no longer real.
* "New directory created with sentinel" → exactly one
  ``on_project_added``. Subsequent file activity inside is the deep
  watcher's problem, not ours.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence

from longctx_daemon.discovery import (
    DiscoveredProject,
    discover_projects,
    is_project_root,
)

logger = logging.getLogger(__name__)


# A small watchfiles import shim. We need the module for the live
# watcher; import at use-site so unit tests that don't exercise the
# full async loop don't pay for it.
try:
    from watchfiles import Change, awatch  # type: ignore
    HAS_WATCHFILES = True
except ImportError:  # pragma: no cover — watchfiles is a Phase 1 dep
    HAS_WATCHFILES = False


AddCallback = Callable[[DiscoveredProject], None]
RemoveCallback = Callable[[str], None]


class ParentDirWatcher:
    """Monitor ``parent_dir`` for new/removed project sentinels.

    Async — designed to run inside the daemon's main asyncio loop. Caller
    creates one instance per parent dir, kicks off ``run`` as a task,
    calls ``stop`` on shutdown.

    Args:
        parent_dir: directory to watch (typically ``corpus.parent_dir``).
        sentinels: priority-ordered list of sentinel filenames.
        on_project_added: called with the ``DiscoveredProject`` when a
            new sentinel-marked subdirectory appears.
        on_project_removed: called with the project name when a
            previously-tracked subdirectory loses its sentinel or is
            removed entirely.
        debounce_ms: events on the same project inside this window
            coalesce; we re-check the final state after the window
            fires.
        forbidden_dirs: project names refused per PRD §13.3.

    Lifecycle:
        watcher = ParentDirWatcher(...)
        task = asyncio.create_task(watcher.run())
        # ... later ...
        watcher.stop()
        await task
    """

    def __init__(
        self,
        parent_dir: Path,
        sentinels: Sequence[str],
        on_project_added: AddCallback,
        on_project_removed: RemoveCallback,
        *,
        debounce_ms: int = 500,
        forbidden_dirs: frozenset[str] = frozenset(),
        exclude: Sequence[str] = (),
    ) -> None:
        self._parent_dir = Path(parent_dir).expanduser().resolve()
        self._sentinels = tuple(sentinels)
        self._on_added = on_project_added
        self._on_removed = on_project_removed
        self._debounce_ms = max(0, int(debounce_ms))
        self._forbidden_dirs = frozenset(forbidden_dirs)
        self._exclude = tuple(exclude)
        self._stop = asyncio.Event()
        # Track currently-known projects to compute add/remove deltas.
        # Key = project name (immediate-subdir basename); value =
        # DiscoveredProject. Re-populated on first scan + maintained
        # incrementally.
        self._known: dict[str, DiscoveredProject] = {}

    # --------------------------------------------------------- public API

    async def run(self) -> None:
        """Run the watcher until ``stop`` is called.

        On entry:
            * Compute the initial project set and treat anything new
              as added (so the daemon picks up projects that appeared
              while it was offline).
        On loop:
            * Stream filesystem events from watchfiles.
            * Filter to events whose path is at depth ≤ 2 from
              parent_dir (immediate subdir or one of its sentinels).
            * Coalesce by project basename within ``debounce_ms``.
            * Re-evaluate ``is_project_root`` at debounce-fire time;
              fire add/remove deltas.
        """
        if not HAS_WATCHFILES:
            logger.error(
                "parent_watcher: watchfiles not installed; live add/remove "
                "is disabled. Add `watchfiles` to deps."
            )
            return

        if not self._parent_dir.is_dir():
            logger.warning(
                "parent_watcher: parent_dir does not exist: %s. Will not run.",
                self._parent_dir,
            )
            return

        # Seed known set from initial scan
        await self._initial_scan()

        try:
            async for changes in awatch(
                str(self._parent_dir),
                recursive=True,
                stop_event=self._stop,
                debounce=self._debounce_ms,
                step=max(50, self._debounce_ms // 4),
            ):
                affected = self._affected_basenames(changes)
                for basename in affected:
                    self._reconcile_basename(basename)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("parent_watcher: loop error: %s", exc, exc_info=True)

    def stop(self) -> None:
        """Signal the watch loop to exit. Idempotent."""
        self._stop.set()

    @property
    def known_projects(self) -> tuple[DiscoveredProject, ...]:
        """Snapshot of the currently-tracked projects.

        Useful for the daemon's startup path (compare against the
        SQLite ``projects`` table to compute initial diff). Returned
        in a stable name-sorted order.
        """
        return tuple(sorted(self._known.values(), key=lambda p: p.name))

    # --------------------------------------------------------- internals

    async def _initial_scan(self) -> None:
        """Populate ``_known`` and fire ``on_project_added`` for each.

        Runs once at start. Subsequent reconciles are delta-driven so
        we don't double-fire.
        """
        try:
            initial = discover_projects(
                self._parent_dir,
                sentinels=self._sentinels,
                exclude=self._exclude,
                forbidden_dirs=self._forbidden_dirs,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("parent_watcher: initial scan failed: %s", exc)
            return
        for proj in initial:
            self._known[proj.name] = proj
            try:
                self._on_added(proj)
            except Exception as exc:  # pragma: no cover - callback bug
                logger.error(
                    "parent_watcher: on_project_added raised for %s: %s",
                    proj.name, exc, exc_info=True,
                )

    def _affected_basenames(
        self, changes: "set[tuple[Change, str]]",
    ) -> set[str]:
        """Map raw watchfiles events to immediate-subdir basenames
        relative to ``parent_dir``.

        We only care about events at depth 1 (the subdir itself was
        added/removed) or depth 2 (a sentinel file inside the subdir
        was added/removed). Deeper events belong to the per-project
        deep watcher; we ignore them so we don't duplicate work or
        spam reconcile cycles for every file-touch inside a project.
        """
        out: set[str] = set()
        parent = self._parent_dir
        for _change, raw_path in changes:
            try:
                p = Path(raw_path).resolve()
            except OSError:
                continue
            try:
                rel = p.relative_to(parent)
            except ValueError:
                # Outside parent_dir; ignore (some watchers emit
                # symlink-target paths).
                continue
            parts = rel.parts
            if not parts:
                continue
            basename = parts[0]
            if basename in self._forbidden_dirs:
                continue
            # Depth 1 = subdir itself created/removed.
            # Depth 2 = sentinel file beneath it.
            # > 2 = file inside project; not our problem.
            if len(parts) <= 2:
                out.add(basename)
        return out

    def _reconcile_basename(self, basename: str) -> None:
        """Recompute project state for one immediate subdir; fire
        callbacks on transitions.

        Four cases:
            previously known + still a project → no-op (sentinel
                inside changed but the project itself didn't transition)
            previously known + no longer a project → remove
            previously unknown + now a project → add
            previously unknown + still not → no-op (event for a
                non-project subdir, e.g. ``~/dev/scratch/``)
        """
        path = self._parent_dir / basename
        previously_known = basename in self._known

        # Excluded? Treat as "not a project" for state purposes.
        if basename in self._forbidden_dirs or _matches_exclude_simple(
            basename, path, self._exclude,
        ):
            sentinel = None
        else:
            try:
                sentinel = is_project_root(path, self._sentinels)
            except Exception as exc:  # pragma: no cover
                logger.debug("reconcile probe failed for %s: %s", path, exc)
                return

        if previously_known and sentinel is None:
            # Removal path
            self._known.pop(basename, None)
            try:
                self._on_removed(basename)
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "parent_watcher: on_project_removed raised for %s: %s",
                    basename, exc, exc_info=True,
                )
            return

        if not previously_known and sentinel is not None:
            # Addition path
            try:
                ignore_marker = (path / ".longctxignore").is_file()
            except OSError:
                ignore_marker = False
            proj = DiscoveredProject(
                name=basename,
                root_path=path.resolve(),
                sentinel=sentinel,
                has_longctxignore=ignore_marker,
            )
            self._known[basename] = proj
            try:
                self._on_added(proj)
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "parent_watcher: on_project_added raised for %s: %s",
                    basename, exc, exc_info=True,
                )
            return
        # else: state unchanged at the project granularity.


# --------------------------------------------------------------- helpers


def _matches_exclude_simple(
    basename: str, full_path: Path, patterns: Sequence[str],
) -> bool:
    """Trimmed-down version of ``discovery._matches_exclude`` for the
    hot reconcile path. Same semantics, just inlined to avoid an
    import cycle if discovery's module changes shape."""
    if not patterns:
        return False
    from fnmatch import fnmatch
    p = full_path.as_posix()
    for pat in patterns:
        if pat == basename:
            return True
        if "/" in pat:
            if fnmatch(p, pat):
                return True
        else:
            if fnmatch(basename, pat):
                return True
    return False


# --------------------------------------------------------------- testing hooks


async def run_with_external_changes(
    watcher: ParentDirWatcher,
    changes_iter: "Iterable[set[tuple[Change, str]]]",  # type: ignore[name-defined]
) -> None:
    """Test-only entrypoint: drive the watcher's reconcile path with
    pre-cooked event batches. Skips the watchfiles dependency so unit
    tests don't have to spin up real FS events.

    Each yielded batch is treated as one debounce window's worth of
    events; the watcher fires callbacks accordingly.
    """
    await watcher._initial_scan()
    for batch in changes_iter:
        affected = watcher._affected_basenames(batch)
        for basename in affected:
            watcher._reconcile_basename(basename)


# TODO(phase 2.2): respect Tier 2 watcher idle pause (PRD §5.6) — when
# a project goes cold, deep watcher unsubscribes; parent watcher stays
# on so we still see remove events.
# TODO(phase 2.4): on Linux, watch the parent only (depth=1) via
# ``watch_filter`` to avoid inotify slot exhaustion at scale; current
# recursive=True is fine for ~50 sub-projects.

__all__ = [
    "ParentDirWatcher",
    "AddCallback",
    "RemoveCallback",
    "run_with_external_changes",
]
