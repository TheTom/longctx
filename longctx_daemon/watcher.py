"""File watcher + auto-quiescence + cleanup tiers for the longctx daemon.

Phase 2.2 component. Watches each registered project's root recursively
with ``watchfiles`` (Rust-backed, atomic-rename aware), debounces the
event stream into batched indexer calls, runs a periodic ``mtime`` sweep
as a "watcher-missed-it" fallback, and piggybacks two cleanup duties on
that sweep:

* **Tier 1 — auto-drop missing root** (PRD §12.4.1): if a project's
  ``root_path`` has been gone for ``cleanup.missing_root_grace_days``
  consecutive observations (default 7d), call
  :py:meth:`Indexer.delete_project`. Conservative-by-default: a USB
  unmount or ``git worktree remove`` shouldn't nuke real data.
* **Tier 2 — cold-project pause** (PRD §12.4.2): unsubscribe watches
  for projects not queried for ``watcher_idle_pause_days`` (default 30)
  AND with no recent file events. The mtime sweep still catches their
  changes; the next ``search`` against the project pays a one-time
  walk + auto-resume.

The watcher also exposes the :py:class:`Searcher`'s
``quiescence_check_callback`` hook so the headline invariant in PRD §6.5
holds — a ``search`` call that arrives mid-batch blocks (up to a
caller-supplied timeout) until the watcher's queue drains, so the agent
sees the filesystem state from its last write.

Design notes:

* ``asyncio`` throughout — one event loop drives all per-project
  ``watchfiles.awatch`` tasks, the debounce/batch worker, and the
  periodic mtime sweep + cleanup loop.
* The debounce coalesces duplicate events on the same path within the
  debounce window, and converts a rapid DELETE+CREATE on the same path
  (vim-style atomic rename) into a single MODIFY so we don't waste an
  embed pass.
* Storm guard: if a project's queue exceeds ``queue_size`` (default
  10000), we flip it into ``rescan-pending`` mode. The worker discards
  the queued events and runs an mtime walk against the DB, scheduling
  fresh updates. Avoids OOM during ``git pull`` / branch swaps.
* ``last_query_at`` lives in-memory here (Tier 2 only needs it for
  watcher policy). Searcher / daemon glue calls
  :py:meth:`Watcher.note_query` so storage doesn't have to evolve.

The whole module is structured so the daemon imports it lazily and
degrades gracefully if ``watchfiles`` isn't installed (tests can drive
the worker directly via :py:meth:`Watcher._enqueue_event`).

TODO(phase 2.3): wire Tier 3 disk-budget eviction into the same
periodic loop (PRD §12.4.3). Out of scope for 2.2.
TODO(phase 2.3): emit watcher metrics (queue depth, dropped events,
storm flips) into the §14 logging schema.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from longctx_daemon.storage.protocol import ChunkStore
from longctx_daemon.types import Project

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- constants

# Mirror the §5.1 default debounce so spec callers don't have to remember it.
DEFAULT_DEBOUNCE_MS = 200

# Max events per project before we flip into rescan-pending. Tuned by
# spec §5.3; revisit if real-world ``git pull`` traces blow past this.
DEFAULT_QUEUE_SIZE = 10_000


# ------------------------------------------------------------------ events

class EventType(str, Enum):
    """Coalesced event types fed into the indexer worker.

    These are post-debounce — the raw ``watchfiles`` Change values are
    folded down into this set. RENAME is synthesized when a DELETE+CREATE
    pair within the debounce window targets a different basename; same-
    basename pairs become MODIFY (vim atomic-rename).
    """

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"  # carries old_rel_path + new_rel_path


@dataclass(frozen=True)
class WatcherEvent:
    """One post-debounce event ready for the indexer.

    For RENAME, ``old_path`` is the source path and ``path`` is the
    destination — the worker calls ``indexer.rename_file`` with both.
    For all other types ``old_path`` is None.
    """

    project: str
    type: EventType
    path: str  # absolute path
    old_path: Optional[str] = None
    # Wall-clock when the event was first observed; used by debounce.
    observed_at: float = field(default_factory=time.monotonic)


# ------------------------------------------------------------------- config

@dataclass(frozen=True)
class WatcherConfig:
    """Tunables for the watcher + cleanup loops.

    Defaults track PRD §5 + §12.4. Frozen so tests can pass alternate
    configs by construction without surprising in-place mutations.

    Attributes:
        debounce_ms: max age of a coalesce window. Smaller = lower
            latency to first commit, larger = bigger batches per pass.
            200ms is the spec's "agent-pace" sweet spot.
        queue_size: per-project soft cap before storm-mode kicks in.
        periodic_mtime_sweep_seconds: how often to mtime-walk every
            project (§5.5). 30s is a good fallback cadence — quick
            enough to catch missed events before the next agent round,
            cheap enough not to thrash an SSD.
        watcher_idle_pause_days: Tier 2 threshold (§12.4.2).
        missing_root_grace_days: Tier 1 grace window (§12.4.1).
        periodic_check_seconds: cadence for the cleanup checks
            piggybacked on the sweep loop. 1h matches §12.4.1.
        cleanup_jitter_pct: spread the cleanup tick by ±N% so multi-
            daemon installs don't synchronize. Phase 2.2 leaves this at
            0; the hook is here for ops to tune later.
    """

    debounce_ms: int = DEFAULT_DEBOUNCE_MS
    queue_size: int = DEFAULT_QUEUE_SIZE
    periodic_mtime_sweep_seconds: int = 30
    watcher_idle_pause_days: int = 30
    missing_root_grace_days: int = 7
    periodic_check_seconds: int = 3600
    cleanup_jitter_pct: float = 0.0


# ------------------------------------------------------------------ helpers

def _utc_now_iso() -> str:
    """ISO-8601 UTC with trailing Z — matches §6.3 freshness format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now_seconds() -> float:
    """Wall-clock seconds. Tests monkeypatch ``time.time`` to advance
    the clock; we call ``time.time`` (not ``time.monotonic``) so cleanup
    decisions track real-world calendar age."""
    return time.time()


# ---------------------------------------------------- per-project state


@dataclass
class _ProjectState:
    """Live watcher state for one project. Mutable; the run loop owns it.

    ``rescan_pending`` flips on storm overflow. ``paused`` flips on Tier
    2 idle-time threshold. ``first_missing_at`` records when we first
    saw the root vanish so Tier 1 can compute a continuous-grace window.
    """

    name: str
    root_path: Path
    queue: deque[WatcherEvent] = field(default_factory=deque)
    last_event_at: float = 0.0
    last_query_at: float = field(default_factory=_now_seconds)
    paused: bool = False
    rescan_pending: bool = False
    # Tier 1 grace tracking; reset to None when root reappears.
    first_missing_at: Optional[float] = None
    # ``watchfiles.awatch`` task — None when paused / removed / never started.
    watch_task: Optional[asyncio.Task[Any]] = None
    # Stop signal for the per-project watch task.
    stop_event: Optional[asyncio.Event] = None


# --------------------------------------------------------- main class


class Watcher:
    """Per-project recursive watcher + global periodic mtime sweep
    + cold-project pause + Tier 1 missing-root auto-drop.

    Lifecycle:

        watcher = Watcher(indexer, chunk_store, WatcherConfig())
        for proj in chunk_store.list_projects():
            watcher.add_project(proj)
        await watcher.run()    # blocks until .stop() is called

    The watcher does not own any threads; everything is asyncio. Where
    the indexer call is sync (CPU + SQLite write), we offload it to
    ``asyncio.to_thread`` so the event loop stays responsive.

    Test escape hatches (private but stable, used by `tests/daemon/`):

      * ``_enqueue_event`` — directly enqueue a synthetic event,
        bypassing ``watchfiles``. Lets unit tests drive the debounce
        worker without spinning up real filesystem activity.
      * ``_drain_now`` — force the worker to drain pending events
        immediately, ignoring the debounce window. Pairs with the above.
      * ``_mtime_sweep_once`` — run one sweep + cleanup tick without
        waiting on the periodic interval. Used by Tier 1/2 tests.
    """

    def __init__(
        self,
        indexer: Any,
        chunk_store: ChunkStore,
        config: WatcherConfig,
    ) -> None:
        """Wire dependencies.

        Args:
            indexer: a ``longctx_daemon.indexer.Indexer``-compatible
                object. We call ``update_file``, ``delete_file``,
                ``delete_project``, and (if present) ``rename_file`` /
                ``update_files``. Duck-typed so test doubles work.
            chunk_store: persistent project + file metadata. Used by
                the mtime sweep and Tier 2 idle detection.
            config: tunables. See :class:`WatcherConfig`.
        """
        self._indexer = indexer
        self._store = chunk_store
        self._cfg = config

        self._states: dict[str, _ProjectState] = {}
        # Guards _states for cross-thread reads from sync API
        # (e.g., note_query called from the searcher worker thread).
        self._states_lock = threading.RLock()

        # asyncio.Event signaled whenever the queue grows or drains;
        # waiters block on this in wait_for_quiescence.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cond: Optional[asyncio.Condition] = None
        # Worker + sweep tasks; populated by run().
        self._worker_task: Optional[asyncio.Task[Any]] = None
        self._sweep_task: Optional[asyncio.Task[Any]] = None
        self._stop_flag: Optional[asyncio.Event] = None
        self._last_indexed_through: str = _utc_now_iso()
        # Wall-clock of the last cleanup tick; throttled by
        # cleanup_check_seconds, separate from the mtime sweep cadence.
        self._last_cleanup_at: float = 0.0
        # Lazy-checked watchfiles import; None until first run() call.
        self._watchfiles: Any = None

    # -------------------------------------------------- public registry

    def add_project(self, project: Project) -> None:
        """Begin watching ``project.root_path`` recursively.

        Idempotent — re-adding a name updates the root and resets
        Tier-1 / Tier-2 state. Safe to call before :py:meth:`run`; the
        actual ``watchfiles.awatch`` task is spawned on the next loop
        tick.
        """
        with self._states_lock:
            existing = self._states.get(project.name)
            if existing is not None:
                # Project root may have moved; reset grace tracking.
                existing.root_path = Path(project.root_path)
                existing.first_missing_at = None
                # If the project was paused / dropped, resume on re-add.
                existing.paused = False
                existing.rescan_pending = False
            else:
                self._states[project.name] = _ProjectState(
                    name=project.name,
                    root_path=Path(project.root_path),
                )

        # If the loop is already running, schedule a watch task.
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                self._spawn_watch_task_safe, project.name,
            )

    def remove_project(self, name: str) -> None:
        """Stop watching and forget all watcher-side state.

        Storage cleanup is the indexer's job (parent_watcher / Tier 1
        call ``indexer.delete_project`` separately).
        """
        with self._states_lock:
            state = self._states.pop(name, None)
        if state is not None:
            self._cancel_watch_task(state)

    def pause_project(self, name: str) -> None:
        """Tier 2: unsubscribe but keep the project's index intact.

        Idempotent; a paused project stays paused until either
        :py:meth:`resume_project` or a real query (via
        :py:meth:`note_query`) flips it back.
        """
        with self._states_lock:
            state = self._states.get(name)
            if state is None or state.paused:
                return
            state.paused = True
        self._cancel_watch_task(state)
        logger.info("watcher: paused cold project %s", name)

    def resume_project(self, name: str) -> None:
        """Re-subscribe after a paused project becomes hot again.

        Pays a one-time mtime walk before re-watching to catch any
        changes that happened while paused. Implementation reuses
        ``_mtime_walk_project`` so the catch-up logic is shared with
        the periodic sweep.
        """
        with self._states_lock:
            state = self._states.get(name)
            if state is None or not state.paused:
                return
            state.paused = False

        # Best-effort catch-up walk — schedule on the loop so we don't
        # block the caller (resume_project is sync, called from
        # note_query which itself runs on the searcher thread).
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                self._spawn_resume_task_safe, name,
            )
        logger.info("watcher: resumed project %s", name)

    def note_query(self, project: Optional[str]) -> None:
        """Update ``last_query_at`` for a project. Called by the
        searcher (or its glue layer) on every search call.

        ``None`` (fanout-no-primary) is a no-op — fanout searches don't
        single out one project. Auto-resumes a paused project.
        """
        if project is None:
            return
        with self._states_lock:
            state = self._states.get(project)
            if state is None:
                return
            state.last_query_at = _now_seconds()
            was_paused = state.paused
        if was_paused:
            self.resume_project(project)

    # -------------------------------------------------- top-level run

    async def run(self) -> None:
        """Top-level event loop. Returns when :py:meth:`stop` is called.

        Spawns:
          * one ``watchfiles.awatch`` task per project (lazy: skipped
            for paused projects)
          * the debounce / batch worker task
          * the periodic mtime sweep + cleanup task

        All sub-tasks are cancelled cooperatively on stop. Exceptions
        from any sub-task are logged at ERROR but do not bring down the
        run loop — the watcher is meant to be best-effort.
        """
        self._loop = asyncio.get_running_loop()
        self._cond = asyncio.Condition()
        self._stop_flag = asyncio.Event()

        # Lazy-import watchfiles so the daemon can boot even if the
        # native lib is broken; tests rely on this fallback too.
        try:
            import watchfiles as _wf

            self._watchfiles = _wf
        except ImportError as exc:
            logger.warning(
                "watchfiles not installed; watcher running in "
                "mtime-sweep-only mode (%s)", exc,
            )
            self._watchfiles = None

        # Boot per-project watch tasks.
        with self._states_lock:
            names = list(self._states.keys())
        for name in names:
            self._spawn_watch_task_safe(name)

        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="longctx-watcher-worker",
        )
        self._sweep_task = asyncio.create_task(
            self._sweep_loop(), name="longctx-watcher-sweep",
        )

        try:
            await self._stop_flag.wait()
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Signal the run loop to exit. Thread-safe.

        Cancels watch / worker / sweep tasks on the next tick.
        """
        if self._loop is None or self._stop_flag is None:
            return
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._stop_flag.set)

    async def _shutdown(self) -> None:
        """Cancel + await all sub-tasks. Idempotent."""
        # Cancel per-project watch tasks first so they stop producing
        # events while we drain the worker.
        with self._states_lock:
            states = list(self._states.values())
        for state in states:
            self._cancel_watch_task(state)

        for task in (self._worker_task, self._sweep_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # -------------------------------------------------- quiescence API

    def pending_updates(self, project: Optional[str] = None) -> int:
        """Sum of queued events across one or all projects."""
        with self._states_lock:
            if project is not None:
                state = self._states.get(project)
                return len(state.queue) if state is not None else 0
            return sum(len(s.queue) for s in self._states.values())

    def indexed_through_iso(self) -> str:
        """ISO-8601 UTC timestamp of the most recent committed batch.

        Updated by the worker after every drain. Phase 2.0 stub returned
        ``utc_now()``; the watcher's value is more honest because it
        reflects the LATEST drain, not the latest call to ``search``.
        """
        return self._last_indexed_through

    async def wait_for_quiescence(
        self,
        project: Optional[str] = None,
        timeout_ms: int = 2000,
    ) -> tuple[int, str]:
        """Block up to ``timeout_ms`` for the queue to drain.

        Returns ``(pending_updates_at_return, indexed_through_iso)``.
        Used by:
          * the searcher's auto-quiescence default
          * the ``wait_for_quiescence`` MCP tool

        Behaviors worth knowing about:
          * Already-empty queue returns immediately with the current
            ``indexed_through``.
          * Timeout returns the current pending count (caller decides
            whether to retry / warn the agent via §6.3 freshness).
          * Cancellation propagates — callers running under their own
            timeout cancel scopes get clean exits.
        """
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        # Fast-path: already quiet.
        pending = self.pending_updates(project)
        if pending == 0:
            return 0, self._last_indexed_through

        if self._cond is None:
            # run() hasn't started — nothing to wait on.
            return pending, self._last_indexed_through

        async with self._cond:
            while True:
                pending = self.pending_updates(project)
                if pending == 0:
                    return 0, self._last_indexed_through
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return pending, self._last_indexed_through
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pending = self.pending_updates(project)
                    return pending, self._last_indexed_through

    # ------------------------------------- per-project watch task plumbing

    def _spawn_watch_task_safe(self, name: str) -> None:
        """Loop-thread-safe wrapper. Idempotent — refuses to spawn a
        task when the project is paused or already watched, or when
        ``watchfiles`` is unavailable."""
        if self._loop is None or self._watchfiles is None:
            return
        with self._states_lock:
            state = self._states.get(name)
            if state is None or state.paused:
                return
            if state.watch_task is not None and not state.watch_task.done():
                return
            state.stop_event = asyncio.Event()
            state.watch_task = asyncio.create_task(
                self._watch_one_project(state),
                name=f"longctx-watch-{name}",
            )

    def _spawn_resume_task_safe(self, name: str) -> None:
        """Schedule the resume's catch-up walk + re-spawn watch."""
        if self._loop is None:
            return

        async def _resume() -> None:
            try:
                with self._states_lock:
                    state = self._states.get(name)
                if state is not None:
                    await asyncio.to_thread(self._mtime_walk_project, state)
            except Exception:  # noqa: BLE001
                logger.exception("watcher resume catch-up failed for %s", name)
            self._spawn_watch_task_safe(name)

        asyncio.create_task(_resume(), name=f"longctx-resume-{name}")

    def _cancel_watch_task(self, state: _ProjectState) -> None:
        """Stop a per-project watch task. Idempotent."""
        task = state.watch_task
        ev = state.stop_event
        if ev is not None:
            ev.set()
        if task is not None:
            task.cancel()
        state.watch_task = None
        state.stop_event = None

    async def _watch_one_project(self, state: _ProjectState) -> None:
        """Drain ``watchfiles.awatch`` into the per-project queue.

        ``watchfiles.Change`` values map to:
          * Change.added   → CREATE
          * Change.modified → MODIFY
          * Change.deleted → DELETE

        Atomic-rename is detected later, in the worker, where we have
        the full debounce window.

        Errors from the native watcher are logged but don't bring the
        loop down; we re-enter ``awatch`` after a short delay so a
        transient unmount doesn't kill the watcher permanently.
        """
        wf = self._watchfiles
        if wf is None:
            return
        backoff = 1.0
        while True:
            if state.stop_event is not None and state.stop_event.is_set():
                return
            try:
                async for changes in wf.awatch(
                    str(state.root_path),
                    stop_event=state.stop_event,
                    recursive=True,
                ):
                    for change, raw_path in changes:
                        ev_type = self._map_change(change)
                        if ev_type is None:
                            continue
                        self._enqueue_event(WatcherEvent(
                            project=state.name,
                            type=ev_type,
                            path=str(raw_path),
                        ))
                    backoff = 1.0  # successful run resets backoff
            except FileNotFoundError:
                # Root vanished — Tier 1 sweep handles this. Pause this
                # task until run() re-spawns us (e.g., after add_project).
                logger.info(
                    "watcher: root vanished for %s, suspending task",
                    state.name,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("watchfiles error for %s; backing off", state.name)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _map_change(self, change: Any) -> Optional[EventType]:
        """Translate a ``watchfiles.Change`` enum to our EventType.

        Returns None for unrecognized values so a future watchfiles
        version that adds new change kinds doesn't crash us.
        """
        wf = self._watchfiles
        if wf is None:
            return None
        if change == wf.Change.added:
            return EventType.CREATE
        if change == wf.Change.modified:
            return EventType.MODIFY
        if change == wf.Change.deleted:
            return EventType.DELETE
        return None

    # ------------------------------------------- enqueue + storm guard

    def _enqueue_event(self, event: WatcherEvent) -> None:
        """Push one event onto its project's debounce queue.

        Called both from the per-project watcher (real fs events) and
        from tests (synthetic events). Trips storm-mode if the queue
        exceeds ``queue_size``. Once flipped, additional events are
        dropped (the rescan walk will catch them) until the worker
        clears the flag during drain.
        """
        with self._states_lock:
            state = self._states.get(event.project)
            if state is None:
                return
            if state.rescan_pending:
                # Storm-mode is in effect — let the rescan walk handle
                # the truth on disk. Updating last_event_at keeps the
                # cold-pause logic from misclassifying us as quiet.
                state.last_event_at = time.monotonic()
                return
            state.queue.append(event)
            state.last_event_at = time.monotonic()
            if len(state.queue) > self._cfg.queue_size:
                # Storm: drop events, mark for full rescan.
                state.queue.clear()
                state.rescan_pending = True
                logger.warning(
                    "watcher: queue overflow for %s (>%d events); "
                    "flipping to rescan-pending",
                    state.name, self._cfg.queue_size,
                )

        # Notify quiescence waiters that something is queued.
        self._notify_cond()

    def _notify_cond(self) -> None:
        """Wake up wait_for_quiescence waiters. Safe across threads."""
        if self._cond is None or self._loop is None:
            return
        if self._loop.is_closed():
            return

        async def _notify() -> None:
            assert self._cond is not None
            async with self._cond:
                self._cond.notify_all()

        try:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_notify()),
            )
        except RuntimeError:
            # Loop is shutting down. Best effort.
            pass

    # ----------------------------------------------------- worker loop

    async def _worker_loop(self) -> None:
        """Drain all per-project queues whenever any debounce expires.

        Sleeps until the earliest ``observed_at + debounce`` across all
        projects, then drains every project that's ready. Drain runs in
        a worker thread (``asyncio.to_thread``) so the indexer's SQLite
        + embed work doesn't block the event loop.
        """
        debounce = self._cfg.debounce_ms / 1000.0
        while True:
            try:
                # Find earliest deadline.
                with self._states_lock:
                    deadlines: list[float] = []
                    for state in self._states.values():
                        if not state.queue and not state.rescan_pending:
                            continue
                        if state.queue:
                            deadlines.append(
                                state.queue[0].observed_at + debounce,
                            )
                        else:
                            # rescan_pending — drain immediately
                            deadlines.append(time.monotonic())

                if not deadlines:
                    # Idle — wait for any new event to arrive.
                    if self._cond is not None:
                        async with self._cond:
                            await self._cond.wait()
                    else:
                        await asyncio.sleep(0.05)
                    continue

                wait = max(0.0, min(deadlines) - time.monotonic())
                if wait > 0:
                    try:
                        if self._cond is not None:
                            async with self._cond:
                                await asyncio.wait_for(
                                    self._cond.wait(), timeout=wait,
                                )
                        else:
                            await asyncio.sleep(wait)
                    except asyncio.TimeoutError:
                        pass

                await self._drain_now()
            except asyncio.CancelledError:
                # On shutdown, drain one more time to flush in-flight
                # events. Best effort — swallow exceptions here.
                try:
                    await self._drain_now()
                except Exception:  # noqa: BLE001
                    logger.exception("watcher final drain failed")
                raise
            except Exception:  # noqa: BLE001
                logger.exception("watcher worker loop iteration failed")
                # Avoid hot-looping on persistent failures.
                await asyncio.sleep(0.1)

    async def _drain_now(self) -> None:
        """Drain every project whose debounce has expired (or that's
        flagged rescan_pending). Runs the per-project drain off-loop."""
        debounce = self._cfg.debounce_ms / 1000.0
        now = time.monotonic()

        ready: list[tuple[_ProjectState, list[WatcherEvent], bool]] = []
        with self._states_lock:
            for state in self._states.values():
                rescan = state.rescan_pending
                events: list[WatcherEvent] = []
                if state.queue and (state.queue[0].observed_at + debounce <= now):
                    events = list(state.queue)
                    state.queue.clear()
                if events or rescan:
                    state.rescan_pending = False
                    ready.append((state, events, rescan))

        for state, events, rescan in ready:
            if rescan:
                # Storm path: ignore per-event queue, walk the project.
                try:
                    await asyncio.to_thread(self._mtime_walk_project, state)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "watcher rescan-pending walk failed for %s", state.name,
                    )
                continue

            coalesced = self._coalesce_events(events)
            if not coalesced:
                continue
            try:
                await asyncio.to_thread(
                    self._apply_events, state.name, coalesced,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "watcher apply_events failed for %s", state.name,
                )

        if ready:
            self._last_indexed_through = _utc_now_iso()
            self._notify_cond()

    def _coalesce_events(
        self, events: list[WatcherEvent],
    ) -> list[WatcherEvent]:
        """Collapse duplicate/atomic-rename events into one event per path.

        Rules:

        * Same-path events: the LAST event wins (CREATE→MODIFY→DELETE
          ends as DELETE; CREATE→MODIFY ends as the latest MODIFY).
        * DELETE on path A followed by CREATE on path B inside the
          window → RENAME (preserves embeddings, just updates rel_path).
        * DELETE+CREATE on the same path → MODIFY (vim atomic-rename).

        Order-preserving for the deduped set.
        """
        # Phase 1: per-path deduplication, last-event-wins.
        per_path: dict[str, WatcherEvent] = {}
        order: list[str] = []
        for ev in events:
            key = ev.path
            if key not in per_path:
                order.append(key)
            per_path[key] = ev

        # Phase 2: detect vim-style atomic rename — a DELETE followed
        # by a CREATE on the SAME path within the debounce window. The
        # last-event-wins pass above already collapsed CREATE→DELETE
        # to a single DELETE; we only want to upgrade DELETE→CREATE
        # back into a MODIFY (the file's still there).
        first_index: dict[str, int] = {}
        for i, ev in enumerate(events):
            first_index.setdefault((ev.path, ev.type), i)
        for path, surviving in list(per_path.items()):
            if surviving.type != EventType.CREATE:
                continue
            del_idx = first_index.get((path, EventType.DELETE))
            cre_idx = first_index.get((path, EventType.CREATE))
            if del_idx is not None and cre_idx is not None and del_idx < cre_idx:
                per_path[path] = WatcherEvent(
                    project=surviving.project,
                    type=EventType.MODIFY,
                    path=path,
                )

        # Phase 3: pair distinct DELETE/CREATE → RENAME.
        # We only fold a rename if the file CREATE is on a DIFFERENT
        # path than any surviving DELETE — otherwise we'd lose the
        # delete signal. Keep it simple: if the batch has exactly one
        # DELETE and one CREATE on different paths, treat it as a
        # rename. This is the common refactor case (``git mv``); more
        # complex moves (one CREATE, many DELETEs) fall through to
        # individual events.
        deletes = [
            ev for k, ev in per_path.items() if ev.type == EventType.DELETE
        ]
        creates = [
            ev for k, ev in per_path.items() if ev.type == EventType.CREATE
        ]
        if (
            len(deletes) == 1
            and len(creates) == 1
            and deletes[0].path != creates[0].path
        ):
            old = deletes[0]
            new = creates[0]
            rename_event = WatcherEvent(
                project=old.project,
                type=EventType.RENAME,
                path=new.path,
                old_path=old.path,
            )
            del per_path[old.path]
            per_path[new.path] = rename_event

        # Preserve insertion order so tests can assert deterministic dispatch.
        return [per_path[k] for k in order if k in per_path]

    # ----------------------------------------------------- indexer dispatch

    def _apply_events(
        self,
        project_name: str,
        events: list[WatcherEvent],
    ) -> None:
        """Sync entry point for the worker thread.

        Each event maps to one indexer call. We prefer ``update_files``
        (bulk path) when the indexer exposes it; otherwise fall back to
        per-file calls. Errors are caught + logged per event so one
        bad file doesn't sink an entire batch.
        """
        # Bulk path: collect all CREATE/MODIFY into one call when
        # supported. RENAME / DELETE always go through the per-call path
        # because they don't share the embed batch.
        with self._states_lock:
            state = self._states.get(project_name)
            if state is None:
                return
            root = state.root_path

        bulk_paths: list[Path] = []
        per_event_tail: list[WatcherEvent] = []
        for ev in events:
            if ev.type in (EventType.CREATE, EventType.MODIFY):
                try:
                    p = Path(ev.path)
                    if p.is_absolute():
                        bulk_paths.append(p)
                    else:
                        bulk_paths.append((root / ev.path).resolve())
                except Exception:  # noqa: BLE001
                    logger.exception("bad event path %s", ev.path)
            else:
                per_event_tail.append(ev)

        update_files = getattr(self._indexer, "update_files", None)
        if callable(update_files) and bulk_paths:
            try:
                update_files(project_name, bulk_paths)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "update_files failed for %s; falling back to per-file",
                    project_name,
                )
                for path in bulk_paths:
                    self._safe_update_one(project_name, path)
        else:
            for path in bulk_paths:
                self._safe_update_one(project_name, path)

        for ev in per_event_tail:
            try:
                if ev.type == EventType.DELETE:
                    rel = self._to_rel(root, ev.path)
                    if rel is not None:
                        self._indexer.delete_file(project_name, rel)
                elif ev.type == EventType.RENAME:
                    rel_old = self._to_rel(root, ev.old_path or "")
                    rel_new = self._to_rel(root, ev.path)
                    if rel_old is None or rel_new is None:
                        continue
                    rename = getattr(self._indexer, "rename_file", None)
                    if callable(rename):
                        rename(project_name, rel_old, rel_new)
                    else:
                        # Conservative fallback: drop+reindex preserves
                        # correctness even without the cheap path.
                        self._indexer.delete_file(project_name, rel_old)
                        self._safe_update_one(
                            project_name, (root / rel_new).resolve(),
                        )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "watcher apply event %s failed", ev,
                )

    def _safe_update_one(self, project_name: str, path: Path) -> None:
        try:
            self._indexer.update_file(project_name, path)
        except Exception:  # noqa: BLE001
            logger.exception(
                "watcher update_file failed for %s in %s", path, project_name,
            )

    def _to_rel(self, root: Path, abs_path: str) -> Optional[str]:
        """Convert an event path to a project-rel path (POSIX)."""
        try:
            p = Path(abs_path)
            if not p.is_absolute():
                p = (root / p).resolve()
            return p.relative_to(root).as_posix()
        except (ValueError, OSError):
            return None

    # ---------------------------------------- periodic mtime + cleanup

    async def _sweep_loop(self) -> None:
        """Periodic mtime sweep + Tier 1/2 cleanup ticker.

        Two cadences:

        * mtime sweep every ``periodic_mtime_sweep_seconds``
        * cleanup tick every ``periodic_check_seconds`` (~hourly by
          default; piggybacked on the same task to avoid two timers)
        """
        interval = max(1, self._cfg.periodic_mtime_sweep_seconds)
        while True:
            try:
                await asyncio.sleep(interval)
                await self._mtime_sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("watcher sweep iteration failed")

    async def _mtime_sweep_once(self) -> None:
        """Run one sweep + cleanup tick.

        Public for tests. Walks every project's file table, schedules
        update / delete events for any drift, runs Tier 1 (missing root
        grace) and Tier 2 (cold pause) checks if the cleanup interval
        has elapsed.
        """
        with self._states_lock:
            states = list(self._states.values())
        for state in states:
            if state.paused:
                # Cold pause skips fs walk; cleanup tick still runs.
                continue
            try:
                await asyncio.to_thread(self._mtime_walk_project, state)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "mtime walk failed for %s", state.name,
                )

        now = _now_seconds()
        if now - self._last_cleanup_at >= self._cfg.periodic_check_seconds:
            self._last_cleanup_at = now
            self._cleanup_tick(now)

    def _mtime_walk_project(self, state: _ProjectState) -> None:
        """Diff the project's file table against the live filesystem.

        Schedules update events for files whose mtime moved forward
        (catches watcher-missed events) and delete events for files
        whose path no longer exists.

        New files (untracked by the DB) are also scheduled — caught by
        comparing the DB list to a fresh ``os.walk``.

        Skips ``.git`` and other always-noisy directories so the sweep
        doesn't dominate on big repos.
        """
        if not state.root_path.exists():
            # Tier 1's job — handled by _cleanup_tick. Skip the walk.
            return

        recorded: dict[str, int] = {}
        try:
            files = self._store.list_files(project=state.name)
        except Exception:  # noqa: BLE001
            logger.exception("list_files failed for %s", state.name)
            return
        for rec in files:
            recorded[rec.rel_path] = rec.mtime

        seen: set[str] = set()
        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".pytest_cache", "target", "build", "dist", ".next",
            ".svelte-kit", ".tox", ".mypy_cache", ".ruff_cache",
        }
        for dirpath, dirs, fnames in os.walk(state.root_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in fnames:
                full = Path(dirpath) / fname
                try:
                    rel = full.relative_to(state.root_path).as_posix()
                except ValueError:
                    continue
                seen.add(rel)
                try:
                    mtime = int(full.stat().st_mtime)
                except OSError:
                    continue
                rec_mtime = recorded.get(rel)
                if rec_mtime is None or mtime > rec_mtime:
                    self._enqueue_event(WatcherEvent(
                        project=state.name,
                        type=EventType.MODIFY if rec_mtime is not None
                        else EventType.CREATE,
                        path=str(full),
                    ))

        # DELETEs: paths in DB but not on disk anymore.
        for rel in recorded.keys() - seen:
            self._enqueue_event(WatcherEvent(
                project=state.name,
                type=EventType.DELETE,
                path=str(state.root_path / rel),
            ))

    def _cleanup_tick(self, now: float) -> None:
        """Tier 1 (missing root) + Tier 2 (cold pause) checks."""
        grace_seconds = self._cfg.missing_root_grace_days * 86_400.0
        idle_seconds = self._cfg.watcher_idle_pause_days * 86_400.0

        with self._states_lock:
            snapshot = list(self._states.values())

        for state in snapshot:
            # Tier 1 — auto-drop missing root_path.
            if not state.root_path.exists():
                if state.first_missing_at is None:
                    state.first_missing_at = now
                    logger.info(
                        "watcher: project %s root missing — starting "
                        "Tier 1 grace window",
                        state.name,
                    )
                # Re-evaluate after stamping so a grace=0 config drops
                # on the same tick where we first noticed.
                if (
                    state.first_missing_at is not None
                    and now - state.first_missing_at >= grace_seconds
                ):
                    logger.warning(
                        "watcher: project %s root missing for >%dd; "
                        "auto-dropping",
                        state.name, self._cfg.missing_root_grace_days,
                    )
                    try:
                        self._indexer.delete_project(state.name)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "delete_project failed for %s", state.name,
                        )
                    self.remove_project(state.name)
                    continue
            else:
                if state.first_missing_at is not None:
                    logger.info(
                        "watcher: project %s root reappeared", state.name,
                    )
                    state.first_missing_at = None

            # Tier 2 — cold-project watch pause.
            if state.paused:
                continue
            idle = now - state.last_query_at
            # "No recent file events" is approximated as "no events in
            # the live queue" — events that already drained shouldn't
            # block a pause. ``last_event_at`` uses monotonic time and
            # we don't try to mix it with the wall-clock ``now`` here;
            # the queue-empty check is the meaningful signal.
            quiet = len(state.queue) == 0
            if idle >= idle_seconds and quiet:
                self.pause_project(state.name)


# ------------------------------------------------ searcher integration


def make_quiescence_callback(
    watcher: Watcher,
) -> Callable[[Optional[str], int], tuple[int, str]]:
    """Build a sync ``(project, timeout_ms) -> (pending, indexed_through)``
    callback for :py:class:`longctx_daemon.searcher.Searcher`.

    Searcher.search runs synchronously (the daemon offloads it to a
    worker thread), so we bridge to the watcher's async API by
    submitting a coroutine to the watcher's loop and blocking on the
    returned ``concurrent.futures.Future``.

    Edge cases:

    * No running loop / loop closed → return ``(pending, now)``
      synchronously without blocking.
    * Caller already on the watcher's event loop (rare, would deadlock)
      → fall back to non-blocking pending count.
    """

    def cb(project: Optional[str], timeout_ms: int) -> tuple[int, str]:
        loop = watcher._loop  # noqa: SLF001 — module-internal coupling
        if loop is None or loop.is_closed():
            return watcher.pending_updates(project), watcher.indexed_through_iso()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            # Same loop → can't block; return current pending instead.
            return (
                watcher.pending_updates(project),
                watcher.indexed_through_iso(),
            )
        try:
            fut = asyncio.run_coroutine_threadsafe(
                watcher.wait_for_quiescence(project, timeout_ms), loop,
            )
            return fut.result(timeout=(timeout_ms / 1000.0) + 1.0)
        except Exception:  # noqa: BLE001
            logger.exception("quiescence callback failed; returning best-effort")
            return (
                watcher.pending_updates(project),
                watcher.indexed_through_iso(),
            )

    return cb
