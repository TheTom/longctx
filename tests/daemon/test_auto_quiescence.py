"""Auto-quiescence integration tests.

The watcher exposes ``wait_for_quiescence`` (async) and a sync bridge
via :py:func:`make_quiescence_callback`. These tests assert:

* ``search`` (synchronous) blocks while the queue has pending events
  and returns once the worker drains.
* ``wait_for_quiescence`` returns immediately when idle.
* Timeout doesn't deadlock when the queue never drains within the
  budget.
* Headline invariant (PRD §6.5): an event enqueued before a search
  is reflected in the freshness fields by the time the search returns.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from longctx_daemon.searcher import Searcher, SearcherConfig
from longctx_daemon.types import Project
from longctx_daemon.watcher import (
    EventType,
    Watcher,
    WatcherConfig,
    WatcherEvent,
    make_quiescence_callback,
)


# Reuse the test indexer from the watcher tests.
class _FakeIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.delay_seconds: float = 0.0

    def update_file(self, project: str, abs_path: Path) -> None:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        self.calls.append(("update_file", project, str(abs_path)))

    def update_files(self, project: str, abs_paths) -> None:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        paths = list(abs_paths)
        self.calls.append(
            ("update_files", project, tuple(str(p) for p in paths)),
        )

    def delete_file(self, project: str, rel_path: str) -> None:
        self.calls.append(("delete_file", project, rel_path))

    def delete_project(self, name: str) -> None:
        self.calls.append(("delete_project", name))


def _project(root: Path, name: str = "demo") -> Project:
    return Project(name=name, root_path=str(root), last_full_scan_at=0)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a(): pass\n")
    return tmp_path


# ---------------------------------------- async wait_for_quiescence


@pytest.mark.asyncio
async def test_wait_returns_immediately_when_idle(
    fake_chunk_store, project_root: Path,
) -> None:
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    pending, ts = await w.wait_for_quiescence("demo", timeout_ms=100)
    assert pending == 0
    assert ts.endswith("Z")


@pytest.mark.asyncio
async def test_wait_returns_after_drain(
    fake_chunk_store, project_root: Path,
) -> None:
    """run() drains the queue → wait_for_quiescence returns 0 once
    the worker commits."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, queue_size=10_000,
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    run_task = asyncio.create_task(w.run())
    try:
        # Wait for run() to set up.
        await asyncio.sleep(0.02)
        # Queue an event.
        w._enqueue_event(WatcherEvent(  # noqa: SLF001
            project="demo", type=EventType.MODIFY,
            path=str(project_root / "src" / "a.py"),
        ))
        assert w.pending_updates() == 1
        # Worker drains within debounce; budget covers it.
        pending, _ = await w.wait_for_quiescence("demo", timeout_ms=2000)
        assert pending == 0
        # Indexer ran exactly once.
        assert any(c[0] == "update_files" for c in indexer.calls), indexer.calls
    finally:
        w.stop()
        await asyncio.sleep(0.02)
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_wait_times_out_with_pending_count(
    fake_chunk_store, project_root: Path,
) -> None:
    """When the worker can't drain in time, wait returns the current
    pending count rather than blocking indefinitely."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=200, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    # No run() loop → queue never drains naturally.
    w._enqueue_event(WatcherEvent(  # noqa: SLF001
        project="demo", type=EventType.MODIFY,
        path=str(project_root / "src" / "a.py"),
    ))
    pending, _ = await w.wait_for_quiescence("demo", timeout_ms=20)
    # Without run() the cond can't fire; quick path returns synchronously.
    assert pending == 1


@pytest.mark.asyncio
async def test_wait_returns_quickly_for_unknown_project(
    fake_chunk_store, project_root: Path,
) -> None:
    """Unknown project name → 0 pending immediately."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    pending, _ = await w.wait_for_quiescence("does-not-exist", timeout_ms=5)
    assert pending == 0


# ----------------------------------------- sync bridge for Searcher


def test_callback_returns_idle_when_no_loop(
    fake_chunk_store, project_root: Path,
) -> None:
    """Without a running watcher loop, the callback degrades to
    a synchronous read of pending_updates / indexed_through."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    cb = make_quiescence_callback(w)
    pending, ts = cb("demo", 50)
    assert pending == 0
    assert ts.endswith("Z")


def test_callback_returns_pending_when_no_loop_with_queue(
    fake_chunk_store, project_root: Path,
) -> None:
    """Best-effort: returns the queue size without blocking."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))
    w._enqueue_event(WatcherEvent(  # noqa: SLF001
        project="demo", type=EventType.MODIFY,
        path=str(project_root / "src" / "a.py"),
    ))

    cb = make_quiescence_callback(w)
    pending, _ = cb("demo", 50)
    assert pending == 1


# --------------------------------- Searcher integration end-to-end


class _StubChunkStore:
    """Empty store — searcher returns 0 hits, but the freshness path
    still runs through the quiescence callback."""

    def __init__(self) -> None:
        self._projects: list[Project] = []

    def list_projects(self):
        return tuple(self._projects)

    def upsert_project(self, p: Project) -> None:
        self._projects.append(p)

    def list_chunk_ids_in_scope(self, scope):
        return ()

    def search_lexical(self, terms, k, scope=None):
        return ()

    def get_chunks_by_id(self, ids):
        return ()

    def list_files(self, project=None):
        return ()


class _StubEmbedStore:
    @property
    def dim(self) -> int:
        return 4

    def search_dense(self, query_emb, k, chunk_ids_in_scope=None):
        return ()


class _StubEmbedder:
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.zeros((len(texts), 4), dtype="float32")


@pytest.mark.asyncio
async def test_searcher_freshness_reflects_watcher_pending(
    fake_chunk_store, project_root: Path,
) -> None:
    """PRD §6.5 headline invariant: a search call sees the watcher's
    pending count via the freshness fields."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10_000,  # never drains in this test
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    # Spin up run() so the loop exists for the threadsafe bridge.
    run_task = asyncio.create_task(w.run())
    try:
        await asyncio.sleep(0.02)

        # Enqueue an event so pending_updates > 0.
        w._enqueue_event(WatcherEvent(  # noqa: SLF001
            project="demo", type=EventType.MODIFY,
            path=str(project_root / "src" / "a.py"),
        ))

        # Searcher with watcher-backed callback. Run synchronously off-loop.
        store = _StubChunkStore()
        store.upsert_project(_project(project_root))
        searcher = Searcher(
            chunk_store=store,
            embed_store=_StubEmbedStore(),
            embedder=_StubEmbedder(),
            config=SearcherConfig(default_wait_for_quiescence_ms=50),
            quiescence_check_callback=make_quiescence_callback(w),
        )

        result = await asyncio.to_thread(
            searcher.search,
            query="anything",
            project="demo",
            wait_for_quiescence_ms=50,
        )
        # Pending events are surfaced in the freshness payload.
        assert result.freshness.pending_updates == 1
        assert not result.freshness.is_fully_fresh
    finally:
        w.stop()
        await asyncio.sleep(0.02)
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_search_blocks_until_drain(
    fake_chunk_store, project_root: Path,
) -> None:
    """Headline invariant in the happy path: an event enqueued just
    before a search call drains within ``wait_for_quiescence_ms``, so
    the search returns is_fully_fresh=True."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=20, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    run_task = asyncio.create_task(w.run())
    try:
        await asyncio.sleep(0.02)
        w._enqueue_event(WatcherEvent(  # noqa: SLF001
            project="demo", type=EventType.MODIFY,
            path=str(project_root / "src" / "a.py"),
        ))

        store = _StubChunkStore()
        store.upsert_project(_project(project_root))
        searcher = Searcher(
            chunk_store=store,
            embed_store=_StubEmbedStore(),
            embedder=_StubEmbedder(),
            config=SearcherConfig(default_wait_for_quiescence_ms=2000),
            quiescence_check_callback=make_quiescence_callback(w),
        )
        result = await asyncio.to_thread(
            searcher.search,
            query="anything",
            project="demo",
            wait_for_quiescence_ms=2000,
        )
        assert result.freshness.pending_updates == 0
        assert result.freshness.is_fully_fresh
    finally:
        w.stop()
        await asyncio.sleep(0.02)
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_callback_does_not_deadlock_when_empty(
    fake_chunk_store, project_root: Path,
) -> None:
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    run_task = asyncio.create_task(w.run())
    try:
        await asyncio.sleep(0.02)
        cb = make_quiescence_callback(w)
        # Queue is empty — should return ~immediately.
        t0 = time.perf_counter()
        pending, _ = await asyncio.to_thread(cb, "demo", 500)
        elapsed = time.perf_counter() - t0
        assert pending == 0
        assert elapsed < 0.1, f"callback blocked too long: {elapsed:.3f}s"
    finally:
        w.stop()
        await asyncio.sleep(0.02)
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_callback_called_from_same_loop_returns_pending(
    fake_chunk_store, project_root: Path,
) -> None:
    """If the search ever runs on the watcher's own loop (rare —
    daemon offloads to a thread), the bridge MUST NOT block on itself.
    Returns the pending count synchronously instead."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10_000,
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    run_task = asyncio.create_task(w.run())
    try:
        await asyncio.sleep(0.02)
        w._enqueue_event(WatcherEvent(  # noqa: SLF001
            project="demo", type=EventType.MODIFY,
            path=str(project_root / "src" / "a.py"),
        ))
        cb = make_quiescence_callback(w)
        # Calling the cb on the same loop would deadlock if the bridge
        # tried to block. The fallback returns pending sync.
        pending, _ = cb("demo", 500)
        assert pending == 1
    finally:
        w.stop()
        await asyncio.sleep(0.02)
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass
