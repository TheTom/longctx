"""Watcher tests — synthetic events, debounce coalescing, storm overflow,
cold-project pause + resume, atomic-rename detection.

These tests drive the worker logic directly via the watcher's internal
synchronous APIs (``_enqueue_event`` + ``_drain_now``) rather than
spinning up a real ``watchfiles`` task. Filesystem-driven smokes live in
``test_watcher_fs.py`` (added in 2.3 if needed) — for unit tests, the
synthetic path is faster and less flaky.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from longctx_daemon.types import Project
from longctx_daemon.watcher import (
    EventType,
    Watcher,
    WatcherConfig,
    WatcherEvent,
    make_quiescence_callback,
)


# ---------------------------------------------------------------- doubles


class _FakeIndexer:
    """Tracks every call so tests can assert dispatch shape.

    Implements every public method the watcher reaches for, with no
    side effects beyond appending to ``calls`` and (for the project
    lifecycle methods) updating an in-memory project set.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.projects: set[str] = set()
        # Set when a test wants update_files to NOT be advertised
        # (forcing the watcher onto the per-file path).
        self.expose_bulk: bool = True
        self.fail_next_update: bool = False

    def add_project(self, name: str, root_path: Path) -> None:
        self.projects.add(name)
        self.calls.append(("add_project", name, str(root_path)))

    def delete_project(self, name: str) -> None:
        self.projects.discard(name)
        self.calls.append(("delete_project", name))

    def update_file(self, project: str, abs_path: Path) -> None:
        if self.fail_next_update:
            self.fail_next_update = False
            raise RuntimeError("simulated index error")
        self.calls.append(("update_file", project, str(abs_path)))

    def delete_file(self, project: str, rel_path: str) -> None:
        self.calls.append(("delete_file", project, rel_path))

    def rename_file(
        self, project: str, old_rel_path: str, new_rel_path: str,
    ) -> bool:
        self.calls.append(("rename_file", project, old_rel_path, new_rel_path))
        return True

    # update_files exposed conditionally so tests can flip the path.
    def __getattr__(self, name: str):  # noqa: D401
        if name == "update_files":
            if not getattr(self, "expose_bulk", True):
                raise AttributeError(name)
            return self._update_files
        raise AttributeError(name)

    def _update_files(self, project: str, abs_paths) -> None:
        # Materialize so test-side assertions see the path list, not
        # a generator that the worker already exhausted.
        paths = list(abs_paths)
        self.calls.append(("update_files", project, tuple(str(p) for p in paths)))


@pytest.fixture
def fake_indexer() -> _FakeIndexer:
    return _FakeIndexer()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Build a tiny project tree: src/a.py + README.md."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a(): pass\n")
    (tmp_path / "README.md").write_text("# repo\n")
    return tmp_path


@pytest.fixture
def watcher(
    fake_chunk_store,
    fake_indexer,
) -> Watcher:
    cfg = WatcherConfig(
        debounce_ms=50,
        queue_size=20,
        periodic_mtime_sweep_seconds=999,  # off — tests drive it manually
        watcher_idle_pause_days=30,
        missing_root_grace_days=7,
        periodic_check_seconds=999,
    )
    return Watcher(
        indexer=fake_indexer,
        chunk_store=fake_chunk_store,
        config=cfg,
    )


def _project(root: Path, name: str = "demo") -> Project:
    return Project(name=name, root_path=str(root), last_full_scan_at=0)


# ------------------------------------------------------------- core API


def test_add_remove_project_idempotent(watcher: Watcher, project_root: Path) -> None:
    proj = _project(project_root)
    watcher.add_project(proj)
    # Second add resets state, doesn't double-register.
    watcher.add_project(proj)
    assert "demo" in watcher._states  # noqa: SLF001
    watcher.remove_project("demo")
    assert "demo" not in watcher._states  # noqa: SLF001


def test_remove_unknown_project_no_error(watcher: Watcher) -> None:
    # Should not raise; just a no-op.
    watcher.remove_project("not-here")


def test_pending_updates_zero_when_idle(watcher: Watcher, project_root: Path) -> None:
    watcher.add_project(_project(project_root))
    assert watcher.pending_updates() == 0
    assert watcher.pending_updates("demo") == 0
    assert watcher.pending_updates("does-not-exist") == 0


def test_indexed_through_iso_format(watcher: Watcher) -> None:
    val = watcher.indexed_through_iso()
    assert val.endswith("Z")
    # Sanity: parseable as ISO with the trailing Z.
    from datetime import datetime
    datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------ enqueue + dispatch


def _enqueue(watcher: Watcher, project: str, etype: EventType, path: str) -> None:
    watcher._enqueue_event(WatcherEvent(  # noqa: SLF001
        project=project,
        type=etype,
        path=path,
    ))


@pytest.mark.asyncio
async def test_create_dispatches_update_files(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"
    _enqueue(watcher, "demo", EventType.CREATE, str(file_a))

    # Wait past the debounce window then drain.
    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    bulk_calls = [c for c in fake_indexer.calls if c[0] == "update_files"]
    assert bulk_calls, fake_indexer.calls
    assert bulk_calls[0][1] == "demo"
    assert str(file_a) in bulk_calls[0][2]


@pytest.mark.asyncio
async def test_modify_dispatches_update_file_when_no_bulk(
    fake_chunk_store,
    project_root: Path,
) -> None:
    indexer = _FakeIndexer()
    indexer.expose_bulk = False
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    file_a = project_root / "src" / "a.py"
    _enqueue(w, "demo", EventType.MODIFY, str(file_a))
    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    update_calls = [c for c in indexer.calls if c[0] == "update_file"]
    assert update_calls
    assert update_calls[0] == ("update_file", "demo", str(file_a))


@pytest.mark.asyncio
async def test_delete_dispatches_delete_file(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"
    _enqueue(watcher, "demo", EventType.DELETE, str(file_a))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    delete_calls = [c for c in fake_indexer.calls if c[0] == "delete_file"]
    assert delete_calls == [("delete_file", "demo", "src/a.py")]


@pytest.mark.asyncio
async def test_rename_pair_dispatches_rename(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """One DELETE + one CREATE on different paths within the debounce
    window collapses to a single RENAME event (PRD §5.4 cheap path)."""
    watcher.add_project(_project(project_root))
    old_path = project_root / "src" / "a.py"
    new_path = project_root / "src" / "b.py"

    _enqueue(watcher, "demo", EventType.DELETE, str(old_path))
    _enqueue(watcher, "demo", EventType.CREATE, str(new_path))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    rename_calls = [c for c in fake_indexer.calls if c[0] == "rename_file"]
    assert rename_calls == [("rename_file", "demo", "src/a.py", "src/b.py")]
    # No raw update_file / delete_file calls survived.
    assert not [c for c in fake_indexer.calls if c[0] == "delete_file"]
    assert not [c for c in fake_indexer.calls if c[0] == "update_files"]


@pytest.mark.asyncio
async def test_atomic_rename_collapses_to_modify(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """vim-style DELETE+CREATE on the SAME path within the debounce
    window must become a single MODIFY (PRD §5.4 atomic-rename row)."""
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"

    _enqueue(watcher, "demo", EventType.DELETE, str(file_a))
    _enqueue(watcher, "demo", EventType.CREATE, str(file_a))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    # Should NOT see a delete_file call.
    deletes = [c for c in fake_indexer.calls if c[0] == "delete_file"]
    assert deletes == []
    bulk = [c for c in fake_indexer.calls if c[0] == "update_files"]
    assert bulk and str(file_a) in bulk[0][2]


@pytest.mark.asyncio
async def test_duplicate_modify_coalesces(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"

    for _ in range(5):
        _enqueue(watcher, "demo", EventType.MODIFY, str(file_a))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    bulk = [c for c in fake_indexer.calls if c[0] == "update_files"]
    assert len(bulk) == 1
    assert bulk[0][2].count(str(file_a)) == 1


@pytest.mark.asyncio
async def test_create_then_delete_collapses_to_delete(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """CREATE followed by DELETE on same path — last event wins."""
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"

    _enqueue(watcher, "demo", EventType.CREATE, str(file_a))
    _enqueue(watcher, "demo", EventType.DELETE, str(file_a))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    deletes = [c for c in fake_indexer.calls if c[0] == "delete_file"]
    bulk = [c for c in fake_indexer.calls if c[0] == "update_files"]
    # CREATE+DELETE is the same-path collapse → DELETE survives.
    # (RENAME pairing only applies when delete and create are on
    # DIFFERENT paths.)
    assert deletes == [("delete_file", "demo", "src/a.py")]
    assert not bulk


@pytest.mark.asyncio
async def test_storm_flips_to_rescan_pending(
    fake_chunk_store,
    project_root: Path,
) -> None:
    """Queueing >queue_size events triggers rescan-pending mode and
    drains by walking the project filesystem (one update per real file)."""
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, queue_size=3,
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    # Flood with events.
    for i in range(10):
        _enqueue(w, "demo", EventType.MODIFY, str(project_root / f"f{i}.py"))

    state = w._states["demo"]  # noqa: SLF001
    assert state.rescan_pending
    # Queue is cleared on overflow.
    assert len(state.queue) == 0

    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    # The rescan walk emitted CREATE events for the on-disk files.
    # Drain again so those events become indexer calls.
    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001
    bulk = [c for c in indexer.calls if c[0] == "update_files"]
    assert bulk, indexer.calls
    paths = bulk[-1][2]
    # README.md + src/a.py both indexed via the rescan walk.
    assert any("a.py" in p for p in paths)
    assert any("README.md" in p for p in paths)


@pytest.mark.asyncio
async def test_storm_clears_after_drain(
    fake_chunk_store,
    project_root: Path,
) -> None:
    indexer = _FakeIndexer()
    cfg = WatcherConfig(debounce_ms=10, queue_size=2,
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    for i in range(5):
        _enqueue(w, "demo", EventType.MODIFY, str(project_root / f"f{i}.py"))

    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001
    state = w._states["demo"]  # noqa: SLF001
    assert not state.rescan_pending


@pytest.mark.asyncio
async def test_pause_resume_cycle(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """Tier 2 pause/resume preserves index but skips watchfiles."""
    watcher.add_project(_project(project_root))
    state = watcher._states["demo"]  # noqa: SLF001
    assert not state.paused

    watcher.pause_project("demo")
    assert state.paused

    # Pausing twice is idempotent.
    watcher.pause_project("demo")
    assert state.paused

    watcher.resume_project("demo")
    assert not state.paused


def test_pause_unknown_project_no_error(watcher: Watcher) -> None:
    watcher.pause_project("does-not-exist")  # no-op


def test_resume_unknown_project_no_error(watcher: Watcher) -> None:
    watcher.resume_project("does-not-exist")


def test_resume_when_not_paused_is_noop(
    watcher: Watcher, project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    watcher.resume_project("demo")  # not paused; no-op


def test_note_query_resets_idle(
    watcher: Watcher, project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    state = watcher._states["demo"]  # noqa: SLF001
    state.last_query_at = 0.0
    watcher.note_query("demo")
    assert state.last_query_at > 0.0


def test_note_query_none_is_noop(watcher: Watcher) -> None:
    watcher.note_query(None)  # fanout case


def test_note_query_resumes_paused_project(
    watcher: Watcher, project_root: Path,
) -> None:
    watcher.add_project(_project(project_root))
    watcher.pause_project("demo")
    assert watcher._states["demo"].paused  # noqa: SLF001
    watcher.note_query("demo")
    assert not watcher._states["demo"].paused  # noqa: SLF001


# -------------------------------------------- coalesce edge cases


@pytest.mark.asyncio
async def test_modify_after_create_keeps_create_intent(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """CREATE then MODIFY on same path → one update_file call (last
    event wins, but both are bulk-eligible)."""
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"

    _enqueue(watcher, "demo", EventType.CREATE, str(file_a))
    _enqueue(watcher, "demo", EventType.MODIFY, str(file_a))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    bulk = [c for c in fake_indexer.calls if c[0] == "update_files"]
    assert len(bulk) == 1
    assert bulk[0][2].count(str(file_a)) == 1


@pytest.mark.asyncio
async def test_multi_project_isolated_queues(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
    tmp_path: Path,
) -> None:
    """Two projects, two queues — drain works independently."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("x = 1\n")
    watcher.add_project(_project(project_root, name="demo"))
    watcher.add_project(_project(other, name="other"))

    _enqueue(watcher, "demo", EventType.MODIFY, str(project_root / "src" / "a.py"))
    _enqueue(watcher, "other", EventType.MODIFY, str(other / "x.py"))

    assert watcher.pending_updates("demo") == 1
    assert watcher.pending_updates("other") == 1
    assert watcher.pending_updates() == 2

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001
    assert watcher.pending_updates() == 0


@pytest.mark.asyncio
async def test_rename_falls_back_when_indexer_lacks_method(
    fake_chunk_store,
    project_root: Path,
) -> None:
    """If the indexer doesn't expose ``rename_file``, the watcher falls
    back to delete + update without crashing."""

    class _NoRenameIndexer(_FakeIndexer):
        def __getattribute__(self, name: str):
            if name == "rename_file":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    indexer = _NoRenameIndexer()
    cfg = WatcherConfig(debounce_ms=10, periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=999)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(_project(project_root))

    old_path = project_root / "src" / "a.py"
    new_path = project_root / "src" / "b.py"
    _enqueue(w, "demo", EventType.DELETE, str(old_path))
    _enqueue(w, "demo", EventType.CREATE, str(new_path))

    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    deletes = [c for c in indexer.calls if c[0] == "delete_file"]
    updates = [c for c in indexer.calls if c[0] == "update_file"]
    assert deletes == [("delete_file", "demo", "src/a.py")]
    assert updates == [("update_file", "demo", str(new_path))]


@pytest.mark.asyncio
async def test_dispatch_swallows_per_event_errors(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """One file blowing up shouldn't sink the whole batch."""
    watcher.add_project(_project(project_root))
    fake_indexer.expose_bulk = False
    fake_indexer.fail_next_update = True

    file_a = project_root / "src" / "a.py"
    file_b = project_root / "README.md"
    _enqueue(watcher, "demo", EventType.MODIFY, str(file_a))
    _enqueue(watcher, "demo", EventType.MODIFY, str(file_b))

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001

    # Second file still got dispatched even though first raised.
    updates = [c for c in fake_indexer.calls if c[0] == "update_file"]
    assert any(file_b.name in u[2] for u in updates)


@pytest.mark.asyncio
async def test_drain_when_paused_skipped(
    watcher: Watcher,
    fake_indexer: _FakeIndexer,
    project_root: Path,
) -> None:
    """Paused project queue still drains on drain_now (only watch task
    pauses; the worker keeps committing whatever's already queued)."""
    watcher.add_project(_project(project_root))
    file_a = project_root / "src" / "a.py"
    _enqueue(watcher, "demo", EventType.MODIFY, str(file_a))
    watcher.pause_project("demo")

    await asyncio.sleep(0.06)
    await watcher._drain_now()  # noqa: SLF001
    bulk = [c for c in fake_indexer.calls if c[0] == "update_files"]
    assert bulk, "queued events should still drain even after pause"


def test_make_quiescence_callback_no_loop(
    watcher: Watcher, project_root: Path,
) -> None:
    """Callback in a no-loop world returns ``(pending, indexed_through)``
    synchronously without blocking."""
    watcher.add_project(_project(project_root))
    cb = make_quiescence_callback(watcher)
    pending, ts = cb("demo", 100)
    assert pending == 0
    assert ts.endswith("Z")


def test_to_rel_handles_relative_input(
    watcher: Watcher, project_root: Path,
) -> None:
    rel = watcher._to_rel(project_root, "src/a.py")  # noqa: SLF001
    assert rel == "src/a.py"


def test_to_rel_returns_none_for_external_path(
    watcher: Watcher, project_root: Path, tmp_path_factory,
) -> None:
    other_root = tmp_path_factory.mktemp("elsewhere")
    rel = watcher._to_rel(project_root, str(other_root / "x.py"))  # noqa: SLF001
    assert rel is None
