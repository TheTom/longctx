"""Periodic mtime-sweep + Tier 1/2 cleanup tests.

These exercise the watcher's sweep loop directly (``_mtime_sweep_once``)
so we don't have to wait on the real ``periodic_mtime_sweep_seconds``
interval. Time control for grace-window assertions uses ``freezegun`` so
we can advance the wall clock by days in milliseconds of test time.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from longctx_daemon.types import FileRecord, Project
from longctx_daemon.watcher import (
    EventType,
    Watcher,
    WatcherConfig,
    WatcherEvent,
)


# ---- minimal fakes shared with test_watcher.py via composition ----

class _FakeIndexer:
    """See ``test_watcher.py``. Duplicated here to keep the two test
    files self-contained — pytest doesn't allow cross-module imports of
    test classes without import-path gymnastics."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.deleted_projects: list[str] = []

    def update_file(self, project: str, abs_path: Path) -> None:
        self.calls.append(("update_file", project, str(abs_path)))

    def update_files(self, project: str, abs_paths) -> None:
        paths = list(abs_paths)
        self.calls.append(
            ("update_files", project, tuple(str(p) for p in paths)),
        )

    def delete_file(self, project: str, rel_path: str) -> None:
        self.calls.append(("delete_file", project, rel_path))

    def delete_project(self, name: str) -> None:
        self.deleted_projects.append(name)
        self.calls.append(("delete_project", name))

    def rename_file(self, project: str, old: str, new: str) -> bool:
        self.calls.append(("rename_file", project, old, new))
        return True


def _project(root: Path, name: str = "demo") -> Project:
    return Project(name=name, root_path=str(root), last_full_scan_at=0)


def _make_watcher(
    fake_chunk_store,
    *,
    missing_root_grace_days: int = 7,
    watcher_idle_pause_days: int = 30,
    periodic_check_seconds: int = 0,
) -> tuple[Watcher, _FakeIndexer]:
    indexer = _FakeIndexer()
    cfg = WatcherConfig(
        debounce_ms=10,
        queue_size=10_000,
        periodic_mtime_sweep_seconds=999,
        watcher_idle_pause_days=watcher_idle_pause_days,
        missing_root_grace_days=missing_root_grace_days,
        # Default 0 lets every sweep tick run cleanup; set to >0 to test
        # throttling.
        periodic_check_seconds=periodic_check_seconds,
    )
    return Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg), indexer


# ----------------------------------------------------- mtime-walk catch-up


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a(): pass\n")
    (tmp_path / "README.md").write_text("# repo\n")
    return tmp_path


@pytest.mark.asyncio
async def test_mtime_walk_schedules_create_for_untracked_files(
    fake_chunk_store, project_root: Path,
) -> None:
    """Files that exist on disk but not in the DB → CREATE event."""
    w, indexer = _make_watcher(fake_chunk_store)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))

    await w._mtime_sweep_once()  # noqa: SLF001

    # Sweep enqueues; need to drain to dispatch.
    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    bulk = [c for c in indexer.calls if c[0] == "update_files"]
    assert bulk
    paths = bulk[-1][2]
    assert any("a.py" in p for p in paths)
    assert any("README.md" in p for p in paths)


@pytest.mark.asyncio
async def test_mtime_walk_schedules_modify_when_mtime_advances(
    fake_chunk_store, project_root: Path, monkeypatch,
) -> None:
    """Existing file with newer on-disk mtime → MODIFY event."""
    w, indexer = _make_watcher(fake_chunk_store)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    # Pretend we already indexed a.py with mtime=0.
    fake_chunk_store.upsert_file(FileRecord(
        id=0, project="demo", rel_path="src/a.py",
        mtime=0, size_bytes=10, content_hash="x",
    ))
    fake_chunk_store.upsert_file(FileRecord(
        id=0, project="demo", rel_path="README.md",
        mtime=0, size_bytes=10, content_hash="y",
    ))

    await w._mtime_sweep_once()  # noqa: SLF001
    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    bulk = [c for c in indexer.calls if c[0] == "update_files"]
    assert bulk
    # Both files have on-disk mtime > recorded 0 → both queued for update.
    paths = bulk[-1][2]
    assert any("a.py" in p for p in paths)


@pytest.mark.asyncio
async def test_mtime_walk_schedules_delete_for_missing_files(
    fake_chunk_store, project_root: Path,
) -> None:
    """File in DB but not on disk → DELETE event."""
    w, indexer = _make_watcher(fake_chunk_store)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    # Record a phantom file that was never on disk.
    fake_chunk_store.upsert_file(FileRecord(
        id=0, project="demo", rel_path="ghost.py",
        mtime=0, size_bytes=10, content_hash="g",
    ))

    await w._mtime_sweep_once()  # noqa: SLF001
    await asyncio.sleep(0.02)
    await w._drain_now()  # noqa: SLF001

    deletes = [c for c in indexer.calls if c[0] == "delete_file"]
    assert ("delete_file", "demo", "ghost.py") in deletes


@pytest.mark.asyncio
async def test_mtime_walk_skips_paused_project(
    fake_chunk_store, project_root: Path,
) -> None:
    w, indexer = _make_watcher(fake_chunk_store)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    w.pause_project("demo")

    await w._mtime_sweep_once()  # noqa: SLF001

    # Paused → no events queued.
    assert w.pending_updates() == 0


@pytest.mark.asyncio
async def test_mtime_walk_handles_missing_root(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """Sweep tolerates a missing root_path without exploding (Tier 1
    handles the actual cleanup)."""
    missing = tmp_path / "doesnotexist"
    w, _ = _make_watcher(fake_chunk_store)
    w.add_project(Project(name="missing", root_path=str(missing)))
    fake_chunk_store.upsert_project(
        Project(name="missing", root_path=str(missing)),
    )

    # Should not raise.
    await w._mtime_sweep_once()  # noqa: SLF001
    assert w.pending_updates() == 0


# ------------------------------------------------- Tier 1: missing root


@pytest.mark.asyncio
async def test_tier1_marks_first_missing(
    fake_chunk_store, tmp_path: Path,
) -> None:
    missing = tmp_path / "gone"
    w, indexer = _make_watcher(fake_chunk_store, missing_root_grace_days=7)
    w.add_project(Project(name="gone", root_path=str(missing)))

    await w._mtime_sweep_once()  # noqa: SLF001

    state = w._states["gone"]  # noqa: SLF001
    assert state.first_missing_at is not None
    # Not yet expired → no delete_project.
    assert "gone" not in indexer.deleted_projects


@pytest.mark.asyncio
async def test_tier1_drops_after_grace(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """After ``missing_root_grace_days`` of consecutive missing root
    observations, Tier 1 calls ``delete_project``."""
    missing = tmp_path / "gone"
    w, indexer = _make_watcher(fake_chunk_store, missing_root_grace_days=7)
    w.add_project(Project(name="gone", root_path=str(missing)))

    # First sweep marks; subsequent sweeps after grace expire should drop.
    state = w._states["gone"]  # noqa: SLF001
    state.first_missing_at = (
        # 8 days ago.
        __import__("time").time() - (8 * 86_400.0)
    )

    await w._mtime_sweep_once()  # noqa: SLF001
    assert "gone" in indexer.deleted_projects
    # The watcher's local state is also dropped.
    assert "gone" not in w._states  # noqa: SLF001


@pytest.mark.asyncio
async def test_tier1_resets_when_root_reappears(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """Root reappears mid-grace → ``first_missing_at`` resets to None."""
    missing = tmp_path / "gone"
    w, indexer = _make_watcher(fake_chunk_store, missing_root_grace_days=7)
    w.add_project(Project(name="gone", root_path=str(missing)))

    await w._mtime_sweep_once()  # noqa: SLF001
    assert w._states["gone"].first_missing_at is not None  # noqa: SLF001

    # Recreate the directory.
    missing.mkdir()
    await w._mtime_sweep_once()  # noqa: SLF001
    assert w._states["gone"].first_missing_at is None  # noqa: SLF001
    assert "gone" not in indexer.deleted_projects


@pytest.mark.asyncio
async def test_tier1_grace_can_be_zero(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """grace=0 → drop immediately (helpful for the manual-clean CLI)."""
    missing = tmp_path / "gone"
    w, indexer = _make_watcher(fake_chunk_store, missing_root_grace_days=0)
    w.add_project(Project(name="gone", root_path=str(missing)))

    await w._mtime_sweep_once()  # noqa: SLF001
    assert "gone" in indexer.deleted_projects


@pytest.mark.asyncio
async def test_tier1_delete_project_failure_doesnt_crash(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """Indexer exception propagates as logged exception, not a crash."""
    missing = tmp_path / "gone"

    class _BoomIndexer(_FakeIndexer):
        def delete_project(self, name: str) -> None:  # noqa: D401
            raise RuntimeError("delete_project blew up")

    indexer = _BoomIndexer()
    cfg = WatcherConfig(missing_root_grace_days=0,
                        periodic_mtime_sweep_seconds=999,
                        periodic_check_seconds=0)
    w = Watcher(indexer=indexer, chunk_store=fake_chunk_store, config=cfg)
    w.add_project(Project(name="gone", root_path=str(missing)))

    await w._mtime_sweep_once()  # noqa: SLF001
    # Project still removed from the watcher's state even though
    # delete_project blew up — better to lose the OS-level subscription
    # than to keep retrying in a tight loop.
    assert "gone" not in w._states  # noqa: SLF001


# ------------------------------------------------ Tier 2: cold pause


@pytest.mark.asyncio
async def test_tier2_pauses_idle_project(
    fake_chunk_store, project_root: Path,
) -> None:
    """Project not queried for >watcher_idle_pause_days → paused."""
    w, _ = _make_watcher(fake_chunk_store, watcher_idle_pause_days=30)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    # Pre-record files so the mtime walk doesn't queue catch-up events
    # (which would defer the pause).
    import os as _os
    for rel in ("src/a.py", "README.md"):
        full = project_root / rel
        fake_chunk_store.upsert_file(FileRecord(
            id=0, project="demo", rel_path=rel,
            mtime=int(full.stat().st_mtime) + 100, size_bytes=10,
            content_hash="x",
        ))
    state = w._states["demo"]  # noqa: SLF001
    state.last_query_at = __import__("time").time() - (31 * 86_400.0)

    await w._mtime_sweep_once()  # noqa: SLF001
    assert state.paused


@pytest.mark.asyncio
async def test_tier2_does_not_pause_recent_query(
    fake_chunk_store, project_root: Path,
) -> None:
    w, _ = _make_watcher(fake_chunk_store, watcher_idle_pause_days=30)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))

    await w._mtime_sweep_once()  # noqa: SLF001
    assert not w._states["demo"].paused  # noqa: SLF001


@pytest.mark.asyncio
async def test_tier2_does_not_pause_when_queue_busy(
    fake_chunk_store, project_root: Path,
) -> None:
    """If the project has live events queued, hold off on pausing —
    activity is still happening."""
    w, _ = _make_watcher(fake_chunk_store, watcher_idle_pause_days=30)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    state = w._states["demo"]  # noqa: SLF001
    state.last_query_at = __import__("time").time() - (31 * 86_400.0)
    # Queue an event that will hold off the pause.
    w._enqueue_event(WatcherEvent(  # noqa: SLF001
        project="demo", type=EventType.MODIFY,
        path=str(project_root / "src" / "a.py"),
    ))

    await w._mtime_sweep_once()  # noqa: SLF001
    assert not state.paused, (
        "active queue should defer Tier 2 pause"
    )


@pytest.mark.asyncio
async def test_tier2_resume_via_note_query(
    fake_chunk_store, project_root: Path,
) -> None:
    """First query after a pause auto-resumes the watch."""
    w, _ = _make_watcher(fake_chunk_store, watcher_idle_pause_days=30)
    w.add_project(_project(project_root))
    fake_chunk_store.upsert_project(_project(project_root))
    w.pause_project("demo")

    w.note_query("demo")
    assert not w._states["demo"].paused  # noqa: SLF001


# ---------------------------------------------------- cleanup throttle


@pytest.mark.asyncio
async def test_cleanup_throttle_skips_intra_window(
    fake_chunk_store, tmp_path: Path,
) -> None:
    """Cleanup tick respects ``periodic_check_seconds`` so back-to-back
    sweeps don't both run the (potentially expensive) Tier 1/2 logic.
    """
    missing = tmp_path / "gone"
    w, indexer = _make_watcher(
        fake_chunk_store,
        missing_root_grace_days=0,
        periodic_check_seconds=3600,
    )
    w.add_project(Project(name="gone", root_path=str(missing)))

    await w._mtime_sweep_once()  # noqa: SLF001
    # First tick ran cleanup → grace=0 dropped the project.
    assert "gone" in indexer.deleted_projects

    # Re-add to test that a second tick within the throttle window
    # doesn't re-fire cleanup.
    indexer.deleted_projects.clear()
    w.add_project(Project(name="gone", root_path=str(missing)))
    await w._mtime_sweep_once()  # noqa: SLF001
    # Throttle should keep us from re-running cleanup.
    assert "gone" not in indexer.deleted_projects
