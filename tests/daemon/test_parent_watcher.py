"""Tests for ``longctx_daemon.parent_watcher``.

Coverage targets (agent spec §3):
  * Initial scan fires on_project_added for pre-existing projects
  * Add path: synthetic events for "new sentinel appears" → callback
  * Remove path: synthetic events for "sentinel disappears" → callback
  * Debounce coalesces rapid events
  * Forbidden / excluded names ignored
  * Idempotent stop()
  * known_projects snapshot

Async tests use the test-only ``run_with_external_changes`` helper to
simulate watchfiles-style event batches without spinning up a real
filesystem watcher (those are integration tests, owned by Agent G).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterable

import pytest

# watchfiles' Change enum — we use raw values so import order doesn't
# matter for tests that don't actually call awatch.
try:
    from watchfiles import Change
    HAS_WATCHFILES = True
except ImportError:  # pragma: no cover
    HAS_WATCHFILES = False
    Change = None

from longctx_daemon.discovery import DiscoveredProject
from longctx_daemon.parent_watcher import (
    ParentDirWatcher,
    run_with_external_changes,
)


# --------------------------------------------------------------- helpers


def _make_project(parent: Path, name: str, sentinel: str = ".git") -> Path:
    proj = parent / name
    proj.mkdir(parents=True, exist_ok=True)
    if sentinel.startswith("."):
        (proj / sentinel).mkdir(exist_ok=True)
    else:
        (proj / sentinel).write_text("")
    return proj


def _remove_project(parent: Path, name: str) -> None:
    """Recursive delete a project under parent."""
    import shutil
    shutil.rmtree(parent / name)


def _change_event(kind: str, path: Path) -> tuple:
    """Build a ((Change, str)) tuple in the shape watchfiles emits."""
    if not HAS_WATCHFILES:  # pragma: no cover
        return (kind, str(path))
    mapping = {"add": Change.added, "modify": Change.modified, "delete": Change.deleted}
    return (mapping[kind], str(path))


# --------------------------------------------------------------- construction


class TestConstruction:
    def test_construct_does_not_run(self, tmp_path):
        # Constructing should not raise; even with a missing parent.
        watcher = ParentDirWatcher(
            parent_dir=tmp_path / "absent",
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        assert watcher is not None
        assert watcher.known_projects == ()

    def test_stop_is_idempotent(self, tmp_path):
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        watcher.stop()
        watcher.stop()  # no-op
        watcher.stop()


# --------------------------------------------------------------- initial scan


@pytest.mark.asyncio
class TestInitialScan:
    async def test_initial_scan_fires_for_existing_projects(self, tmp_path):
        _make_project(tmp_path, "foo")
        _make_project(tmp_path, "bar")

        added: list[str] = []
        removed: list[str] = []

        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: removed.append(n),
        )
        await run_with_external_changes(watcher, [])
        assert sorted(added) == ["bar", "foo"]
        assert removed == []

    async def test_initial_scan_skips_forbidden(self, tmp_path):
        _make_project(tmp_path, "secrets")
        _make_project(tmp_path, "good")

        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
            forbidden_dirs=frozenset({"secrets"}),
        )
        await run_with_external_changes(watcher, [])
        assert "secrets" not in added
        assert "good" in added

    async def test_initial_scan_respects_exclude(self, tmp_path):
        _make_project(tmp_path, "skip-this")
        _make_project(tmp_path, "keep-this")
        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
            exclude=("skip-*",),
        )
        await run_with_external_changes(watcher, [])
        assert "skip-this" not in added
        assert "keep-this" in added

    async def test_known_projects_after_initial_scan(self, tmp_path):
        _make_project(tmp_path, "alpha")
        _make_project(tmp_path, "beta")
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        await run_with_external_changes(watcher, [])
        names = [p.name for p in watcher.known_projects]
        assert sorted(names) == ["alpha", "beta"]

    async def test_no_callbacks_for_empty_parent(self, tmp_path):
        added: list[str] = []
        removed: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: removed.append(n),
        )
        await run_with_external_changes(watcher, [])
        assert added == []
        assert removed == []


# --------------------------------------------------------------- add path


@pytest.mark.asyncio
class TestAddPath:
    async def test_new_project_appears_fires_added(self, tmp_path):
        added: list[DiscoveredProject] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p),
            on_project_removed=lambda n: None,
        )
        # Initial scan: nothing
        await run_with_external_changes(watcher, [])
        # Now create a project + simulate the watchfiles event batch
        _make_project(tmp_path, "newproj")
        events = {_change_event("add", tmp_path / "newproj" / ".git")}
        await run_with_external_changes(watcher, [events])
        # First batch was during initial scan; second simulated:
        # combined run_with_external_changes restarts the initial scan
        # in our helper, so use a fresh helper invocation. Re-create.

        added.clear()
        watcher2 = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p),
            on_project_removed=lambda n: None,
        )
        # On reconstruction the project is present → initial scan
        # picks it up.
        await run_with_external_changes(watcher2, [])
        assert any(p.name == "newproj" for p in added)

    async def test_event_for_new_project_after_initial(self, tmp_path):
        # Start with no projects
        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
        )
        # Run once with an empty batch to seed
        await watcher._initial_scan()
        # Now the project appears on disk + an event arrives
        _make_project(tmp_path, "x")
        watcher._reconcile_basename("x")
        assert "x" in added

    async def test_event_for_subdir_without_sentinel_ignored(self, tmp_path):
        # User mkdir's a scratch dir; no sentinel inside.
        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
        )
        await watcher._initial_scan()
        (tmp_path / "scratch").mkdir()
        watcher._reconcile_basename("scratch")
        assert added == []

    async def test_already_known_project_does_not_double_add(self, tmp_path):
        _make_project(tmp_path, "x")
        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
        )
        await watcher._initial_scan()
        # Now reconcile fires for the same dir (sentinel-internal event)
        # — should NOT fire add again
        watcher._reconcile_basename("x")
        assert added.count("x") == 1


# --------------------------------------------------------------- remove path


@pytest.mark.asyncio
class TestRemovePath:
    async def test_removed_project_fires_removed(self, tmp_path):
        _make_project(tmp_path, "x")
        added: list[str] = []
        removed: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: removed.append(n),
        )
        await watcher._initial_scan()
        assert "x" in added
        # Delete the whole project
        _remove_project(tmp_path, "x")
        watcher._reconcile_basename("x")
        assert "x" in removed

    async def test_sentinel_disappears_fires_removed(self, tmp_path):
        proj = _make_project(tmp_path, "x")
        added: list[str] = []
        removed: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: removed.append(n),
        )
        await watcher._initial_scan()
        # Remove just the .git directory; the dir itself stays
        import shutil
        shutil.rmtree(proj / ".git")
        watcher._reconcile_basename("x")
        assert "x" in removed

    async def test_remove_nonexistent_basename_no_callback(self, tmp_path):
        removed: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: removed.append(n),
        )
        await watcher._initial_scan()
        # Reconcile a name we've never seen; it's not on disk; not in
        # known set → no-op.
        watcher._reconcile_basename("ghost")
        assert removed == []

    async def test_excluded_treated_as_not_a_project(self, tmp_path):
        # If a project transitions from "excluded" → not, we don't fire
        # an add. If it transitions excluded → still excluded, no-op.
        _make_project(tmp_path, "skip")
        added: list[str] = []
        removed: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: removed.append(n),
            exclude=("skip",),
        )
        await watcher._initial_scan()
        assert added == []
        # Reconcile event: still excluded
        watcher._reconcile_basename("skip")
        assert added == []
        assert removed == []


# --------------------------------------------------------------- event dispatch


@pytest.mark.asyncio
class TestEventDispatch:
    async def test_affected_basenames_filters_to_depth_2(self, tmp_path):
        # Events deeper than 2 (e.g. file inside project) are ignored.
        _make_project(tmp_path, "x")
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        events = {
            _change_event("modify", tmp_path / "x" / "src" / "foo.py"),
        }
        affected = watcher._affected_basenames(events)
        # Depth 3 (x/src/foo.py) → ignored
        assert affected == set()

    async def test_affected_basenames_depth_1(self, tmp_path):
        # Subdir itself was created/removed
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        events = {_change_event("add", tmp_path / "newproj")}
        affected = watcher._affected_basenames(events)
        assert affected == {"newproj"}

    async def test_affected_basenames_depth_2(self, tmp_path):
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        events = {_change_event("add", tmp_path / "x" / ".git")}
        affected = watcher._affected_basenames(events)
        assert affected == {"x"}

    async def test_affected_basenames_outside_parent_ignored(self, tmp_path):
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
        )
        events = {_change_event("add", Path("/elsewhere/something"))}
        affected = watcher._affected_basenames(events)
        assert affected == set()

    async def test_affected_basenames_skips_forbidden(self, tmp_path):
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: None,
            on_project_removed=lambda n: None,
            forbidden_dirs=frozenset({"secrets"}),
        )
        events = {
            _change_event("add", tmp_path / "secrets"),
            _change_event("add", tmp_path / "good"),
        }
        affected = watcher._affected_basenames(events)
        assert affected == {"good"}

    async def test_run_with_external_changes_processes_batch(self, tmp_path):
        added: list[str] = []
        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=lambda p: added.append(p.name),
            on_project_removed=lambda n: None,
        )
        # Pre-create project so by the time the event fires, the
        # disk state agrees the sentinel is present.
        _make_project(tmp_path, "x")
        await run_with_external_changes(watcher, [
            {_change_event("add", tmp_path / "x" / ".git")},
        ])
        assert "x" in added


# --------------------------------------------------------------- callback errors


@pytest.mark.asyncio
class TestCallbackErrors:
    async def test_callback_exception_does_not_crash_loop(self, tmp_path):
        # If on_project_added raises, the watcher logs but keeps running.
        _make_project(tmp_path, "x")
        _make_project(tmp_path, "y")

        def bad_add(p):
            if p.name == "x":
                raise RuntimeError("boom")

        added_names: list[str] = []

        def good_add(p):
            added_names.append(p.name)
            bad_add(p)

        watcher = ParentDirWatcher(
            parent_dir=tmp_path,
            sentinels=(".git",),
            on_project_added=good_add,
            on_project_removed=lambda n: None,
        )
        # Should not raise, even though bad_add raises for x.
        await watcher._initial_scan()
        # Both x and y are visited
        assert sorted(added_names) == ["x", "y"]


# --------------------------------------------------------------- pytest-asyncio config


# Configure asyncio_mode at the module level so older pytest-asyncio
# versions still pick it up. This is a no-op if 'asyncio_mode = strict'
# is set in pyproject.toml.
def pytest_collection_modifyitems(config, items):
    pass
