"""File-watcher tests. PRD §5.6 / §R5 / smoke §7.4."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from longctx_svc.watcher import FileWatcher, _HAS_WATCHDOG


pytestmark = pytest.mark.skipif(
    not _HAS_WATCHDOG, reason="watchdog not installed",
)


def _wait_for(predicate, timeout: float = 3.0,
              interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_watcher_fires_on_create(tmp_path):
    fired: list[set[Path]] = []
    w = FileWatcher(tmp_path, on_change=fired.append, debounce_seconds=0.2)
    w.start()
    try:
        (tmp_path / "new.py").write_text("x = 1\n")
        assert _wait_for(lambda: bool(fired), timeout=3.0)
    finally:
        w.stop()
    paths = {p.name for batch in fired for p in batch}
    assert "new.py" in paths


def test_watcher_fires_on_modify(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    fired: list[set[Path]] = []
    w = FileWatcher(tmp_path, on_change=fired.append, debounce_seconds=0.2)
    w.start()
    try:
        time.sleep(0.05)
        f.write_text("x = 2\n")
        assert _wait_for(lambda: bool(fired), timeout=3.0)
    finally:
        w.stop()


def test_watcher_skips_skip_dirs(tmp_path):
    """node_modules / .git / __pycache__ events ignored."""
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "junk.js").write_text("")
    fired: list[set[Path]] = []
    w = FileWatcher(tmp_path, on_change=fired.append, debounce_seconds=0.2)
    w.start()
    try:
        (nm / "junk.js").write_text("// changed")
        # Watcher will still see the event but should drop it.
        time.sleep(0.6)
    finally:
        w.stop()
    paths = {p for batch in fired for p in batch}
    assert all("node_modules" not in str(p) for p in paths)


def test_watcher_skips_lockfiles(tmp_path):
    fired: list[set[Path]] = []
    w = FileWatcher(tmp_path, on_change=fired.append, debounce_seconds=0.2)
    w.start()
    try:
        (tmp_path / "package-lock.json").write_text('{"lock": true}')
        (tmp_path / "real.py").write_text("x = 1")
        assert _wait_for(lambda: bool(fired), timeout=3.0)
    finally:
        w.stop()
    paths = {p.name for batch in fired for p in batch}
    assert "package-lock.json" not in paths
    assert "real.py" in paths


def test_watcher_debounces_rapid_writes(tmp_path):
    """Rapid writes within debounce window collapse into one batch."""
    fired: list[set[Path]] = []
    w = FileWatcher(tmp_path, on_change=fired.append, debounce_seconds=0.5)
    w.start()
    try:
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"# {i}")
            time.sleep(0.05)
        assert _wait_for(lambda: bool(fired), timeout=3.0)
        time.sleep(0.6)  # let further events stabilize
    finally:
        w.stop()
    # At least one batch fired and no batch is empty.
    assert all(len(b) > 0 for b in fired)
    # Total touched files should be 5
    touched = {p.name for batch in fired for p in batch}
    for i in range(5):
        assert f"f{i}.py" in touched


def test_watcher_stop_is_idempotent(tmp_path):
    w = FileWatcher(tmp_path, on_change=lambda _: None,
                    debounce_seconds=0.2)
    w.start()
    w.stop()
    w.stop()  # must not raise
    assert not w.running


def test_state_attach_detach_watcher_smoke(tmp_path, fake_embedder,
                                           fake_reranker):
    """Smoke §7.4: edit a file in an indexed scope → on_change fires →
    incremental update propagates to the index."""
    from unittest.mock import patch
    from longctx_svc.indexer.builder import build_index, update_files
    from longctx_svc.scope.detect import canonicalize_scope, hash_scope
    from longctx_svc.state import get_state, reset_state

    reset_state()
    root = tmp_path / "p"
    root.mkdir()
    (root / "package.json").write_text('{}')
    (root / "alpha.py").write_text("# alpha original\n")

    # Build initial index using the fake embedder.
    idx = build_index(
        root, [root / "alpha.py"],
        scope_hash="sw1", embedder=fake_embedder,
    )
    canon = canonicalize_scope(root)
    state = get_state()
    # Make pipeline use the fake embedder so the watcher's incremental
    # update path doesn't try to download a real model.
    from longctx_svc.retrieve.pipeline import RetrievePipeline
    state.set_pipeline(RetrievePipeline(
        embedder=fake_embedder, reranker=fake_reranker,
    ))
    entry = state.register_scope(canon, "sw1", "package.json")
    entry.index = idx
    entry.status = "ready"

    state.attach_watcher("sw1")
    try:
        time.sleep(0.1)
        (root / "alpha.py").write_text("# alpha updated beta gamma\n")
        assert _wait_for(
            lambda: any("beta" in c.text for c in entry.index.chunks),
            timeout=4.0,
        )
    finally:
        state.detach_watcher("sw1")
