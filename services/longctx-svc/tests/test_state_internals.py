# SPDX-License-Identifier: Apache-2.0
"""Coverage for `longctx_svc.state._State` lifecycle methods.

Existing test files cover the happy paths for register/promote/evict.
This file targets the "early return" branches, LRU cap eviction,
watcher attach/detach edge cases, and reset_state cleanup — which are
load-bearing for graceful shutdown but rarely tripped in happy-path
integration tests.
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from longctx_svc.state import _State, get_state, reset_state


@pytest.fixture(autouse=True)
def _isolate_state():
    """Each test gets a fresh _State singleton."""
    reset_state()
    yield
    reset_state()


def _mk_entry(state: _State, scope_hash: str, *, index=None,
              mode: str = "hot", status: str = "ready"):
    """Insert a synthetic ScopeEntry without going through disk reload.

    load_index is imported inside register_scope, so patch its source module.
    """
    with patch("longctx_svc.cache.disk.load_index", return_value=None):
        entry = state.register_scope(
            Path("/fake/root"), scope_hash, sentinel=".git",
        )
    entry.index = index
    entry.mode = mode
    entry.status = status
    return entry


# ----------------------------------------------------- pipeline lazy load


def test_pipeline_lazy_initializes_on_first_access():
    state = _State()
    assert state._pipeline is None
    pipe = state.pipeline
    assert pipe is not None
    # Second access returns the same instance.
    assert state.pipeline is pipe


def test_set_pipeline_overrides_lazy_init():
    state = _State()
    sentinel = MagicMock()
    state.set_pipeline(sentinel)
    assert state.pipeline is sentinel


# ----------------------------------------------------- evict_idle_indexes


def test_evict_idle_skips_scope_without_index():
    """Entry with .index = None must be skipped (line 105)."""
    state = _State()
    _mk_entry(state, "hash-a", index=None)
    # Nothing to evict.
    assert state.evict_idle_indexes(now=1e18) == 0


def test_evict_idle_skips_scope_without_embeddings():
    """Entry with index.embeddings is None must be skipped (line 107)."""
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = None
    _mk_entry(state, "hash-b", index=fake_index)
    assert state.evict_idle_indexes(now=1e18) == 0


def test_evict_idle_swallows_watcher_stop_exception():
    """If watcher.stop() raises during idle eviction, the eviction must
    still complete (lines 113-114)."""
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()  # truthy
    entry = _mk_entry(state, "hash-c", index=fake_index)
    entry.last_query_at = 0.0  # very old → idle
    entry.watcher = MagicMock()
    entry.watcher.stop.side_effect = RuntimeError("watcher boom")

    n = state.evict_idle_indexes(now=1e18)
    assert n == 1
    assert entry.index is None
    assert entry.status == "evicted"
    assert entry.watcher is None


# ----------------------------------------------------- maybe_promote


def test_maybe_promote_returns_false_when_index_missing():
    """Line 287: index is None → can't promote."""
    state = _State()
    _mk_entry(state, "h-noidx", index=None, mode="hot")
    assert state.maybe_promote("h-noidx", mentioned=["/some/path"]) is False


def test_maybe_promote_returns_false_when_no_indexed_files():
    """Line 293: indexed_files empty → no point checking outside."""
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    entry = _mk_entry(state, "h-empty-idx", index=fake_index, mode="hot")
    entry.indexed_files = frozenset()
    assert state.maybe_promote("h-empty-idx", mentioned=["/some/x"]) is False


def test_maybe_promote_returns_false_when_status_bad():
    """status in (indexing, promoting, error, evicted) → bail."""
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    entry = _mk_entry(state, "h-err", index=fake_index, mode="hot")
    entry.status = "error"
    assert state.maybe_promote("h-err", mentioned=["/x"]) is False


def test_maybe_promote_returns_false_when_nothing_outside_hot():
    """All mentioned paths already in the hot index → no promotion."""
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    entry = _mk_entry(state, "h-allin", index=fake_index, mode="hot")
    entry.indexed_files = frozenset(["/scope/a.py", "/scope/b.py"])
    assert state.maybe_promote(
        "h-allin", mentioned=["/scope/a.py"],
    ) is False


def test_maybe_promote_returns_false_when_no_mentioned():
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    entry = _mk_entry(state, "h-nom", index=fake_index, mode="hot")
    entry.indexed_files = frozenset(["/scope/a.py"])
    assert state.maybe_promote("h-nom", mentioned=[]) is False


# ----------------------------------------------------- watcher attach/detach


def test_attach_watcher_noop_on_missing_scope():
    state = _State()
    # Should not raise.
    state.attach_watcher("never-existed")


def test_attach_watcher_idempotent_when_already_attached():
    state = _State()
    entry = _mk_entry(state, "h-w", index=MagicMock())
    entry.watcher = MagicMock(name="existing-watcher")
    state.attach_watcher("h-w")
    # Existing watcher untouched.
    assert entry.watcher is entry.watcher


def test_detach_watcher_noop_on_missing_scope():
    state = _State()
    state.detach_watcher("never-existed")  # must not raise


def test_detach_watcher_noop_when_already_detached():
    state = _State()
    entry = _mk_entry(state, "h-d", index=MagicMock())
    entry.watcher = None
    state.detach_watcher("h-d")  # must not raise


def test_detach_watcher_swallows_stop_exception():
    state = _State()
    entry = _mk_entry(state, "h-flaky", index=MagicMock())
    entry.watcher = MagicMock()
    entry.watcher.stop.side_effect = RuntimeError("boom")
    state.detach_watcher("h-flaky")
    assert entry.watcher is None


# ----------------------------------------------------- _begin_package_rebuild


def test_begin_package_rebuild_skips_when_already_promoting():
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    entry = _mk_entry(state, "h-busyp", index=fake_index, mode="hot")
    entry.status = "promoting"
    assert state._begin_package_rebuild(
        "h-busyp", reason="test"
    ) is False


def test_begin_package_rebuild_skips_when_already_package():
    state = _State()
    fake_index = MagicMock()
    fake_index.embeddings = MagicMock()
    _mk_entry(state, "h-pkg2", index=fake_index, mode="package")
    assert state._begin_package_rebuild(
        "h-pkg2", reason="test"
    ) is False


def test_begin_package_rebuild_skips_when_scope_missing():
    state = _State()
    assert state._begin_package_rebuild(
        "never-existed", reason="test"
    ) is False


# ----------------------------------------------------- reset_state


def test_reset_state_swallows_watcher_stop_exception():
    """Reset must succeed even if a watcher.stop() raises (lines 379-380)."""
    state = get_state()
    entry = _mk_entry(state, "h-reset", index=MagicMock())
    entry.watcher = MagicMock()
    entry.watcher.stop.side_effect = RuntimeError("flap")
    # Should not raise.
    reset_state()
    # Singleton is gone.
    fresh = get_state()
    assert fresh is not state
