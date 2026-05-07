"""Idle eviction tests. PRD §5.7."""
from __future__ import annotations

import time

import numpy as np
import pytest

from longctx_svc.config import Limits, ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk
from longctx_svc.session.manager import SessionManager
from longctx_svc.state import get_state, reset_state


def _index_with_embeddings(scope_root, scope_hash="h"):
    chunks = [Chunk(
        text="x = 1\n", file_path=str(scope_root / "a.py"),
        start_line=1, end_line=1, file_type="code",
    )]
    embs = np.ones((1, 4), dtype=np.float32)
    return ScopeIndex(
        scope_root=scope_root, scope_hash=scope_hash,
        chunks=chunks, embeddings=embs, file_count=1,
        built_at=time.time(),
    )


def test_evict_idle_indexes_drops_stale(tmp_path):
    reset_state()
    # Tight idle window
    set_config(ServiceConfig(
        cache_dir=tmp_path / "cache",
        limits=Limits(index_idle_timeout_seconds=1),
    ))
    state = get_state()
    root = tmp_path / "p"
    root.mkdir()
    entry = state.register_scope(root, "h-stale", "package.json")
    entry.index = _index_with_embeddings(root, "h-stale")
    entry.status = "ready"
    entry.last_query_at = time.time() - 10  # well past 1s

    n = state.evict_idle_indexes()
    assert n == 1
    assert entry.index is None
    assert entry.status == "evicted"
    set_config(ServiceConfig())


def test_evict_idle_indexes_keeps_recent(tmp_path):
    reset_state()
    set_config(ServiceConfig(
        cache_dir=tmp_path / "cache",
        limits=Limits(index_idle_timeout_seconds=3600),
    ))
    state = get_state()
    root = tmp_path / "p2"
    root.mkdir()
    entry = state.register_scope(root, "h-fresh", "package.json")
    entry.index = _index_with_embeddings(root, "h-fresh")
    entry.last_query_at = time.time()  # just now

    assert state.evict_idle_indexes() == 0
    assert entry.index is not None
    set_config(ServiceConfig())


def test_session_evict_idle():
    sm = SessionManager()
    sm.bind("alice", "scope-a")
    sm.bind("bob", "scope-b")
    set_config(ServiceConfig(limits=Limits(
        session_idle_timeout_seconds=1,
    )))
    # Backdate alice
    sm._sessions["alice"].last_seen_at = time.time() - 10
    assert sm.evict_idle() == 1
    assert "alice" not in sm._sessions
    assert "bob" in sm._sessions
    set_config(ServiceConfig())


def test_evict_clears_watcher_too(tmp_path):
    """When evicting an indexed scope, attached watcher must stop."""
    reset_state()
    set_config(ServiceConfig(
        cache_dir=tmp_path / "cache",
        limits=Limits(index_idle_timeout_seconds=1),
    ))
    state = get_state()
    root = tmp_path / "p"
    root.mkdir()
    entry = state.register_scope(root, "h-w", "package.json")
    entry.index = _index_with_embeddings(root, "h-w")
    entry.last_query_at = time.time() - 10

    class FakeWatcher:
        def __init__(self):
            self.stopped = False
        def stop(self):
            self.stopped = True

    fw = FakeWatcher()
    entry.watcher = fw
    state.evict_idle_indexes()
    assert fw.stopped is True
    assert entry.watcher is None
    set_config(ServiceConfig())
