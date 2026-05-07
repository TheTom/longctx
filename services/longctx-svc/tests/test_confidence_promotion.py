"""Confidence-driven promotion tests. PRD §6.2 / v0.3.2."""
from __future__ import annotations

import time

import numpy as np
import pytest

from longctx_svc.config import Limits, ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk
from longctx_svc.retrieve.pipeline import RetrievePipeline
from longctx_svc.session.manager import SessionEntry, SessionManager
from longctx_svc.state import get_state, reset_state


def _seed(tmp_path, fake_embedder, fake_reranker, *,
          threshold=0.30, consecutive=2):
    reset_state()
    set_config(ServiceConfig(
        cache_dir=tmp_path / "cache",
        limits=Limits(
            confidence_threshold=threshold,
            confidence_consecutive_low=consecutive,
        ),
    ))
    root = tmp_path / "p"
    root.mkdir()
    (root / "package.json").write_text('{}')
    (root / "alpha.py").write_text("# alpha\n")
    state = get_state()
    state.set_pipeline(RetrievePipeline(
        embedder=fake_embedder, reranker=fake_reranker,
    ))
    entry = state.register_scope(root, "conf1", "package.json")
    entry.index = ScopeIndex(
        scope_root=root, scope_hash="conf1",
        chunks=[Chunk(text="a\n", file_path=str(root / "alpha.py"),
                      start_line=1, end_line=1, file_type="code")],
        embeddings=np.ones((1, 4), dtype=np.float32) / 2,
        file_count=1, built_at=time.time(),
    )
    entry.indexed_files = frozenset({str(root / "alpha.py")})
    entry.mode = "hot"
    entry.status = "ready"
    state.sessions.bind("alice", "conf1")
    return state


def test_record_confidence_appends_to_window():
    sess = SessionEntry(
        session_id="x", scope_hash=None, last_seen_at=0.0,
        detected_via="ephemeral", confidence_window_size=3,
    )
    sess.record_confidence(0.5)
    sess.record_confidence(0.3)
    assert sess.confidence_window == [0.5, 0.3]


def test_record_confidence_truncates_to_window_size():
    sess = SessionEntry(
        session_id="x", scope_hash=None, last_seen_at=0.0,
        detected_via="ephemeral", confidence_window_size=3,
    )
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        sess.record_confidence(v)
    assert sess.confidence_window == [0.3, 0.4, 0.5]


def test_no_promotion_when_score_above_threshold(tmp_path, fake_embedder,
                                                    fake_reranker):
    state = _seed(tmp_path, fake_embedder, fake_reranker, threshold=0.30)
    promoted = state.maybe_promote_on_confidence("conf1", "alice", 0.50)
    assert promoted is False
    assert state.get_scope("conf1").mode == "hot"
    set_config(ServiceConfig())


def test_promotion_after_consecutive_low_scores(tmp_path, fake_embedder,
                                                   fake_reranker):
    state = _seed(tmp_path, fake_embedder, fake_reranker,
                  threshold=0.30, consecutive=2)
    # First low score: counter=1, no promotion yet
    p1 = state.maybe_promote_on_confidence("conf1", "alice", 0.10)
    assert p1 is False
    # Second consecutive low: should fire
    p2 = state.maybe_promote_on_confidence("conf1", "alice", 0.05)
    assert p2 is True
    assert state.get_scope("conf1").status == "promoting"
    set_config(ServiceConfig())


def test_high_score_resets_streak(tmp_path, fake_embedder, fake_reranker):
    """A single good turn breaks the consecutive-low streak so we don't
    over-promote on transient bad queries."""
    state = _seed(tmp_path, fake_embedder, fake_reranker, threshold=0.30)
    state.maybe_promote_on_confidence("conf1", "alice", 0.10)  # low (1)
    state.maybe_promote_on_confidence("conf1", "alice", 0.50)  # high → reset
    p = state.maybe_promote_on_confidence("conf1", "alice", 0.10)  # low (1, not 2)
    assert p is False
    assert state.get_scope("conf1").mode == "hot"
    set_config(ServiceConfig())


def test_no_promotion_when_session_unknown(tmp_path, fake_embedder,
                                              fake_reranker):
    state = _seed(tmp_path, fake_embedder, fake_reranker)
    p = state.maybe_promote_on_confidence("conf1", "ghost", 0.0)
    assert p is False
    set_config(ServiceConfig())


def test_no_promotion_when_session_id_empty(tmp_path, fake_embedder,
                                               fake_reranker):
    state = _seed(tmp_path, fake_embedder, fake_reranker)
    p = state.maybe_promote_on_confidence("conf1", "", 0.0)
    assert p is False
    set_config(ServiceConfig())


def test_no_promotion_in_already_package_mode(tmp_path, fake_embedder,
                                                 fake_reranker):
    state = _seed(tmp_path, fake_embedder, fake_reranker,
                  threshold=0.30, consecutive=2)
    state.get_scope("conf1").mode = "package"
    state.maybe_promote_on_confidence("conf1", "alice", 0.05)
    p = state.maybe_promote_on_confidence("conf1", "alice", 0.05)
    assert p is False
    set_config(ServiceConfig())


def test_streak_resets_after_firing(tmp_path, fake_embedder, fake_reranker):
    """After firing, consecutive_low resets so we don't re-promote next turn."""
    state = _seed(tmp_path, fake_embedder, fake_reranker,
                  threshold=0.30, consecutive=2)
    state.maybe_promote_on_confidence("conf1", "alice", 0.10)
    fired = state.maybe_promote_on_confidence("conf1", "alice", 0.10)
    assert fired is True
    sess = state.sessions.get("alice")
    assert sess.consecutive_low == 0
    set_config(ServiceConfig())


# --- end-to-end via /retrieve ---

def test_retrieve_emits_confidence_header(client, project_dir):
    """PRD §6.2 visibility: x-longctx-confidence on every reply."""
    r = client.post(
        "/retrieve",
        headers={"x-session-affinity": "conf-e2e"},
        json={"prefill_text": f"see {project_dir}/src/auth.ts",
              "query": "auth", "top_k": 4},
    )
    assert r.status_code == 200
    assert "x-longctx-confidence" in r.headers
    score = float(r.headers["x-longctx-confidence"])
    assert 0.0 <= score <= 1.0
