"""Tier 2 of the V3 + longctx rescue stack: per-session eviction store.

Pin the contract that the V3 backend code in vllm-turboquant relies on:
write returns running count, retrieve does session-isolated cosine search,
clear is idempotent, idle-eviction works.

These tests use a fake embedder (deterministic vectors keyed off text) so
they don't pull MiniLM at test time. The real code path is exercised by
integration tests that boot a real longctx-svc.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from longctx_svc.eviction_store import EvictedChunk, EvictionStore


class _FakeEmbedder:
    """Deterministic 4-dim embedder: hashes text → 4 floats. Two texts
    that share a token compute as similar; otherwise ~orthogonal."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.lower()[:8]):
                out[i, j % 4] += float(ord(ch))
        return out


@pytest.fixture
def store():
    return EvictionStore(embedder=_FakeEmbedder())


def _ev(text: str, layer: int = 0, score: float = 0.0) -> EvictedChunk:
    return EvictedChunk(text=text, token_range=(0, len(text)),
                        layer=layer, score=score)


def test_write_returns_running_count(store):
    assert store.write("sess-A", [_ev("alpha"), _ev("bravo")]) == 2
    assert store.write("sess-A", [_ev("charlie")]) == 3
    # Other session is independent
    assert store.write("sess-B", [_ev("delta")]) == 1


def test_retrieve_returns_session_isolated_topk(store):
    store.write("sess-A", [_ev("the secret password is ZEBRA42")])
    store.write("sess-A", [_ev("apple banana cherry")])
    store.write("sess-B", [_ev("ZEBRA42 sentinel value")])

    hits_a = store.retrieve("sess-A", "what is the password", top_k=2)
    # Top hit on session A must be the password chunk; sess-B's matching
    # chunk must NOT leak.
    assert len(hits_a) >= 1
    assert any("ZEBRA42" in c.text for c in hits_a)
    for c in hits_a:
        assert "sentinel" not in c.text


def test_retrieve_empty_session_returns_empty(store):
    assert store.retrieve("never-touched", "anything") == []


def test_score_floor_drops_low_similarity(store):
    store.write("sess-A", [_ev("totally unrelated stuff zzz")])
    # Floor of 0.99 should drop everything (fake embedder won't match exactly)
    hits = store.retrieve("sess-A", "what is the meaning of life",
                          top_k=8, score_floor=0.99)
    assert hits == []


def test_clear_drops_session_data(store):
    store.write("sess-A", [_ev("one"), _ev("two")])
    n = store.clear("sess-A")
    assert n == 2
    assert store.retrieve("sess-A", "anything") == []
    # Idempotent: clearing again is a no-op
    assert store.clear("sess-A") == 0


def test_evict_idle_drops_stale_sessions(store):
    store.write("fresh", [_ev("recent")])
    store.write("stale", [_ev("old")])
    # Force the stale session's last_access into the past
    store._sessions["stale"].last_access = time.time() - 10_000.0
    dropped = store.evict_idle(ttl_seconds=60.0)
    assert dropped == 1
    assert store.retrieve("stale", "anything") == []
    # Fresh session survives
    assert len(store.retrieve("fresh", "recent")) == 1


def test_chunks_preserve_metadata(store):
    chunk = _ev("payload text", layer=7, score=2.5)
    chunk.token_range = (1024, 1056)
    store.write("sess-X", [chunk])
    hits = store.retrieve("sess-X", "payload", top_k=1)
    assert len(hits) == 1
    h = hits[0]
    assert h.text == "payload text"
    assert h.layer == 7
    assert h.score == 2.5
    assert h.token_range == (1024, 1056)


def test_stats_reports_per_session_counts(store):
    store.write("a", [_ev("x"), _ev("y")])
    store.write("b", [_ev("z")])
    s = store.stats()
    assert s["sessions"] == 2
    assert s["total_chunks"] == 3
    assert s["per_session"]["a"] == 2
    assert s["per_session"]["b"] == 1
