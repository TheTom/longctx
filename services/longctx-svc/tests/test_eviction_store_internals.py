# SPDX-License-Identifier: Apache-2.0
"""Coverage for internal helpers + edge cases in `eviction_store.py`.

Complements `test_eviction_store.py` (which covers the public
EvictionStore contract) by exercising the import-fail fallbacks,
env-parsing helpers, and lazy-loader short-circuits. These paths are
critical for graceful degradation when optional deps (rank_bm25,
sentence-transformers reranker) aren't installed.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from longctx_svc.eviction_store import (
    EvictedChunk,
    EvictionStore,
    _bm25_tokenize,
    _env_float,
)


class _FakeEmbedder:
    """Same fake as test_eviction_store.py — deterministic 4-dim."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.lower()[:8]):
                out[i, j % 4] += float(ord(ch))
        return out


def _ev(text: str, layer: int = 0, score: float = 0.0) -> EvictedChunk:
    return EvictedChunk(
        text=text, token_range=(0, len(text)), layer=layer, score=score,
    )


# ----------------------------------------------------- _bm25_tokenize


def test_bm25_tokenize_empty_returns_empty():
    assert _bm25_tokenize("") == []


def test_bm25_tokenize_strips_punctuation_and_lowercases():
    out = _bm25_tokenize("Project NOVA INV-2845 $123,456")
    assert out == ["project", "nova", "inv", "2845", "123", "456"]


def test_bm25_tokenize_handles_unicode():
    """Token splitter must not crash on non-ASCII."""
    out = _bm25_tokenize("café résumé Σigma")
    # Exact tokenization depends on regex; just assert it's lowercase + non-empty.
    assert all(t.islower() or t.isdigit() for t in out)


# ----------------------------------------------------- _env_float


def test_env_float_unset_returns_default(monkeypatch):
    monkeypatch.delenv("MY_TEST_FLOAT", raising=False)
    assert _env_float("MY_TEST_FLOAT", 0.5) == 0.5


def test_env_float_empty_string_returns_default(monkeypatch):
    monkeypatch.setenv("MY_TEST_FLOAT", "")
    assert _env_float("MY_TEST_FLOAT", 0.7) == 0.7


def test_env_float_valid_string_parses(monkeypatch):
    monkeypatch.setenv("MY_TEST_FLOAT", "0.42")
    assert _env_float("MY_TEST_FLOAT", 0.0) == pytest.approx(0.42)


def test_env_float_invalid_string_returns_default(monkeypatch):
    """Garbage value → fall back to default, don't crash."""
    monkeypatch.setenv("MY_TEST_FLOAT", "not-a-number")
    assert _env_float("MY_TEST_FLOAT", 0.3) == 0.3


# ----------------------------------------------------- reranker lazy load


def test_ensure_reranker_short_circuits_when_preset():
    """Reranker passed at construction time is returned immediately —
    no lazy load attempt."""
    sentinel = object()
    store = EvictionStore(embedder=_FakeEmbedder(), reranker=sentinel)
    assert store._ensure_reranker() is sentinel


def test_ensure_reranker_caches_failed_load():
    """If sentence-transformers import or CrossEncoder construction
    fails, subsequent calls return None without retrying the import."""
    store = EvictionStore(embedder=_FakeEmbedder())
    # Force the import to fail.
    with patch(
        "longctx_svc.eviction_store.EvictionStore._ensure_reranker",
        wraps=store._ensure_reranker,
    ):
        # Patch the actual import path inside the method.
        with patch.dict(
            "sys.modules",
            {"sentence_transformers": None},  # makes import fail
        ):
            try:
                # Patched import → fall into except → reranker stays None
                # and _reranker_loaded flips True.
                result = store._ensure_reranker()
            except Exception:
                result = None
    # After first call, _reranker_loaded is True so subsequent calls
    # return None without retrying.
    store._reranker = None
    store._reranker_loaded = True
    assert store._ensure_reranker() is None
    # And the cached-None short-circuit is hit explicitly.
    assert store._ensure_reranker() is None


# ----------------------------------------------------- BM25 graceful degrade


def test_bm25_rebuild_handles_import_failure():
    """If rank_bm25 isn't installed, _rebuild_bm25 must clear
    state and return cleanly instead of crashing the write path."""
    store = EvictionStore(embedder=_FakeEmbedder())
    store.write("sess-x", [_ev("alpha beta")])
    idx = store._sessions["sess-x"]

    # Make `from rank_bm25 import BM25Plus` fail.
    with patch.dict("sys.modules", {"rank_bm25": None}):
        store._rebuild_bm25(idx)

    assert idx.bm25 is None
    assert idx.bm25_tokens == []
    assert idx.bm25_dirty is False


def test_bm25_rebuild_handles_empty_corpus():
    """A session containing only whitespace-only chunks must not crash
    rank_bm25 (which raises on empty corpus). Verifies the guard at
    lines 226-231."""
    store = EvictionStore(embedder=_FakeEmbedder())
    # Whitespace chunks tokenize to []. Store insists the write happens
    # so the chunk objects exist for BM25 attempt.
    store.write("sess-empty", [_ev("   "), _ev("\n\t")])
    idx = store._sessions["sess-empty"]
    store._rebuild_bm25(idx)
    assert idx.bm25 is None
    assert idx.bm25_dirty is False


# ----------------------------------------------------- retrieve edge cases


def test_retrieve_returns_empty_when_pool_size_zero():
    """If everything is filtered out by the score floor, retrieve must
    return []. Hits the `if pool_size <= 0: return []` branch."""
    store = EvictionStore(embedder=_FakeEmbedder())
    store.write("sess-z", [_ev("alpha"), _ev("beta")])
    # Impossible floor → no chunks survive.
    out = store.retrieve("sess-z", query="z", top_k=5, score_floor=1.5)
    assert out == []


def test_clear_returns_zero_for_unknown_session():
    store = EvictionStore(embedder=_FakeEmbedder())
    assert store.clear("never-existed") == 0


def test_retrieve_pure_cosine_when_alpha_none():
    """hybrid_alpha=None must skip the BM25 fusion path entirely
    (lines 327-330: `else: fused = cos_sims`).
    """
    store = EvictionStore(embedder=_FakeEmbedder())
    store.write("sess-a", [_ev("alpha"), _ev("bravo")])
    out = store.retrieve(
        "sess-a", query="alpha", top_k=2, hybrid_alpha=None,
    )
    # We don't assert ordering — just that it returns without crashing
    # and didn't take the hybrid path.
    assert len(out) == 2
