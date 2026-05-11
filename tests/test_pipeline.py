"""Tests for RetrievalPipeline using a mocked embedder.

Mocks the sentence-transformers SentenceTransformer + CrossEncoder so
tests run instantly without downloading a model or consuming GPU/RAM.
The retrieval logic (top-K selection, parent dedup, ordering) is what
needs testing, not the embedder math itself.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class FakeEmbedder:
    """Returns deterministic embeddings keyed by substring presence."""

    def encode(self, texts, convert_to_numpy=True,
               normalize_embeddings=True, batch_size=32, **_):
        out = []
        for t in texts:
            t_lower = t.lower()
            # 4-dim vector based on keyword presence
            vec = np.array([
                1.0 if "alpha" in t_lower else 0.0,
                1.0 if "beta" in t_lower else 0.0,
                1.0 if "gamma" in t_lower else 0.0,
                1.0 if "query" in t_lower else 0.5,
            ], dtype=np.float32)
            n = np.linalg.norm(vec) + 1e-8
            out.append(vec / n)
        return np.stack(out)


class FakeReranker:
    """Returns scores based on substring overlap."""

    def predict(self, pairs, batch_size=16, show_progress_bar=False, **_):
        scores = []
        for q, c in pairs:
            shared = sum(1 for w in q.lower().split()
                         if w in c.lower())
            scores.append(float(shared))
        return np.array(scores, dtype=np.float32)


def _make_pipeline(reranker=False):
    """Build a RetrievalPipeline with mocked encoder(s)."""
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=FakeEmbedder()
    ), patch(
        "sentence_transformers.CrossEncoder",
        return_value=FakeReranker()
    ):
        from longctx import RetrievalPipeline
        return RetrievalPipeline(
            reranker_model="fake/reranker" if reranker else None,
            cache_dir=None,  # avoid disk-cache dim-mismatch under mocks
        )


def test_imports():
    from longctx import LongCtxClient, RetrievalPipeline
    assert RetrievalPipeline is not None
    assert LongCtxClient is not None


def test_retrieve_basic():
    pipeline = _make_pipeline()
    candidates = ["mostly alpha here", "all about beta", "gamma everywhere"]
    result = pipeline.retrieve(query="alpha thing",
                               candidates=candidates, top_k=1)
    assert len(result.candidates) == 1
    assert "alpha" in result.candidates[0]


def test_retrieve_preserves_order():
    pipeline = _make_pipeline()
    candidates = [
        "third item gamma",
        "first item alpha",
        "second item beta",
    ]
    result = pipeline.retrieve(query="alpha beta gamma",
                               candidates=candidates,
                               top_k=3, preserve_order=True)
    assert result.indices == sorted(result.indices)


def test_retrieve_preserve_order_false_returns_score_order():
    pipeline = _make_pipeline()
    candidates = ["x irrelevant", "alpha alpha alpha", "beta only"]
    result = pipeline.retrieve(query="alpha",
                               candidates=candidates,
                               top_k=3, preserve_order=False)
    # With preserve_order=False, "alpha alpha alpha" should be first
    assert result.candidates[0].count("alpha") == 3


def test_retrieve_with_reranker_uses_two_stage():
    pipeline = _make_pipeline(reranker=True)
    candidates = ["alpha doc", "beta doc", "gamma doc", "delta doc"]
    result = pipeline.retrieve(query="alpha doc",
                               candidates=candidates,
                               top_k=2, retrieve_k=3)
    # Reranker scores by word overlap; "alpha doc" matches both words
    assert "alpha" in result.candidates[0] or "alpha" in result.candidates[1]


def test_retrieve_top_k_larger_than_candidates():
    pipeline = _make_pipeline()
    candidates = ["alpha", "beta"]
    result = pipeline.retrieve(query="alpha", candidates=candidates,
                               top_k=10)
    # Should return all available candidates without crashing
    assert len(result.candidates) == 2


def test_retrieve_chunked_basic():
    pipeline = _make_pipeline()
    candidates = [
        "irrelevant content here. " * 10,
        "this paragraph has alpha keyword in it. " * 5,
        "different gamma topic discussion. " * 8,
    ]
    result = pipeline.retrieve_chunked(
        query="alpha",
        candidates=candidates,
        top_k=1,
        chunk_size=20,
    )
    assert result.indices[0] == 1
    assert "alpha" in result.candidates[0]


def test_retrieve_chunked_dedupes_parents():
    pipeline = _make_pipeline()
    candidates = [
        "irrelevant",
        "alpha alpha alpha. " * 20,  # long; many alpha-rich chunks
        "irrelevant 2",
    ]
    result = pipeline.retrieve_chunked(
        query="alpha", candidates=candidates,
        top_k=2, chunk_size=10,
    )
    # Even though candidate 1 has many alpha-matching chunks,
    # we should get 2 UNIQUE parents.
    assert len(set(result.indices)) == len(result.indices)
    assert len(result.indices) == 2
    assert 1 in result.indices


def test_retrieve_chunked_preserves_order():
    pipeline = _make_pipeline()
    candidates = [
        "third item gamma alpha",
        "first item nothing",
        "second item beta alpha",
    ]
    result = pipeline.retrieve_chunked(
        query="alpha", candidates=candidates,
        top_k=2, chunk_size=10, preserve_order=True,
    )
    assert result.indices == sorted(result.indices)


def test_templates_present():
    from longctx.templates import TEMPLATES
    for name, template in TEMPLATES.items():
        assert isinstance(template, str), name
        assert len(template) > 50, name
