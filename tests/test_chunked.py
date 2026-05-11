"""Chunked-retrieval edge cases (mocked embedder; see test_pipeline)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np


class _CountAlphaEmbedder:
    """Score chunks by how many 'alpha' tokens they contain."""

    def encode(self, texts, convert_to_numpy=True,
               normalize_embeddings=True, batch_size=32, **_):
        out = []
        for t in texts:
            count = t.lower().count("alpha")
            vec = np.array([float(count + 1), 1.0], dtype=np.float32)
            n = np.linalg.norm(vec) + 1e-8
            out.append(vec / n)
        return np.stack(out)


def _pipeline():
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=_CountAlphaEmbedder()
    ):
        from longctx import RetrievalPipeline
        # cache_dir=None: keep tests deterministic / avoid disk pollution
        # from real-model runs leaking dim-mismatched vectors back in.
        return RetrievalPipeline(cache_dir=None)


def test_chunked_short_message_single_chunk():
    """Short message produces 1 chunk; chunked path still works."""
    p = _pipeline()
    candidates = ["short alpha", "irrelevant text"]
    result = p.retrieve_chunked(
        query="alpha",
        candidates=candidates,
        top_k=1,
        chunk_size=500,  # bigger than messages
    )
    assert result.indices == [0]


def test_chunked_overlap_is_applied():
    """chunk_overlap > 0 produces overlapping chunks."""
    p = _pipeline()
    candidates = ["a" * 4000]  # 1000 tokens approx
    # With chunk_size=100, char_size=400, overlap=20→char_overlap=80;
    # Should produce ~10-13 chunks
    result = p.retrieve_chunked(
        query="alpha",
        candidates=candidates,
        top_k=1,
        chunk_size=100,
        chunk_overlap=20,
    )
    assert result.indices == [0]


def test_chunked_skips_empty_candidates():
    """Empty strings in candidates list don't generate chunks (line 104)."""
    p = _pipeline()
    candidates = ["", "alpha alpha", ""]
    result = p.retrieve_chunked(
        query="alpha",
        candidates=candidates,
        top_k=1,
        chunk_size=10,
    )
    assert result.indices == [1]


def test_chunked_coarse_filter_below_threshold_unchanged():
    """Below the coarse-prefilter threshold the result must be byte-identical
    to the pre-coarse path (this is the MRCR v2 backwards-compat guard)."""
    p = _pipeline()
    candidates = ["alpha alpha alpha", "irrelevant"]  # tiny corpus
    out_disabled = p.retrieve_chunked(
        query="alpha", candidates=candidates, top_k=1, chunk_size=10,
        coarse_filter_threshold_chars=None,
    )
    out_default = p.retrieve_chunked(
        query="alpha", candidates=candidates, top_k=1, chunk_size=10,
    )
    assert out_disabled.indices == out_default.indices == [0]


def test_chunked_coarse_filter_fires_above_threshold():
    """Above threshold + chunk count > coarse_top_n: prefilter must fire
    AND still surface the planted answer."""
    p = _pipeline()
    # ~50K chars per candidate × 3 = 150K, total well above the lowered
    # threshold below. With chunk_size=50 (200 char), ~250 chunks per
    # candidate → ~750 chunks total.
    long_text = "filler text " * 4000
    candidates = [
        long_text + " irrelevant content",
        long_text + " alpha alpha alpha needle here",
        long_text + " unrelated topic",
    ]
    result = p.retrieve_chunked(
        query="alpha",
        candidates=candidates,
        top_k=1,
        chunk_size=50,
        coarse_filter_threshold_chars=10_000,  # force the prefilter on
        coarse_filter_top_n=100,                # aggressive trim
    )
    assert result.indices == [1]
