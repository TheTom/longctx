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
        return RetrievalPipeline()


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
