"""Tests for the BM25 + dense RRF coarse-filter fusion lane in
``RetrievePipeline``.

Covers:
- below threshold: behavior is byte-identical to today's path
  (used_coarse_filter=False on every result)
- above threshold (with synthesized many-chunk index): fusion fires
- BM25 cache reuses across calls; cache invalidates on chunk-count
  change
- rare-term query that cosine alone would miss: fusion surfaces it
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from longctx_svc.config import Limits, ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk
from longctx_svc.retrieve.pipeline import RetrievePipeline


def _index_with_chunks(chunks: list[Chunk], embedder) -> ScopeIndex:
    """Build a ScopeIndex with embeddings already computed (no model
    download). Used for tests that need a known chunk corpus."""
    embs = embedder.encode(
        [c.text for c in chunks],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return ScopeIndex(
        scope_root=Path("/tmp/fake"),
        scope_hash="fake",
        chunks=chunks,
        embeddings=embs,
        embedder_name="fake",
    )


def _chunk(text: str, file: str = "/tmp/x.py") -> Chunk:
    return Chunk(text=text, file_path=file, start_line=1, end_line=1,
                 file_type="code")


# ------------------------------------------------------ below threshold

def test_below_threshold_does_not_engage_fusion(fake_embedder):
    """Tiny scope: below coarse_filter_min_chunks → fusion is skipped
    even when use_coarse_filter=True. used_coarse_filter must be False."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=1000),
    ))
    chunks = [_chunk(f"alpha content {i}") for i in range(5)]
    chunks.append(_chunk("beta content with auth"))
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    result = pipe.retrieve("alpha", idx, top_k=3)
    assert result.used_coarse_filter is False
    assert result.chunks


# ------------------------------------------------------ above threshold

def test_above_threshold_engages_fusion(fake_embedder):
    """Scope with chunks ≥ coarse_filter_min_chunks → fusion fires."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=10),
    ))
    chunks = [_chunk(f"unrelated noise number {i}") for i in range(20)]
    chunks.append(_chunk("alpha needle here in this chunk"))
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    result = pipe.retrieve("alpha needle", idx, top_k=3)
    assert result.used_coarse_filter is True
    assert any("alpha needle" in c.text for c in result.chunks)


def test_above_threshold_off_flag_skips_fusion(fake_embedder):
    """``use_coarse_filter=False`` (per-call override) skips fusion even
    above the threshold. Flag plumbing must respect the explicit no."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=10),
    ))
    chunks = [_chunk(f"text {i} alpha" if i == 0 else f"text {i}")
              for i in range(20)]
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    result = pipe.retrieve("alpha", idx, top_k=3, use_coarse_filter=False)
    assert result.used_coarse_filter is False


# --------------------------------------------------- BM25 cache reuse

def test_bm25_cache_is_reused_across_retrievals(fake_embedder):
    """Calling retrieve twice on the same index should reuse the BM25
    structure. Verified by inspecting the pipeline's internal cache."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=5),
    ))
    chunks = [_chunk(f"alpha text {i}") for i in range(15)]
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    pipe.retrieve("alpha", idx, top_k=3)
    cached_first = pipe._bm25_cache[id(idx)]
    pipe.retrieve("alpha again", idx, top_k=3)
    cached_second = pipe._bm25_cache[id(idx)]
    assert cached_first is cached_second  # exact same tuple object


def test_bm25_cache_rebuilds_on_chunk_count_change(fake_embedder):
    """If the index gains chunks, the cached BM25 must be discarded —
    otherwise stale BM25 indexes would silently miss new content."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=5),
    ))
    chunks = [_chunk(f"alpha text {i}") for i in range(10)]
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    pipe.retrieve("alpha", idx, top_k=3)
    cached_first, n_first = pipe._bm25_cache[id(idx)]
    # Mutate the index to add a new chunk + extend embeddings to match
    new_chunk = _chunk("alpha brand new content")
    new_emb = fake_embedder.encode([new_chunk.text], normalize_embeddings=True)
    idx.chunks.append(new_chunk)
    idx.embeddings = np.vstack([idx.embeddings, new_emb])
    pipe.retrieve("alpha", idx, top_k=3)
    cached_second, n_second = pipe._bm25_cache[id(idx)]
    assert n_first != n_second
    assert cached_first is not cached_second


# ------------------------------------------------ rare-term recovery

def test_fusion_recovers_rare_term_query(fake_embedder):
    """A query whose ONLY discriminative term is a rare identifier —
    one that the fake cosine embedder doesn't see (no 'alpha'/'beta'/
    'gamma'/'auth' overlap) — must still be findable via BM25."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=True,
        limits=Limits(rerank_min_chunks=10_000, multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=5),
    ))
    chunks = [_chunk(f"alpha alpha alpha alpha {i}") for i in range(20)]
    chunks.append(_chunk("the_rare_function_name_x9z is defined here"))
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None,
                            use_coarse_filter=True)
    # Query has zero cosine signal (no alpha/beta/gamma/auth) — only
    # BM25 will rank the planted chunk highly.
    result = pipe.retrieve("the_rare_function_name_x9z", idx, top_k=3)
    assert result.used_coarse_filter is True
    texts = [c.text for c in result.chunks]
    assert any("the_rare_function_name_x9z" in t for t in texts)
