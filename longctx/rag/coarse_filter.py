"""Coarse filter — BM25 + dense embedding hybrid with RRF fusion.

Stage 1 of the hierarchical-selector pipeline.

Trims O(N) candidate chunks down to top-K (default 1000) cheaply, so the
existing BGE cross-encoder rerank stage stays bounded regardless of input
size. Designed for inputs up to 12M tokens (~6,000 chunks at 2K each) on
commodity hardware.

Why hybrid:
  - BM25 is lexical — catches exact keyword matches, milliseconds, no model
  - Dense embedding is semantic — catches paraphrases, seconds, needs a model
  - Each fails what the other catches. RRF combines them deterministically.

This module is a NEW STAGE in front of the existing
RetrievalPipeline.retrieve() — the upstream pipeline now reads:

    chunks (any size) → coarse_filter (top-1000) → existing rerank+pick

Below `CoarseFilter.threshold_tokens` (default 100K) the coarse filter is
skipped — RetrievalPipeline runs as it does today, preserving the existing
MRCR v2 numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from longctx.rag.embed_cache import EmbedCache


@dataclass
class Chunk:
    """A retrieval-addressable text chunk with stable identity.

    Stable id is derived from position-in-input + content-hash so the
    same chunk produced by re-running the chunker on identical input
    gets the same id (helps the embed cache).
    """
    id: str
    text: str
    start_offset: int = 0
    end_offset: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def _word_tokenize(text: str) -> list[str]:
    """Word-level tokenizer for BM25 (NOT the model tokenizer).

    BM25 wants lexical tokens; subword tokenization would over-shard
    code identifiers and break term-frequency counts. A simple regex
    word tokenizer with lowercasing is the standard recipe.
    """
    return re.findall(r"\w+", text.lower())


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal-rank-fusion score for one ranking source.

    Standard formulation from Cormack, Clarke, Buettcher 2009:
        score(d) = sum over rankings r of  1 / (k + r(d))

    rank is 1-indexed (top-ranked = 1).
    """
    return 1.0 / (k + rank)


class CoarseFilter:
    """BM25 + dense-embedding hybrid pre-filter for very long contexts.

    Usage:
        cf = CoarseFilter()
        top_chunks = cf.filter(chunks, query, top_k=1000)

    The filter runs whenever input is "large enough" to need it; smaller
    inputs should go through the existing RetrievalPipeline directly. The
    threshold is configured at the pipeline level (not in this class).

    Performance targets on M5 128GB for 6,000 chunks (≈12M tokens):
        BM25 build + score:      <1 s
        Dense embed (cached):    <2 s
        Dense embed (cold):      <30 s
        RRF + top-K extraction:  <0.1 s
        Total cold:              <30 s
        Total cached:            <2 s

    Memory: chunk embeddings at 384 dims × N chunks × 4 bytes; for 6K
    chunks ≈ 9 MB.
    """

    def __init__(
        self,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        cache_dir: str | Path | None = "default",
        rrf_k: int = 60,
        batch_size: int = 64,
        max_seq_length: int | None = None,
    ) -> None:
        """
        Args:
            embedder_model: HF id of the dense encoder. Default
                bge-small-en-v1.5 (33M params, 384-dim, ~130 MB).
                Spec rationale: good quality-per-dollar for the coarse
                stage; bge-large is overkill here.
            device: "auto" → CUDA / MPS / CPU; or explicit override.
            cache_dir: see EmbedCache. None to disable cache.
            rrf_k: RRF dampening constant. 60 is the canonical value.
            batch_size: encode batch size. Default 64 is fine for
                384-dim models like bge-small / MiniLM-L6. Heavy
                multilingual encoders (bge-m3, 1024-dim, 8K context)
                blow the device budget at 64 because BERT attention
                is O(seq²) per layer and the activations are
                materialised for the whole batch — drop to 1–4 to
                keep them sequential.
            max_seq_length: optional override of the embedder's
                default token cap. bge-m3 ships with 8192; capping at
                512 cuts the per-chunk activation memory ~256× at the
                cost of truncating long chunks. Leave as ``None`` to
                use the model default.
        """
        from longctx.rag.pipeline import _resolve_device  # share the picker
        from sentence_transformers import SentenceTransformer

        self._device = _resolve_device(device)
        self._embedder = SentenceTransformer(
            embedder_model, device=self._device,
        )
        if max_seq_length is not None:
            self._embedder.max_seq_length = max_seq_length
        # Different max_seq_length truncates at different token boundaries
        # so the cached embeddings differ — bake it into the cache subdir
        # name so a 512-cap run can never silently reuse 8192-cap entries.
        cache_name = embedder_model
        if max_seq_length is not None:
            cache_name = f"{embedder_model}__seq{max_seq_length}"
        self._cache = EmbedCache(
            cache_dir=cache_dir, embedder_name=cache_name,
        )
        self._rrf_k = rrf_k
        self._batch_size = batch_size

    @property
    def device(self) -> str:
        return self._device

    def filter(
        self,
        chunks: Sequence[Chunk],
        query: str,
        top_k: int = 1000,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> list[tuple[Chunk, float]]:
        """Return top-K chunks fused by BM25 + dense cosine.

        Args:
            chunks: candidate chunks
            query: user query
            top_k: number of chunks to return
            bm25_weight, dense_weight: optional reweighting if one
                ranking source consistently outperforms the other.
                Set one to 0.0 to disable that ranking source entirely
                (e.g. ``dense_weight=0`` for pure-BM25 mode). Default
                is equal-weight RRF.

        Returns:
            list of (chunk, fused_score) sorted descending by score.
        """
        return self.filter_multi_query(
            chunks, [query], top_k=top_k,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
        )

    def filter_multi_query(
        self,
        chunks: Sequence[Chunk],
        queries: Sequence[str],
        top_k: int = 1000,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> list[tuple[Chunk, float]]:
        """RRF-fuse rankings across multiple paraphrase queries.

        The hard-mode NIAH sweep on 2026-05-08 found that paraphrase
        queries with shrinking surface-token overlap outranked the
        literal query when the haystack contained topically-overlapping
        filler. This method turns that finding into a feature: pass N
        paraphrases and the fusion absorbs each one's strengths via
        RRF, raising recall above any single query in isolation.

        Equivalent to calling ``filter`` once per query and summing the
        per-query RRF scores — but does only one BM25 build and one
        dense embed pass over chunks, then scores each query against
        the cached representations. So cost is roughly
        ``embed(N_chunks) + N_queries × (cheap rank pass)``, not
        ``N_queries × full filter``.

        Args:
            chunks: candidate chunks
            queries: 1+ paraphrase queries
            top_k: chunks to keep
            bm25_weight, dense_weight: per-stage reweighting (same
                semantics as ``filter``).
        """
        if not chunks:
            return []
        if not queries:
            raise ValueError("queries must be non-empty")
        if bm25_weight <= 0 and dense_weight <= 0:
            raise ValueError("at least one of bm25_weight / dense_weight must be > 0")

        fused: dict[str, float] = {}

        if bm25_weight > 0:
            from rank_bm25 import BM25Okapi

            corpus_tokens = [_word_tokenize(c.text) for c in chunks]
            bm25 = BM25Okapi(corpus_tokens)
            for q in queries:
                qt = _word_tokenize(q)
                scores = bm25.get_scores(qt)
                order = np.argsort(-scores)
                for rank, idx in enumerate(order, start=1):
                    cid = chunks[idx].id
                    fused[cid] = fused.get(cid, 0.0) + (
                        bm25_weight * _rrf_score(rank, self._rrf_k)
                    )

        if dense_weight > 0:
            chunk_texts = [c.text for c in chunks]
            chunk_embs = self._cache.encode_with_cache(
                self._embedder, chunk_texts, batch_size=self._batch_size,
            )
            # All queries embedded in one batch pass.
            query_embs = self._embedder.encode(
                list(queries),
                convert_to_numpy=True, normalize_embeddings=True,
            )
            sim_matrix = chunk_embs @ query_embs.T  # (N_chunks, N_queries)
            for qi in range(len(queries)):
                order = np.argsort(-sim_matrix[:, qi])
                for rank, idx in enumerate(order, start=1):
                    cid = chunks[idx].id
                    fused[cid] = fused.get(cid, 0.0) + (
                        dense_weight * _rrf_score(rank, self._rrf_k)
                    )

        by_id = {c.id: c for c in chunks}
        sorted_pairs = sorted(
            ((by_id[cid], score) for cid, score in fused.items() if cid in by_id),
            key=lambda x: -x[1],
        )
        return sorted_pairs[:top_k] if top_k < len(chunks) else sorted_pairs

    # ------------------------------------------------------------------ utils

    def _bm25_rankings(
        self, chunks: Sequence[Chunk], query: str
    ) -> dict[str, int]:
        """Return {chunk_id: 1-indexed rank} from BM25."""
        from rank_bm25 import BM25Okapi

        corpus_tokens = [_word_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _word_tokenize(query)
        scores = bm25.get_scores(query_tokens)
        # argsort descending; rank 1 is best
        order = np.argsort(-scores)
        return {chunks[idx].id: rank + 1 for rank, idx in enumerate(order)}

    def _dense_rankings(
        self, chunks: Sequence[Chunk], query: str
    ) -> dict[str, int]:
        """Return {chunk_id: 1-indexed rank} from dense cosine."""
        chunk_texts = [c.text for c in chunks]
        chunk_embs = self._cache.encode_with_cache(
            self._embedder, chunk_texts, batch_size=64,
        )
        # Query embedded fresh each call (short, never cached).
        query_emb = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        )
        sims = (chunk_embs @ query_emb.T).flatten()
        order = np.argsort(-sims)
        return {chunks[idx].id: rank + 1 for rank, idx in enumerate(order)}
