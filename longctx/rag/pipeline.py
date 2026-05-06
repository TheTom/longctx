"""RetrievalPipeline: bi-encoder retrieval with optional cross-encoder rerank.

Drop-in retrieval component for long-context workloads. Plug in your
candidate text chunks, get back the top-K most relevant ones for any query.

Validated on MRCR v2 8-needle: pure faiss top-K=8 + Qwen2.5-14B-Instruct-1M
hits 0.760 on the 8K bin. See https://github.com/TheTom/longctx for the
full benchmark reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from longctx.rag.embed_cache import EmbedCache


def _resolve_device(device: str = "auto") -> str:
    """Pick the best available embedder device.

    `auto` walks: CUDA → MPS (Apple Silicon) → CPU. Explicit device strings
    are passed through unchanged so callers can force CPU-only for
    determinism / debugging.
    """
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class RetrievalResult:
    """Result from a RetrievalPipeline.retrieve() call."""

    indices: list[int]  # original positions in the input candidate list
    candidates: list[str]  # candidate texts in retrieval order
    scores: list[float]  # similarity scores from the bi-encoder


class RetrievalPipeline:
    """Bi-encoder + optional cross-encoder retrieval over a list of candidates.

    Usage:
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve(query, candidates, top_k=8)
        # result.candidates is the top 8 most relevant, in original order

    For MRCR-style "find the Nth message of type X" tasks, the bi-encoder
    alone is sufficient at short to medium context. Cross-encoder reranking
    with off-the-shelf models did NOT improve MRCR at 64K bin (validated
    2026-05-06). Fine-tuning a domain-specific reranker is on the roadmap.
    """

    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str | None = None,
        device: str = "auto",
        cache_dir: str | Path | None = "default",
    ) -> None:
        """Build a retrieval pipeline.

        Args:
            embedder_model: HF id or path of the bi-encoder
            reranker_model: optional cross-encoder for two-stage rerank
            device: "auto" (default — picks CUDA/MPS/CPU), or explicit
                "cuda" / "mps" / "cpu" to force a specific backend
            cache_dir: None (disable) | "default" (~/.cache/longctx) |
                path. Per-chunk content-hash cache; subsequent calls with
                identical chunks skip embedding entirely. Override with
                env var LONGCTX_CACHE_DIR.
        """
        from sentence_transformers import SentenceTransformer

        resolved = _resolve_device(device)
        self._device = resolved
        self._embedder = SentenceTransformer(embedder_model, device=resolved)
        self._reranker = None
        if reranker_model is not None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                reranker_model, device=resolved, max_length=512
            )
        self._cache = EmbedCache(
            cache_dir=cache_dir, embedder_name=embedder_model,
        )

    @property
    def device(self) -> str:
        """Device the embedder is running on (after auto-resolution)."""
        return self._device

    @property
    def cache_enabled(self) -> bool:
        """True if disk-backed embedding cache is active."""
        return self._cache.enabled

    def retrieve_chunked(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 8,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        preserve_order: bool = True,
    ) -> RetrievalResult:
        """Hierarchical chunked retrieval over long candidates.

        Splits each candidate into chunk_size-token sub-chunks (approx),
        embeds each chunk, scores all chunks against the query, then
        ranks each parent candidate by its best chunk's similarity.
        Returns the top_k unique parent candidates.

        At 64K MRCR bin with 32B-Instruct generator, chunked retrieval
        improved score from 0.673 to 0.718 vs full-message retrieval
        (validated 2026-05-06). Chunk-level cosine is more discriminative
        than full-message cosine when individual messages are very long.

        chunk_size is approximate; we use a 4-char-per-token proxy for
        speed. Set to 500 for default. Larger chunks reduce
        discriminative power; smaller chunks add overhead.

        Args:
            query: question / target
            candidates: list of candidate texts (typically long messages)
            top_k: number of unique parent candidates to return
            chunk_size: approximate tokens per sub-chunk
            chunk_overlap: token overlap between adjacent chunks
            preserve_order: return parents in original input order

        Returns:
            RetrievalResult with indices, candidates, scores
        """
        char_size = max(chunk_size * 4, 1)
        # Clamp overlap so it can't equal or exceed chunk size
        # (would create an infinite loop where start never advances).
        char_overlap = max(0, min(chunk_overlap * 4, char_size - 1))
        stride = char_size - char_overlap

        chunks_text: list[str] = []
        chunks_parent: list[int] = []
        for parent_idx, msg in enumerate(candidates):
            if not msg:
                continue
            start = 0
            while start < len(msg):
                end = min(start + char_size, len(msg))
                chunks_text.append(msg[start:end])
                chunks_parent.append(parent_idx)
                if end == len(msg):
                    break
                start += stride

        # Query is short and one-shot; never cache it.
        query_emb = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        # Chunks are large and often repeated across calls — cache them.
        chunk_embs = self._cache.encode_with_cache(
            self._embedder, chunks_text, batch_size=64,
        )
        sims = (chunk_embs @ query_emb.T).flatten()

        n_parents = len(candidates)
        best_per_parent = np.full(n_parents, -np.inf, dtype=np.float32)
        for chunk_i, parent_i in enumerate(chunks_parent):
            if sims[chunk_i] > best_per_parent[parent_i]:
                best_per_parent[parent_i] = sims[chunk_i]

        order = np.argsort(-best_per_parent)[:top_k]
        scores = [float(best_per_parent[i]) for i in order]
        order = list(int(i) for i in order)

        if preserve_order:
            paired = sorted(zip(order, scores), key=lambda x: x[0])
            order = [p[0] for p in paired]
            scores = [p[1] for p in paired]

        return RetrievalResult(
            indices=order,
            candidates=[candidates[i] for i in order],
            scores=scores,
        )

    def retrieve(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 8,
        retrieve_k: int | None = None,
        preserve_order: bool = True,
    ) -> RetrievalResult:
        """Retrieve top_k candidates by relevance to query.

        Args:
            query: the question / target the user is looking for
            candidates: list of candidate texts to rank
            top_k: number of final candidates to return
            retrieve_k: bi-encoder pre-filter size if reranker is set;
                ignored without reranker. Defaults to 2 * top_k.
            preserve_order: if True, return candidates in their original
                input order (preserves position semantics for tasks like
                "the Nth assistant message"). If False, return in score
                order.

        Returns:
            RetrievalResult with indices, candidates, scores.
        """
        if retrieve_k is None:
            retrieve_k = max(top_k * 2, top_k + 4)

        query_emb = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        cand_embs = self._cache.encode_with_cache(
            self._embedder, list(candidates), batch_size=32,
        )
        sims = (cand_embs @ query_emb.T).flatten()

        if self._reranker is None or retrieve_k >= len(candidates):
            order = np.argsort(-sims)[:top_k]
            scores = [float(sims[i]) for i in order]
        else:
            # Stage 1: bi-encoder gets top retrieve_k
            stage1 = np.argsort(-sims)[:retrieve_k]
            # Stage 2: cross-encoder rerank
            pairs = [(query, candidates[i][:2000]) for i in stage1]
            rerank = self._reranker.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
            top_within = np.argsort(-rerank)[:top_k]
            order = [int(stage1[i]) for i in top_within]
            scores = [float(rerank[i]) for i in top_within]

        order = list(order)
        if preserve_order:
            paired = sorted(zip(order, scores), key=lambda x: x[0])
            order = [p[0] for p in paired]
            scores = [p[1] for p in paired]

        return RetrievalResult(
            indices=[int(i) for i in order],
            candidates=[candidates[i] for i in order],
            scores=scores,
        )
