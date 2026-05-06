"""RetrievalPipeline: bi-encoder retrieval with optional cross-encoder rerank.

Drop-in retrieval component for long-context workloads. Plug in your
candidate text chunks, get back the top-K most relevant ones for any query.

Validated on MRCR v2 8-needle: pure faiss top-K=8 + Qwen2.5-14B-Instruct-1M
hits 0.760 on the 8K bin. See https://github.com/TheTom/longctx for the
full benchmark reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


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
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(embedder_model, device=device)
        self._reranker = None
        if reranker_model is not None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                reranker_model, device=device, max_length=512
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
        cand_embs = self._embedder.encode(
            list(candidates),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
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
