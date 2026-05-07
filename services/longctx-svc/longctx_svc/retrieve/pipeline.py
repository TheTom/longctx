"""Retrieve top-K chunks from a ScopeIndex.

Default recipe (validated 2026-05-06 on MRCR v2 1M, n=80):
  multi-query (4 template paraphrases) → cosine top-100 union →
  bge-reranker-v2-m3 → top-K.

Multi-query and rerank can each be disabled via config.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from longctx_svc.config import get_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk


@dataclass(frozen=True)
class RetrieveResult:
    """Top-K retrieval result."""
    chunks: list[Chunk]
    scores: list[float]
    query: str
    paraphrases: list[str]      # the queries that were embedded (incl. original)
    used_rerank: bool


# Generic paraphrase templates — work for code, prose, and config queries.
def _paraphrase(query: str, n: int = 3) -> list[str]:
    """Template-based paraphrases. Caller may extend with LLM-generated ones."""
    q = query.strip()
    if not q:
        return [q]
    paraphrases = [q]
    # Direct restatements that vary noun/verb order without changing semantics.
    paraphrases.append(f"about {q}")
    paraphrases.append(f"{q} explanation")
    paraphrases.append(f"how does {q} work")
    paraphrases.append(f"definition of {q}")
    seen = set()
    unique = []
    for p in paraphrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[: n + 1]


class RetrievePipeline:
    """Stateful retrieve pipeline. Owns the embedder + (optional) reranker.

    Built once per service instance and reused across scopes.
    """

    def __init__(self, embedder=None, reranker=None,
                 use_multi_query: bool | None = None,
                 n_paraphrases: int = 3):
        cfg = get_config()
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_loaded = reranker is not None
        self.use_multi_query = (
            cfg.use_multi_query if use_multi_query is None else use_multi_query
        )
        self.n_paraphrases = n_paraphrases

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            cfg = get_config()
            self._embedder = SentenceTransformer(cfg.embedder_model)
        return self._embedder

    def _ensure_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if self._reranker_loaded:
            return None
        cfg = get_config()
        if not cfg.reranker_model:
            self._reranker_loaded = True
            return None
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(cfg.reranker_model, max_length=512)
        except Exception:
            self._reranker = None
        self._reranker_loaded = True
        return self._reranker

    def retrieve_multi(self, query: str, indexes: list[ScopeIndex],
                       top_k: int = 8,
                       prefilter: int = 100,
                       use_rerank: bool = True) -> RetrieveResult:
        """PRD §6.3 / v0.3.3: query several scope indexes and merge by
        score. Used for workspace queries (ws:) and multi-scope routing
        within a single conversation.
        """
        if not indexes:
            return RetrieveResult(
                chunks=[], scores=[], query=query,
                paraphrases=[query], used_rerank=False,
            )
        per_scope = []
        for idx in indexes:
            r = self.retrieve(query, idx, top_k=top_k,
                              prefilter=prefilter, use_rerank=use_rerank)
            per_scope.append(r)
        # Flatten, sort by score, take top-K
        all_pairs: list[tuple[float, object]] = []
        used_rerank = any(r.used_rerank for r in per_scope)
        paraphrases = per_scope[0].paraphrases if per_scope else [query]
        for r in per_scope:
            for c, s in zip(r.chunks, r.scores):
                all_pairs.append((float(s), c))
        all_pairs.sort(key=lambda p: -p[0])
        picks = all_pairs[:top_k]
        return RetrieveResult(
            chunks=[c for _, c in picks],
            scores=[s for s, _ in picks],
            query=query,
            paraphrases=paraphrases,
            used_rerank=used_rerank,
        )

    def retrieve(self, query: str, index: ScopeIndex, top_k: int = 8,
                 prefilter: int = 100,
                 use_rerank: bool = True) -> RetrieveResult:
        """Multi-query + rerank retrieval over a ScopeIndex.

        Scale-aware: at small scope sizes (<rerank_min_chunks) the rerank
        is skipped (cosine R@8 is already ~95%+ on small repos and the
        cross-encoder costs ~5s on CPU). Multi-query is similarly skipped
        when chunks < multiquery_min_chunks. Override per-call by passing
        use_rerank=True/False explicitly.
        """
        if index.embeddings is None or len(index.chunks) == 0:
            return RetrieveResult(
                chunks=[], scores=[], query=query,
                paraphrases=[query], used_rerank=False,
            )
        cfg = get_config()
        n_chunks = len(index.chunks)
        embedder = self._ensure_embedder()

        # Scale-aware multi-query: only worth the 4× embed cost when the
        # scope is large enough that paraphrase variance helps recall.
        do_multi_query = (
            self.use_multi_query
            and n_chunks >= cfg.limits.multiquery_min_chunks
        )
        if do_multi_query:
            paraphrases = _paraphrase(query, n=self.n_paraphrases)
        else:
            paraphrases = [query]

        q_embs = embedder.encode(
            paraphrases, convert_to_numpy=True, normalize_embeddings=True,
        )
        # max-cosine across paraphrases for each chunk
        sims_matrix = index.embeddings @ q_embs.T   # (N, num_queries)
        max_sims = sims_matrix.max(axis=1)

        # Scale-aware prefilter: cross-encoder cost is linear in pairs,
        # so smaller prefilter at small scopes keeps rerank latency low.
        if prefilter == 100:  # caller didn't override
            prefilter = (cfg.limits.rerank_prefilter_large
                         if n_chunks >= 10_000
                         else cfg.limits.rerank_prefilter_small)
        cos_top_n = min(prefilter, len(index.chunks))
        cos_top = np.argsort(-max_sims)[:cos_top_n].tolist()

        # Scale-aware rerank gate. The cross-encoder is the dominant
        # latency at small scope sizes — cosine alone is already strong
        # there, so skip the rerank.
        do_rerank = use_rerank and n_chunks >= cfg.limits.rerank_min_chunks
        used_rerank = False
        if do_rerank:
            reranker = self._ensure_reranker()
            if reranker is not None:
                pairs = [(query, index.chunks[i].text[:2000])
                         for i in cos_top]
                try:
                    scores = reranker.predict(
                        pairs, batch_size=16, show_progress_bar=False,
                    )
                    order_within = np.argsort(-scores).tolist()
                    cos_top = [cos_top[i] for i in order_within]
                    used_rerank = True
                except Exception:
                    used_rerank = False

        picks = cos_top[:top_k]
        if not used_rerank:
            scores_out = [float(max_sims[i]) for i in picks]
        else:
            # When reranked, we don't keep the per-pair float (and order is
            # already reflected in `picks`). Surface cosine for visibility.
            scores_out = [float(max_sims[i]) for i in picks]

        index.touch()
        return RetrieveResult(
            chunks=[index.chunks[i] for i in picks],
            scores=scores_out,
            query=query,
            paraphrases=paraphrases,
            used_rerank=used_rerank,
        )
