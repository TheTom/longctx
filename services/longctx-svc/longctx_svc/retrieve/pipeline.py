"""Retrieve top-K chunks from a ScopeIndex.

Default recipe (validated 2026-05-06 on MRCR v2 1M, n=80):
  multi-query (4 template paraphrases) → cosine top-100 union →
  bge-reranker-v2-m3 → top-K.

Multi-query and rerank can each be disabled via config. An optional
BM25 + dense RRF coarse-filter fusion lane fires when the scope is
huge (≥`coarse_filter_min_chunks`) and `use_coarse_filter=True` —
mirrors `longctx.rag.coarse_filter` plumbing for catching rare-term
queries (identifiers, codes, names) that lose to cosine alone. See
docs/PRD-12m-coarse-filter.md.

Symbol-aware augment (2026-05-11): when the caller provides a scope
root, extract Python identifiers from the query, grep the scope for
`class X` / `def X` definition sites, promote chunks from those files
into the prefilter pool, and apply a file-type prior (.py boost, docs
demote) when the query has a code signal. Recovers retrieval cases
where bge-m3 dense ranks docs/changelogs above source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from longctx.rag.symbol_augment import (
    extract_symbols, file_type_weight, query_features, symbol_grep_repo,
)
from longctx_svc.config import get_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk

# Tokenizer for BM25 — word-level lowercasing, identical recipe to
# longctx.rag.coarse_filter._word_tokenize. Subword tokenization would
# over-shard code identifiers and break term-frequency counts.
_BM25_WORD_RE = re.compile(r"\w+")


def _bm25_tokenize(text: str) -> list[str]:
    return _BM25_WORD_RE.findall(text.lower())


def _rrf_score(rank: int, k: int) -> float:
    """Reciprocal-rank-fusion score for one ranking source. ``rank`` is
    1-indexed; standard k=60 from Cormack/Clarke/Buettcher 2009."""
    return 1.0 / (k + rank)


@dataclass(frozen=True)
class RetrieveResult:
    """Top-K retrieval result."""
    chunks: list[Chunk]
    scores: list[float]
    query: str
    paraphrases: list[str]      # the queries that were embedded (incl. original)
    used_rerank: bool
    used_coarse_filter: bool = False   # BM25 ⊕ dense RRF fusion lane
    used_symbol_augment: bool = False  # symbol-grep + file-type prior


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
                 n_paraphrases: int = 3,
                 use_coarse_filter: bool | None = None):
        cfg = get_config()
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_loaded = reranker is not None
        self.use_multi_query = (
            cfg.use_multi_query if use_multi_query is None else use_multi_query
        )
        self.n_paraphrases = n_paraphrases
        self.use_coarse_filter = (
            cfg.use_coarse_filter
            if use_coarse_filter is None else use_coarse_filter
        )
        # Lazy per-ScopeIndex BM25 cache. Weak-keyed so eviction of an
        # index also drops its BM25 structure without explicit cleanup.
        # Keyed by id(index) since ScopeIndex is a mutable dataclass and
        # we want strict identity equality.
        self._bm25_cache: dict[int, object] = {}

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
                used_coarse_filter=False,
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
        used_coarse_filter = any(r.used_coarse_filter for r in per_scope)
        return RetrieveResult(
            chunks=[c for _, c in picks],
            scores=[s for s, _ in picks],
            query=query,
            paraphrases=paraphrases,
            used_rerank=used_rerank,
            used_coarse_filter=used_coarse_filter,
        )

    def _get_or_build_bm25(self, index: ScopeIndex):
        """Lazy per-index BM25Okapi. Built on first call, cached by
        identity, invalidated whenever the chunk count changes (a quick
        heuristic that catches add/remove without watching every field).
        """
        from rank_bm25 import BM25Okapi

        key = id(index)
        cached = self._bm25_cache.get(key)
        if cached is not None:
            bm25, n_seen = cached
            if n_seen == len(index.chunks):
                return bm25
        corpus_tokens = [_bm25_tokenize(c.text) for c in index.chunks]
        bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        self._bm25_cache[key] = (bm25, len(index.chunks))
        return bm25

    def retrieve(self, query: str, index: ScopeIndex, top_k: int = 8,
                 prefilter: int = 100,
                 use_rerank: bool = True,
                 use_coarse_filter: bool | None = None,
                 scope_root: Path | None = None,
                 use_symbol_augment: bool | None = None) -> RetrieveResult:
        """Multi-query + rerank retrieval over a ScopeIndex.

        Scale-aware: at small scope sizes (<rerank_min_chunks) the rerank
        is skipped (cosine R@8 is already ~95%+ on small repos and the
        cross-encoder costs ~5s on CPU). Multi-query is similarly skipped
        when chunks < multiquery_min_chunks. Override per-call by passing
        use_rerank=True/False explicitly.

        ``use_coarse_filter`` (default config-driven) enables a BM25 +
        dense RRF fusion lane that fires when the scope is huge
        (≥`coarse_filter_min_chunks`). Catches rare-term queries that
        cosine alone misses; cost is one BM25 build per index +
        per-query rank pass.
        """
        if index.embeddings is None or len(index.chunks) == 0:
            return RetrieveResult(
                chunks=[], scores=[], query=query,
                paraphrases=[query], used_rerank=False,
                used_coarse_filter=False,
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

        # Scale-aware coarse filter: BM25 ⊕ dense RRF fusion. Mirrors
        # longctx.rag.coarse_filter.filter_multi_query — runs BM25 over
        # the same paraphrases that drove the cosine pass and fuses
        # ranks via RRF. Catches rare-term queries (identifiers, codes,
        # names) that cosine alone misses. Off below the threshold so
        # we don't perturb established MRCR v2 numbers on small/medium
        # scopes.
        do_coarse_filter = (
            (use_coarse_filter
             if use_coarse_filter is not None else self.use_coarse_filter)
            and n_chunks >= cfg.limits.coarse_filter_min_chunks
        )
        used_coarse_filter = False
        if do_coarse_filter:
            bm25 = self._get_or_build_bm25(index)
            if bm25 is not None:
                # Cosine ranks (1-indexed by descending similarity)
                cos_order = np.argsort(-max_sims)
                cos_rank = np.empty(n_chunks, dtype=np.int64)
                cos_rank[cos_order] = np.arange(1, n_chunks + 1)
                # BM25 ranks across paraphrases — keep the best (lowest)
                # rank each chunk earns from any paraphrase.
                bm25_best_rank = np.full(n_chunks, n_chunks + 1,
                                         dtype=np.int64)
                for p in paraphrases:
                    qt = _bm25_tokenize(p)
                    if not qt:
                        continue
                    scores = bm25.get_scores(qt)
                    order = np.argsort(-scores)
                    rank = np.empty(n_chunks, dtype=np.int64)
                    rank[order] = np.arange(1, n_chunks + 1)
                    np.minimum(bm25_best_rank, rank, out=bm25_best_rank)
                # RRF fuse cosine + BM25
                k = cfg.limits.rrf_k
                fused = (1.0 / (k + cos_rank)) + (1.0 / (k + bm25_best_rank))
                cos_top = np.argsort(-fused)[:cos_top_n].tolist()
                used_coarse_filter = True
            else:
                cos_top = np.argsort(-max_sims)[:cos_top_n].tolist()
        else:
            cos_top = np.argsort(-max_sims)[:cos_top_n].tolist()

        # Symbol-aware augment. Greps the scope for `class X` / `def X`
        # definitions of identifiers in the query and promotes chunks
        # from those files into the candidate pool. Then applies a
        # file-type prior (.py boost, doc demote) when the query has a
        # code signal. No-op without scope_root or when the query lacks
        # extractable identifiers.
        do_symbol_augment = (
            (use_symbol_augment if use_symbol_augment is not None
             else cfg.use_symbol_augment)
            and scope_root is not None
        )
        used_symbol_augment = False
        if do_symbol_augment:
            symbols = extract_symbols(query)
            if symbols:
                sym_paths = symbol_grep_repo(symbols, scope_root)
                if sym_paths:
                    sym_path_set = {str(Path(p).resolve()) for p in sym_paths}
                    chunk_paths_norm = [
                        str(Path(index.chunks[i].file_path).resolve())
                        for i in range(n_chunks)
                    ]
                    sym_chunk_idx = [
                        i for i, p in enumerate(chunk_paths_norm)
                        if p in sym_path_set
                    ]
                    seen = set(cos_top)
                    for i in sym_chunk_idx:
                        if i not in seen:
                            cos_top.append(i)
                            seen.add(i)
                    used_symbol_augment = True
            qf = query_features(query)
            cos_top.sort(
                key=lambda i: -file_type_weight(
                    index.chunks[i].file_path, qf,
                ),
            )

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
            used_coarse_filter=used_coarse_filter,
            used_symbol_augment=used_symbol_augment,
        )
