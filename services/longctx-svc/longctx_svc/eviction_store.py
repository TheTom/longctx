"""Per-session in-memory eviction store for the V3 + longctx rescue stack.

V3 (the TriAttention V3 KV-cache eviction policy in vLLM) evicts tokens
during decode. Without this rescue, the evicted KV is gone — irrecoverable
from cache. This store gives V3 a place to dump the *text* of evicted
spans, embedded for retrieval, so a future query can recover them.

Design lives in obsidian:
  TriAttention V3 — 3-Tier Eviction Rescue Architecture
  - Tier 1: query-aware eviction protection (in-engine, no I/O)
  - Tier 2 (this module): evict-to-vector — write evicted spans here
  - Tier 3: rehydrate-on-next-prefill — pull from this store

Two endpoints in app.py route here:
  - POST /evict/write    — V3 calls this after each eviction round
  - POST /evict/retrieve — vLLM prefill hook calls this on each new turn

Per-session lifecycle:
  - First /evict/write for a session creates the index entry
  - /evict/retrieve searches that session's chunks only (strict isolation)
  - Janitor (in app.py lifespan) evicts idle sessions

Phase-A scope: brute-force cosine similarity. No faiss. The store is
expected to hold O(100s-1000s) of evicted chunks per session. Brute force
on 384-dim MiniLM embeddings is microseconds at that scale; faiss would
be premature optimization. Swap to faiss if a session's chunk count grows
past ~10K (workshop / coding-agent territory).

v0.3.1 (2026-05-07) — hybrid scoring + cross-encoder rerank:
  Cosine alone misses verbatim entity hits ("Project NOVA", "INV-2845")
  because MiniLM averages over the sentence. Hybrid fuses normalized
  BM25 keyword score with cosine. Optional cross-encoder rerank runs on
  top of the hybrid prefilter when chunk count is high enough to
  justify the CPU cost.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EvictedChunk:
    """One evicted span recorded by V3.

    `text` is the raw decoded text of the span (V3 caller is responsible
    for decoding token IDs → text via the model's tokenizer; longctx-svc
    deliberately doesn't ship a tokenizer).

    `token_range` is (start, end) in the original prompt token-id sequence.
    `layer` is the V3 attention layer that evicted this span (V3 evicts
    per-layer; same position can be evicted by multiple layers, deduped
    on the V3 side before write).

    `score` is the V3 eviction score at write time — useful for telemetry
    and for the "should retrieval surface this?" relevance gate at read
    time. Higher score = V3 wanted this gone harder.
    """
    text: str
    token_range: tuple[int, int]
    layer: int
    score: float
    embedding: Optional[np.ndarray] = None


@dataclass
class _SessionIndex:
    """Per-session bag of evicted chunks + their embeddings."""
    chunks: list[EvictedChunk] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)
    # BM25 lazy-built on first hybrid retrieve, invalidated on write.
    bm25_dirty: bool = True
    bm25: object = None  # rank_bm25.BM25Okapi or None
    bm25_tokens: list[list[str]] = field(default_factory=list)


# Simple word tokenizer for BM25. Lowercases, splits on non-alphanumeric,
# preserves entity-bearing tokens like "INV-2845" by treating '-' as a
# splitter then re-joining adjacent alnum chunks. We deliberately do
# NOT stem — verbatim matches on codenames and invoice numbers are the
# whole point of adding BM25.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token splitter. Preserves digits.

    Examples:
      "Project NOVA INV-2845 $123,456" →
          ["project", "nova", "inv", "2845", "123", "456"]
    """
    if not text:
        return []
    tokens = [t for t in _TOKEN_SPLIT.split(text.lower()) if t]
    return tokens


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class EvictionStore:
    """Module-level singleton keyed by session_id.

    Thread-safe (V3 may write from a worker subprocess; /evict/retrieve
    runs on the FastAPI loop). Embedder is shared across sessions to
    avoid loading MiniLM N times.
    """

    # Default rerank-eligibility threshold. Below this chunk count, the
    # cross-encoder rerank path is skipped even when use_rerank=True
    # (CPU rerank costs ~5s and isn't worth it on small sessions).
    # TODO: env-tune via VLLM_TRIATT_RESCUE_RERANK_MIN_CHUNKS if testers
    # want to override.
    DEFAULT_RERANK_MIN_CHUNKS = 100

    def __init__(self, embedder=None, reranker=None):
        self._sessions: dict[str, _SessionIndex] = {}
        self._lock = threading.Lock()
        self._embedder = embedder  # lazy-loaded on first write
        self._reranker = reranker
        self._reranker_loaded = reranker is not None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return self._embedder
        # Lazy import — sentence-transformers is already a longctx-svc
        # dependency for the main /retrieve path; reuse the same model.
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        return self._embedder

    def _ensure_reranker(self):
        """Lazy-load the cross-encoder. Returns None if unavailable.

        Loading is deferred until the first hybrid retrieve that asks
        for rerank AND has enough chunks. CPU cross-encoder is ~5s/call
        so we don't want to pay the import cost on small sessions.
        """
        if self._reranker is not None:
            return self._reranker
        if self._reranker_loaded:
            return None
        try:
            # Use the same model the main /retrieve pipeline does so
            # we don't double-load weights. Model name lifted from
            # longctx_svc.config defaults.
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512,
            )
        except Exception:
            self._reranker = None
        self._reranker_loaded = True
        return self._reranker

    def write(self, session_id: str, chunks: list[EvictedChunk]) -> int:
        """Add `chunks` to the session's index. Embeds any chunks that
        don't already have an embedding. Returns the post-write count.

        Idempotency: caller is responsible for deduping cross-layer
        evictions at the same token range before posting; the store
        does not deduplicate. Different layers evicting overlapping
        spans is the caller's call (typical: one chunk per merged
        contiguous span across layers, attached to the LAYER that scored
        it the highest).
        """
        if not chunks:
            return 0
        embedder = self._ensure_embedder()
        # Embed only chunks that don't already have one
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed:
            texts = [c.text for c in to_embed]
            vecs = embedder.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            for c, v in zip(to_embed, vecs):
                # L2-normalize for cosine via dot product later
                norm = float(np.linalg.norm(v) + 1e-9)
                c.embedding = (v / norm).astype(np.float32)

        with self._lock:
            idx = self._sessions.setdefault(session_id, _SessionIndex())
            idx.chunks.extend(chunks)
            idx.last_access = time.time()
            # Mark BM25 stale; will be rebuilt lazily on next hybrid read.
            idx.bm25_dirty = True
            return len(idx.chunks)

    def _rebuild_bm25(self, idx: _SessionIndex) -> None:
        """Rebuild BM25 index from the session's current chunks. Called
        under the store lock or on a snapshot. Idempotent: clears the
        dirty flag once done. No-op if rank_bm25 is missing — hybrid
        callers fall back to cosine-only.

        Uses BM25Plus (not BM25Okapi) because BM25Okapi's IDF goes
        negative when df > N/2 — on tiny corpora (N < ~10 chunks)
        every common term has negative IDF and the ranking inverts.
        BM25Plus adds a +1 IDF floor that keeps scores rank-meaningful
        even for very small corpora.
        """
        try:
            from rank_bm25 import BM25Plus
        except Exception:
            idx.bm25 = None
            idx.bm25_tokens = []
            idx.bm25_dirty = False
            return
        tokens_per_chunk = [_bm25_tokenize(c.text) for c in idx.chunks]
        # rank_bm25 crashes on a fully empty corpus; guard.
        if not tokens_per_chunk or not any(tokens_per_chunk):
            idx.bm25 = None
            idx.bm25_tokens = tokens_per_chunk
            idx.bm25_dirty = False
            return
        idx.bm25 = BM25Plus(tokens_per_chunk)
        idx.bm25_tokens = tokens_per_chunk
        idx.bm25_dirty = False

    def retrieve(
        self, session_id: str, query: str, top_k: int = 8,
        score_floor: float = 0.0,
        hybrid_alpha: Optional[float] = None,
        use_rerank: bool = False,
        prefilter: int = 32,
    ) -> list[EvictedChunk]:
        """Return the top-K most query-relevant evicted chunks for the
        session, ranked by:

            score = alpha * cosine_sim + (1 - alpha) * bm25_normalized

        BM25 scores are min-max normalized inside this call; cosine is
        already in [-1, 1] (~[0, 1] on this corpus). When alpha=1.0 or
        BM25 is unavailable, behavior collapses to pure cosine —
        backwards compatible with v0.3.0.

        Args:
            session_id: V3 session id (strict isolation).
            query: user question / decode-time signal.
            top_k: number of chunks to return after final ranking.
            score_floor: drop chunks below this final score.
            hybrid_alpha: blend weight for cosine vs BM25. None →
                read from `VLLM_TRIATT_RESCUE_HYBRID_ALPHA` (default 0.5).
                Set to 1.0 to disable BM25 (cosine-only legacy path).
            use_rerank: when True AND chunk count ≥
                DEFAULT_RERANK_MIN_CHUNKS, run a cross-encoder rerank
                over the prefilter pool to refine the top-K.
            prefilter: candidate pool size BEFORE rerank. Defaults to
                32 (= 4× a typical top_k=8). When rerank is off, this
                is ignored — we just take top-K by hybrid score.

        Returns [] when the session has no evictions (cold) or when no
        chunk clears the floor.

        TODO: faiss-backed cosine when N > 10K. Right now N is
        microseconds-cheap so brute-force wins on simplicity.
        """
        # Resolve hybrid alpha from env if not explicit
        if hybrid_alpha is None:
            hybrid_alpha = _env_float(
                "VLLM_TRIATT_RESCUE_HYBRID_ALPHA", 0.5,
            )
        # Clamp alpha to [0, 1]
        alpha = float(max(0.0, min(1.0, hybrid_alpha)))

        with self._lock:
            idx = self._sessions.get(session_id)
            if idx is None or not idx.chunks:
                return []
            chunks_snapshot = list(idx.chunks)  # copy out of lock
            idx.last_access = time.time()
            # Rebuild BM25 if dirty (under lock so we don't race writes).
            if alpha < 1.0 and idx.bm25_dirty:
                self._rebuild_bm25(idx)
            bm25 = idx.bm25
            # Note: bm25_tokens not needed at retrieve time

        # Pure-cosine path — preserves v0.3.0 behavior bit-for-bit when
        # alpha=1.0. Used by the existing test_eviction_store.py suite.
        embedder = self._ensure_embedder()
        q_vec = embedder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
        q_norm = float(np.linalg.norm(q_vec) + 1e-9)
        q_vec = (q_vec / q_norm).astype(np.float32)

        embs = np.stack(
            [c.embedding for c in chunks_snapshot], axis=0
        )  # [N, D] — already L2-normalized at write
        cos_sims = embs @ q_vec  # [N] cosine since both are unit-norm

        # BM25 component (only if alpha < 1 and BM25 is available).
        if alpha < 1.0 and bm25 is not None:
            q_tokens = _bm25_tokenize(query)
            if q_tokens:
                bm25_scores = np.asarray(
                    bm25.get_scores(q_tokens), dtype=np.float32,
                )
                # BM25Plus keeps scores ≥ 0, so a simple max-divide
                # normalizes into [0, 1]. When all scores are equal
                # (no per-chunk discrimination — query terms occur in
                # every chunk identically), fall back to cosine.
                bmax = float(bm25_scores.max())
                bmin = float(bm25_scores.min())
                spread = bmax - bmin
                if spread > 1e-9:
                    bm25_norm = (bm25_scores - bmin) / spread
                    fused = alpha * cos_sims + (1.0 - alpha) * bm25_norm
                else:
                    fused = cos_sims
            else:
                fused = cos_sims
        else:
            fused = cos_sims

        # Floor
        keep = fused >= score_floor
        if not bool(keep.any()):
            return []
        masked = np.where(keep, fused, -np.inf)

        # Determine prefilter pool — at least top_k, capped at chunk count.
        n_chunks = len(chunks_snapshot)
        do_rerank = (
            use_rerank
            and n_chunks >= self.DEFAULT_RERANK_MIN_CHUNKS
            and int(keep.sum()) > top_k
        )
        pool_size = (
            min(max(int(prefilter), int(top_k)), int(keep.sum()))
            if do_rerank else min(int(top_k), int(keep.sum()))
        )
        # Top-N within mask
        if pool_size <= 0:
            return []
        pool_idx = np.argpartition(-masked, kth=pool_size - 1)[:pool_size]
        pool_idx = pool_idx[np.argsort(-masked[pool_idx])]
        pool = [chunks_snapshot[int(i)] for i in pool_idx]

        if do_rerank:
            reranker = self._ensure_reranker()
            if reranker is not None:
                # Truncate texts to keep cross-encoder under max_length.
                pairs = [(query, c.text[:2000]) for c in pool]
                try:
                    rr_scores = reranker.predict(
                        pairs, batch_size=16, show_progress_bar=False,
                    )
                    rr_order = np.argsort(-np.asarray(rr_scores)).tolist()
                    pool = [pool[i] for i in rr_order]
                except Exception:
                    # Rerank failure → fall through with hybrid order.
                    pass

        return pool[: int(top_k)]

    def clear(self, session_id: str) -> int:
        """Drop all evictions for a session. Returns count dropped."""
        with self._lock:
            idx = self._sessions.pop(session_id, None)
            return len(idx.chunks) if idx is not None else 0

    def evict_idle(self, ttl_seconds: float = 1800.0) -> int:
        """Janitor hook: drop sessions that haven't been read or written
        in `ttl_seconds`. Returns count of sessions dropped.

        Default 30 min — matches the typical `/retrieve` session-idle TTL
        in the existing scope-binding code.
        """
        cutoff = time.time() - ttl_seconds
        with self._lock:
            stale = [
                sid for sid, idx in self._sessions.items()
                if idx.last_access < cutoff
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
            return len(stale)

    def stats(self) -> dict:
        """Telemetry dump for /longctx/status integration."""
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "total_chunks": sum(
                    len(idx.chunks) for idx in self._sessions.values()
                ),
                "per_session": {
                    sid: len(idx.chunks)
                    for sid, idx in self._sessions.items()
                },
            }


# Module-level singleton, attached to the FastAPI app at startup.
_GLOBAL: Optional[EvictionStore] = None


def get_eviction_store() -> EvictionStore:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = EvictionStore()
    return _GLOBAL
