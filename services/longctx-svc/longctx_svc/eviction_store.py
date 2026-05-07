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
"""
from __future__ import annotations

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


class EvictionStore:
    """Module-level singleton keyed by session_id.

    Thread-safe (V3 may write from a worker subprocess; /evict/retrieve
    runs on the FastAPI loop). Embedder is shared across sessions to
    avoid loading MiniLM N times.
    """

    def __init__(self, embedder=None):
        self._sessions: dict[str, _SessionIndex] = {}
        self._lock = threading.Lock()
        self._embedder = embedder  # lazy-loaded on first write

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
            return len(idx.chunks)

    def retrieve(
        self, session_id: str, query: str, top_k: int = 8,
        score_floor: float = 0.0,
    ) -> list[EvictedChunk]:
        """Return the top-K most query-similar evicted chunks for the
        session, filtered by `score_floor` cosine similarity.

        Strict session isolation: never cross-queries other sessions.
        Returns [] when the session has no evictions (cold) or when no
        chunk clears the floor.
        """
        with self._lock:
            idx = self._sessions.get(session_id)
            if idx is None or not idx.chunks:
                return []
            chunks_snapshot = list(idx.chunks)  # copy out of lock
            idx.last_access = time.time()

        embedder = self._ensure_embedder()
        q_vec = embedder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
        q_norm = float(np.linalg.norm(q_vec) + 1e-9)
        q_vec = (q_vec / q_norm).astype(np.float32)

        # Stack chunk embeddings and brute-force cosine
        embs = np.stack(
            [c.embedding for c in chunks_snapshot], axis=0
        )  # [N, D] — already L2-normalized at write
        sims = embs @ q_vec  # [N] cosine since both are unit-norm
        # Apply floor
        keep = sims >= score_floor
        if not bool(keep.any()):
            return []
        masked_sims = np.where(keep, sims, -np.inf)
        # Top-K
        k = min(int(top_k), int(keep.sum()))
        top_idx = np.argpartition(-masked_sims, kth=k - 1)[:k]
        # Order by similarity descending for nicer logs / response
        top_idx = top_idx[np.argsort(-masked_sims[top_idx])]
        return [chunks_snapshot[int(i)] for i in top_idx]

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
