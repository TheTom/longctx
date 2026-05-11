"""Stress test the EvictionStore at 10M-token-equivalent chunk counts.

Working assumption: 10M tokens / ~8K turn-tokens / few evicted spans
per turn = O(few thousand) evicted chunks per session. This test
pushes to 20K and 50K chunks to confirm brute-force cosine still
satisfies the retrieve-latency budget (target: < 100 ms per query).

Decides whether faiss is needed at 10M-token scope or premature opt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from longctx_svc.eviction_store import EvictionStore, EvictedChunk


def _fake_chunks(n: int, dim: int = 384, seed: int = 0) -> list[EvictedChunk]:
    rng = np.random.default_rng(seed)
    chunks: list[EvictedChunk] = []
    for i in range(n):
        v = rng.standard_normal(dim).astype(np.float32)
        v /= float(np.linalg.norm(v) + 1e-9)
        chunks.append(EvictedChunk(
            text=f"chunk {i}",
            token_range=(i * 16, i * 16 + 16),
            layer=0, score=0.5,
            embedding=v,
        ))
    return chunks


def _fake_query_vec(dim: int = 384, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / float(np.linalg.norm(v) + 1e-9)


def main() -> int:
    # Patch the embedder so we never call sentence-transformers in this
    # micro-bench — we only care about the search path.
    class _StubEmbedder:
        def encode(self, texts, **kw):
            arr = np.stack([_fake_query_vec(seed=hash(t) & 0xFFFF)
                            for t in texts])
            return arr

    sizes = [1_000, 5_000, 10_000, 20_000, 50_000]
    print(f"{'N':>8}  {'write_s':>10}  {'retrieve_ms':>14}  "
          f"{'top1_score':>12}", flush=True)
    for n in sizes:
        store = EvictionStore(embedder=_StubEmbedder())
        chunks = _fake_chunks(n)
        sid = f"sess-{n}"
        t0 = time.time()
        store.write(sid, chunks)
        t_write = time.time() - t0
        # Warm + measure retrieve
        timings: list[float] = []
        for trial in range(5):
            t0 = time.time()
            res = store.retrieve(sid, query=f"query trial {trial}",
                                 top_k=8, score_floor=0.0)
            timings.append((time.time() - t0) * 1000)
        med_ms = sorted(timings)[len(timings) // 2]
        top1 = "-"
        if res:
            top1 = f"{float(np.dot(res[0].embedding, store._ensure_embedder().encode([f'query trial 4'])[0])):.3f}"  # noqa: E501
        print(f"{n:>8d}  {t_write:>10.3f}  {med_ms:>14.2f}  {top1:>12}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
