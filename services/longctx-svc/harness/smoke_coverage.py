"""In-process smoke test for /evict/dump + coverage.compute_coverage.

Avoids spinning up FastAPI: drives the EvictionStore directly, then
calls compute_coverage with a synthetic dump shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from longctx_svc.eviction_store import EvictionStore, EvictedChunk
from harness.coverage import compute_coverage


class _StubEmbedder:
    def encode(self, texts, **kw):
        rng = np.random.default_rng(123)
        return rng.standard_normal((len(texts), 384)).astype(np.float32)


def main() -> int:
    store = EvictionStore(embedder=_StubEmbedder())
    sid = "smoke-cov"
    chunks = [
        EvictedChunk(text="span A", token_range=(100, 250),
                     layer=4, score=0.9),
        EvictedChunk(text="span B", token_range=(800, 950),
                     layer=4, score=0.7),
        EvictedChunk(text="span C", token_range=(2000, 2200),
                     layer=4, score=0.5),
    ]
    store.write(sid, chunks)

    # Build an in-memory "dump" the way the endpoint would.
    with store._lock:
        idx = store._sessions[sid]
        dump = {
            "session_id": sid,
            "session_total": len(idx.chunks),
            "token_ranges": [list(c.token_range) for c in idx.chunks],
            "layers": [c.layer for c in idx.chunks],
            "scores": [c.score for c in idx.chunks],
        }

    facts = [
        {"fact_idx": 0, "entity": "PEGASUS", "kind": "access_code",
         "token_pos": 175},     # inside [100,250]
        {"fact_idx": 1, "entity": "TITAN",   "kind": "renewal_date",
         "token_pos": 500},     # gap
        {"fact_idx": 2, "entity": "DAFFODIL", "kind": "record_count",
         "token_pos": 949},     # right edge of [800,950] — covered
        {"fact_idx": 3, "entity": "TUCANA",  "kind": "access_code",
         "token_pos": 2210},    # within bleed=32 of [2000,2200] hi
        {"fact_idx": 4, "entity": "ATLAS",   "kind": "access_code",
         "token_pos": 9_000_000},  # totally uncovered (10M scale)
    ]
    cov = compute_coverage(facts, dump, bleed=32)
    print(f"[smoke_cov] coverage={cov['coverage_pct']:.1f}% "
          f"({cov['n_covered']}/{cov['n_facts']})")

    expected_covered = {0, 2, 3}
    actual_covered = {r["fact_idx"] for r in cov["per_fact"] if r["covered"]}
    if actual_covered != expected_covered:
        print(f"[smoke_cov] FAIL covered={actual_covered} "
              f"expected={expected_covered}")
        return 1
    print("[smoke_cov] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
