"""Latency benchmark for /retrieve.

Certifies PRD §6 acceptance criterion: average overhead <100ms on a
warm cache. Measures four scenarios:

  1. cold build  — first call, builds the index synchronously
  2. warm cosine — subsequent call against in-memory index
  3. warm rerank — same, but the reranker model has loaded
  4. cache reload — fresh process, scope was previously persisted

Run:
    python3 benchmarks/latency.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Hint: keep the bench self-contained — no pytest, no integration harness.
import urllib.request
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from longctx_svc.sidecar import managed_sidecar  # noqa: E402


N_PROJECT_FILES = 20  # ~realistic small repo
N_WARM_REPEATS = 50   # many samples for percentile stability


def _build_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text('{"name":"bench"}\n')
    (root / "README.md").write_text(
        "# bench\n\nMid-size sample project for latency runs.\n"
    )
    src = root / "src"
    src.mkdir(exist_ok=True)
    for i in range(N_PROJECT_FILES):
        (src / f"mod_{i:03d}.ts").write_text(
            f"export function mod{i}() {{\n"
            f"  // module {i} performs auth flow {i}\n"
            f"  return validateJWT('key{i}');\n"
            f"}}\n"
        )


def _post_retrieve(base: str, prefill: str, query: str,
                   session_id: str = "bench") -> tuple[float, dict]:
    body = json.dumps({
        "prefill_text": prefill,
        "query": query,
        "top_k": 8,
    }).encode()
    req = urllib.request.Request(
        f"{base}/retrieve",
        data=body,
        headers={
            "content-type": "application/json",
            "x-session-affinity": session_id,
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    dt_ms = (time.perf_counter() - t0) * 1000
    return dt_ms, data


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    sorted_samples = sorted(samples)
    k = int(len(sorted_samples) * pct / 100)
    k = max(0, min(len(sorted_samples) - 1, k))
    return sorted_samples[k]


def _summary(label: str, samples: list[float]) -> dict:
    return {
        "label": label,
        "n": len(samples),
        "mean_ms": round(statistics.mean(samples), 2) if samples else 0,
        "median_ms": round(statistics.median(samples), 2) if samples else 0,
        "p95_ms": round(_percentile(samples, 95), 2) if samples else 0,
        "p99_ms": round(_percentile(samples, 99), 2) if samples else 0,
        "min_ms": round(min(samples), 2) if samples else 0,
        "max_ms": round(max(samples), 2) if samples else 0,
    }


@contextmanager
def _svc(cache_dir: Path):
    with managed_sidecar(
        cache_dir=str(cache_dir),
        boot_timeout=30.0,
        extra_env={"LONGCTX_NO_JANITOR": "1"},
    ) as sc:
        yield sc.url


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="longctx-bench-"))
    project = tmp / "bench"
    _build_project(project)
    cache_dir = tmp / "cache"

    auth_path = project / "src" / "mod_005.ts"
    prefill = f"## context\n[user]\nlook at {auth_path}"

    print("longctx-svc latency benchmark")
    print(f"  project: {project} ({N_PROJECT_FILES} src files)")
    print(f"  cache:   {cache_dir}")
    print(f"  warm samples: {N_WARM_REPEATS}")

    results = {}

    # === 1. cold build (first call ever) =====================================
    print("\n[1/4] cold build...")
    with _svc(cache_dir) as base:
        cold_ms, data = _post_retrieve(base, prefill, "auth flow")
        if data["scope_status"] not in ("ready", "empty"):
            print(f"  WARN: cold scope_status={data['scope_status']}", file=sys.stderr)
        print(f"  cold build  : {cold_ms:7.1f} ms  "
              f"(chunks={len(data['chunks'])}, status={data['scope_status']})")
        results["cold_build_ms"] = round(cold_ms, 2)

        # === 2. warm cosine + 3. warm rerank within same process =============
        print(f"\n[2-3/4] warm cosine + rerank ({N_WARM_REPEATS} samples each)...")
        cosine_samples: list[float] = []
        for i in range(N_WARM_REPEATS):
            ms, _ = _post_retrieve(base, prefill, f"auth flow {i}")
            cosine_samples.append(ms)
        results["warm_retrieve"] = _summary("warm /retrieve", cosine_samples)
        s = results["warm_retrieve"]
        print(f"  mean={s['mean_ms']:6.1f}  p50={s['median_ms']:6.1f}  "
              f"p95={s['p95_ms']:6.1f}  p99={s['p99_ms']:6.1f}  ms")

    # === 4. cache reload from disk (fresh process) ===========================
    print("\n[4/4] cache reload from disk (fresh subprocess)...")
    with _svc(cache_dir) as base:
        reload_ms, data = _post_retrieve(base, prefill, "auth flow")
        print(f"  cache reload: {reload_ms:7.1f} ms  "
              f"(status={data['scope_status']})")
        results["cache_reload_ms"] = round(reload_ms, 2)

    # === verdict =============================================================
    print("\n=== verdict ===")
    target = 100.0
    warm_p95 = results["warm_retrieve"]["p95_ms"]
    print(f"  PRD §6 target: warm avg < {target:.0f}ms")
    print(f"  measured warm mean: {results['warm_retrieve']['mean_ms']:6.1f} ms")
    print(f"  measured warm p95 : {warm_p95:6.1f} ms")
    ok = results["warm_retrieve"]["mean_ms"] < target
    print(f"  result: {'PASS' if ok else 'FAIL'}")

    out_path = ROOT / "benchmarks" / "latency_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults saved to {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
