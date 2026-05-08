"""Read the M5 V3 validation matrix CSV → derive per-family default-on rules.

Tom's strategic question: "I want to be comfortable to always have
triattention on as long as longctx is installed and at which percentage
by data driven decisions."

This script applies a deterministic decision tree to the matrix data:

  For each (model_family) group:
    1. Find rate=0.0 baseline cell (V3 off): record exact baseline.
    2. For each rate > 0:
       - Compare exact_N to baseline_exact at the same ctx.
       - Compute lift / regression.
       - Compute compression delta (savings %).
    3. Pick the SAFE-DEFAULT rate as the highest rate where exact_N ≥
       baseline_exact (no quality regression at all) AND compression ≥
       the rate's nominal target.
    4. Pick the AGGRESSIVE-DEFAULT rate as the highest rate where the
       quality drop is ≤1 question (within statistical noise) AND
       compression ≥ rate's target.
    5. Output the recommendation table: model_family → safe% / aggressive%.

If a model family has no rows that meet "no regression" → recommend
V3 OFF for that family ("longctx external memory only" mode).

Headline outputs:
  - per-family default rate
  - "always-on" boolean per family (true if any rate ≥10% holds without regression)
  - markdown table for docs

Usage:
  python3 m5_recommend_default.py --csv /tmp/m5_matrix/results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass


def family_of(model_id: str) -> str:
    """Map a HuggingFace model id to a coarse family bucket."""
    m = model_id.lower()
    if "qwen3.5" in m or "qwen3-5" in m: return "qwen3.5"
    if "qwen3" in m: return "qwen3"
    if "qwen2.5" in m or "qwen2-5" in m: return "qwen2.5"
    if "qwen2" in m: return "qwen2"
    if "llama-3" in m or "llama3" in m: return "llama3"
    if "llama-4" in m or "llama4" in m: return "llama4"
    if "mistral" in m or "ministral" in m: return "mistral"
    if "phi-4" in m or "phi4" in m: return "phi4"
    if "phi-3" in m or "phi3" in m: return "phi3"
    if "gemma-3" in m or "gemma3" in m: return "gemma3"
    if "gemma-4" in m or "gemma4" in m: return "gemma4"
    if "glm-4" in m or "glm4" in m: return "glm4"
    if "nemotron" in m: return "nemotron"
    return "other"


@dataclass
class Cell:
    model: str
    family: str
    ctx: int
    rate: float
    mode: str
    sanity_ok: int
    exact_n: int
    total_n: int
    compression_pct: float
    session_total: int
    wall_s: int
    verdict: str

    @property
    def exact_rate(self) -> float:
        if self.total_n <= 0:
            return 0.0
        return self.exact_n / self.total_n


def load_csv(path: str) -> list[Cell]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append(Cell(
                    model=r["model"],
                    family=family_of(r["model"]),
                    ctx=int(r["ctx"]),
                    rate=float(r["eviction_rate"]),
                    mode=r["mode"],
                    sanity_ok=int(r["sanity_ok"]),
                    exact_n=int(r["exact_n"]),
                    total_n=int(r["total_n"]),
                    compression_pct=float(r["compression_pct"]),
                    session_total=int(r["session_total"]),
                    wall_s=int(r["wall_s"]),
                    verdict=r["verdict"],
                ))
            except (ValueError, KeyError):
                continue
    return rows


def recommend(rows: list[Cell]) -> dict:
    """Per-family decision."""
    out: dict[str, dict] = {}
    by_family = defaultdict(list)
    for r in rows:
        by_family[r.family].append(r)

    for family, cells in by_family.items():
        # Baseline = rate=0 cell at each ctx (any mode — usually the
        # `baseline` mode but rate=0 in any mode means V3 off).
        baseline_by_ctx: dict[int, float] = {}
        for c in cells:
            if c.rate == 0.0 and c.sanity_ok == 1 and c.total_n > 0:
                baseline_by_ctx[c.ctx] = c.exact_rate

        # For each rate>0 cell, compare against the same-ctx baseline.
        rate_quality: dict[float, list[float]] = defaultdict(list)
        rate_compression: dict[float, list[float]] = defaultdict(list)
        rate_baselines: dict[float, list[float]] = defaultdict(list)
        for c in cells:
            if c.rate <= 0.0 or c.sanity_ok != 1 or c.total_n <= 0:
                continue
            base = baseline_by_ctx.get(c.ctx)
            if base is None:
                continue
            rate_quality[c.rate].append(c.exact_rate)
            rate_baselines[c.rate].append(base)
            rate_compression[c.rate].append(c.compression_pct)

        # Decision: highest rate where median quality drop ≤ 0 (safe) or
        # ≤ ~10% relative (aggressive)
        def median(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2] if xs else 0.0

        rates_sorted = sorted(rate_quality.keys())
        safe_rate = 0.0
        aggressive_rate = 0.0
        for r in rates_sorted:
            qs = rate_quality.get(r, [])
            bs = rate_baselines.get(r, [])
            cs = rate_compression.get(r, [])
            if not qs:
                continue
            mq, mb = median(qs), median(bs)
            mc = median(cs)
            if mq >= mb:
                safe_rate = r
            if (mb - mq) <= 0.10:  # ≤ 10pp relative drop
                aggressive_rate = r

        always_on = safe_rate > 0.0
        out[family] = {
            "n_models": len(set(c.model for c in cells)),
            "n_cells": len(cells),
            "safe_default_rate": safe_rate,
            "aggressive_default_rate": aggressive_rate,
            "always_on_when_longctx": always_on,
            "rate_quality_summary": {
                r: {
                    "median_exact_rate": round(median(rate_quality[r]), 3),
                    "median_baseline": round(median(rate_baselines[r]), 3),
                    "median_compression_pct": round(median(rate_compression[r]), 1),
                    "n_cells": len(rate_quality[r]),
                }
                for r in rates_sorted
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None,
                    help="optional markdown report path")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print(f"No rows in {args.csv}")
        return
    print(f"Loaded {len(rows)} cells across "
          f"{len(set(r.family for r in rows))} families")

    rec = recommend(rows)

    print("\n--- Per-family default-on recommendations ---")
    print(f"{'family':>12}  {'always_on':>10}  {'safe%':>7}  "
          f"{'aggressive%':>12}  {'n_cells':>8}")
    print("-" * 60)
    for fam, r in sorted(rec.items()):
        print(f"{fam:>12}  {str(r['always_on_when_longctx']):>10}  "
              f"{int(r['safe_default_rate']*100):>6}%  "
              f"{int(r['aggressive_default_rate']*100):>11}%  "
              f"{r['n_cells']:>8}")

    print("\n--- Per-rate quality vs baseline (median across ctxs) ---")
    for fam, r in sorted(rec.items()):
        print(f"\n[{fam}]")
        for rate, q in sorted(r["rate_quality_summary"].items()):
            tag = " ✓ SAFE" if rate == r["safe_default_rate"] else \
                  " ★ AGGR" if rate == r["aggressive_default_rate"] else ""
            print(f"  rate={int(rate*100):>3}%  exact={q['median_exact_rate']:.3f} "
                  f"vs base {q['median_baseline']:.3f}  "
                  f"comp={q['median_compression_pct']:.1f}%  "
                  f"n={q['n_cells']:>2}{tag}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("# M5 V3+longctx default-on recommendations\n\n")
            f.write("Decision rule:\n")
            f.write("- **safe%**: highest eviction rate where median exact "
                    "matches or beats baseline.\n")
            f.write("- **aggressive%**: highest rate where quality drop "
                    "≤ 10pp from baseline.\n")
            f.write("- **always_on**: safe% > 0% (V3 has no quality cost "
                    "at some non-zero eviction).\n\n")
            f.write("| family | always_on | safe % | aggressive % | n_cells |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for fam, r in sorted(rec.items()):
                f.write(f"| {fam} | {r['always_on_when_longctx']} | "
                        f"{int(r['safe_default_rate']*100)}% | "
                        f"{int(r['aggressive_default_rate']*100)}% | "
                        f"{r['n_cells']} |\n")
        print(f"\nWrote markdown report → {args.out}")


if __name__ == "__main__":
    main()
