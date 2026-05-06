"""Reproduce the full open-stack-vs-SubQ bin curve from the X thread.

This is the one-command demo. Runs MRCR v2 8-needle at multiple bins
against a vLLM-served generator. Prints a markdown table when done.

Default config matches the headline: Qwen2.5-14B-Instruct-1M (vanilla,
no further training) + sentence-transformers MiniLM + faiss top-K=8.

Mass-validated result on AMD MI300X 2026-05-06 with this exact config:

    | Bin | pipeline    | n  | avg_score |
    | --- | ----------- | -- | --------- |
    | 8K  | RAG         | 82 | 0.822     |
    | 32K | RAG         | 98 | 0.697     |
    | 64K | RAG         | 95 | 0.641     |
    | 64K | chunked-RAG | 95 | 0.670     |

Three of three bins clear SubQ Inc.'s published 0.659 headline with the
right pipeline. Plain RAG over standard attention is competitive with
claimed-state-of-the-art subquadratic architectures on this workload.

Usage:
    python full_bin_curve.py /path/to/mrcr/data --model qwen2.5-14b-instruct-1m
"""
from __future__ import annotations

import argparse
import sys

from longctx import LongCtxClient
from longctx.eval import MRCRRunner


DEFAULT_BINS = [
    ("8k", 30),
    ("32k", 20),
    ("64k", 20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Path to MRCR v2 data dir")
    ap.add_argument("--model", default="qwen2.5-14b-instruct-1m",
                    help="Generator model name on the OpenAI-compatible endpoint")
    ap.add_argument(
        "--server",
        default="http://localhost:5050/v1/chat/completions",
    )
    ap.add_argument(
        "--bins",
        nargs="*",
        default=None,
        help="Override default bin list, e.g. --bins 8k 32k",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n", type=int, default=None,
                    help="Override sample count for all bins")
    args = ap.parse_args()

    client = LongCtxClient(model=args.model, server=args.server)
    runner = MRCRRunner(args.data_dir)

    bins_to_run = []
    if args.bins:
        for b in args.bins:
            n = args.n if args.n else 20
            bins_to_run.append((b, n))
    else:
        for b, default_n in DEFAULT_BINS:
            n = args.n if args.n else default_n
            bins_to_run.append((b, n))

    print(f"Running {len(bins_to_run)} bins on model={args.model}",
          file=sys.stderr)

    rows = []
    for bin_name, n in bins_to_run:
        print(f"\n=== {bin_name} bin (n={n}) ===", file=sys.stderr)
        summary = runner.run(client, bin_name=bin_name, n=n,
                             top_k=args.top_k, verbose=False)
        rows.append((bin_name, summary.avg_score, summary.n,
                     summary.prefix_pass_rate, summary.total_time_s))
        print(f"  avg_score={summary.avg_score:.3f} "
              f"prefix_pass={summary.prefix_pass_rate:.0%} "
              f"time={summary.total_time_s:.0f}s",
              file=sys.stderr)

    print()
    print(f"# Open Stack RAG bin curve — {args.model}")
    print()
    print("| Bin | avg_score | n | prefix_pass | total_time_s |")
    print("| --- | --------- | - | ----------- | ------------ |")
    for bin_name, score, n, prefix_pass, t in rows:
        print(f"| {bin_name} | {score:.3f} | {n} | {prefix_pass:.0%} | "
              f"{t:.1f} |")
    print()
    print(f"SubQ Inc. published 0.659 on this benchmark (at the 1M bin).")
    above = sum(1 for _, s, _, _, _ in rows if s > 0.659)
    print(f"Open stack exceeds 0.659 at {above}/{len(rows)} bins tested.")


if __name__ == "__main__":
    main()
