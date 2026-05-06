"""Canonical bench script.

Runs the full open-stack-vs-SubQ comparison: dense baseline, RAG, and
chunked-RAG variants across multiple bins, against a single served
generator. Outputs a markdown table with means and per-cell sample sizes.

Usage:
    longctx-bench --data-dir /path/to/mrcr --model qwen25-32b

This is the single command that reproduces the numbers from the
2026-05-06 thread. Run it on your own hardware against your own
vLLM-served Qwen2.5-32B-Instruct to verify.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from longctx import LongCtxClient, RetrievalPipeline
from longctx.eval.mrcr import MRCRRunner


@dataclass
class BenchCell:
    pipeline_name: str
    bin: str
    n: int
    avg_score: float
    prefix_pass_rate: float
    avg_prompt_tokens: float
    total_time_s: float


def run_dense(client, runner, bin_name, n):
    """Dense baseline: pass ALL assistant messages as if no retrieval."""

    class _DenseClient:
        """Adapter that behaves like LongCtxClient but skips retrieval."""

        def __init__(self, inner):
            self.inner = inner
            self.model = inner.model

        def ask(self, query, candidates, top_k, max_tokens=4096,
                temperature=0.0):
            # Use a pass-through "retrieval" that returns ALL candidates.
            return self.inner.ask(query, candidates, top_k=len(candidates),
                                  max_tokens=max_tokens,
                                  temperature=temperature)

    dense = _DenseClient(client)
    summary = runner.run(dense, bin_name=bin_name, n=n, top_k=99,
                         verbose=False)
    return summary


def run_rag(client, runner, bin_name, n, top_k):
    return runner.run(client, bin_name=bin_name, n=n, top_k=top_k,
                      verbose=False)


def run_chunked_rag(client, runner, bin_name, n, top_k, chunk_size=500):
    """Wrap LongCtxClient to use chunked retrieval."""
    orig_pipeline = client.pipeline

    class _ChunkedClient:
        def __init__(self, inner):
            self.inner = inner
            self.model = inner.model

        def ask(self, query, candidates, top_k, max_tokens=4096,
                temperature=0.0):
            # Use chunked retrieval directly
            import time
            import requests
            retrieval = orig_pipeline.retrieve_chunked(
                query, candidates, top_k=top_k, chunk_size=chunk_size
            )
            body = "\n\n".join(
                f"=== CANDIDATE {pos + 1} (originally item {idx + 1}) ==="
                f"\n{m}"
                for pos, (idx, m) in enumerate(
                    zip(retrieval.indices, retrieval.candidates)
                )
            )
            user = (
                f"{body}\n\n=== USER FINAL QUESTION ===\n{query}\n\n"
                "Output ONLY the requested message."
            )
            t0 = time.time()
            response = requests.post(
                self.inner.server,
                json={
                    "model": self.inner.model,
                    "messages": [
                        {"role": "system",
                         "content": self.inner.system_prompt},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.inner.timeout,
            )
            elapsed = time.time() - t0
            response.raise_for_status()
            data = response.json()
            from longctx.rag.client import LongCtxResponse
            return LongCtxResponse(
                content=data["choices"][0]["message"]["content"],
                retrieved_indices=retrieval.indices,
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                latency_s=elapsed,
            )

    chunked = _ChunkedClient(client)
    return runner.run(chunked, bin_name=bin_name, n=n, top_k=top_k,
                      verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--server", default="http://localhost:5050/v1/chat/completions"
    )
    ap.add_argument("--bins", nargs="*", default=["8k", "32k", "64k"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--include-dense", action="store_true",
                    help="Also run dense baseline (slow at long bins)")
    ap.add_argument("--include-chunked", action="store_true",
                    help="Also run chunked RAG variant")
    ap.add_argument("--out", default=None,
                    help="Write detailed JSON results")
    args = ap.parse_args()

    client = LongCtxClient(model=args.model, server=args.server)
    runner = MRCRRunner(args.data_dir)

    cells: list[BenchCell] = []

    for bin_name in args.bins:
        if args.include_dense:
            print(f"[dense] {bin_name} n={args.n}", file=sys.stderr)
            s = run_dense(client, runner, bin_name, args.n)
            cells.append(BenchCell(
                "dense", bin_name, s.n, s.avg_score,
                s.prefix_pass_rate, s.avg_prompt_tokens, s.total_time_s
            ))

        print(f"[rag] {bin_name} n={args.n}", file=sys.stderr)
        s = run_rag(client, runner, bin_name, args.n, top_k=8)
        cells.append(BenchCell(
            "rag", bin_name, s.n, s.avg_score,
            s.prefix_pass_rate, s.avg_prompt_tokens, s.total_time_s
        ))

        if args.include_chunked:
            print(f"[chunked-rag] {bin_name} n={args.n}", file=sys.stderr)
            s = run_chunked_rag(client, runner, bin_name, args.n, top_k=8)
            cells.append(BenchCell(
                "chunked-rag", bin_name, s.n, s.avg_score,
                s.prefix_pass_rate, s.avg_prompt_tokens, s.total_time_s
            ))

    print()
    print(f"# longctx bench — model={args.model}")
    print()
    print("| pipeline | bin | n | avg_score | prefix_pass | "
          "avg_prompt_tok | time_s |")
    print("| -------- | --- | - | --------- | ----------- | "
          "-------------- | ------ |")
    for c in cells:
        print(f"| {c.pipeline_name} | {c.bin} | {c.n} | "
              f"{c.avg_score:.3f} | {c.prefix_pass_rate:.0%} | "
              f"{c.avg_prompt_tokens:.0f} | {c.total_time_s:.1f} |")

    print()
    above_subq = sum(1 for c in cells
                     if c.pipeline_name in ("rag", "chunked-rag")
                     and c.avg_score > 0.659)
    rag_total = sum(1 for c in cells
                    if c.pipeline_name in ("rag", "chunked-rag"))
    if rag_total:
        print(f"RAG cells exceeding SubQ's 0.659: {above_subq}/{rag_total}")

    if args.out:
        Path(args.out).write_text(
            json.dumps([c.__dict__ for c in cells], indent=2) + "\n"
        )
        print(f"\nDetailed results: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
