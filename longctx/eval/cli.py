"""longctx-eval CLI.

Usage:
    longctx-eval --bin 8k --n 30 --model qwen25-14b
"""
import argparse
import sys

from longctx import LongCtxClient
from longctx.eval.mrcr import MRCRRunner


def main():
    ap = argparse.ArgumentParser(
        prog="longctx-eval",
        description="Run MRCR v2 8-needle eval against a LongCtxClient.",
    )
    ap.add_argument(
        "--bin",
        required=True,
        choices=["8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m"],
    )
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", required=True,
                    help="Served model name on the OpenAI-compatible endpoint")
    ap.add_argument(
        "--server", default="http://localhost:5050/v1/chat/completions"
    )
    ap.add_argument("--data-dir", required=True,
                    help="Path to MRCR v2 data dir (contains 8needle/*.parquet)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Bi-encoder embedder model",
    )
    ap.add_argument(
        "--reranker",
        default=None,
        help="Optional cross-encoder reranker. Note: off-the-shelf rerankers "
        "degraded MRCR-style retrieval in our testing. Default: none.",
    )
    ap.add_argument("--out", default=None,
                    help="Write JSON summary to this path")
    args = ap.parse_args()

    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline(
        embedder_model=args.embedder,
        reranker_model=args.reranker,
    )
    client = LongCtxClient(
        pipeline=pipeline, server=args.server, model=args.model
    )

    runner = MRCRRunner(args.data_dir)
    summary = runner.run(
        client,
        bin_name=args.bin,
        n=args.n,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    if args.out:
        MRCRRunner.write_summary(summary, args.out)
        print(f"\nWrote summary to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
