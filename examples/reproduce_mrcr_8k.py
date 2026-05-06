"""Reproduce the 0.760 result on MRCR v2 8K bin from the open sparse stack thread.

Prerequisites:
    1. vLLM serving Qwen2.5-14B-Instruct-1M on http://localhost:5050
       Recommended: vLLM with the DCA RoPE V1 fallback patch
       (https://github.com/TheTom/vllm/tree/feature/dca-v1-fallback) and
       the workspace pre-grow patch.
    2. MRCR v2 data dir with 8needle/*.parquet files
    3. pip install longctx[eval]

Run:
    python examples/reproduce_mrcr_8k.py /path/to/mrcr/data
"""
import sys

from longctx import LongCtxClient
from longctx.eval import MRCRRunner


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    data_dir = sys.argv[1]

    # Default LongCtxClient uses:
    #   embedder = sentence-transformers/all-MiniLM-L6-v2 (23M params, CPU)
    #   server   = http://localhost:5050/v1/chat/completions
    #   model    = qwen25-14b
    client = LongCtxClient(model="qwen25-14b")

    runner = MRCRRunner(data_dir)
    summary = runner.run(client, bin_name="8k", n=30, top_k=8)

    print(f"\nReproduction target: 0.760")
    print(f"Your run:            {summary.avg_score:.3f}")
    if summary.avg_score >= 0.70:
        print("Within 0.06 of reference. Stack is healthy.")
    else:
        print(
            "Score below reference. Common causes:"
            " (1) MRCR data parsed differently,"
            " (2) Generator not the patched Qwen2.5-14B-Instruct-1M,"
            " (3) Embedder swapped from default."
        )


if __name__ == "__main__":
    main()
