"""Cross-encoder reranker recall test.

For each MRCR sample, measures whether a cross-encoder reranker lands the
gold candidate in top-K. Compares cosine-only retrieval to:
  cosine top-100 -> cross-encoder rerank -> top-K

If the reranker materially beats cosine R@8, it's the lever to wire into
longctx for production retrieval.

Models tried (configurable via --reranker):
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (small, fast, default)
  - BAAI/bge-reranker-v2-m3              (multilingual, stronger)
  - BAAI/bge-reranker-base               (medium)

Pure MPS / CPU. ~3-10 min on M5 Max for n=30.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


_BINS = {
    "8k": (16_000, 32_000), "32k": (64_000, 128_000),
    "64k": (128_000, 256_000), "1m": (2_000_000, 5_000_000),
}


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="1m", choices=list(_BINS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--data-dir", default="/tmp/mrcr")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="HF id of the cross-encoder")
    ap.add_argument("--prefilter", type=int, default=100,
                    help="cosine top-N to feed into the reranker")
    ap.add_argument("--max-pair-chars", type=int, default=2000,
                    help="truncate each candidate to this many chars for "
                    "cross-encoder pair input")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 8, 16, 32])
    args = ap.parse_args()

    import pandas as pd
    parts = []
    for f in sorted(Path(args.data_dir, "8needle").glob("*.parquet")):
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts).reset_index(drop=True)
    lo, hi = _BINS[args.bin]
    df = df[(df.n_chars >= lo) & (df.n_chars < hi)].head(args.n)
    print(f"Loaded {len(df)} samples in bin {args.bin}")

    device = detect_device()
    print(f"Device: {device}")
    print(f"Reranker: {args.reranker}")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device,
    )
    reranker = CrossEncoder(args.reranker, device=device, max_length=512)

    cosine_ranks = []
    rerank_ranks = []
    skipped = 0

    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        msgs = json.loads(row["prompt"])
        candidates = [m["content"] for m in msgs if m.get("role") == "assistant"]
        final_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                final_user = m["content"]
                break
        prefix = row["random_string_to_prepend"]
        answer = row["answer"]
        body = answer[len(prefix):] if answer.startswith(prefix) else answer

        gold_indices = [j for j, c in enumerate(candidates) if body and body in c]
        if len(gold_indices) != 1:
            skipped += 1
            continue
        gold = gold_indices[0]

        # Cosine top-N
        q_emb = embedder.encode([final_user], convert_to_numpy=True,
                                normalize_embeddings=True)
        cand_embs = embedder.encode(candidates, convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=64)
        cos = (cand_embs @ q_emb.T).flatten()
        cos_order = np.argsort(-cos).tolist()
        cosine_ranks.append(cos_order.index(gold))

        # Reranker on top-N
        prefilter_n = min(args.prefilter, len(candidates))
        top_n = cos_order[:prefilter_n]
        if gold not in top_n:
            rerank_ranks.append(prefilter_n + 1000)  # gold lost in prefilter
        else:
            pairs = [(final_user, candidates[idx][: args.max_pair_chars])
                     for idx in top_n]
            scores = reranker.predict(pairs, batch_size=16,
                                      show_progress_bar=False)
            rerank_order_within = np.argsort(-scores).tolist()
            rerank_global = [top_n[i] for i in rerank_order_within]
            rerank_ranks.append(rerank_global.index(gold))

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(df)} done")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(df) - skipped} samples (skipped {skipped} ambiguous gold), "
          f"elapsed {elapsed:.1f}s\n")

    cosine_ranks = np.array(cosine_ranks)
    rerank_ranks = np.array(rerank_ranks)

    print(f"{'lever':<28} | {'   '.join(f'R@{k:<4}' for k in args.ks)} | mean_rank")
    print("-" * 80)

    for label, ranks in [("cosine only", cosine_ranks),
                         (f"cosine top-{args.prefilter} -> rerank", rerank_ranks)]:
        cells = [label]
        for k in args.ks:
            cells.append(f"{int((ranks < k).sum())}/{len(ranks)}".ljust(8))
        finite = ranks[ranks < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        print(f"{cells[0]:<28} | {' '.join(c.ljust(7) for c in cells[1:-1])} | {cells[-1]}")

    print(f"\nGen ceiling reference (true oracle on droplet, n=30): 0.849")


if __name__ == "__main__":
    main()
