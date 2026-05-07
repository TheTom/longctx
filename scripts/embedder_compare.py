"""Compare different bi-encoder embedders for retrieval recall on 1M MRCR.

Tests whether swapping MiniLM-L6 for a bigger / better embedder lifts R@8
without any reranker layer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


_BINS = {
    "8k": (16_000, 32_000), "32k": (64_000, 128_000),
    "64k": (128_000, 256_000), "1m": (2_000_000, 5_000_000),
}


def detect_device():
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
    ap.add_argument("--embedders", nargs="+",
                    default=["sentence-transformers/all-MiniLM-L6-v2",
                             "BAAI/bge-m3"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 8, 16, 32])
    args = ap.parse_args()

    import pandas as pd
    parts = []
    for f in sorted(Path(args.data_dir, "8needle").glob("*.parquet")):
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts).reset_index(drop=True)
    lo, hi = _BINS[args.bin]
    df = df[(df.n_chars >= lo) & (df.n_chars < hi)].head(args.n)
    print(f"Loaded {len(df)} samples in bin {args.bin}\n")

    device = detect_device()
    print(f"Device: {device}\n")

    from sentence_transformers import SentenceTransformer

    results = {}
    for model_name in args.embedders:
        print(f"=== {model_name} ===")
        embedder = SentenceTransformer(model_name, device=device)
        ranks = []
        skipped = 0
        t0 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            msgs = json.loads(row["prompt"])
            candidates = [m["content"]
                          for m in msgs if m.get("role") == "assistant"]
            final_user = ""
            for m in reversed(msgs):
                if m.get("role") == "user":
                    final_user = m["content"]
                    break
            prefix = row["random_string_to_prepend"]
            answer = row["answer"]
            body = (answer[len(prefix):] if answer.startswith(prefix)
                    else answer)
            gold_indices = [j for j, c in enumerate(candidates)
                            if body and body in c]
            if len(gold_indices) != 1:
                skipped += 1
                continue
            gold = gold_indices[0]
            q_emb = embedder.encode([final_user], convert_to_numpy=True,
                                    normalize_embeddings=True)
            cand_embs = embedder.encode(candidates, convert_to_numpy=True,
                                        normalize_embeddings=True,
                                        batch_size=32)
            cos = (cand_embs @ q_emb.T).flatten()
            order = np.argsort(-cos).tolist()
            ranks.append(order.index(gold))
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(df)} done")
        elapsed = time.time() - t0
        results[model_name] = (np.array(ranks), elapsed, skipped)
        print(f"  elapsed {elapsed:.1f}s, skipped {skipped}\n")

    print(f"\n{'embedder':<48} | {'    '.join(f'R@{k:<4}' for k in args.ks)} | mean_rank | sec")
    print("-" * 120)
    for name, (ranks, elapsed, _) in results.items():
        cells = [name[:46]]
        for k in args.ks:
            in_topk = int((ranks < k).sum())
            cells.append(f"{in_topk}/{len(ranks)}".ljust(8))
        finite = ranks[ranks < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        cells.append(f"{elapsed:.0f}")
        print(f"{cells[0]:<48} | {' '.join(c.ljust(7) for c in cells[1:-2])} | {cells[-2]:<9} | {cells[-1]}")


if __name__ == "__main__":
    main()
