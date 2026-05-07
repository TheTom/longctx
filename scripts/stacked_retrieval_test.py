"""Stacked retrieval recall test: BM25 prefilter -> cosine -> reranker.

Tests whether layering retrieval scorers (each catching different signals)
beats single-scorer cosine. Three stages by default:
  1. BM25 prefilter to top-200 (lexical, catches verbatim topic mentions)
  2. Cosine rerank to top-100 (semantic precision)
  3. Cross-encoder rerank to top-K (pairwise scoring)

Reports R@K at multiple K values for: cosine-only, cos→rerank, BM25→cos→rerank.

Pure MPS / CPU. ~5-10 min on M5 Max for n=30.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


_BINS = {
    "8k": (16_000, 32_000), "32k": (64_000, 128_000),
    "64k": (128_000, 256_000), "1m": (2_000_000, 5_000_000),
}

_TOK = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]+")


def tokenize(s):
    return [w.lower() for w in _TOK.findall(s)]


class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_lens = np.array([len(d) for d in corpus_tokens],
                                 dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 1.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for w, n in df.items()
        }

    def score(self, query):
        scores = np.zeros(self.N, dtype=np.float32)
        for q in query:
            idf = self.idf.get(q, 0.0)
            if idf == 0.0:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(q, 0)
                if not f:
                    continue
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b
                                     * self.doc_lens[i] / self.avgdl)
                scores[i] += idf * num / den
        return scores


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
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--bm25-prefilter", type=int, default=200)
    ap.add_argument("--cos-prefilter", type=int, default=100)
    ap.add_argument("--max-pair-chars", type=int, default=2000)
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
    cos_rerank_ranks = []
    stacked_ranks = []
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

        # Cosine
        q_emb = embedder.encode([final_user], convert_to_numpy=True,
                                normalize_embeddings=True)
        cand_embs = embedder.encode(candidates, convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=64)
        cos = (cand_embs @ q_emb.T).flatten()
        cos_order = np.argsort(-cos).tolist()
        cosine_ranks.append(cos_order.index(gold))

        # Cosine -> rerank (baseline from earlier test)
        cos_top = cos_order[: args.cos_prefilter]
        if gold not in cos_top:
            cos_rerank_ranks.append(args.cos_prefilter + 1000)
        else:
            pairs = [(final_user, candidates[idx][: args.max_pair_chars])
                     for idx in cos_top]
            scores = reranker.predict(pairs, batch_size=16,
                                      show_progress_bar=False)
            order_within = np.argsort(-scores).tolist()
            global_order = [cos_top[i] for i in order_within]
            cos_rerank_ranks.append(global_order.index(gold))

        # Stacked: BM25 -> cosine subset -> rerank
        cand_toks = [tokenize(c) for c in candidates]
        bm25 = BM25(cand_toks).score(tokenize(final_user))
        bm25_top = np.argsort(-bm25)[: args.bm25_prefilter].tolist()
        if gold not in bm25_top:
            stacked_ranks.append(args.bm25_prefilter + 1000)
        else:
            sub_cos = cos[bm25_top]
            cos_within = np.argsort(-sub_cos).tolist()[: args.cos_prefilter]
            cos_global = [bm25_top[k] for k in cos_within]
            if gold not in cos_global:
                stacked_ranks.append(args.cos_prefilter + 1000)
            else:
                pairs = [(final_user, candidates[idx][: args.max_pair_chars])
                         for idx in cos_global]
                scores = reranker.predict(pairs, batch_size=16,
                                          show_progress_bar=False)
                order_within = np.argsort(-scores).tolist()
                stacked_global = [cos_global[k] for k in order_within]
                stacked_ranks.append(stacked_global.index(gold))

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(df)} done")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(df) - skipped} samples (skipped {skipped} ambiguous), "
          f"elapsed {elapsed:.1f}s\n")

    cosine_ranks = np.array(cosine_ranks)
    cos_rerank_ranks = np.array(cos_rerank_ranks)
    stacked_ranks = np.array(stacked_ranks)

    print(f"{'lever':<32} | {'   '.join(f'R@{k:<4}' for k in args.ks)} | mean_rank")
    print("-" * 80)

    for label, ranks in [
        ("cosine only", cosine_ranks),
        (f"cos top-{args.cos_prefilter} -> rerank", cos_rerank_ranks),
        (f"BM25-{args.bm25_prefilter} -> cos-{args.cos_prefilter} -> rerank",
         stacked_ranks),
    ]:
        cells = [label]
        for k in args.ks:
            cells.append(f"{int((ranks < k).sum())}/{len(ranks)}".ljust(8))
        finite = ranks[ranks < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        print(f"{cells[0]:<32} | {' '.join(c.ljust(7) for c in cells[1:-1])} | {cells[-1]}")


if __name__ == "__main__":
    main()
