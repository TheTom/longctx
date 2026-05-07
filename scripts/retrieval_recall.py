"""Retrieval recall@K benchmark — no LLM call, pure retrieval precision test.

For each MRCR sample, identifies the gold candidate (the unique assistant
message containing the full answer body) and measures whether each
retrieval lever ranks it within top-K.

If a lever lands the gold in top-K=8 X% of the time, that's the precondition
for downstream LLM accuracy. Helps avoid burning expensive end-to-end runs
on retrieval levers that fail the precondition.

Runs on CPU or Apple Silicon MPS. ~3-5 min for n=30 on M5 Max.

Usage:
    # First, scp the MRCR data from the droplet (or wherever you have it):
    #   mkdir -p /tmp/mrcr/8needle
    #   scp do-amd:/mnt/scratch/mrcr/8needle/*.parquet /tmp/mrcr/8needle/

    python3 scripts/retrieval_recall.py --bin 1m --n 30
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
    "64k": (128_000, 256_000), "128k": (256_000, 512_000),
    "256k": (512_000, 1_000_000), "512k": (1_000_000, 2_000_000),
    "1m": (2_000_000, 5_000_000),
}

_TOK = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]+")


def tokenize(s: str) -> list[str]:
    return [w.lower() for w in _TOK.findall(s)]


_ORDINAL_DIGITS = re.compile(r"\b(\d+)\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
}


def parse_ordinal(query: str) -> int | None:
    m = _ORDINAL_DIGITS.search(query)
    if m:
        return int(m.group(1))
    low = query.lower()
    for word, n in _ORDINAL_WORDS.items():
        if word in low.split():
            return n
    return None


class BM25:
    """Minimal BM25Okapi (no rank_bm25 dep)."""

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


def normalize(x):
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


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


def chunked_rank(emb, candidates, query, gold_idx, chunk_size=500, top_k=8):
    """Score chunks, take best chunk per parent, rank parents."""
    char_size = chunk_size * 4
    chunks_text, chunks_parent = [], []
    for parent_idx, msg in enumerate(candidates):
        if not msg:
            continue
        for start in range(0, len(msg), char_size):
            chunks_text.append(msg[start:start + char_size])
            chunks_parent.append(parent_idx)
    q_emb = emb.encode([query], convert_to_numpy=True,
                       normalize_embeddings=True)
    chunk_embs = emb.encode(chunks_text, convert_to_numpy=True,
                            normalize_embeddings=True, batch_size=64)
    sims = (chunk_embs @ q_emb.T).flatten()
    best_per = np.full(len(candidates), -np.inf, dtype=np.float32)
    for i, p in enumerate(chunks_parent):
        if sims[i] > best_per[p]:
            best_per[p] = sims[i]
    order = np.argsort(-best_per)
    return list(order).index(gold_idx) if gold_idx in order else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="1m", choices=list(_BINS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--data-dir", default="/tmp/mrcr")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 100])
    args = ap.parse_args()

    import pandas as pd
    parts = []
    for f in sorted(Path(args.data_dir, "8needle").glob("*.parquet")):
        parts.append(pd.read_parquet(f))
    if not parts:
        raise SystemExit(
            f"no parquet files at {args.data_dir}/8needle/. "
            "scp do-amd:/mnt/scratch/mrcr/8needle/*.parquet /tmp/mrcr/8needle/"
        )
    df = pd.concat(parts).reset_index(drop=True)
    lo, hi = _BINS[args.bin]
    df = df[(df.n_chars >= lo) & (df.n_chars < hi)].head(args.n)
    print(f"Loaded {len(df)} samples in bin {args.bin}")

    device = detect_device()
    print(f"Device: {device}")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device,
    )

    levers = ["plain", "chunked_500", "position_pf50", "bm25_a0.3",
              "bm25_a0.5", "bm25_a0.7", "layered_bm25_then_cos"]
    ranks = {lv: [] for lv in levers}
    pos_correct = 0
    pos_attempted = 0
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

        # Find unique gold via full-body containment
        gold_indices = [j for j, c in enumerate(candidates) if body and body in c]
        if len(gold_indices) != 1:
            skipped += 1
            continue
        gold = gold_indices[0]

        # Embed once per sample
        q_emb = embedder.encode([final_user], convert_to_numpy=True,
                                normalize_embeddings=True)
        cand_embs = embedder.encode(candidates, convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=64)
        cos = (cand_embs @ q_emb.T).flatten()

        # Plain cosine
        plain_order = np.argsort(-cos).tolist()
        ranks["plain"].append(plain_order.index(gold))

        # Chunked (top parent rank)
        ranks["chunked_500"].append(
            chunked_rank(embedder, candidates, final_user, gold,
                         chunk_size=500),
        )

        # Position-aware: top-50 cosine, sorted by doc position, pick Nth
        ordinal = parse_ordinal(final_user)
        top50 = sorted(np.argsort(-cos)[:50].tolist())
        if ordinal is not None and 1 <= ordinal <= len(top50):
            pred = top50[ordinal - 1]
            pos_attempted += 1
            if pred == gold:
                pos_correct += 1
                ranks["position_pf50"].append(0)
            else:
                # gold's actual rank within top50 (sorted by doc pos)
                if gold in top50:
                    ranks["position_pf50"].append(top50.index(gold))
                else:
                    ranks["position_pf50"].append(999)
        else:
            # Fallback to plain rank
            ranks["position_pf50"].append(plain_order.index(gold))

        # BM25 hybrids
        cand_toks = [tokenize(c) for c in candidates]
        bm25 = BM25(cand_toks).score(tokenize(final_user))
        cos_n = normalize(cos)
        bm25_n = normalize(bm25)
        for alpha in (0.3, 0.5, 0.7):
            combined = alpha * bm25_n + (1.0 - alpha) * cos_n
            order = np.argsort(-combined).tolist()
            ranks[f"bm25_a{alpha}"].append(order.index(gold))

        # Layered: BM25 prefilter top-100 → cosine rerank
        bm25_top100 = np.argsort(-bm25)[:100].tolist()
        if gold in bm25_top100:
            sub_cos = cos[bm25_top100]
            sub_order = np.argsort(-sub_cos).tolist()
            sub_rank = sub_order.index(bm25_top100.index(gold))
            ranks["layered_bm25_then_cos"].append(sub_rank)
        else:
            ranks["layered_bm25_then_cos"].append(999)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(df)} done")

    elapsed = time.time() - t0

    print(f"\nProcessed {len(df) - skipped} samples (skipped {skipped} ambiguous gold), "
          f"elapsed {elapsed:.1f}s\n")

    # Recall@K table
    headers = ["lever"] + [f"R@{k}" for k in args.ks] + ["mean_rank"]
    print(" | ".join(f"{h:<22}" for h in headers))
    print("-" * (24 * len(headers)))
    for lv in levers:
        rs = np.array(ranks[lv])
        cells = [lv]
        for k in args.ks:
            in_topk = int((rs < k).sum())
            cells.append(f"{in_topk}/{len(rs)} ({in_topk / len(rs):.1%})")
        finite = rs[rs < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        print(" | ".join(f"{c:<22}" for c in cells))

    # Position-aware ordinal accuracy
    if pos_attempted:
        print(f"\nposition-aware exact-gold pick: "
              f"{pos_correct}/{pos_attempted} = {pos_correct / pos_attempted:.1%}")
    print(f"\nGen ceiling reference (true oracle on droplet, n=30): 0.849")


if __name__ == "__main__":
    main()
