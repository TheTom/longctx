"""Multi-query recall test: generate paraphrases of the user query,
embed each, take MAX cosine across queries per candidate.

Hypothesis: a single query has limited semantic surface. Multiple
paraphrases catch needles that any single phrasing misses. M5 prediction:
R@8 lifts from 70% to 80%+ with 3-4 paraphrases.

Paraphrases generated via small templates (no LLM call). MRCR queries
follow a strict format that templates can target.

Optionally chains a cross-encoder rerank stage for the full stacked pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


_BINS = {
    "8k": (16_000, 32_000), "16k": (32_000, 64_000),
    "32k": (64_000, 128_000), "64k": (128_000, 256_000),
    "128k": (256_000, 512_000), "256k": (512_000, 1_000_000),
    "512k": (1_000_000, 2_000_000), "1m": (2_000_000, 5_000_000),
}


# MRCR query template: "Prepend XXXX to the Nth (1 indexed) X about Y. Do not include any other text in your response."
_MRCR_QUERY_RE = re.compile(
    r"[Pp]repend\s+\S+\s+to\s+the\s+(?P<ord>\d+)(?:st|nd|rd|th)?\s*(?:\(1\s*indexed\))?\s+"
    r"(?P<form>[\w\- ]+?)\s+about\s+(?P<topic>[\w\- ]+?)\.",
    re.DOTALL,
)


def paraphrase_query(query: str, n: int = 3) -> list[str]:
    """Template-based paraphrases. Falls back to the original if parsing fails."""
    m = _MRCR_QUERY_RE.search(query)
    if not m:
        return [query]
    form = m.group("form").strip()
    topic = m.group("topic").strip()
    paraphrases = [
        # original-flavor
        query,
        # subject-led
        f"{form} discussing {topic}",
        # natural-language
        f"a {form} about {topic}",
        # noun-emphasis
        f"{topic} {form}",
        # the-piece
        f"the {form} about {topic}",
    ]
    seen = set()
    unique = []
    for p in paraphrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[: max(n, 1) + 1]  # original + n paraphrases


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
    ap.add_argument("--n-paraphrases", type=int, default=3)
    ap.add_argument("--with-rerank", action="store_true",
                    help="also run cosine top-100 -> bge-v2-m3 rerank")
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
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device,
    )
    reranker = None
    if args.with_rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device,
                                max_length=512)

    cosine_ranks = []
    multiq_ranks = []
    multiq_rerank_ranks = []
    skipped = 0
    parse_fails = 0
    paraphrase_lens = []

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

        # Embed candidates once
        cand_embs = embedder.encode(candidates, convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=64)

        # Plain cosine baseline
        q_emb = embedder.encode([final_user], convert_to_numpy=True,
                                normalize_embeddings=True)
        cos = (cand_embs @ q_emb.T).flatten()
        cos_order = np.argsort(-cos).tolist()
        cosine_ranks.append(cos_order.index(gold))

        # Multi-query
        paras = paraphrase_query(final_user, n=args.n_paraphrases)
        if len(paras) == 1:
            parse_fails += 1
        paraphrase_lens.append(len(paras))
        para_embs = embedder.encode(paras, convert_to_numpy=True,
                                    normalize_embeddings=True)
        multi_sims = cand_embs @ para_embs.T  # (N, num_queries)
        multi_max = multi_sims.max(axis=1)
        multi_order = np.argsort(-multi_max).tolist()
        multiq_ranks.append(multi_order.index(gold))

        # Multi-query + rerank
        if reranker is not None:
            top100 = multi_order[:100]
            if gold not in top100:
                multiq_rerank_ranks.append(101)
            else:
                pairs = [(final_user, candidates[idx][:2000])
                         for idx in top100]
                scores = reranker.predict(pairs, batch_size=16,
                                          show_progress_bar=False)
                rerank_within = np.argsort(-scores).tolist()
                rerank_global = [top100[k] for k in rerank_within]
                multiq_rerank_ranks.append(rerank_global.index(gold))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(df)} done")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(df) - skipped} (skipped {skipped}, "
          f"parse_fails {parse_fails}), elapsed {elapsed:.1f}s")
    print(f"avg paraphrases per query: {np.mean(paraphrase_lens):.2f}\n")

    cosine_ranks = np.array(cosine_ranks)
    multiq_ranks = np.array(multiq_ranks)
    print(f"{'lever':<32} | {'   '.join(f'R@{k:<4}' for k in args.ks)} | mean_rank")
    print("-" * 80)
    for label, ranks in [
        ("plain cosine", cosine_ranks),
        ("multi-query (max-cosine)", multiq_ranks),
    ]:
        cells = [label]
        for k in args.ks:
            in_topk = int((ranks < k).sum())
            cells.append(f"{in_topk}/{len(ranks)}".ljust(8))
        finite = ranks[ranks < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        print(f"{cells[0]:<32} | {' '.join(c.ljust(7) for c in cells[1:-1])} | {cells[-1]}")

    if reranker is not None:
        rerank_arr = np.array(multiq_rerank_ranks)
        cells = ["multi-query + bge-rerank"]
        for k in args.ks:
            in_topk = int((rerank_arr < k).sum())
            cells.append(f"{in_topk}/{len(rerank_arr)}".ljust(8))
        finite = rerank_arr[rerank_arr < 999]
        cells.append(f"{finite.mean():.1f}" if finite.size else "n/a")
        print(f"{cells[0]:<32} | {' '.join(c.ljust(7) for c in cells[1:-1])} | {cells[-1]}")


if __name__ == "__main__":
    main()
