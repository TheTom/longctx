"""Three "money" queries against the 13.4M-token aggregate corpus.

No planted needles. These are real questions about Tom's own code
where the right answer is known because Tom wrote it.

For each query the script reports the top-5 chunks the coarse filter
returns, plus a verdict on whether any top-5 chunk contains the
ground-truth file/symbol I expect.

The ground truth comes from this session's own work plus the
existing repo state — not from running the model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from longctx.eval.bench_coarse_filter_real import _walk
from longctx.rag.chunker import Chunker
from longctx.rag.coarse_filter import CoarseFilter

CORPORA = [
    (Path('/Users/tom/dev/longctx'),
     ('.py', '.md', '.txt')),
    (Path('/Users/tom/dev/mlx-swift-lm'),
     ('.swift', '.cpp', '.h', '.c', '.metal', '.md', '.py', '.txt')),
    (Path('/Users/tom/local_llms/llama.cpp'),
     ('.cpp', '.h', '.c', '.cu', '.metal', '.md', '.py')),
    (Path('/Users/tom/dev/vllm-swift'),
     ('.swift', '.cpp', '.h', '.c', '.metal', '.md', '.py')),
    (Path('/Users/tom/Documents/obsidian/Self Study'), ('.md',)),
]

# Each query: (id, question, expected substring(s) in retrieved text).
# An answer is "found" if ANY of the top-5 chunks contains ANY expected
# substring. Substrings are chosen to be specific (function names, file
# paths, distinctive phrases) so accidental matches are unlikely.
QUERIES = [
    {
        "id": "Q1_multi_query_fusion",
        "question": (
            "Where is multi-query paraphrase fusion implemented "
            "in longctx? Show me the function and its docstring."
        ),
        "paraphrases": [
            "find the filter_multi_query method in longctx",
            "how does the coarse filter combine multiple paraphrase queries",
            "RRF fusion across N queries longctx coarse filter",
        ],
        "expected": ["filter_multi_query", "RRF-fuse rankings across multiple"],
        "ground_truth": (
            "longctx/rag/coarse_filter.py — CoarseFilter.filter_multi_query, "
            "added in Phase 6 of the 12M coarse filter work."
        ),
    },
    {
        "id": "Q2_bge_small_decision",
        "question": (
            "What was the decision rationale for choosing bge-small "
            "over MiniLM-L6 as the default embedder in longctx?"
        ),
        "paraphrases": [
            "why does longctx default to bge-small-en-v1.5",
            "embedder ablation results bge-small vs MiniLM-L6",
            "bge-small synth hard mode rank 8 MiniLM 26",
        ],
        "expected": [
            "bge-small remains the right default",
            "BAAI/bge-small-en-v1.5",
            "MiniLM-L6",
        ],
        "ground_truth": (
            "benchmark/coarse_filter/RESULTS.md — ablation table: "
            "bge-small wins synth hard rank 8, MiniLM-L6 rank 26; on "
            "real vault MiniLM-L6 rank 1, bge-small rank 3 (close 2nd)."
        ),
    },
    {
        "id": "Q3_centroid_threshold",
        "question": (
            "In mlx-swift-lm, where does the centroid block-sparse "
            "attention check whether the context is large enough to "
            "switch from dense to sparse?"
        ),
        "paraphrases": [
            "centroid sparse attention hybrid policy threshold",
            "minContextForSparse swift code",
            "where does mlx-swift-lm gate the centroid path on context length",
        ],
        "expected": ["minContextForSparse", "BlockSparseConfig"],
        "ground_truth": (
            "mlx-swift-lm Libraries/MLXLMCommon/CentroidRouter.swift — "
            "BlockSparseConfig.minContextForSparse (default 65536). The "
            "gate is checked in each model's attention forward."
        ),
    },
]


def main():
    print("[step 1] walking corpora", file=sys.stderr)
    t0 = time.time()
    parts = []
    for path, exts in CORPORA:
        text, _ = _walk(path, exts, max_files=None, quiet=True)
        parts.append(text)
    text = "\n\n".join(parts)
    print(f"  {len(text):,} chars in {time.time() - t0:.1f}s", file=sys.stderr)

    print("[step 2] chunking", file=sys.stderr)
    t0 = time.time()
    chunks = Chunker(tokens_per_chunk=2048).chunk(text)
    print(f"  {len(chunks):,} chunks in {time.time() - t0:.1f}s",
          file=sys.stderr)

    print("[step 3] loading CoarseFilter (bge-small / MPS)",
          file=sys.stderr)
    cf = CoarseFilter()

    out = []
    for q in QUERIES:
        print(f"\n=== {q['id']}: {q['question'][:80]}", file=sys.stderr)
        all_queries = [q["question"], *q["paraphrases"]]
        t0 = time.time()
        kept = cf.filter_multi_query(chunks, all_queries, top_k=5)
        elapsed = time.time() - t0

        top5 = []
        match_idx = None
        for rank, (chunk, score) in enumerate(kept[:5], start=1):
            hit_terms = [e for e in q["expected"] if e in chunk.text]
            if hit_terms and match_idx is None:
                match_idx = rank
            top5.append({
                "rank": rank,
                "score": round(float(score), 5),
                "preview": chunk.text[:160].replace("\n", " "),
                "hit_terms": hit_terms,
            })
            print(f"  #{rank} score={score:.4f} hit={hit_terms} | "
                  f"{chunk.text[:100].replace(chr(10), ' ')!r}",
                  file=sys.stderr)

        verdict = "FOUND" if match_idx is not None else "MISS"
        print(f"  → {verdict} (best rank {match_idx}) in {elapsed:.2f}s",
              file=sys.stderr)
        out.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_substrings": q["expected"],
            "top5": top5,
            "verdict": verdict,
            "best_rank": match_idx,
            "filter_secs": round(elapsed, 2),
        })

    out_path = Path("benchmark/coarse_filter/money_queries_2026-05-08.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[saved] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
