"""Needle distribution study — 20 needles × 2 modes (single + multi-query).

Measures rank distribution across:
  * 4 character offsets evenly distributed across the 13.4M-token aggregate
  * 5 content types per offset (function def, prose paragraph, config
    value, code comment, error message)

Output: per-needle ranks + summary histogram. Provides the
"P95 rank N, all in top-K" credibility numbers.

Run from longctx repo root:
    python3 benchmark/coarse_filter/needle_distribution_2026-05-08.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from longctx.eval.bench_coarse_filter_real import _walk
from longctx.rag.chunker import Chunker
from longctx.rag.coarse_filter import CoarseFilter

CORPORA = [
    (Path('/Users/tom/dev/mlx-swift-lm'),
     ('.swift', '.cpp', '.h', '.c', '.metal', '.md', '.py', '.txt')),
    (Path('/Users/tom/local_llms/llama.cpp'),
     ('.cpp', '.h', '.c', '.cu', '.metal', '.md', '.py')),
    (Path('/Users/tom/dev/vllm-swift'),
     ('.swift', '.cpp', '.h', '.c', '.metal', '.md', '.py')),
    (Path('/Users/tom/Documents/obsidian/Self Study'), ('.md',)),
]

# 4 target character offsets (rough — actual plant goes at nearest \n\n)
TARGET_OFFSETS = [5_000_000, 15_000_000, 30_000_000, 45_000_000]

# Each entry: (content_type, marker_id, needle_template, primary_query,
#              paraphrase_1, paraphrase_2, paraphrase_3)
# Primary keeps strong surface overlap with the needle so single-query
# isn't a strawman. Paraphrases shrink overlap progressively.
CONTENT_TYPES = [
    ("function_def",
     "FN_TOKEN",
     'func handle_{m}(packet: Packet) -> Bool {{ return true /* TOKEN_{m} */ }}',
     "where is the handle_{m} function defined",
     "find the function named handle_{m}",
     "show me the {m} packet handler",
     "locate the bool-returning method that processes {m} packets"),
    ("prose_paragraph",
     "PROSE_TOKEN",
     "The {m} subsystem owns rotation of the long-lived authentication "
     "tokens; specifically the rotation cadence is documented as "
     "every 90 days under the heading TOKEN_{m}.",
     "how often does the {m} subsystem rotate authentication tokens",
     "what is the cadence for rotating long-lived auth tokens in {m}",
     "tell me about TOKEN_{m} rotation policy",
     "describe authentication token lifecycle for the {m} subsystem"),
    ("config_value",
     "CFG_TOKEN",
     "[runtime]\n{m}_max_threads = 17\n{m}_secret_token = TOKEN_{m}\n",
     "what is the value of {m}_secret_token",
     "what is the {m}_max_threads setting",
     "where is TOKEN_{m} configured",
     "show me the runtime config for the {m} subsystem"),
    ("code_comment",
     "CMT_TOKEN",
     "// IMPORTANT: the {m} reset path must be called once per epoch."
     " See TOKEN_{m} for the architectural rationale.",
     "where is the {m} reset path documented",
     "find the comment about {m} epoch reset",
     "show the architectural rationale for {m} resets",
     "locate the TOKEN_{m} epoch handling note"),
    ("error_message",
     "ERR_TOKEN",
     'raise RuntimeError("PEEK_FAILURE_{m}: corrupted ring buffer at '
     'offset 0xDEAD_BEEF; see runbook section TOKEN_{m}.")',
     "where is the PEEK_FAILURE_{m} runtime error raised",
     "find the corrupted ring buffer error for {m}",
     "show me the {m} runbook reference in source",
     "locate where the 0xDEAD_BEEF check fires for {m}"),
]


def build_corpus():
    """Walk corpora, return concatenated text + per-corpus stats."""
    parts = []
    stats = []
    for path, exts in CORPORA:
        text, n_files = _walk(path, exts, max_files=None, quiet=True)
        parts.append(text)
        stats.append({"path": str(path), "files": n_files,
                      "chars": len(text)})
    return "\n\n".join(parts), stats


def plant_needles(text: str) -> tuple[str, list[dict]]:
    """Plant 20 needles at the nearest \\n\\n boundaries to the targets.

    Returns the modified text and a list of needle metadata dicts. Each
    metadata includes the unique marker, content_type, primary_query,
    and paraphrases.
    """
    needles: list[dict] = []
    # One pass: find ALL \n\n joins up front, then for each (target,
    # content_type) pair pick the nearest unused join.
    joins = sorted(i for i in range(len(text) - 1) if text[i:i+2] == "\n\n")
    used: set[int] = set()

    def nearest_join(target: int) -> int:
        # Linear over filtered list; fast enough for 20 plants.
        candidates = [j for j in joins if j not in used]
        return min(candidates, key=lambda j: abs(j - target))

    parts: list[tuple[int, str, dict]] = []
    for target in TARGET_OFFSETS:
        for ct, prefix, tmpl, pq, p1, p2, p3 in CONTENT_TYPES:
            marker = f"{prefix}_{target // 1_000_000}M_{ct[:4].upper()}"
            join = nearest_join(target)
            used.add(join)
            needle_text = tmpl.format(m=marker)
            parts.append((join, needle_text, {
                "marker": marker,
                "content_type": ct,
                "target_offset": target,
                "join_offset": join,
                "primary": pq.format(m=marker),
                "paras": [p.format(m=marker) for p in (p1, p2, p3)],
            }))

    # Insert in reverse offset order so earlier insertions don't shift
    # later target offsets.
    parts.sort(key=lambda x: -x[0])
    for join, needle_text, meta in parts:
        text = text[:join + 2] + needle_text + "\n\n" + text[join + 2:]
        needles.append(meta)

    needles.sort(key=lambda m: m["target_offset"])
    return text, needles


def main():
    print("[step 1] walking corpora", file=sys.stderr)
    t0 = time.time()
    corpus, corpus_stats = build_corpus()
    print(f"  {sum(s['chars'] for s in corpus_stats):,} chars in "
          f"{time.time() - t0:.1f}s", file=sys.stderr)

    print("[step 2] planting 20 needles", file=sys.stderr)
    text, needles = plant_needles(corpus)
    for n in needles:
        print(f"  marker={n['marker']:<32s} "
              f"target={n['target_offset']:>11,} "
              f"join={n['join_offset']:>11,} "
              f"type={n['content_type']}", file=sys.stderr)

    print(f"[step 3] chunking ({len(text):,} chars)", file=sys.stderr)
    t0 = time.time()
    chunks = Chunker(tokens_per_chunk=2048).chunk(text)
    print(f"  {len(chunks):,} chunks in {time.time() - t0:.1f}s",
          file=sys.stderr)

    print("[step 4] loading CoarseFilter", file=sys.stderr)
    cf = CoarseFilter()  # bge-small / MPS / default cache

    # Per-needle: run single-query and multi-query, record rank.
    results = []
    for i, n in enumerate(needles):
        single_kept = cf.filter(chunks, n["primary"], top_k=1000)
        single_rank = next((r for r, (c, _) in enumerate(single_kept, 1)
                            if n["marker"] in c.text), None)

        multi_queries = [n["primary"], *n["paras"]]
        multi_kept = cf.filter_multi_query(chunks, multi_queries, top_k=1000)
        multi_rank = next((r for r, (c, _) in enumerate(multi_kept, 1)
                           if n["marker"] in c.text), None)

        results.append({
            "marker": n["marker"],
            "content_type": n["content_type"],
            "target_offset": n["target_offset"],
            "join_offset": n["join_offset"],
            "single_rank": single_rank,
            "multi_rank": multi_rank,
        })
        print(f"  [{i+1:>2}/{len(needles)}] {n['marker']:<32s} "
              f"single=#{single_rank or 'MISS':<5} "
              f"multi=#{multi_rank or 'MISS':<5}", file=sys.stderr)

    # ------------------------------------------------------------ histogram
    def histogram(ranks: list[int | None]) -> dict:
        hits = [r for r in ranks if r is not None]
        misses = sum(1 for r in ranks if r is None)
        if not hits:
            return {"hits": 0, "misses": misses}
        # Simple bucket histogram (tuned for typical retrieval ranks)
        buckets = [(1, 5), (6, 10), (11, 25), (26, 50),
                   (51, 100), (101, 250), (251, 500), (501, 1000)]
        hist = {f"{lo}-{hi}": 0 for lo, hi in buckets}
        for r in hits:
            for lo, hi in buckets:
                if lo <= r <= hi:
                    hist[f"{lo}-{hi}"] += 1
                    break
        return {
            "n": len(ranks),
            "hits": len(hits),
            "misses": misses,
            "min": min(hits),
            "median": statistics.median(hits),
            "p90": sorted(hits)[max(int(len(hits) * 0.9) - 1, 0)],
            "p95": sorted(hits)[max(int(len(hits) * 0.95) - 1, 0)],
            "max": max(hits),
            "buckets": hist,
        }

    summary = {
        "corpus_stats": corpus_stats,
        "total_chars": sum(s["chars"] for s in corpus_stats),
        "approx_tokens": sum(s["chars"] for s in corpus_stats) // 4,
        "n_chunks": len(chunks),
        "n_needles": len(needles),
        "single_query": histogram([r["single_rank"] for r in results]),
        "multi_query": histogram([r["multi_rank"] for r in results]),
        "by_content_type": {},
        "results": results,
    }
    # Per-content-type breakdown
    for ct, *_ in CONTENT_TYPES:
        single = [r["single_rank"] for r in results if r["content_type"] == ct]
        multi = [r["multi_rank"] for r in results if r["content_type"] == ct]
        summary["by_content_type"][ct] = {
            "single": histogram(single),
            "multi": histogram(multi),
        }

    out_path = Path("benchmark/coarse_filter/needle_distribution_2026-05-08.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[saved] {out_path}", file=sys.stderr)
    print("\n=== Single-query histogram ===")
    print(json.dumps(summary["single_query"], indent=2))
    print("\n=== Multi-query histogram ===")
    print(json.dumps(summary["multi_query"], indent=2))


if __name__ == "__main__":
    main()
