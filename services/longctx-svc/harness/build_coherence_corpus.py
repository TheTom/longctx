"""CLI to build a coherence-test corpus JSON.

Wraps `synthetic_haystack.build_haystack_with_tasks` + `tasks.generate_tasks`,
writes the corpus + per-task answer key into one JSON file consumable by
`coherence_driver.py`.

Output JSON shape:
    {
      "tokens_target": 100000,
      "tokens_actual":  99713,
      "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
      "seed": 42,
      "n_per_family": {...},
      "haystack": "<full text>",
      "tasks": [
        {
          "task_id": "task_0",
          "task_family": "single_fact",
          "question": "...",
          "answer": "...",
          "answer_kind": "...",
          "evidence_spans": [
              {"evidence_id": "...", "text": "...",
               "marker": "...", "target_token_pos": ...,
               "actual_token_pos": ...},
              ...
          ],
          "required_evidence_ids": [...],
          "min_evidence_count": ...,
          "rationale": "..."
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harness.tasks import generate_tasks
from harness.synthetic_haystack import build_haystack_with_tasks


_DEFAULT_FAMILY_COUNTS = {
    "single_fact": 2,
    "multi_hop": 3,
    "contradiction": 2,
    "aggregation": 2,
    "temporal": 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True,
                    help="Target token count of the haystack")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single_fact", type=int, default=None)
    ap.add_argument("--multi_hop", type=int, default=None)
    ap.add_argument("--contradiction", type=int, default=None)
    ap.add_argument("--aggregation", type=int, default=None)
    ap.add_argument("--temporal", type=int, default=None)
    ap.add_argument(
        "--distribute", type=str, default="spread",
        choices=["spread", "local"],
        help="span distribution strategy across the haystack",
    )
    args = ap.parse_args()

    counts = dict(_DEFAULT_FAMILY_COUNTS)
    for k in counts.keys():
        v = getattr(args, k, None)
        if v is not None:
            counts[k] = int(v)

    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True,
    )

    rng = random.Random(args.seed)
    tasks = generate_tasks(rng, counts)
    print(f"Generated {len(tasks)} tasks: {counts}", file=sys.stderr)

    print(
        f"Building haystack: target={args.tokens:,} tokens, "
        f"distribute={args.distribute}",
        file=sys.stderr,
    )
    text, tasks = build_haystack_with_tasks(
        target_tokens=args.tokens,
        tasks=tasks, tokenizer=tok,
        seed=args.seed,
        distribute=args.distribute,
    )
    actual = len(tok.encode(text, add_special_tokens=False))
    print(
        f"Final: {actual:,} tokens, embedded "
        f"{sum(len(t.evidence_spans) for t in tasks)} spans across "
        f"{len(tasks)} tasks",
        file=sys.stderr,
    )

    out_obj = {
        "tokens_target": args.tokens,
        "tokens_actual": actual,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "distribute": args.distribute,
        "n_per_family": counts,
        "haystack": text,
        "tasks": [
            {
                "task_id": t.task_id,
                "task_family": t.task_family,
                "question": t.question,
                "answer": t.answer,
                "answer_kind": t.answer_kind,
                "evidence_spans": [asdict(s) for s in t.evidence_spans],
                "required_evidence_ids": list(t.required_evidence_ids),
                "min_evidence_count": int(t.min_evidence_count),
                "rationale": t.rationale,
            }
            for t in tasks
        ],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out_obj))
    print(
        f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
