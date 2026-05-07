"""Coverage scorer: did V3 evict the planted-fact token spans?

Pulls /evict/dump for a session and computes, for each planted fact,
whether its `token_pos` falls inside any captured chunk's token_range.
PRD acceptance bar is ≥95% coverage.

Companion to streaming_driver.py's run output. Call after the run
finishes; expects the same session_id used by V3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def fetch_dump(longctx_endpoint: str, session_id: str,
               timeout: float = 60.0) -> dict:
    url = longctx_endpoint.rstrip("/") + "/evict/dump"
    r = requests.get(url, params={"session_id": session_id}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def compute_coverage(facts: list[dict], dump: dict,
                     bleed: int = 32) -> dict:
    """Return per-fact coverage + aggregate stats.

    bleed allows a token_pos to count as covered if any chunk's range
    spans `[token_pos - bleed, token_pos + bleed]`. V3 chunks usually
    encompass dozens of tokens, so a strict point-in-range check is
    fine; bleed exists for off-by-one tolerance.
    """
    ranges = [tuple(r) for r in dump.get("token_ranges", [])]
    per_fact = []
    n_covered = 0
    for f in facts:
        tp = f.get("token_pos")
        if tp is None:
            per_fact.append({**f, "covered": False, "rationale": "no token_pos"})
            continue
        hit = next(
            ((lo, hi) for (lo, hi) in ranges
             if lo - bleed <= tp <= hi + bleed),
            None,
        )
        covered = hit is not None
        per_fact.append({
            "fact_idx": f.get("fact_idx"),
            "entity": f.get("entity"),
            "kind": f.get("kind"),
            "token_pos": tp,
            "covered": covered,
            "covering_range": hit,
        })
        if covered:
            n_covered += 1
    return {
        "n_facts": len(facts),
        "n_covered": n_covered,
        "coverage_pct": (
            100.0 * n_covered / max(1, len(facts))
        ),
        "n_chunks": len(ranges),
        "per_fact": per_fact,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--longctx", required=True,
                    help="longctx-svc endpoint, e.g. http://localhost:5054")
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--haystack", required=True,
                    help="haystack JSON with planted facts")
    ap.add_argument("--out", default=None, help="optional output JSON path")
    ap.add_argument("--bleed", type=int, default=32)
    args = ap.parse_args()

    facts = json.loads(Path(args.haystack).read_text())["facts"]
    dump = fetch_dump(args.longctx, args.session_id)
    cov = compute_coverage(facts, dump, bleed=args.bleed)

    print(f"Coverage: {cov['n_covered']}/{cov['n_facts']} "
          f"({cov['coverage_pct']:.1f}%) over {cov['n_chunks']} chunks",
          file=sys.stderr)
    for r in cov["per_fact"]:
        mark = "+" if r["covered"] else "-"
        rg = r.get("covering_range")
        rg_s = f"[{rg[0]}..{rg[1]}]" if rg else "—"
        print(f"  {mark} fact{r['fact_idx']:>2} {r.get('entity', ''):>16} "
              f"@ tok {r['token_pos']:>9} {rg_s}", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(cov, indent=2))
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0 if cov["coverage_pct"] >= 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
