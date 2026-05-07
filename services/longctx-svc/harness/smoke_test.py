"""End-to-end smoke test of the 10M PRD harness.

Patches `_post_chat` with a canned mock so the chain
  synthetic_haystack -> streaming_driver -> scorer
runs without a GPU or network. Validates plumbing only:
  * driver iterates chunks
  * Phase 2 fires one chat per fact
  * scorer assigns expected classes (exact / coherent_wrong / miss)
  * output JSON has the expected shape

Run:
    python3 harness/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make harness imports work whether invoked from repo root or /
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dataclasses import asdict
from harness import streaming_driver
from harness.synthetic_haystack import build_haystack


def _mock_post_chat_factory(facts: list[dict], plan: dict[int, str]):
    """Return a stub that mimics OpenAI chat-completions output.

    `plan[i]` maps fact index -> mock-answer string. Question phase
    detects the QUESTION: marker and looks up the matching fact by
    substring match on the question text. Haystack-chunk phase just
    returns "ok".
    """
    q_to_idx = {f["question"]: i for i, f in enumerate(facts)}
    truths = {i: f["answer"] for i, f in enumerate(facts)}

    state = {"calls": 0, "chunk_calls": 0, "q_calls": 0}

    def _stub(endpoint, model, messages, max_tokens=64, temperature=0.0,
              timeout=600.0):
        state["calls"] += 1
        last_user = next(
            (m for m in reversed(messages) if m["role"] == "user"), None
        )
        text = last_user["content"] if last_user else ""
        # Question turn?
        if "QUESTION:" in text:
            state["q_calls"] += 1
            q = text.split("QUESTION:", 1)[1].strip()
            idx = q_to_idx.get(q, -1)
            override = plan.get(idx)
            if override is not None:
                ans = override
            else:
                ans = f"The answer is {truths.get(idx, 'unknown')}."
            return {
                "choices": [{"message": {"content": ans}}],
                "usage": {
                    "prompt_tokens": 50_000,  # arbitrary
                    "completion_tokens": 12,
                },
            }
        state["chunk_calls"] += 1
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1024, "completion_tokens": 1},
        }

    return _stub, state


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=10_000)
    ap.add_argument("--facts", type=int, default=5)
    ap.add_argument("--turn_tokens", type=int, default=2048)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="harness_smoke_"))
    haystack_path = tmp / "haystack.json"
    out_path = tmp / "run.json"

    print(f"[smoke] tmp={tmp}", file=sys.stderr)
    print(f"[smoke] building {args.tokens:,}-token haystack with "
          f"N={args.facts} facts...", file=sys.stderr)

    from transformers import AutoTokenizer
    tok_name = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    text, planted = build_haystack(
        target_tokens=args.tokens, n_facts=args.facts,
        tokenizer=tok, seed=1234,
    )
    final_tokens = len(tok.encode(text))
    haystack_path.write_text(json.dumps({
        "tokens_target": args.tokens,
        "tokens_actual": final_tokens,
        "n_facts": args.facts,
        "tokenizer": tok_name,
        "seed": 1234,
        "haystack": text,
        "facts": [asdict(f) for f in planted],
    }))

    data = json.loads(haystack_path.read_text())
    facts = data["facts"]
    print(f"[smoke] haystack tokens={data['tokens_actual']:,}, "
          f"facts={len(facts)}", file=sys.stderr)

    # Plan: 0=exact (default), 1=coherent_wrong (right shape, wrong digits),
    # 2=miss (refusal), 3=degenerate (token spew), 4=exact via default.
    plan: dict[int, str] = {}
    if len(facts) >= 5:
        plan[1] = _coherent_wrong(facts[1])
        plan[2] = "I don't have that information."
        plan[3] = "8!!!!!!!!!!!"

    stub, state = _mock_post_chat_factory(facts, plan)
    streaming_driver._post_chat = stub  # monkeypatch

    streaming_driver.run(
        haystack_path=haystack_path,
        endpoint="http://mock",
        model="mock-model",
        turn_tokens=args.turn_tokens,
        out_path=out_path,
        mode="streaming",
    )

    out = json.loads(out_path.read_text())
    summary = out["summary"]
    print(f"[smoke] summary={summary}", file=sys.stderr)
    print(f"[smoke] mock state={state}", file=sys.stderr)

    # Assertions
    fails: list[str] = []
    if state["q_calls"] != len(facts):
        fails.append(
            f"expected {len(facts)} question calls, got {state['q_calls']}"
        )
    if state["chunk_calls"] < 1:
        fails.append("no chunk calls fired")

    # Plan injects 1 coherent_wrong + 1 miss + 1 degenerate when N>=5.
    # The remaining (N-3) facts return correct text; classifier promotes
    # any whose token_pos falls inside [total_tokens - max_kv_active,
    # total_tokens] to native_hit, so check the union, not exact alone.
    if len(facts) >= 5:
        n_correct = (
            summary.get("exact", 0) + summary.get("native_hit", 0)
        )
        if n_correct != len(facts) - 3:
            fails.append(
                f"exact+native_hit={n_correct} expected {len(facts) - 3}"
            )
        for k, v in {"coherent_wrong": 1, "miss": 1, "degenerate": 1}.items():
            if summary.get(k, 0) != v:
                fails.append(
                    f"summary[{k}]={summary.get(k, 0)} expected {v}"
                )

    # Spot-check that question results carry truth + extracted_answer
    q_results = [r for r in out["results"] if r["role"] == "question"]
    if len(q_results) != len(facts):
        fails.append(f"q_results len={len(q_results)} expected {len(facts)}")
    if not all(r["truth"] for r in q_results):
        fails.append("some q_results missing truth")

    if fails:
        for f in fails:
            print(f"[smoke] FAIL: {f}", file=sys.stderr)
        return 1
    print("[smoke] PASS — harness chain end-to-end clean", file=sys.stderr)
    return 0


def _coherent_wrong(f: dict) -> str:
    """Build a plausible-shaped wrong answer for `f`."""
    kind = f["kind"]
    truth = f["answer"]
    if kind == "access_code":
        return "The access code is 482000."
    if kind == "record_count":
        # Pick a different 5-digit count
        return "There are 73,412 records."
    if kind == "renewal_date":
        return "Renews on 2027-05-01."
    return f"Approximately {truth[:-1]}9."


if __name__ == "__main__":
    sys.exit(main())
