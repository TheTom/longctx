"""Admission ladder probe — pre-flight for the 10M PRD.

Submits prompts of 8K, 32K, 64K, 256K, 1M, 4M, 10M tokens to a
vLLM-compatible OpenAI endpoint and checks each one is ADMITTED
without a max-model-len rejection. We don't care about answer
quality — only "did the engine accept the prompt?".

Failure modes detected:
  * 400 Bad Request from vLLM: "max_model_len exceeded"
  * 5xx engine crash on extreme position-id values
  * timeout (default 600s) — flag as UNKNOWN, may still be running

Usage:
    python admission_ladder.py \\
        --endpoint http://10.245.71.5:8000/v1 \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --tokenizer Qwen/Qwen2.5-32B-Instruct \\
        --max_tokens 16
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests


# Filler that tokenizes to ~one token per char on Qwen — actual
# tokenizer ratio is closer to 0.25-0.5 tok/char so we'll over-shoot.
_FILLER_WORD = "alpha "


def _build_prompt_text(target_tokens: int, tokenizer) -> str:
    """Build a filler string that encodes to ~target_tokens tokens.

    Iteratively grow until the encode length matches. ~Linear in
    target — fine for scales up to 10M (a few seconds for tokenize).
    """
    base = _FILLER_WORD * 200
    base_tokens = len(tokenizer.encode(base, add_special_tokens=False))
    repeats = max(1, target_tokens // base_tokens + 1)
    text = _FILLER_WORD * (200 * repeats)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual


def _post(endpoint: str, model: str, prompt: str,
          max_tokens: int, timeout: float) -> tuple[int, str]:
    url = endpoint.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        r = requests.post(url, json=body, timeout=timeout)
    except requests.exceptions.Timeout:
        return -1, "timeout"
    except requests.exceptions.RequestException as exc:
        return -2, f"connect_error: {exc}"
    return r.status_code, (r.text[:500] if r.status_code >= 400 else "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name (defaults to --model)")
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--rungs", type=str,
                    default="8000,32000,64000,256000,1000000,4000000,10000000")
    args = ap.parse_args()

    print(f"[ladder] loading tokenizer: {args.tokenizer or args.model}",
          file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )

    rungs = [int(x) for x in args.rungs.split(",")]
    print(f"{'target':>10}  {'actual':>10}  {'status':>8}  "
          f"{'elapsed_s':>10}  detail", flush=True)
    summary: list[dict[str, Any]] = []
    for target in rungs:
        prompt, actual = _build_prompt_text(target, tok)
        t0 = time.time()
        code, detail = _post(args.endpoint, args.model, prompt,
                              args.max_tokens, args.timeout)
        elapsed = time.time() - t0
        verdict = (
            "ADMIT" if code == 200
            else "REJECT" if code in (400, 422)
            else "TIMEOUT" if code == -1
            else "ERROR"
        )
        print(f"{target:>10d}  {actual:>10d}  {verdict:>8}  "
              f"{elapsed:>10.1f}  {detail[:80]}", flush=True)
        summary.append({
            "target": target, "actual": actual, "verdict": verdict,
            "elapsed_s": elapsed, "detail": detail[:200],
        })
        if verdict == "REJECT" or verdict == "ERROR":
            # First reject is the cap; no point trying larger.
            print(f"[ladder] cap detected at target={target}",
                  file=sys.stderr)
            break

    print("\n[ladder] summary:", file=sys.stderr)
    for s in summary:
        print(f"  {s['target']:>10d} → {s['verdict']}", file=sys.stderr)


if __name__ == "__main__":
    main()
