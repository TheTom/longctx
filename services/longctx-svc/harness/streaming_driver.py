"""Streaming-chat driver for the 10M PRD.

Submits a haystack as multi-turn chat to a vLLM-compatible OpenAI
endpoint. Each turn appends a chunk of the haystack to the
conversation. After the haystack is fully fed, asks the question
battery and scores answers.

Three modes:
  "single_long" — submit the entire haystack + all questions in
                  ONE chat completion. Fails outside max_model_len.
                  Useful for the fp16 baseline arm.
  "streaming"   — chunked turns, the cumulative-history mode. By
                  turn N the request carries N×turn_tokens tokens,
                  so this requires --max-model-len ≥ haystack size
                  and cannot demonstrate the "32K active + longctx
                  external memory" PRD claim today.
  "compact"     — the today-feasible 10M-effective-context mode.
                  Phase 1 writes haystack chunks directly to longctx-
                  svc /evict/write, indexed under --session_id. Phase
                  2 asks each question as a fresh single-message chat;
                  vLLM's Tier 3 rehydrate prepends rescued chunks
                  server-side. Per-request active prompt stays ≤32K
                  regardless of haystack size. Bypasses V3 entirely
                  — the receipt is "longctx external memory at scale,"
                  not "V3 eviction-and-recovery."

Key design choices:
  * Per-turn chunk size tunable (default 8192 tokens) — must be
    ≤ vLLM's chunked-prefill chunk size and ≤ max_model_len.
  * Question phase: each question is a separate chat completion
    in streaming mode (so we can score per-question and observe
    Tier 3 firing per turn).
  * Active-window calculation: tracks turn N's KV state so the
    scorer can mark `native_hit` correctly.

Usage:
    python streaming_driver.py \\
        --haystack /tmp/haystack_100k_v3.json \\
        --endpoint http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --turn_tokens 8192 --mode streaming \\
        --out /tmp/run_100k.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass
class TurnResult:
    turn_idx: int
    role: str  # "haystack_chunk" / "question"
    question: Optional[str] = None
    fact_idx: Optional[int] = None
    truth: Optional[str] = None
    kind: Optional[str] = None
    fact_token_pos: Optional[int] = None
    answer_text: Optional[str] = None
    classification: Optional[str] = None
    extracted_answer: Optional[str] = None
    elapsed_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    rationale: Optional[str] = None


def _post_chat(
    endpoint: str, model: str, messages: list[dict],
    max_tokens: int = 64, temperature: float = 0.0,
    timeout: float = 600.0,
    session_id: Optional[str] = None,
) -> dict:
    url = endpoint.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {}
    if session_id:
        headers["X-Longctx-Session"] = session_id
    r = requests.post(
        url, json=body, headers=headers or None, timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _split_haystack(text: str, tokenizer, turn_tokens: int) -> list[str]:
    """Split a haystack into turn-sized chunks. Boundary-aligned to
    sentence ends where possible."""
    # Encode once, split by token, decode chunks.
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    for i in range(0, len(ids), turn_tokens):
        chunk_ids = ids[i:i + turn_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


def run(
    haystack_path: Path,
    endpoint: str,
    model: str,
    turn_tokens: int,
    out_path: Path,
    max_kv_active: int = 32768,
    mode: str = "streaming",
    longctx_endpoint: Optional[str] = None,
    session_id: Optional[str] = None,
    system_prefix: str = (
        "You are reading a long document chunk-by-chunk. "
        "Internalize each chunk; you will be asked questions later. "
        "Acknowledge each chunk with one short sentence."
    ),
):
    print(f"Loading haystack: {haystack_path}", file=sys.stderr)
    data = json.loads(haystack_path.read_text())
    facts = data["facts"]
    print(f"Tokens: {data['tokens_actual']:,}, facts: {len(facts)}",
          file=sys.stderr)

    print(f"Loading tokenizer: {data['tokenizer']}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        data["tokenizer"], trust_remote_code=True
    )

    chunks = _split_haystack(data["haystack"], tokenizer, turn_tokens)
    print(f"Streaming in {len(chunks)} turns of ~{turn_tokens} tokens each",
          file=sys.stderr)

    from harness.scorer import classify, ClassifyArgs

    results: list[TurnResult] = []
    messages: list[dict] = [{"role": "system", "content": system_prefix}]

    if mode == "streaming":
        # Phase 1: stream haystack
        for turn_idx, chunk_text in enumerate(chunks):
            user_msg = (
                f"[document chunk {turn_idx + 1}/{len(chunks)}]:\n"
                f"{chunk_text}"
            )
            messages.append({"role": "user", "content": user_msg})
            t0 = time.time()
            try:
                resp = _post_chat(
                    endpoint, model, messages,
                    max_tokens=32, temperature=0.0,
                    session_id=session_id,
                )
            except Exception as exc:
                print(f"[turn {turn_idx}] FAIL: {exc}", file=sys.stderr)
                results.append(TurnResult(
                    turn_idx=turn_idx, role="haystack_chunk",
                    rationale=f"http_error: {exc}",
                ))
                continue
            t1 = time.time()
            ack = resp["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": ack})
            usage = resp.get("usage", {})
            results.append(TurnResult(
                turn_idx=turn_idx, role="haystack_chunk",
                answer_text=ack[:200], elapsed_s=t1 - t0,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ))
            if turn_idx % max(1, len(chunks) // 10) == 0:
                print(f"  turn {turn_idx + 1}/{len(chunks)} "
                      f"prompt_tokens={usage.get('prompt_tokens')} "
                      f"({t1 - t0:.1f}s)", file=sys.stderr)
        haystack_turns = len(results)

        # Phase 2: ask each fact question as a fresh chat completion
        # but with the FULL conversation history (prior chunks +
        # acks). Tier 3 prefill rehydrate fires on each.
        for q_idx, f in enumerate(facts):
            question = f["question"]
            messages_q = messages + [
                {"role": "user", "content": f"QUESTION: {question}"}
            ]
            t0 = time.time()
            try:
                resp = _post_chat(
                    endpoint, model, messages_q,
                    max_tokens=64, temperature=0.0,
                    session_id=session_id,
                )
            except Exception as exc:
                print(f"[Q{q_idx}] FAIL: {exc}", file=sys.stderr)
                results.append(TurnResult(
                    turn_idx=haystack_turns + q_idx, role="question",
                    question=question, fact_idx=f["fact_idx"],
                    truth=f["answer"], kind=f["kind"],
                    fact_token_pos=f["token_pos"],
                    rationale=f"http_error: {exc}",
                ))
                continue
            t1 = time.time()
            ans = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})

            # Compute active-window for native_hit detection.
            # Heuristic: the active KV at question time has the LAST
            # max_kv_active tokens of the conversation. If the fact's
            # token_pos is within (total_tokens - max_kv_active) of
            # the end, it's still in active KV.
            total_tokens = usage.get("prompt_tokens", 0)
            active_lo = max(0, total_tokens - max_kv_active)
            cls = classify(ClassifyArgs(
                kind=f["kind"], truth=f["answer"],
                answer_text=ans, fact_token_pos=f["token_pos"],
                active_window_lo=active_lo,
                active_window_hi=total_tokens,
            ))
            results.append(TurnResult(
                turn_idx=haystack_turns + q_idx, role="question",
                question=question, fact_idx=f["fact_idx"],
                truth=f["answer"], kind=f["kind"],
                fact_token_pos=f["token_pos"],
                answer_text=ans, classification=cls.classification,
                extracted_answer=cls.extracted_answer,
                rationale=cls.rationale, elapsed_s=t1 - t0,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ))
            print(f"  Q{q_idx + 1}/{len(facts)} {f['entity']:>16} → "
                  f"{cls.classification:>14} ({t1 - t0:.1f}s)",
                  file=sys.stderr)
    elif mode == "compact":
        # Phase 1: ingest haystack directly to longctx-svc as chunks
        # tagged with the harness session id. Each chunk carries its
        # token range in the original haystack so coverage works.
        if not longctx_endpoint:
            raise ValueError("compact mode requires --longctx")
        if not session_id:
            raise ValueError("compact mode requires --session_id")
        ingest_url = longctx_endpoint.rstrip("/") + "/evict/write"
        # Reconstruct token-aligned ranges by re-encoding each chunk.
        # synthetic_haystack split chunks at exact turn_tokens boundaries
        # in token space, so chunk N occupies [N*turn_tokens, (N+1)*turn_tokens).
        chunks_payload: list[dict] = []
        for chunk_idx, chunk_text in enumerate(chunks):
            tok_start = chunk_idx * turn_tokens
            tok_end = tok_start + turn_tokens
            chunks_payload.append({
                "text": chunk_text,
                "token_range": (tok_start, tok_end),
                "layer": -3,    # synthetic external-memory marker
                "score": 0.0,
            })
        BATCH = 32
        t0 = time.time()
        for i in range(0, len(chunks_payload), BATCH):
            batch = chunks_payload[i:i + BATCH]
            r = requests.post(ingest_url, json={
                "session_id": session_id,
                "chunks": batch,
            }, timeout=120.0)
            r.raise_for_status()
            if (i // BATCH) % max(1, len(chunks_payload) // BATCH // 10) == 0:
                print(
                    f"  ingest {i + len(batch)}/{len(chunks_payload)} "
                    f"chunks ({time.time() - t0:.1f}s)",
                    file=sys.stderr,
                )
        t1 = time.time()
        print(f"Phase 1 ingest: {len(chunks_payload)} chunks in "
              f"{t1 - t0:.1f}s", file=sys.stderr)

        # Phase 2: per-question fresh single-message chat. Server-side
        # Tier 3 rehydrate (prefill_rehydrate.py:maybe_rehydrate_messages)
        # retrieves from /evict/retrieve and prepends rescued chunks as a
        # leading system message. Each request stays compact.
        compact_system = (
            "You answer factual questions about a long document. "
            "Use only the recovered context provided to you. "
            "Answer in one short sentence."
        )
        for q_idx, f in enumerate(facts):
            question = f["question"]
            messages_q = [
                {"role": "system", "content": compact_system},
                {"role": "user", "content": f"QUESTION: {question}"},
            ]
            t0 = time.time()
            try:
                resp = _post_chat(
                    endpoint, model, messages_q,
                    max_tokens=64, temperature=0.0,
                    session_id=session_id,
                )
            except Exception as exc:
                print(f"[Q{q_idx}] FAIL: {exc}", file=sys.stderr)
                results.append(TurnResult(
                    turn_idx=q_idx, role="question",
                    question=question, fact_idx=f["fact_idx"],
                    truth=f["answer"], kind=f["kind"],
                    fact_token_pos=f["token_pos"],
                    rationale=f"http_error: {exc}",
                ))
                continue
            t1 = time.time()
            ans = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            # Compact mode: no active KV holds the haystack. Disable
            # native_hit detection by setting the active window to a
            # zero-length range — every correct answer is therefore
            # attributed to rescue retrieval, which is what we want.
            cls = classify(ClassifyArgs(
                kind=f["kind"], truth=f["answer"],
                answer_text=ans, fact_token_pos=f["token_pos"],
                active_window_lo=0, active_window_hi=0,
            ))
            results.append(TurnResult(
                turn_idx=q_idx, role="question",
                question=question, fact_idx=f["fact_idx"],
                truth=f["answer"], kind=f["kind"],
                fact_token_pos=f["token_pos"],
                answer_text=ans, classification=cls.classification,
                extracted_answer=cls.extracted_answer,
                rationale=cls.rationale, elapsed_s=t1 - t0,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ))
            print(f"  Q{q_idx + 1}/{len(facts)} {f['entity']:>16} → "
                  f"{cls.classification:>14} ({t1 - t0:.1f}s)",
                  file=sys.stderr)
    elif mode == "single_long":
        # Submit haystack + all questions in ONE turn. Will fail
        # past max_model_len; useful as a baseline contrast.
        all_q = "\n".join(
            f"Q{i + 1}. {f['question']}" for i, f in enumerate(facts)
        )
        prompt = (
            f"Document:\n{data['haystack']}\n\n"
            f"Answer each of the following questions in one short "
            f"sentence:\n{all_q}"
        )
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        try:
            resp = _post_chat(
                endpoint, model, messages,
                max_tokens=128 * len(facts), temperature=0.0,
                session_id=session_id,
            )
            ans_block = resp["choices"][0]["message"]["content"]
        except Exception as exc:
            ans_block = f"ERROR: {exc}"
        t1 = time.time()
        # Naive split on Q1./Q2./etc.
        import re
        per_q = re.split(r"(?:^|\n)\s*Q\d+\.?\s*", ans_block)
        per_q = [p for p in per_q if p.strip()]
        for q_idx, f in enumerate(facts):
            ans = per_q[q_idx] if q_idx < len(per_q) else ""
            cls = classify(ClassifyArgs(
                kind=f["kind"], truth=f["answer"], answer_text=ans,
            ))
            results.append(TurnResult(
                turn_idx=q_idx, role="question",
                question=f["question"], fact_idx=f["fact_idx"],
                truth=f["answer"], kind=f["kind"],
                fact_token_pos=f["token_pos"],
                answer_text=ans, classification=cls.classification,
                extracted_answer=cls.extracted_answer,
                rationale=cls.rationale,
                elapsed_s=(t1 - t0) / max(1, len(facts)),
            ))
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Summary
    summary: dict[str, int] = {
        "exact": 0, "coherent_wrong": 0, "degenerate": 0,
        "miss": 0, "native_hit": 0,
    }
    for r in results:
        if r.role != "question": continue
        if r.classification:
            summary[r.classification] = summary.get(r.classification, 0) + 1

    coverage: Optional[dict] = None
    if longctx_endpoint and session_id:
        try:
            from harness.coverage import fetch_dump, compute_coverage
            dump = fetch_dump(longctx_endpoint, session_id)
            coverage = compute_coverage(facts, dump)
            print(f"Coverage: {coverage['n_covered']}/{coverage['n_facts']} "
                  f"({coverage['coverage_pct']:.1f}%) over "
                  f"{coverage['n_chunks']} chunks", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] coverage fetch failed: {exc}", file=sys.stderr)

    out = {
        "haystack_path": str(haystack_path),
        "tokens_actual": data["tokens_actual"],
        "n_facts": len(facts),
        "endpoint": endpoint,
        "model": model,
        "turn_tokens": turn_tokens,
        "mode": mode,
        "session_id": session_id,
        "summary": summary,
        "coverage": coverage,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSummary: {summary}", file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haystack", type=str, required=True)
    ap.add_argument("--endpoint", type=str,
                    default="http://localhost:5054/v1")
    ap.add_argument("--model", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--turn_tokens", type=int, default=8192)
    ap.add_argument("--max_kv_active", type=int, default=32768)
    ap.add_argument("--mode", type=str, default="streaming",
                    choices=["streaming", "compact", "single_long"])
    ap.add_argument("--longctx", type=str, default=None,
                    help="longctx-svc endpoint for coverage dump")
    ap.add_argument("--session_id", type=str, default=None,
                    help="V3 session id (matches the X-Longctx-Session header)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    run(
        haystack_path=Path(args.haystack),
        endpoint=args.endpoint, model=args.model,
        turn_tokens=args.turn_tokens, out_path=Path(args.out),
        max_kv_active=args.max_kv_active, mode=args.mode,
        longctx_endpoint=args.longctx, session_id=args.session_id,
    )


if __name__ == "__main__":
    main()
