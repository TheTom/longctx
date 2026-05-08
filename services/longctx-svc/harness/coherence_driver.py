"""Coherence-test driver for the 10M PRD.

End-to-end runner:
  1. Load haystack JSON (built by `synthetic_haystack.build_haystack_with_tasks`).
  2. Phase 1: POST haystack chunks to longctx-svc `/evict/write`,
     tagged with `--session_id`. Each chunk is a turn_tokens-sized slice.
  3. Phase 2: per task, POST to longctx-svc `/evict/retrieve` with the
     task's question; record which evidence ids were surfaced
     (substring match on each EvidenceSpan.marker).
  4. Phase 3: build the model prompt = system + retrieved_chunks + question.
     Submit a single chat completion. Per-request stays ≤32K.
  5. Phase 4: classify each task with `coherence_scorer.classify_coherence`.
  6. Aggregate per-family + global metrics: exact_rate, retrieval_recall,
     evidence_complete_wrong rate, evidence_missing rate, latency.

Usage:
    python -m harness.coherence_driver \\
        --corpus /tmp/coherence_100k.json \\
        --endpoint http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --longctx http://localhost:5054 \\
        --session_id coh-100k-{timestamp} \\
        --turn_tokens 8192 --top_k 8 \\
        --out /tmp/coherence_run.json

Stage gating: this driver is the same shape regardless of haystack size,
so a 100K smoke and a 10M cloud run differ only in --corpus + wall time.
Always dry-run smoke first (mocked endpoint) before AMD cycles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harness.tasks import EvidenceSpan, TaskInstance, generate_tasks  # noqa: F401
from harness.coherence_scorer import (
    classify_coherence, CoherenceArgs, CoherenceResult,
)


@dataclass
class TaskResult:
    task_id: str
    task_family: str
    question: str
    answer_truth: str
    answer_text: str
    classification: str
    evidence_present_count: int
    evidence_required_count: int
    evidence_present_ids: list[str]
    evidence_missing_ids: list[str]
    retrieved_chunk_ids: list[int]            # chunk_idx of each surfaced
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_s: Optional[float] = None
    rationale: str = ""


def _post_chat(
    endpoint: str, model: str, messages: list[dict],
    max_tokens: int = 64, temperature: float = 0.0,
    timeout: float = 600.0, session_id: Optional[str] = None,
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
    r = requests.post(url, json=body, headers=headers or None, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _split_haystack(text: str, tokenizer, turn_tokens: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    for i in range(0, len(ids), turn_tokens):
        chunks.append(tokenizer.decode(
            ids[i:i + turn_tokens], skip_special_tokens=True,
        ))
    return chunks


def _ingest_to_longctx(
    longctx_url: str, session_id: str, chunks: list[str],
    turn_tokens: int, batch: int = 32,
) -> int:
    """Phase 1: write each chunk to longctx /evict/write. Returns count."""
    write_url = longctx_url.rstrip("/") + "/evict/write"
    payload = []
    for i, ct in enumerate(chunks):
        tok_start = i * turn_tokens
        tok_end = tok_start + turn_tokens
        payload.append({
            "text": ct,
            "token_range": (tok_start, tok_end),
            "layer": -3,
            "score": 0.0,
        })
    posted = 0
    for i in range(0, len(payload), batch):
        r = requests.post(write_url, json={
            "session_id": session_id,
            "chunks": payload[i:i + batch],
        }, timeout=120.0)
        r.raise_for_status()
        posted += min(batch, len(payload) - i)
    return posted


def _retrieve_for_task(
    longctx_url: str, session_id: str, query: str,
    top_k: int, score_floor: float = 0.0,
) -> list[dict]:
    """Phase 2: ask longctx for top-K chunks for this question."""
    url = longctx_url.rstrip("/") + "/evict/retrieve"
    r = requests.post(url, json={
        "session_id": session_id,
        "query": query,
        "top_k": int(top_k),
        "score_floor": float(score_floor),
    }, timeout=60.0)
    r.raise_for_status()
    return list(r.json().get("chunks") or [])


def _build_compact_prompt(
    retrieved_chunk_texts: list[str], question: str, max_chars: int,
) -> list[dict]:
    """Phase 3: stitch retrieved context + question into a small prompt.

    Caps total chars to keep per-request prompt ≤32K-ish. When a single
    chunk would exceed the cap on first iteration, truncate it instead
    of returning an empty body (the bug Scout-100K-v1 hit: 8K-token
    chunks decode to ~50K chars with Qwen tokenizer, which exceeded
    the prior 24K default and skipped the only retrieved chunk)."""
    sys_prompt = (
        "You answer factual questions about a long document using only "
        "the recovered context provided below. Answer in one short "
        "sentence. If the context contradicts itself, prefer the most "
        "recent value. If the context is insufficient, say so."
    )
    body_parts = []
    used = 0
    HEADER_BUDGET = 80  # bytes per chunk header
    for i, c in enumerate(retrieved_chunk_texts):
        header = f"\n--- recovered chunk {i + 1} ---\n"
        block = f"{header}{c}\n"
        if used + len(block) > max_chars:
            room = max_chars - used - len(header) - 16
            if room > 256:
                truncated_header = (
                    f"\n--- recovered chunk {i + 1} (truncated) ---\n"
                )
                body_parts.append(f"{truncated_header}{c[:room]}\n")
            break
        body_parts.append(block)
        used += len(block)
    user_msg = (
        "Recovered context:\n" + "".join(body_parts)
        + f"\n\nQUESTION: {question}\n\nAnswer:"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]


def run(
    corpus_path: Path,
    endpoint: str,
    model: str,
    longctx_url: str,
    session_id: str,
    turn_tokens: int,
    top_k: int,
    score_floor: float,
    max_recovered_chars: int,
    out_path: Path,
):
    print(f"[coh] loading corpus: {corpus_path}", file=sys.stderr)
    data = json.loads(corpus_path.read_text())
    tasks_raw = data["tasks"]
    haystack_text = data["haystack"]

    print(f"[coh] tokens={data['tokens_actual']:,}, "
          f"tasks={len(tasks_raw)}", file=sys.stderr)

    print(f"[coh] loading tokenizer: {data['tokenizer']}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        data["tokenizer"], trust_remote_code=True,
    )

    chunks = _split_haystack(haystack_text, tokenizer, turn_tokens)
    print(f"[coh] {len(chunks)} chunks @ ~{turn_tokens} tokens",
          file=sys.stderr)

    # Phase 1
    t0 = time.time()
    n = _ingest_to_longctx(longctx_url, session_id, chunks, turn_tokens)
    t1 = time.time()
    print(f"[coh] ingested {n} chunks in {t1 - t0:.1f}s", file=sys.stderr)

    # Phase 2-4 per task
    results: list[TaskResult] = []
    family_counts: dict[str, dict[str, int]] = {}
    for tr in tasks_raw:
        # Reconstitute a TaskInstance-like object for the scorer
        task = TaskInstance(
            task_id=tr["task_id"],
            task_family=tr["task_family"],
            question=tr["question"],
            answer=tr["answer"],
            answer_kind=tr["answer_kind"],
            evidence_spans=[
                EvidenceSpan(**s) for s in tr["evidence_spans"]
            ],
            required_evidence_ids=list(tr["required_evidence_ids"]),
            min_evidence_count=int(tr["min_evidence_count"]),
            rationale=tr.get("rationale", ""),
        )

        # Phase 2: retrieve
        retrieved = _retrieve_for_task(
            longctx_url, session_id, task.question,
            top_k=top_k, score_floor=score_floor,
        )
        retrieved_texts = [r.get("text", "") for r in retrieved]
        retrieved_chunk_ids = []
        for r in retrieved:
            tr_range = r.get("token_range") or [0, 0]
            ci = int(tr_range[0]) // turn_tokens
            retrieved_chunk_ids.append(ci)

        # Phase 3: model answer
        messages = _build_compact_prompt(
            retrieved_texts, task.question, max_recovered_chars,
        )
        t0q = time.time()
        try:
            resp = _post_chat(
                endpoint, model, messages,
                max_tokens=64, temperature=0.0,
                session_id=session_id,
            )
            ans_text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {}) or {}
        except Exception as exc:
            ans_text = f"<error: {exc}>"
            usage = {}
        elapsed = time.time() - t0q

        # Phase 4: classify
        cr = classify_coherence(CoherenceArgs(
            task=task, answer_text=ans_text,
            retrieved_chunk_texts=retrieved_texts,
        ))
        results.append(TaskResult(
            task_id=task.task_id,
            task_family=task.task_family,
            question=task.question,
            answer_truth=task.answer,
            answer_text=ans_text[:300],
            classification=cr.classification,
            evidence_present_count=cr.evidence_present_count,
            evidence_required_count=cr.evidence_required_count,
            evidence_present_ids=cr.evidence_present_ids,
            evidence_missing_ids=cr.evidence_missing_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            elapsed_s=elapsed,
            rationale=cr.rationale,
        ))
        fc = family_counts.setdefault(
            task.task_family,
            {"exact": 0, "exact_lucky": 0, "reasoning_fail": 0,
             "retrieval_miss": 0, "miss": 0, "degenerate": 0,
             "coherent_wrong": 0, "_total": 0},
        )
        fc[cr.classification] = fc.get(cr.classification, 0) + 1
        fc["_total"] += 1
        print(
            f"  [{task.task_family:>14}] {task.task_id} → "
            f"{cr.classification:>15} "
            f"ev={cr.evidence_present_count}/{cr.evidence_required_count} "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )

    # Aggregates
    by_family = {}
    overall = {"exact": 0, "exact_lucky": 0, "reasoning_fail": 0,
               "retrieval_miss": 0, "miss": 0, "degenerate": 0,
               "coherent_wrong": 0, "_total": 0}
    for fam, fc in family_counts.items():
        total = fc["_total"]
        denom = max(1, total)
        by_family[fam] = {
            **fc,
            "exact_rate": fc.get("exact", 0) / denom,
            "exact_or_lucky_rate":
                (fc.get("exact", 0) + fc.get("exact_lucky", 0)) / denom,
            "retrieval_miss_rate": fc.get("retrieval_miss", 0) / denom,
            "reasoning_fail_rate": fc.get("reasoning_fail", 0) / denom,
        }
        for k, v in fc.items():
            overall[k] = overall.get(k, 0) + v

    # Retrieval recall: total_present_required / total_required across tasks
    total_required = sum(r.evidence_required_count for r in results)
    total_present = sum(r.evidence_present_count for r in results)
    retrieval_recall = (
        total_present / max(1, total_required)
    )

    out = {
        "corpus_path": str(corpus_path),
        "tokens_actual": data["tokens_actual"],
        "n_tasks": len(results),
        "endpoint": endpoint, "model": model,
        "longctx": longctx_url, "session_id": session_id,
        "turn_tokens": turn_tokens, "top_k": top_k,
        "score_floor": score_floor,
        "n_chunks_ingested": n,
        "ingest_seconds": t1 - t0,
        "results": [asdict(r) for r in results],
        "by_family": by_family,
        "overall": overall,
        "retrieval_recall_atK": retrieval_recall,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[coh] retrieval_recall@{top_k}: "
          f"{retrieval_recall * 100:.1f}%", file=sys.stderr)
    print(f"[coh] overall: {overall}", file=sys.stderr)
    print(f"[coh] wrote {out_path}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--endpoint",
                    default="http://localhost:5054/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--longctx", required=True)
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--turn_tokens", type=int, default=8192)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--score_floor", type=float, default=0.0)
    ap.add_argument("--max_recovered_chars", type=int, default=24000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(
        corpus_path=Path(args.corpus),
        endpoint=args.endpoint, model=args.model,
        longctx_url=args.longctx, session_id=args.session_id,
        turn_tokens=args.turn_tokens, top_k=args.top_k,
        score_floor=args.score_floor,
        max_recovered_chars=args.max_recovered_chars,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
