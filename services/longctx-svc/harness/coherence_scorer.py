"""Coherence scorer for the 10M-context PRD.

Distinct from `scorer.py` (single-fact NIAH classifier). This one knows
about task families and evidence-presence so we can separate retrieval
failures from reasoning failures.

Four mutually-exclusive outcomes per question:

  exact            — answer matches truth AND retrieval surfaced ≥
                     min_evidence_count of required spans
  retrieval_miss   — retrieval did NOT surface enough required evidence
                     (the answer was unanswerable from what the model saw)
  reasoning_fail   — retrieval was complete; model still answered wrong
                     (this is the model's coherence/reasoning failure)
  coherent_wrong   — model produced a plausible-shaped wrong answer; we
                     only emit this when evidence presence is incomplete
                     AND the answer matches a "shape but wrong value"
                     pattern. Rare, mostly diagnostic.

Auxiliary outcomes (suspicious / debug):
  exact_lucky      — answer correct but retrieval was incomplete; could
                     be model memorization or guess. Reported separately.
  degenerate       — `<digit>!!!!` collapse, etc. Always a model failure.
  miss             — empty / refusal output.

Usage:
    cls = classify_coherence(CoherenceArgs(
        task=task_instance,                     # harness.tasks.TaskInstance
        answer_text=model_output,
        retrieved_chunk_texts=[...],            # what the harness fed back
    ))
    cls.classification, cls.evidence_present_count, ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoherenceArgs:
    task: object                            # harness.tasks.TaskInstance
    answer_text: str
    retrieved_chunk_texts: list[str]


@dataclass
class CoherenceResult:
    classification: str
    evidence_present_count: int             # how many required spans were surfaced
    evidence_required_count: int            # min_evidence_count
    evidence_present_ids: list[str] = field(default_factory=list)
    evidence_missing_ids: list[str] = field(default_factory=list)
    extracted_answer: Optional[str] = None
    rationale: str = ""


_DEGEN_REGEX = re.compile(
    r"!{3,}|\.{4,}|(\b\w{1,3}\b\W*){8,}"
)
_MISS_REGEX = re.compile(
    r"\b(don'?t (know|have)|do(n'?t)? not know|cannot|"
    r"unable to|no (specific )?information|not (provided|sure|specified)|"
    r"isn'?t (in|mentioned|provided)|wasn'?t (provided|given)|"
    r"no record|i'?m not sure|i don'?t see|insufficient information)\b",
    re.IGNORECASE,
)


def _is_degenerate(text: str) -> bool:
    return bool(_DEGEN_REGEX.search(text))


def _is_miss(text: str) -> bool:
    return bool(_MISS_REGEX.search(text))


def _evidence_in_chunks(
    span_marker: str, chunks: list[str],
) -> bool:
    """Substring match: marker is a unique-in-corpus discriminator from
    the span. Robust to chunk boundaries because markers are short."""
    if not span_marker:
        return False
    for c in chunks:
        if span_marker in c:
            return True
    return False


def _answer_matches(answer: str, truth: str, kind: str) -> bool:
    """Family-agnostic answer match. Strips commas + currency markers."""
    a = answer.strip().replace(",", "").replace("$", "")
    t = truth.strip().replace(",", "").replace("$", "")
    if not t:
        return False
    if t in a:
        return True
    # For string answers, case-insensitive whole-word match
    if kind == "string":
        if re.search(r"\b" + re.escape(t) + r"\b", a, re.IGNORECASE):
            return True
    return False


def classify_coherence(args: CoherenceArgs) -> CoherenceResult:
    """Score a model output on a single task using the 4-outcome taxonomy."""
    task = args.task
    text = args.answer_text.strip()

    # 1) Compute evidence presence first — we need it for all branches
    chunks = args.retrieved_chunk_texts
    present_ids: list[str] = []
    missing_ids: list[str] = []
    span_by_id = {s.evidence_id: s for s in task.evidence_spans}
    for eid in task.required_evidence_ids:
        span = span_by_id.get(eid)
        if span is not None and _evidence_in_chunks(span.marker, chunks):
            present_ids.append(eid)
        else:
            missing_ids.append(eid)
    evidence_complete = len(present_ids) >= int(task.min_evidence_count)

    # 2) Pre-classify obvious failure modes
    if _is_degenerate(text):
        return CoherenceResult(
            classification="degenerate",
            evidence_present_count=len(present_ids),
            evidence_required_count=task.min_evidence_count,
            evidence_present_ids=present_ids,
            evidence_missing_ids=missing_ids,
            rationale="degenerate token sequence",
        )
    if not text:
        return CoherenceResult(
            classification="miss",
            evidence_present_count=len(present_ids),
            evidence_required_count=task.min_evidence_count,
            evidence_present_ids=present_ids,
            evidence_missing_ids=missing_ids,
            rationale="empty output",
        )
    if _is_miss(text) and task.answer not in text.replace(",", ""):
        return CoherenceResult(
            classification="miss",
            evidence_present_count=len(present_ids),
            evidence_required_count=task.min_evidence_count,
            evidence_present_ids=present_ids,
            evidence_missing_ids=missing_ids,
            rationale="abstention pattern",
        )

    # 3) Did the model produce the right answer?
    correct = _answer_matches(text, task.answer, task.answer_kind)

    # 4) Combine evidence + correctness into the 4-outcome taxonomy
    if correct and evidence_complete:
        cls = "exact"
        rationale = "answer correct and required evidence retrieved"
    elif correct and not evidence_complete:
        cls = "exact_lucky"
        rationale = (
            "answer correct but only "
            f"{len(present_ids)}/{task.min_evidence_count} required spans "
            "retrieved — model may be memorizing / lucky"
        )
    elif (not correct) and evidence_complete:
        cls = "reasoning_fail"
        rationale = (
            "all required evidence retrieved, model still answered wrong"
        )
    else:
        # not correct, not evidence_complete
        cls = "retrieval_miss"
        rationale = (
            f"only {len(present_ids)}/{task.min_evidence_count} required "
            f"spans retrieved; answer also wrong"
        )

    extracted = text[:200]

    return CoherenceResult(
        classification=cls,
        evidence_present_count=len(present_ids),
        evidence_required_count=task.min_evidence_count,
        evidence_present_ids=present_ids,
        evidence_missing_ids=missing_ids,
        extracted_answer=extracted,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import random
    from harness.tasks import generate_tasks

    rng = random.Random(0)
    tasks = generate_tasks(rng, {
        "single_fact": 1, "multi_hop": 1, "contradiction": 1,
        "aggregation": 1, "temporal": 1,
    })

    cases = []
    # 1) single_fact perfect
    t = tasks[0]
    cases.append(("single_fact perfect",
                  CoherenceArgs(task=t, answer_text=t.answer,
                                retrieved_chunk_texts=[t.evidence_spans[0].text]),
                  "exact"))
    # 2) multi_hop with all evidence + correct
    t = tasks[1]
    cases.append(("multi_hop perfect",
                  CoherenceArgs(task=t, answer_text=t.answer,
                                retrieved_chunk_texts=[s.text for s in t.evidence_spans]),
                  "exact"))
    # 3) multi_hop with only 1 of 3 spans → retrieval_miss
    cases.append(("multi_hop partial → retrieval_miss",
                  CoherenceArgs(task=t, answer_text="$0",
                                retrieved_chunk_texts=[t.evidence_spans[0].text]),
                  "retrieval_miss"))
    # 4) multi_hop all spans + wrong answer → reasoning_fail
    cases.append(("multi_hop reasoning_fail",
                  CoherenceArgs(task=t, answer_text="$1",
                                retrieved_chunk_texts=[s.text for s in t.evidence_spans]),
                  "reasoning_fail"))
    # 5) contradiction with both spans + correct
    t = tasks[2]
    cases.append(("contradiction perfect",
                  CoherenceArgs(task=t, answer_text=t.answer,
                                retrieved_chunk_texts=[s.text for s in t.evidence_spans]),
                  "exact"))
    # 6) contradiction with only old span → retrieval_miss
    old_span = next(s for s in t.evidence_spans
                    if s.evidence_id.endswith(".original"))
    cases.append(("contradiction only-original → retrieval_miss",
                  CoherenceArgs(task=t, answer_text="000000",
                                retrieved_chunk_texts=[old_span.text]),
                  "retrieval_miss"))
    # 7) aggregation perfect
    t = tasks[3]
    cases.append(("aggregation perfect",
                  CoherenceArgs(task=t, answer_text=t.answer,
                                retrieved_chunk_texts=[s.text for s in t.evidence_spans]),
                  "exact"))
    # 8) temporal perfect
    t = tasks[4]
    cases.append(("temporal perfect",
                  CoherenceArgs(task=t, answer_text=t.answer,
                                retrieved_chunk_texts=[s.text for s in t.evidence_spans]),
                  "exact"))
    # 9) degenerate
    t = tasks[0]
    cases.append(("degenerate",
                  CoherenceArgs(task=t, answer_text="!!!!!!!!!!!!",
                                retrieved_chunk_texts=[t.evidence_spans[0].text]),
                  "degenerate"))
    # 10) miss
    cases.append(("miss / refusal",
                  CoherenceArgs(task=t, answer_text="I don't have that information.",
                                retrieved_chunk_texts=[t.evidence_spans[0].text]),
                  "miss"))

    fail = 0
    for name, args, expected in cases:
        r = classify_coherence(args)
        ok = r.classification == expected
        mark = "✓" if ok else "✗"
        if not ok:
            fail += 1
        print(f"{mark} {name}: got {r.classification:>15} "
              f"(expected {expected:>15}) "
              f"evidence={r.evidence_present_count}/{r.evidence_required_count}")
    sys.exit(0 if fail == 0 else 1)
