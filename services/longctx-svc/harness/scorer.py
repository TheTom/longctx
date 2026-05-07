"""Failure-class scorer for the 10M-effective-context PRD.

Classifies a model's answer to a question against the planted truth:

  exact          — the answer is fully correct
  coherent_wrong — the model produced a plausible-shaped answer but
                   wrong digits/date/etc. Indicates rescue reached
                   the model but retrieval surfaced wrong content,
                   or the model hallucinated within the right shape.
  degenerate     — `<digit>!!!!!` collapse, or non-numeric token
                   spew. The model's KV is too damaged to decode
                   coherently. V3 over-compression with no rescue.
  miss           — model says "I don't know" / refused / produced
                   nothing related. Honest abstention.
  native_hit     — the answer was correct AND the planted fact's
                   token range is still in the active KV window
                   (caller-provided). Not proof of rescue mechanism.

The classifier is intentionally narrow — exact match on extracted
numerics/dates. Looser matches fall to coherent_wrong. The point is
NOT to be a generic LLM grader; it's to distinguish mechanism
classes for the rescue receipts.

Usage:
    from harness.scorer import classify, ClassifyArgs
    cls = classify(ClassifyArgs(
        kind="access_code",
        truth="481729",
        answer_text="The access code is 481729.",
    ))
    cls.classification == "exact"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Tokens that signal degeneracy (token spew). The threshold is "at
# least 3 of these in a row" — a single ! could be a real punctuation.
_DEGEN_PATTERNS = [
    r"!{3,}",
    r"\.{4,}",
    r"(\b\w{1,3}\b\W*){8,}",  # 8+ tiny tokens in a row = noise
]
_DEGEN_REGEX = re.compile("|".join(_DEGEN_PATTERNS))

# Refusal / abstention markers.
_MISS_PATTERNS = [
    r"\b(don'?t know|cannot|unable to|no information|not provided|"
    r"isn'?t (in|mentioned)|wasn'?t (provided|given)|no record)\b",
]
_MISS_REGEX = re.compile("|".join(_MISS_PATTERNS), re.IGNORECASE)


@dataclass
class ClassifyArgs:
    kind: str               # "access_code" / "renewal_date" / "record_count"
    truth: str              # ground-truth answer
    answer_text: str        # full model output
    fact_token_pos: Optional[int] = None
    active_window_lo: Optional[int] = None
    active_window_hi: Optional[int] = None


@dataclass
class ClassifyResult:
    classification: str
    extracted_answer: Optional[str] = None
    rationale: str = ""


def _extract_6digit(text: str) -> list[str]:
    return re.findall(r"\b\d{6}\b", text)


def _extract_5digit(text: str) -> list[str]:
    # Allow comma-separated like "98,696" or plain "98696".
    return re.findall(r"\b\d{2,3}[,]?\d{3}\b", text)


def _extract_iso_date(text: str) -> list[str]:
    # YYYY-MM-DD plus YYYY/MM/DD plus "April 17, 2028" style.
    return re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)


def _is_degenerate(text: str) -> bool:
    return bool(_DEGEN_REGEX.search(text))


def _is_miss(text: str) -> bool:
    return bool(_MISS_REGEX.search(text))


def _is_native_hit(args: ClassifyArgs) -> bool:
    if args.fact_token_pos is None: return False
    if args.active_window_lo is None or args.active_window_hi is None:
        return False
    return args.active_window_lo <= args.fact_token_pos <= args.active_window_hi


def classify(args: ClassifyArgs) -> ClassifyResult:
    """Classify a model's answer against the planted truth."""
    text = args.answer_text.strip()

    if _is_degenerate(text):
        return ClassifyResult(
            classification="degenerate",
            rationale="degenerate token sequence detected",
        )
    if not text:
        return ClassifyResult(
            classification="miss", rationale="empty output"
        )
    if _is_miss(text) and args.truth not in text.replace(",", ""):
        return ClassifyResult(
            classification="miss",
            rationale="abstention pattern matched",
        )

    truth_norm = args.truth.replace(",", "")
    text_norm = text.replace(",", "")

    if args.kind == "access_code":
        candidates = _extract_6digit(text_norm)
    elif args.kind == "record_count":
        candidates = [c.replace(",", "") for c in _extract_5digit(text)]
    elif args.kind == "renewal_date":
        candidates = _extract_iso_date(text)
    else:
        # Generic: substring match.
        if truth_norm in text_norm:
            return ClassifyResult(
                classification="exact" if not _is_native_hit(args)
                                       else "native_hit",
                extracted_answer=args.truth,
                rationale="generic substring match",
            )
        candidates = []

    if truth_norm in candidates:
        return ClassifyResult(
            classification="exact" if not _is_native_hit(args)
                                   else "native_hit",
            extracted_answer=truth_norm,
            rationale="extracted candidate matches truth",
        )

    if candidates:
        return ClassifyResult(
            classification="coherent_wrong",
            extracted_answer=candidates[0],
            rationale=f"got {candidates[0]!r}, expected {truth_norm!r}",
        )
    if truth_norm in text_norm:
        return ClassifyResult(
            classification="exact" if not _is_native_hit(args)
                                   else "native_hit",
            extracted_answer=args.truth,
            rationale="loose substring match",
        )
    return ClassifyResult(
        classification="miss",
        rationale="no candidates extracted, truth not found",
    )


# Compact self-test — run this file directly to sanity-check.
if __name__ == "__main__":
    cases = [
        ("access_code", "481729", "The access code is 481729.", "exact"),
        ("access_code", "481729", "The access code is 482000.", "coherent_wrong"),
        ("access_code", "481729", "8!!!!!!!!!!", "degenerate"),
        ("access_code", "481729", "I don't have that information.", "miss"),
        ("access_code", "481729", "", "miss"),
        ("record_count", "98,696", "98,696 records.", "exact"),
        ("record_count", "98,696", "98696 records.", "exact"),
        ("record_count", "98,696", "73,412 records.", "coherent_wrong"),
        ("renewal_date", "2028-04-17", "Renews on 2028-04-17.", "exact"),
        ("renewal_date", "2028-04-17", "Renews on 2027-05-01.", "coherent_wrong"),
    ]
    for kind, truth, ans, expected in cases:
        result = classify(ClassifyArgs(
            kind=kind, truth=truth, answer_text=ans
        ))
        ok = "✓" if result.classification == expected else "✗"
        print(f"{ok} {kind} truth={truth!r} ans={ans!r} → "
              f"{result.classification} (expected {expected}) "
              f"[{result.rationale}]")
