"""Tests for the Messy MRCR transform algorithms.

Each transform is a pure function ``(query, seed) -> str``; the tests
hit the SHAPE of the transformation deterministically — same input +
seed must produce the same output, plus per-transform algorithm
invariants.
"""
from __future__ import annotations

import re

import pytest

from longctx_daemon.eval import messy_mrcr as mm
from longctx_daemon.eval.messy_mrcr import (
    ABSTENTION_POOL,
    TRANSFORMS,
    apply_transform,
)


# ─────────────────────────────────────── shared fixtures

CLEAN_QUERY = (
    "Please write me the 5th poem about a tabby cat verbatim, "
    "prepended with QXR-"
)


# ═══════════════════════════════════════ registry shape

def test_registry_has_all_seven_transforms() -> None:
    """The seven canonical transforms must all be registered."""
    expected = {
        "clean", "typo", "multi-question", "run-on",
        "fragments", "blob+question", "off-corpus",
    }
    assert set(TRANSFORMS) == expected


def test_registry_entries_are_callable() -> None:
    """Each transform must be callable with the (query, seed) sig."""
    for name, t in TRANSFORMS.items():
        out = t.transform("hello world", 42)
        assert isinstance(out, str), f"{name} must return str"


def test_apply_transform_unknown_raises() -> None:
    """Unknown transform name must raise KeyError with the known list."""
    with pytest.raises(KeyError) as exc:
        apply_transform("not-a-real-transform", "x", 0)
    assert "not-a-real-transform" in str(exc.value)


# ═══════════════════════════════════════ clean (control)

def test_clean_is_strict_identity() -> None:
    """Critical contract: clean MUST be a true no-op so the control
    column is unambiguous. It must not even strip whitespace."""
    weird = "  Please write me the 5th poem.  "
    assert apply_transform("clean", weird, 0) == weird


def test_clean_ignores_seed() -> None:
    """Different seeds → same output; clean is deterministic and
    seed-independent."""
    q = "any query"
    assert apply_transform("clean", q, 0) == apply_transform("clean", q, 1)
    assert apply_transform("clean", q, 0) == apply_transform("clean", q, 99999)


# ═══════════════════════════════════════ typo

def test_typo_keeps_length_close() -> None:
    """Adjacent-letter swap doesn't change length."""
    out = apply_transform("typo", CLEAN_QUERY, seed=42)
    assert len(out) == len(CLEAN_QUERY) - (1 if CLEAN_QUERY.endswith("?") else 0)


def test_typo_drops_trailing_question_mark() -> None:
    """Spec: trailing ? gets dropped."""
    out = apply_transform(
        "typo", "Where is the file located?", seed=0,
    )
    assert not out.endswith("?")


def test_typo_no_trailing_question_mark_unchanged_in_that_aspect() -> None:
    """Without a trailing ?, no removal happens."""
    base = "no question mark here"
    out = apply_transform("typo", base, seed=0)
    # Same length (no ? to drop, swaps preserve length).
    assert len(out) == len(base)


def test_typo_introduces_some_swaps_on_long_queries() -> None:
    """At seed=0 with the canonical MRCR query, at least one swap
    should be made — 3-5% rate × ~14 words = 0.4-0.7 expected swaps,
    so for SOME seed it must actually fire."""
    # Try several seeds — at least one should produce a difference.
    base = " ".join(["alpha", "bravo", "charlie", "delta", "echo"] * 5)
    differs = [
        apply_transform("typo", base, seed=s) != base
        for s in range(20)
    ]
    assert any(differs), "typo never fired on any of 20 seeds"


def test_typo_is_deterministic() -> None:
    """Same seed → same output."""
    a = apply_transform("typo", CLEAN_QUERY, seed=7)
    b = apply_transform("typo", CLEAN_QUERY, seed=7)
    assert a == b


def test_typo_protects_short_words() -> None:
    """Words <4 chars are protected — the transform shouldn't mangle
    short tokens (e.g. "is", "a", "the")."""
    base = "is a it on by my we he do go up"
    out = apply_transform("typo", base, seed=42)
    # Same words back, in same order. Some seeds may drop a trailing ?
    # but here there isn't one.
    assert out == base


def test_typo_protects_first_and_last_chars() -> None:
    """Interior-only swap means the first and last letters of any
    swapped word stay put. With seed=42 and a long word, verify."""
    long_word = "magnificent"
    base = " ".join([long_word] * 20)
    # Try multiple seeds to find one where a swap fires.
    for s in range(30):
        out = apply_transform("typo", base, seed=s)
        if out != base:
            for tok in out.split():
                assert tok[0] == long_word[0]
                assert tok[-1] == long_word[-1]
            return
    pytest.fail("typo never fired on 30 seeds with long-word input")


# ═══════════════════════════════════════ multi-question

def test_multi_question_prepends_distractors() -> None:
    """The transform must prepend at least 2 questions before
    the real query."""
    out = apply_transform("multi-question", CLEAN_QUERY, seed=0)
    # The original query should be at the END.
    assert out.endswith(CLEAN_QUERY)
    # And there must be content BEFORE it.
    assert out != CLEAN_QUERY
    assert len(out) > len(CLEAN_QUERY) + 10


def test_multi_question_uses_2_or_3_distractors() -> None:
    """k ∈ {2, 3} per the spec."""
    out = apply_transform("multi-question", CLEAN_QUERY, seed=0)
    # Count ". " separators before the real query.
    head = out[: -len(CLEAN_QUERY)]
    # Strip trailing ". " — the join leaves it as the boundary.
    head = head.rstrip()
    if head.endswith("."):
        head = head[:-1]
    n_distractors = head.count(". ") + 1 if head else 0
    assert n_distractors in (2, 3), (
        f"expected 2 or 3 distractors, got {n_distractors} from: {head!r}"
    )


def test_multi_question_is_deterministic() -> None:
    """Same seed → same output."""
    a = apply_transform("multi-question", CLEAN_QUERY, seed=7)
    b = apply_transform("multi-question", CLEAN_QUERY, seed=7)
    assert a == b


def test_multi_question_varies_across_seeds() -> None:
    """Different seeds should produce different distractor sets at
    least some of the time."""
    outs = {
        apply_transform("multi-question", CLEAN_QUERY, seed=s)
        for s in range(10)
    }
    assert len(outs) > 1


# ═══════════════════════════════════════ run-on

def test_run_on_strips_punctuation() -> None:
    """No punctuation should remain after run-on."""
    out = apply_transform("run-on", CLEAN_QUERY, seed=0)
    assert "?" not in out
    assert "," not in out
    assert "." not in out
    assert "-" not in out


def test_run_on_lowercases() -> None:
    """All chars must be lowercase or whitespace/digits."""
    out = apply_transform("run-on", CLEAN_QUERY, seed=0)
    for ch in out:
        if ch.isalpha():
            assert ch.islower(), f"uppercase char in run-on output: {ch!r}"


def test_run_on_collapses_whitespace() -> None:
    """No double spaces, no newlines."""
    out = apply_transform("run-on", "hello   world  \n  foo", seed=0)
    assert "  " not in out
    assert "\n" not in out


def test_run_on_seed_irrelevant() -> None:
    """run-on is deterministic regardless of seed."""
    a = apply_transform("run-on", CLEAN_QUERY, seed=0)
    b = apply_transform("run-on", CLEAN_QUERY, seed=99)
    assert a == b


# ═══════════════════════════════════════ fragments

def test_fragments_drops_stopwords() -> None:
    """``the``, ``a``, ``please`` should be stripped."""
    out = apply_transform("fragments", CLEAN_QUERY, seed=0)
    tokens = set(out.split())
    assert "the" not in tokens
    assert "a" not in tokens
    assert "please" not in tokens


def test_fragments_keeps_content_words() -> None:
    """``poem``, ``tabby``, ``cat``, ``5th``, ``qxr`` should survive."""
    out = apply_transform("fragments", CLEAN_QUERY, seed=0)
    tokens = set(out.split())
    # 5th becomes "5th" (lowercased; alnum survives the regex).
    assert "5th" in tokens
    assert "poem" in tokens
    assert "tabby" in tokens
    assert "cat" in tokens
    assert "verbatim" in tokens


def test_fragments_no_punctuation() -> None:
    """Output is run-on style stripped down further; no punctuation."""
    out = apply_transform("fragments", CLEAN_QUERY, seed=0)
    assert "?" not in out
    assert "," not in out


def test_fragments_idempotent() -> None:
    """Running fragments twice on the same input gives the same output."""
    out1 = apply_transform("fragments", CLEAN_QUERY, seed=0)
    out2 = apply_transform("fragments", out1, seed=0)
    assert out1 == out2


# ═══════════════════════════════════════ blob+question

def test_blob_question_contains_traceback_marker() -> None:
    """The fake stack-trace marker must appear in the output."""
    out = apply_transform("blob+question", CLEAN_QUERY, seed=0)
    assert "Traceback" in out
    assert "File '" in out


def test_blob_question_ends_with_real_query() -> None:
    """The real query should be at the end so the searcher can still
    parse it; the blob is dilution before the question."""
    out = apply_transform("blob+question", CLEAN_QUERY, seed=0)
    assert out.endswith(CLEAN_QUERY)


def test_blob_question_is_deterministic() -> None:
    a = apply_transform("blob+question", CLEAN_QUERY, seed=0)
    b = apply_transform("blob+question", CLEAN_QUERY, seed=99)
    assert a == b   # blob is constant by design


def test_blob_question_is_multiline() -> None:
    """Stack traces span multiple lines (used by classify_query_type)."""
    out = apply_transform("blob+question", CLEAN_QUERY, seed=0)
    assert "\n" in out


# ═══════════════════════════════════════ off-corpus

def test_off_corpus_returns_pool_query() -> None:
    """Output must be one of the 30 pool entries verbatim."""
    out = apply_transform("off-corpus", CLEAN_QUERY, seed=0)
    assert out in ABSTENTION_POOL


def test_off_corpus_pool_size_is_thirty() -> None:
    """Spec calls for a 30-query pool."""
    assert len(ABSTENTION_POOL) == 30


def test_off_corpus_pool_has_no_duplicates() -> None:
    """No duplicates in the pool — would skew seed selection."""
    assert len(set(ABSTENTION_POOL)) == len(ABSTENTION_POOL)


def test_off_corpus_seed_picks_modulo() -> None:
    """Seed N picks pool[N % len(pool)] — so seed 0 == seed 30."""
    a = apply_transform("off-corpus", CLEAN_QUERY, seed=0)
    b = apply_transform("off-corpus", CLEAN_QUERY, seed=30)
    assert a == b


def test_off_corpus_is_deterministic_per_seed() -> None:
    a = apply_transform("off-corpus", CLEAN_QUERY, seed=7)
    b = apply_transform("off-corpus", CLEAN_QUERY, seed=7)
    assert a == b


def test_off_corpus_ignores_input_query() -> None:
    """The off-corpus transform replaces the query entirely; the input
    string content shouldn't influence the output."""
    a = apply_transform("off-corpus", "totally different text here", seed=3)
    b = apply_transform("off-corpus", CLEAN_QUERY, seed=3)
    assert a == b


# ═══════════════════════════════════════ regression — module-level check

def test_all_transforms_accept_empty_string() -> None:
    """No transform should crash on an empty input. Exception:
    the typo transform is fine because the loop is over an empty list."""
    for name in TRANSFORMS:
        out = apply_transform(name, "", seed=0)
        assert isinstance(out, str)


def test_all_transforms_handle_unicode() -> None:
    """Sanity check — unicode shouldn't crash anything."""
    q = "where is café résumé naïve"
    for name in TRANSFORMS:
        out = apply_transform(name, q, seed=1)
        assert isinstance(out, str)
