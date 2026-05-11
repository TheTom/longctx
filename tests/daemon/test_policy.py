"""Tests for the auto-policy router.

Covers:
  * Size bucketing (boundaries + interior points).
  * UNKNOWN query_shape → production-safe default at every size.
  * Each (bucket, shape) cell returns its expected policy shape.
  * detect_query_shape heuristic on representative inputs.
"""
from __future__ import annotations

import pytest

from longctx_daemon.policy import (
    EXTRA_LONG,
    LONG,
    MID,
    SHORT,
    QueryShape,
    RetrievalPolicy,
    _DEFAULT_POLICY,
    _POLICY_TABLE,
    _size_bucket,
    detect_query_shape,
    select_policy,
)


# ─────────────────────────────────────────────────────── size buckets


class TestSizeBuckets:
    def test_short_at_zero(self):
        assert _size_bucket(0) == SHORT

    def test_short_just_under_threshold(self):
        assert _size_bucket(63_999) == SHORT

    def test_mid_at_lower_threshold(self):
        assert _size_bucket(64_000) == MID

    def test_mid_interior(self):
        assert _size_bucket(150_000) == MID

    def test_long_at_lower_threshold(self):
        assert _size_bucket(256_000) == LONG

    def test_long_interior(self):
        assert _size_bucket(800_000) == LONG

    def test_xlong_at_lower_threshold(self):
        assert _size_bucket(1_000_000) == EXTRA_LONG

    def test_xlong_at_5m(self):
        assert _size_bucket(5_000_000) == EXTRA_LONG


# ─────────────────────────────────────────────────────── select_policy


class TestSelectPolicyDefaults:
    def test_unknown_shape_returns_default_at_every_size(self):
        for size in (0, 32_000, 64_000, 256_000, 1_000_000, 5_000_000):
            pol = select_policy(
                corpus_size_chars=size, query_shape=QueryShape.UNKNOWN,
            )
            assert pol == _DEFAULT_POLICY

    def test_default_policy_is_production_safe(self):
        # BM25+dense+RRF with no extras — won't break existing callers.
        assert _DEFAULT_POLICY.bm25_weight == 1.0
        assert _DEFAULT_POLICY.dense_weight == 1.0
        assert _DEFAULT_POLICY.chunk_tokens is None
        assert _DEFAULT_POLICY.rerank_pre_k is None

    def test_unknown_at_extreme_size_still_safe(self):
        pol = select_policy(
            corpus_size_chars=10_000_000_000,  # 10B chars
            query_shape=QueryShape.UNKNOWN,
        )
        assert pol == _DEFAULT_POLICY


class TestSelectPolicySymbolic:
    def test_short_symbolic_keeps_bm25(self):
        pol = select_policy(
            corpus_size_chars=10_000, query_shape=QueryShape.SYMBOLIC,
        )
        # Symbolic queries always keep BM25 — that's the whole point.
        assert pol.bm25_weight > 0.0
        assert pol.dense_weight > 0.0

    def test_long_symbolic_adds_chunking_and_rerank(self):
        pol = select_policy(
            corpus_size_chars=500_000, query_shape=QueryShape.SYMBOLIC,
        )
        assert pol.chunk_tokens is not None
        assert pol.rerank_pre_k is not None
        assert pol.bm25_weight > 0.0


class TestSelectPolicyProse:
    def test_short_prose_disables_bm25(self):
        pol = select_policy(
            corpus_size_chars=10_000, query_shape=QueryShape.PROSE,
        )
        # MRCR data: BM25 hurts prose disambiguation by ~21 abs pts.
        assert pol.bm25_weight == 0.0
        assert pol.dense_weight > 0.0

    def test_short_prose_hints_bge_m3(self):
        pol = select_policy(
            corpus_size_chars=20_000, query_shape=QueryShape.PROSE,
        )
        # Short prose → bge-m3 wins (full essay fits in 8K maxlen).
        assert pol.embedder_hint == "BAAI/bge-m3"

    def test_long_prose_switches_to_minilm(self):
        pol = select_policy(
            corpus_size_chars=500_000, query_shape=QueryShape.PROSE,
        )
        # Long prose → MiniLM's 256-token anchor beats bge-m3's full embedding.
        assert "MiniLM" in (pol.embedder_hint or "")

    def test_long_prose_uses_chunk_and_rerank(self):
        pol = select_policy(
            corpus_size_chars=500_000, query_shape=QueryShape.PROSE,
        )
        assert pol.chunk_tokens is not None
        assert pol.rerank_pre_k is not None

    def test_xlong_prose_keeps_minilm(self):
        pol = select_policy(
            corpus_size_chars=3_000_000, query_shape=QueryShape.PROSE,
        )
        assert pol.bm25_weight == 0.0
        assert "MiniLM" in (pol.embedder_hint or "")


class TestSelectPolicyMixed:
    def test_short_mixed_keeps_hybrid(self):
        pol = select_policy(
            corpus_size_chars=10_000, query_shape=QueryShape.MIXED,
        )
        assert pol.bm25_weight > 0.0  # mixed queries keep BM25 in play
        assert pol.dense_weight > 0.0

    def test_long_mixed_chunks_and_reranks(self):
        pol = select_policy(
            corpus_size_chars=500_000, query_shape=QueryShape.MIXED,
        )
        assert pol.chunk_tokens is not None
        assert pol.rerank_pre_k is not None


class TestPolicyTableCoverage:
    def test_all_known_buckets_covered_for_each_shape(self):
        # Every size_bucket × {SYMBOLIC, PROSE, MIXED} must have an entry.
        for shape in (QueryShape.SYMBOLIC, QueryShape.PROSE, QueryShape.MIXED):
            for bucket in (SHORT, MID, LONG, EXTRA_LONG):
                assert (bucket, shape) in _POLICY_TABLE, (
                    f"missing policy entry for ({bucket!r}, {shape.value})"
                )

    def test_every_policy_has_rationale(self):
        for key, pol in _POLICY_TABLE.items():
            assert pol.rationale, (
                f"policy at {key} missing rationale text"
            )


# ─────────────────────────────────────────────────────── query-shape


class TestDetectQueryShape:
    def test_empty_returns_unknown(self):
        assert detect_query_shape("") == QueryShape.UNKNOWN
        assert detect_query_shape("   ") == QueryShape.UNKNOWN
        assert detect_query_shape("hi") == QueryShape.UNKNOWN

    def test_camel_case_identifier_is_symbolic(self):
        assert detect_query_shape("getUserById") == QueryShape.SYMBOLIC

    def test_snake_case_identifier_is_symbolic(self):
        assert detect_query_shape("process_payment") == QueryShape.SYMBOLIC

    def test_screaming_snake_is_symbolic(self):
        assert detect_query_shape("SECRET_KEY_HERE") == QueryShape.SYMBOLIC

    def test_dotted_path_is_symbolic(self):
        assert detect_query_shape("app.config.settings") == QueryShape.SYMBOLIC

    def test_natural_language_is_prose(self):
        assert detect_query_shape(
            "explain how the payment system works"
        ) == QueryShape.PROSE

    def test_question_is_prose(self):
        assert detect_query_shape(
            "what does this code do"
        ) == QueryShape.PROSE

    def test_identifier_plus_NL_is_mixed(self):
        # "find the getUserById function and explain it" — both signals.
        assert detect_query_shape(
            "find the getUserById function"
        ) == QueryShape.MIXED

    def test_long_keyword_bag_defaults_to_prose(self):
        assert detect_query_shape(
            "needle find secret hidden where"
        ) == QueryShape.PROSE

    def test_short_unknown(self):
        assert detect_query_shape("hi") == QueryShape.UNKNOWN


# ─────────────────────────────────────────────────────── full integration


class TestEndToEnd:
    """Full path: detect → select → use the policy."""
    def test_short_prose_query_picks_dense_only(self):
        shape = detect_query_shape(
            "reproduce the poem about pencils"
        )
        assert shape == QueryShape.PROSE
        pol = select_policy(corpus_size_chars=20_000, query_shape=shape)
        assert pol.bm25_weight == 0.0
        assert pol.dense_weight == 1.0

    def test_long_symbolic_query_picks_chunked_rerank_hybrid(self):
        shape = detect_query_shape("UserAuthMiddleware")
        assert shape == QueryShape.SYMBOLIC
        pol = select_policy(corpus_size_chars=500_000, query_shape=shape)
        assert pol.bm25_weight > 0.0
        assert pol.chunk_tokens is not None
        assert pol.rerank_pre_k is not None

    def test_default_for_unknown_shape(self):
        # User typed something we can't classify confidently.
        shape = detect_query_shape("?")
        assert shape == QueryShape.UNKNOWN
        pol = select_policy(corpus_size_chars=100_000, query_shape=shape)
        # Conservative default; both retrieval channels live.
        assert pol.bm25_weight == 1.0
        assert pol.dense_weight == 1.0
