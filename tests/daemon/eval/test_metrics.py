"""Tests for ``longctx_daemon.eval._metrics``.

Recall@K is a 1-line definition (1 if rank<=k else 0, averaged across
samples) — but the harness uses it across every messy-query and MRCR
sample, so the corner cases need to behave deterministically.

Covered corners:
  * miss (rank=None) → all zeros
  * rank below first cutoff → only larger-K cutoffs flip on
  * rank exactly at K → counts as a hit (closed interval)
  * rank exceeding all K → all zeros
  * empty + degenerate aggregate inputs
  * floats round-trip after aggregation
"""
from __future__ import annotations

import pytest

from longctx_daemon.eval._metrics import aggregate_recall, recall_at_k


# ────────────────────────────────────────── recall_at_k

def test_recall_at_k_top1():
    assert recall_at_k(1, (1, 3, 5, 8)) == {1: 1, 3: 1, 5: 1, 8: 1}


def test_recall_at_k_miss_returns_all_zeros():
    assert recall_at_k(None, (1, 3, 5)) == {1: 0, 3: 0, 5: 0}


def test_recall_at_k_rank_at_cutoff_counts_as_hit():
    # Closed interval: rank == k → indicator is 1 at depth k.
    assert recall_at_k(3, (1, 3, 5)) == {1: 0, 3: 1, 5: 1}


def test_recall_at_k_rank_above_largest_k_all_zero():
    assert recall_at_k(99, (1, 3, 5, 8)) == {1: 0, 3: 0, 5: 0, 8: 0}


def test_recall_at_k_rank_zero_treated_as_miss():
    # Defensive: 0 isn't a valid 1-indexed rank; treat as miss.
    assert recall_at_k(0, (1, 3, 5)) == {1: 0, 3: 0, 5: 0}


def test_recall_at_k_negative_rank_treated_as_miss():
    assert recall_at_k(-2, (1, 3, 5)) == {1: 0, 3: 0, 5: 0}


def test_recall_at_k_single_k():
    assert recall_at_k(2, (5,)) == {5: 1}
    assert recall_at_k(7, (5,)) == {5: 0}


def test_recall_at_k_handles_unsorted_k_values():
    # Caller may pass (8, 1, 5, 3); output keys reflect inputs.
    out = recall_at_k(4, (8, 1, 5, 3))
    assert out == {8: 1, 1: 0, 5: 1, 3: 0}


# ────────────────────────────────────────── aggregate_recall

def test_aggregate_recall_simple_average():
    rows = [
        {1: 1, 3: 1, 5: 1},
        {1: 0, 3: 1, 5: 1},
        {1: 0, 3: 0, 5: 1},
        {1: 1, 3: 1, 5: 1},
    ]
    out = aggregate_recall(rows)
    assert out[1] == 0.5
    assert out[3] == 0.75
    assert out[5] == 1.0


def test_aggregate_recall_empty_input_returns_empty():
    # Distinct from all-zeros: empty == "no samples evaluated".
    assert aggregate_recall([]) == {}


def test_aggregate_recall_single_row_passes_through():
    out = aggregate_recall([{1: 1, 3: 1}])
    assert out == {1: 1.0, 3: 1.0}
    out2 = aggregate_recall([{1: 0, 3: 0}])
    assert out2 == {1: 0.0, 3: 0.0}


def test_aggregate_recall_missing_key_treated_as_zero():
    # If a per-sample dict is missing a key the aggregator template
    # has, count it as a miss for that depth (lenient by design).
    rows = [
        {1: 1, 3: 1, 5: 1},
        {1: 1},  # missing 3 and 5
    ]
    out = aggregate_recall(rows)
    assert out[1] == 1.0
    assert out[3] == 0.5
    assert out[5] == 0.5


def test_aggregate_recall_round_trip_via_recall_at_k():
    # End-to-end: build per-sample indicators with recall_at_k, then
    # aggregate. This is exactly what the harness does.
    ranks = [1, 3, None, 6, 2]   # 5 samples
    per_sample = [recall_at_k(r, (1, 3, 5)) for r in ranks]
    agg = aggregate_recall(per_sample)
    # R@1 hits: only rank 1            → 1/5 = 0.2
    # R@3 hits: ranks {1, 3, 2}        → 3/5 = 0.6
    # R@5 hits: ranks {1, 3, 2}        → 3/5 = 0.6 (rank 6 is past 5)
    assert agg[1] == pytest.approx(0.2)
    assert agg[3] == pytest.approx(0.6)
    assert agg[5] == pytest.approx(0.6)


def test_aggregate_recall_all_misses():
    rows = [recall_at_k(None, (1, 3, 5)) for _ in range(10)]
    out = aggregate_recall(rows)
    assert out == {1: 0.0, 3: 0.0, 5: 0.0}


def test_aggregate_recall_preserves_int_keys():
    # Caller iterates output by int — make sure we don't accidentally
    # promote keys to str.
    out = aggregate_recall([{1: 1, 3: 0}, {1: 0, 3: 1}])
    assert all(isinstance(k, int) for k in out)
