"""Tests for the coarse-filter bench module.

Mocks ``CoarseFilter`` so tests run instantly. The actual end-to-end
runs at 100K → 12M tokens are documented in the PRD; here we only
exercise the haystack builder + bench-runner plumbing so a refactor
won't quietly break the bench script.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from longctx.eval.bench_coarse_filter import (
    BenchResult,
    NEEDLE_TEMPLATE,
    _build_haystack,
    run_bench,
)
from longctx.rag.coarse_filter import Chunk


# ----------------------------------------------- haystack builder

def test_haystack_contains_needle():
    text, pos = _build_haystack(target_tokens=2_000, code="ABC", seed=0)
    expected = NEEDLE_TEMPLATE.format(code="ABC")
    assert expected in text
    assert text[pos:pos + len(expected)] == expected


def test_haystack_size_roughly_matches_target():
    """Char count should land within ±20% of the target × 4 chars/token
    proxy. Loose tolerance because we add whole sentences at a time."""
    target = 5_000
    text, _ = _build_haystack(target_tokens=target, code="ABC", seed=0)
    target_chars = target * 4
    assert 0.8 * target_chars <= len(text) <= 1.4 * target_chars


def test_haystack_seed_changes_filler_arrangement():
    a, _ = _build_haystack(2_000, code="X", seed=0)
    b, _ = _build_haystack(2_000, code="X", seed=1)
    assert a != b


# ------------------------------------------------- run_bench plumbing

class _StubCoarseFilter:
    """Stand-in for CoarseFilter that always surfaces the needle at #1."""

    device = "stub"

    def __init__(self, *args, **kwargs):
        pass

    def filter(self, chunks, query, top_k=1000):
        # Find any chunk whose text contains "NOVA" — the bench's needle
        # marker — and put it on top. Fill the rest in input order.
        first = next((c for c in chunks if "NOVA" in c.text), None)
        rest = [c for c in chunks if c is not first]
        out: list = []
        if first is not None:
            out.append((first, 9.99))
        out.extend((c, 1.0) for c in rest[: max(top_k - 1, 0)])
        return out


def test_run_bench_returns_result_struct():
    with patch("longctx.eval.bench_coarse_filter.CoarseFilter",
               _StubCoarseFilter):
        res = run_bench(target_tokens=2_000, top_k=10, quiet=True)
    assert isinstance(res, BenchResult)
    assert res.needle_in_topk is True
    assert res.needle_rank == 1
    assert res.n_chunks >= 1
    assert res.total_secs >= 0


def test_run_bench_misses_when_filter_drops_needle():
    """If the stub filter omits the needle entirely, bench reports MISS."""

    class _LosingFilter(_StubCoarseFilter):
        def filter(self, chunks, query, top_k=1000):
            # Drop anything matching NOVA on purpose
            keep = [c for c in chunks if "NOVA" not in c.text][: top_k]
            return [(c, 1.0) for c in keep]

    with patch("longctx.eval.bench_coarse_filter.CoarseFilter",
               _LosingFilter):
        res = run_bench(target_tokens=2_000, top_k=10, quiet=True)
    assert res.needle_in_topk is False
    assert res.needle_rank is None


def test_bench_result_report_string():
    res = BenchResult(
        target_tokens=10_000, n_chunks=5, n_kept=5,
        needle_position_chars=20_000, needle_in_topk=True,
        needle_rank=2, chunk_secs=0.01, filter_secs=0.5,
        total_secs=0.51, top_k=10, embedder_model="x", device="cpu",
    )
    s = res.report()
    assert "HIT" in s
    assert "rank #2" in s
    assert "10,000" in s
