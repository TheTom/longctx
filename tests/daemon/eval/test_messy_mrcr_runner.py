"""Runner tests for Messy MRCR — mocked dataset + mocked scorer.

These tests bypass the heavy storage stack + embedder by patching
:func:`longctx_daemon.eval.messy_mrcr._score_one_sample`. The runner's
job is the matrix shape (transform × bin → recall), the loop, the
JSON shape, and the markdown rendering — none of which need a real
embedder. The transform algorithms have their own dedicated test
file (``test_messy_mrcr_transforms.py``).
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from longctx_daemon.eval import messy_mrcr as mm
from longctx_daemon.eval.messy_mrcr import (
    BIN_RANGES_CHAR,
    DEFAULT_BINS,
    MessyMRCRReport,
    MessyMRCRSample,
    TRANSFORMS,
    render_matrix,
    render_results_md,
    report_to_dict,
    run_messy_mrcr,
)


# ─────────────────────────────────────── fake sample / scorer

def _fake_sample(n_chars: int = 20_000) -> dict:
    """Build one fake MRCR-shaped sample. Just enough to drive the
    runner; the real ``_score_one_sample`` would index conversation
    messages but our mock skips that."""
    return {
        "messages": [
            {"role": "user", "content": "intro question"},
            {"role": "assistant", "content": "ABC123 some intro response"},
            {"role": "user", "content": "Please write me the 5th poem about a tabby cat verbatim, prepended with QXR-"},
        ],
        "answer": "QXR-poem about a tabby cat",
        "random_string_to_prepend": "QXR-",
        "n_needles": 8,
        "n_chars": n_chars,
        "final_user_query": "Please write me the 5th poem about a tabby cat verbatim, prepended with QXR-",
    }


def _fake_loader(n_samples_per_cell: int = 3):
    """Return a deterministic loader closure suitable for tests."""

    def _loader(needles: int, bin_name: str, n_samples: int, seed: int) -> list[dict]:
        return [_fake_sample(n_chars=BIN_RANGES_CHAR[bin_name][0] + 100 * i)
                for i in range(n_samples_per_cell)]

    return _loader


def _fake_embedder_factory(model_id: str, device: str):
    """Factory that returns a stub embedder + dim + sha. Never called
    by the runner because the scoring is mocked too."""
    return ("EMBEDDER_STUB", 4, "fake-sha")


def _make_fake_scorer(*, hit_recipe: dict[tuple[str, str], list[bool]]):
    """Returns a fake _score_one_sample whose hit value is dictated by
    ``hit_recipe[(transform, bin)][sample_idx]``."""

    def _fake(*, sample, transform_name, transformed_query, bin_name,
              sample_idx, embedder, embed_dim, embedder_sha, embedder_id,
              chunk_store_factory, embed_store_factory, chunker,
              indexer_factory, searcher_factory, top_k):
        recipe = hit_recipe.get((transform_name, bin_name), [])
        hit = recipe[sample_idx] if sample_idx < len(recipe) else False
        return MessyMRCRSample(
            transform=transform_name,
            bin=bin_name,
            sample_idx=sample_idx,
            n_chars=sample["n_chars"],
            n_needles=sample["n_needles"],
            clean_query=sample["final_user_query"][:80],
            transformed_query=transformed_query[:80],
            gold_chunk_idx=2,
            top_k_chunk_indices=(2, 0, 1),
            top_k_scores=(0.8, 0.7, 0.6),
            hit=hit,
            no_relevant_results=False,
            top1_dense_cosine=0.7 if hit else 0.4,
            confidence_gap=0.1,
            query_type="natural_language",
        )

    return _fake


# ═══════════════════════════════════════ matrix shape

def test_runner_returns_report_with_correct_shape() -> None:
    """Three transforms × two bins × two samples = 6 cells, 12 rows."""
    transforms = ("clean", "typo", "off-corpus")
    bins = ("8K", "32K")
    recipe = {
        (t, b): [True, False] for t in transforms for b in bins
    }
    fake_scorer = _make_fake_scorer(hit_recipe=recipe)

    with patch.object(mm, "_score_one_sample", fake_scorer):
        report = run_messy_mrcr(
            needles=8, samples_per_cell=2,
            bins=bins, transforms=transforms,
            sample_loader=_fake_loader(2),
            embedder_factory=_fake_embedder_factory,
            seed=0,
        )

    assert isinstance(report, MessyMRCRReport)
    assert report.transforms == transforms
    assert report.bins == bins
    assert report.n_samples_per_cell == 2
    assert report.needles == 8
    # 6 cells.
    assert len(report.matrix) == 6
    # 2 hits + 0 hits per cell × 6 = recall 0.5 everywhere.
    for v in report.matrix.values():
        assert v == 0.5
    # 12 per-sample rows.
    assert len(report.per_sample) == 12


def test_runner_uses_seed_offset_per_sample() -> None:
    """Per-sample seed is ``seed + sample_idx`` so re-runs with the
    same top-level seed produce the same transformed queries."""
    captured: list[str] = []

    def _capture_scorer(*, sample, transform_name, transformed_query, **_kw):
        captured.append(transformed_query)
        return MessyMRCRSample(
            transform=transform_name, bin=_kw["bin_name"],
            sample_idx=_kw["sample_idx"], n_chars=sample["n_chars"],
            n_needles=8, clean_query=sample["final_user_query"][:80],
            transformed_query=transformed_query[:80],
            gold_chunk_idx=0, top_k_chunk_indices=(),
            top_k_scores=(), hit=False, no_relevant_results=False,
            top1_dense_cosine=0.0, confidence_gap=0.0,
            query_type="natural_language",
        )

    with patch.object(mm, "_score_one_sample", _capture_scorer):
        run_messy_mrcr(
            needles=8, samples_per_cell=3,
            bins=("8K",), transforms=("off-corpus",),
            sample_loader=_fake_loader(3),
            embedder_factory=_fake_embedder_factory,
            seed=10,
        )

    # off-corpus picks ABSTENTION_POOL[(seed + i) % 30] — three distinct
    # seeds (10, 11, 12) → three (likely-distinct) pool entries.
    assert len(captured) == 3
    assert len(set(captured)) == 3


def test_runner_recall_1_when_all_hit() -> None:
    """All-hit recipe → recall = 1.0."""
    recipe = {("clean", "8K"): [True, True, True]}
    fake_scorer = _make_fake_scorer(hit_recipe=recipe)

    with patch.object(mm, "_score_one_sample", fake_scorer):
        report = run_messy_mrcr(
            needles=8, samples_per_cell=3,
            bins=("8K",), transforms=("clean",),
            sample_loader=_fake_loader(3),
            embedder_factory=_fake_embedder_factory, seed=0,
        )
    assert report.matrix[("clean", "8K")] == 1.0


def test_runner_recall_0_when_no_hits() -> None:
    """All-miss recipe → recall = 0.0. Mirrors the off-corpus expectation."""
    recipe = {("off-corpus", "8K"): [False, False, False]}
    fake_scorer = _make_fake_scorer(hit_recipe=recipe)

    with patch.object(mm, "_score_one_sample", fake_scorer):
        report = run_messy_mrcr(
            needles=8, samples_per_cell=3,
            bins=("8K",), transforms=("off-corpus",),
            sample_loader=_fake_loader(3),
            embedder_factory=_fake_embedder_factory, seed=0,
        )
    assert report.matrix[("off-corpus", "8K")] == 0.0


def test_runner_handles_empty_bin() -> None:
    """If the loader returns no samples for a bin, recall is 0.0
    (no division-by-zero crash)."""

    def _empty_loader(needles, bin_name, n_samples, seed):
        return []

    fake_scorer = _make_fake_scorer(hit_recipe={})

    with patch.object(mm, "_score_one_sample", fake_scorer):
        report = run_messy_mrcr(
            needles=8, samples_per_cell=3,
            bins=("8K",), transforms=("clean",),
            sample_loader=_empty_loader,
            embedder_factory=_fake_embedder_factory, seed=0,
        )

    assert report.matrix[("clean", "8K")] == 0.0
    assert report.per_sample == []


def test_runner_progress_callback_is_invoked() -> None:
    """``progress`` is called at least once per bin load + cell."""
    calls: list[str] = []
    fake_scorer = _make_fake_scorer(hit_recipe={})

    with patch.object(mm, "_score_one_sample", fake_scorer):
        run_messy_mrcr(
            needles=8, samples_per_cell=1,
            bins=("8K", "32K"), transforms=("clean", "typo"),
            sample_loader=_fake_loader(1),
            embedder_factory=_fake_embedder_factory,
            progress=calls.append,
        )

    # 2 bins × (load + done) at minimum.
    assert len(calls) >= 4


# ═══════════════════════════════════════ rendering / serialization

def _trivial_report(transforms=("clean", "off-corpus"),
                    bins=("8K", "32K"),
                    matrix_recipe=None) -> MessyMRCRReport:
    """Build a minimal report for rendering tests."""
    if matrix_recipe is None:
        matrix_recipe = {
            ("clean", "8K"): 0.95, ("clean", "32K"): 0.81,
            ("off-corpus", "8K"): 0.04, ("off-corpus", "32K"): 0.05,
        }
    return MessyMRCRReport(
        transforms=transforms, bins=bins, n_samples_per_cell=20,
        needles=8, embedder_id="BAAI/bge-small-en-v1.5", seed=0,
        matrix=matrix_recipe, per_sample=[],
    )


def test_render_matrix_includes_all_cells() -> None:
    """Every transform × bin combination must appear in the rendered table."""
    out = render_matrix(_trivial_report())
    assert "clean" in out
    assert "off-corpus" in out
    assert "8K" in out
    assert "32K" in out
    assert "0.950" in out
    assert "0.040" in out


def test_render_matrix_has_separator_row() -> None:
    """Markdown table needs a ``|---|---|`` separator after the header."""
    out = render_matrix(_trivial_report())
    lines = out.strip().split("\n")
    assert len(lines) >= 4   # header, sep, two body rows
    assert "---" in lines[1]


def test_render_results_md_has_sections() -> None:
    """RESULTS.md should have all required methodology sections.

    Headers are lowercase per Tom's voice convention.
    """
    out = render_results_md(_trivial_report())
    assert "## what this measures" in out
    assert "## why it's novel" in out
    assert "## method" in out
    assert "## algorithms" in out
    assert "## results" in out
    assert "## reading guide" in out
    assert "## caveats" in out
    assert "## reproduce" in out


def test_render_results_md_has_matrix_table() -> None:
    """The matrix table must be embedded in the writeup."""
    out = render_results_md(_trivial_report())
    assert "0.950" in out
    assert "off-corpus" in out


def test_report_to_dict_round_trips_through_json() -> None:
    """Report serializes to JSON cleanly — needed for the per-sample
    raw output."""
    report = _trivial_report()
    payload = report_to_dict(report)
    blob = json.dumps(payload)
    parsed = json.loads(blob)
    assert parsed["transforms"] == ["clean", "off-corpus"]
    assert parsed["bins"] == ["8K", "32K"]
    assert parsed["needles"] == 8
    assert any(
        c["transform"] == "clean" and c["bin"] == "8K" and c["recall"] == 0.95
        for c in parsed["matrix"]
    )


def test_report_per_sample_is_list_of_dicts_in_json() -> None:
    """Per-sample rows must be JSON-friendly."""
    sample = MessyMRCRSample(
        transform="clean", bin="8K", sample_idx=0,
        n_chars=20000, n_needles=8,
        clean_query="q", transformed_query="q",
        gold_chunk_idx=2, top_k_chunk_indices=(2, 0, 1),
        top_k_scores=(0.8, 0.7, 0.6), hit=True,
        no_relevant_results=False, top1_dense_cosine=0.7,
        confidence_gap=0.1, query_type="natural_language",
    )
    report = MessyMRCRReport(
        transforms=("clean",), bins=("8K",), n_samples_per_cell=1,
        needles=8, embedder_id="x", seed=0,
        matrix={("clean", "8K"): 1.0}, per_sample=[sample],
    )
    blob = json.dumps(report_to_dict(report))
    parsed = json.loads(blob)
    assert len(parsed["per_sample"]) == 1
    assert parsed["per_sample"][0]["transform"] == "clean"
    assert parsed["per_sample"][0]["hit"] is True


# ═══════════════════════════════════════ regression — defaults

def test_default_bins_excludes_1m() -> None:
    """1M is opt-in; never in DEFAULT_BINS."""
    assert "1M" not in DEFAULT_BINS


def test_bin_ranges_are_monotonic() -> None:
    """Each bin's lo < hi, and bin lo's are strictly increasing in the
    canonical order 8K, 32K, 64K, ... 1M."""
    canonical_order = ["8K", "32K", "64K", "128K", "256K", "1M"]
    last_lo = -1
    for b in canonical_order:
        lo, hi = BIN_RANGES_CHAR[b]
        assert lo < hi, f"{b}: lo {lo} >= hi {hi}"
        assert lo > last_lo, f"{b}: lo {lo} not strictly greater than prev {last_lo}"
        last_lo = lo
