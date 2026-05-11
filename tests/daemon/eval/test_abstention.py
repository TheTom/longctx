"""Tests for the abstention-rate eval harness.

Most tests use a fake searcher that returns canned ``no_relevant_results``
+ ``top1_dense_cosine`` per query so we can exercise the metric
computation deterministically without standing up an embedder.

The real-embedder smoke is gated behind ``LONGCTX_RUN_REAL_EMBEDDER=1``
so CI doesn't pull a 90 MB model on every run. To run it locally:

    LONGCTX_RUN_REAL_EMBEDDER=1 pytest tests/daemon/eval/test_abstention.py -v
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import pytest

from longctx_daemon.eval import abstention as abst
from longctx_daemon.eval.abstention import (
    AbstentionReport,
    SweepReport,
    _compute_metrics,
    _IN_CORPUS_QUERIES,
    _OFF_CORPUS_QUERIES,
    _run_query_suite,
    recommend_floor,
    render_abstention_report,
    render_sweep_table,
)


# ════════════════════════════════════════════════ fakes


@dataclass
class _FakeResult:
    """Mimics ``SearchResult`` for the searcher mock — only the two
    fields the abstention harness actually reads."""
    no_relevant_results: bool
    top1_dense_cosine: float


@dataclass
class _FakeSearcher:
    """Searcher stub keyed on (query, floor) → canned response.

    Lets each test plant the exact behavior it needs to exercise
    (e.g. "off-corpus query at floor 0.50 returns top1=0.42 and
    no_relevant_results=True"). If a query isn't in the dict, default
    to a moderate-confidence positive result so the in-corpus suite
    doesn't accidentally measure FNR=100% in tests that don't care.
    """
    responses: dict[tuple[str, float], _FakeResult] = field(default_factory=dict)
    default_cosine: float = 0.70  # well above any default floor

    def search(self, *, query: str, cwd: str, relevance_floor: float):
        key = (query, relevance_floor)
        if key in self.responses:
            return self.responses[key]
        # Default: high cosine, NOT suppressed
        cos = self.default_cosine
        no_rel = (relevance_floor > 0 and cos < relevance_floor)
        return _FakeResult(no_relevant_results=no_rel, top1_dense_cosine=cos)


# ════════════════════════════════════════════════ suite shape


class TestSuiteShape:
    """The query suites are part of the contract — keep them stable."""

    def test_thirty_off_corpus_queries(self):
        assert len(_OFF_CORPUS_QUERIES) == 30
        # Diversity sanity: no duplicates
        assert len(set(_OFF_CORPUS_QUERIES)) == 30

    def test_at_least_ten_in_corpus_queries(self):
        assert len(_IN_CORPUS_QUERIES) >= 10
        assert len(_IN_CORPUS_QUERIES) <= 20
        assert len(set(_IN_CORPUS_QUERIES)) == len(_IN_CORPUS_QUERIES)

    def test_off_corpus_spans_multiple_domains(self):
        """Spec: queries should span world history, geography, pop
        culture, math, sports, science, etc. We test for hallmarks."""
        joined = " ".join(_OFF_CORPUS_QUERIES).lower()
        # At least one query from each broad domain
        assert "wall" in joined or "war" in joined or "roman" in joined
        assert "capital" in joined or "river" in joined or "everest" in joined
        assert "mona lisa" in joined or "hamlet" in joined or "beatles" in joined
        assert "speed of light" in joined or "pi" in joined
        assert "world cup" in joined or "jordan" in joined or "baseball" in joined

    def test_in_corpus_targets_real_files(self):
        """Each in-corpus query must reference a token that actually
        appears in the synthetic corpus from messy_queries_real."""
        joined = " ".join(_IN_CORPUS_QUERIES).lower()
        # The synthetic corpus has: needle, auth, billing, inventory, charge_card
        assert "needle" in joined
        assert "auth" in joined
        assert "billing" in joined or "charge" in joined


# ════════════════════════════════════════════════ metrics core


class TestComputeMetrics:
    def test_all_off_corpus_correctly_suppressed(self):
        per_query = [
            {"label": "off_corpus", "no_relevant_results": True},
            {"label": "off_corpus", "no_relevant_results": True},
            {"label": "off_corpus", "no_relevant_results": True},
        ]
        supp, fnr = _compute_metrics(per_query)
        assert supp == 1.0
        assert fnr == 0.0

    def test_all_off_corpus_wrongly_passed(self):
        per_query = [
            {"label": "off_corpus", "no_relevant_results": False},
            {"label": "off_corpus", "no_relevant_results": False},
        ]
        supp, fnr = _compute_metrics(per_query)
        assert supp == 0.0
        assert fnr == 0.0

    def test_mixed_off_corpus(self):
        per_query = [
            {"label": "off_corpus", "no_relevant_results": True},
            {"label": "off_corpus", "no_relevant_results": True},
            {"label": "off_corpus", "no_relevant_results": False},
            {"label": "off_corpus", "no_relevant_results": False},
        ]
        supp, _ = _compute_metrics(per_query)
        assert supp == 0.5

    def test_in_corpus_fnr_zero_when_all_pass(self):
        per_query = [
            {"label": "in_corpus", "no_relevant_results": False},
            {"label": "in_corpus", "no_relevant_results": False},
        ]
        _, fnr = _compute_metrics(per_query)
        assert fnr == 0.0

    def test_in_corpus_fnr_full_when_all_suppressed(self):
        per_query = [
            {"label": "in_corpus", "no_relevant_results": True},
            {"label": "in_corpus", "no_relevant_results": True},
        ]
        _, fnr = _compute_metrics(per_query)
        assert fnr == 1.0

    def test_in_corpus_partial_fnr(self):
        per_query = [
            {"label": "in_corpus", "no_relevant_results": True},
            {"label": "in_corpus", "no_relevant_results": True},
            {"label": "in_corpus", "no_relevant_results": False},
            {"label": "in_corpus", "no_relevant_results": False},
        ]
        _, fnr = _compute_metrics(per_query)
        assert fnr == 0.5

    def test_combined_off_and_in(self):
        per_query = [
            {"label": "off_corpus", "no_relevant_results": True},
            {"label": "off_corpus", "no_relevant_results": False},
            {"label": "in_corpus", "no_relevant_results": False},
            {"label": "in_corpus", "no_relevant_results": True},
        ]
        supp, fnr = _compute_metrics(per_query)
        assert supp == 0.5
        assert fnr == 0.5

    def test_empty_off_corpus_does_not_divide_by_zero(self):
        per_query = [
            {"label": "in_corpus", "no_relevant_results": False},
        ]
        supp, _ = _compute_metrics(per_query)
        assert supp == 0.0

    def test_empty_in_corpus_does_not_divide_by_zero(self):
        per_query = [
            {"label": "off_corpus", "no_relevant_results": True},
        ]
        _, fnr = _compute_metrics(per_query)
        assert fnr == 0.0

    def test_empty_per_query_no_crash(self):
        supp, fnr = _compute_metrics([])
        assert supp == 0.0
        assert fnr == 0.0


# ════════════════════════════════════════════════ recommend_floor


class TestRecommendFloor:
    def test_pick_highest_under_max_fnr(self):
        points = [
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.02),
            (0.50, 0.68, 0.05),
            (0.60, 0.81, 0.20),
            (0.70, 0.89, 0.45),
        ]
        # max_fnr=0.05 → 0.50 wins (FNR exactly at boundary)
        assert recommend_floor(points, max_fnr=0.05) == 0.50

    def test_tighter_max_fnr_picks_lower_floor(self):
        points = [
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.02),
            (0.50, 0.68, 0.05),
            (0.60, 0.81, 0.20),
        ]
        # max_fnr=0.03 → only 0.30 and 0.40 qualify; pick 0.40
        assert recommend_floor(points, max_fnr=0.03) == 0.40

    def test_relaxed_max_fnr_picks_higher(self):
        points = [
            (0.30, 0.42, 0.00),
            (0.50, 0.68, 0.05),
            (0.70, 0.89, 0.45),
        ]
        # max_fnr=0.50 → all qualify; pick the highest
        assert recommend_floor(points, max_fnr=0.50) == 0.70

    def test_no_floor_qualifies_falls_back_to_lowest(self):
        points = [
            (0.30, 0.42, 0.10),
            (0.50, 0.68, 0.20),
            (0.70, 0.89, 0.50),
        ]
        # max_fnr=0.05 → nothing qualifies; lowest floor is the most
        # permissive (and least likely to suppress real questions)
        assert recommend_floor(points, max_fnr=0.05) == 0.30

    def test_single_point_input(self):
        assert recommend_floor([(0.50, 0.6, 0.04)], max_fnr=0.05) == 0.50

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            recommend_floor([], max_fnr=0.05)

    def test_boundary_inclusive(self):
        """FNR exactly at max_fnr should qualify."""
        points = [(0.30, 0.4, 0.04), (0.50, 0.7, 0.05)]
        assert recommend_floor(points, max_fnr=0.05) == 0.50

    def test_unsorted_input_still_picks_highest(self):
        """Recommendation logic mustn't depend on input ordering."""
        points = [
            (0.50, 0.68, 0.05),
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.02),
        ]
        assert recommend_floor(points, max_fnr=0.05) == 0.50


# ════════════════════════════════════════════════ _run_query_suite


class TestRunQuerySuite:
    def test_off_corpus_correctness_flag(self):
        searcher = _FakeSearcher(responses={
            ("q1", 0.5): _FakeResult(no_relevant_results=True, top1_dense_cosine=0.30),
            ("q2", 0.5): _FakeResult(no_relevant_results=False, top1_dense_cosine=0.60),
        })
        out = _run_query_suite(
            searcher, Path("/tmp/x"),
            ("q1", "q2"), 0.5, "off_corpus",
        )
        assert len(out) == 2
        # q1 was suppressed → "correct" for off-corpus
        assert out[0]["correct"] is True
        assert out[0]["no_relevant_results"] is True
        # q2 wasn't suppressed → "incorrect" for off-corpus
        assert out[1]["correct"] is False
        assert out[1]["no_relevant_results"] is False

    def test_in_corpus_correctness_flag(self):
        searcher = _FakeSearcher(responses={
            ("q1", 0.5): _FakeResult(no_relevant_results=False, top1_dense_cosine=0.70),
            ("q2", 0.5): _FakeResult(no_relevant_results=True, top1_dense_cosine=0.40),
        })
        out = _run_query_suite(
            searcher, Path("/tmp/x"),
            ("q1", "q2"), 0.5, "in_corpus",
        )
        # q1 surfaced → "correct" for in-corpus
        assert out[0]["correct"] is True
        # q2 suppressed → "incorrect" for in-corpus
        assert out[1]["correct"] is False

    def test_per_query_carries_label_and_cosine(self):
        searcher = _FakeSearcher(responses={
            ("q1", 0.5): _FakeResult(no_relevant_results=True, top1_dense_cosine=0.42),
        })
        out = _run_query_suite(
            searcher, Path("/tmp/x"), ("q1",), 0.5, "off_corpus",
        )
        assert out[0]["label"] == "off_corpus"
        assert out[0]["top1_dense_cosine"] == pytest.approx(0.42)
        assert out[0]["query"] == "q1"

    def test_floor_passes_through_to_searcher(self):
        """The harness must pass ``relevance_floor`` to ``search`` so
        the floor under test is the one being measured."""
        seen: list[float] = []

        class _Recording:
            def search(self, *, query, cwd, relevance_floor):
                seen.append(relevance_floor)
                return _FakeResult(False, 0.7)

        _run_query_suite(_Recording(), Path("/tmp/x"),
                         ("a", "b"), 0.42, "off_corpus")
        assert seen == [0.42, 0.42]


# ════════════════════════════════════════════════ run_abstention (mocked)


class TestRunAbstentionMocked:
    """Integration of _run_query_suite + _compute_metrics via the
    public ``run_abstention`` entry point, with the embedder + indexer
    mocked out so it runs in milliseconds."""

    def test_perfect_floor(self, tmp_path):
        """A floor that suppresses every off-corpus and lets every
        in-corpus through gives suppression=1.0, FNR=0.0."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "x.py").write_text("# placeholder\n")

        # Map all off-corpus → suppressed; in-corpus → not suppressed
        responses = {}
        for q in _OFF_CORPUS_QUERIES:
            responses[(q, 0.5)] = _FakeResult(True, 0.20)
        for q in _IN_CORPUS_QUERIES:
            responses[(q, 0.5)] = _FakeResult(False, 0.75)

        fake_searcher = _FakeSearcher(responses=responses)

        def _fake_make_searcher(*a, **kw):
            return (object(), object(), fake_searcher, "corpus", lambda: None)

        with patch.object(abst, "_make_searcher", _fake_make_searcher):
            report = abst.run_abstention(corpus, floor=0.5)

        assert report.suppression_rate == 1.0
        assert report.false_negative_rate == 0.0
        assert report.n_off_corpus == 30
        assert report.n_in_corpus == 12
        assert len(report.per_query) == 42

    def test_pessimal_floor(self, tmp_path):
        """A floor that suppresses everything → suppression=1.0 but
        FNR=1.0 (the trade-off the floor is supposed to manage)."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()

        responses = {}
        for q in _OFF_CORPUS_QUERIES:
            responses[(q, 0.99)] = _FakeResult(True, 0.20)
        for q in _IN_CORPUS_QUERIES:
            responses[(q, 0.99)] = _FakeResult(True, 0.75)

        fake_searcher = _FakeSearcher(responses=responses)

        def _fake_make_searcher(*a, **kw):
            return (object(), object(), fake_searcher, "corpus", lambda: None)

        with patch.object(abst, "_make_searcher", _fake_make_searcher):
            report = abst.run_abstention(corpus, floor=0.99)
        assert report.suppression_rate == 1.0
        assert report.false_negative_rate == 1.0

    def test_invalid_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            abst.run_abstention(tmp_path / "does_not_exist", floor=0.5)


# ════════════════════════════════════════════════ sweep_floor (mocked)


class TestSweepFloorMocked:
    def test_sweep_emits_one_point_per_floor(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()

        # Floor monotonicity in the model: as floor rises, both metrics
        # rise (more suppression, more FNR). Stage canned responses to
        # match that shape.
        floors = (0.30, 0.50, 0.70)

        def _resp_for(q: str, floor: float) -> _FakeResult:
            # Off-corpus: cos always 0.40; in-corpus: cos always 0.55
            cos = 0.40 if q in _OFF_CORPUS_QUERIES else 0.55
            return _FakeResult(no_relevant_results=cos < floor,
                               top1_dense_cosine=cos)

        class _PiecewiseSearcher:
            def search(self, *, query, cwd, relevance_floor):
                return _resp_for(query, relevance_floor)

        def _fake_make_searcher(*a, **kw):
            return (object(), object(), _PiecewiseSearcher(), "corpus",
                    lambda: None)

        with patch.object(abst, "_make_searcher", _fake_make_searcher):
            report = abst.sweep_floor(corpus, floors=floors)

        assert isinstance(report, SweepReport)
        assert len(report.points) == 3
        floors_seen = [p[0] for p in report.points]
        assert floors_seen == [0.30, 0.50, 0.70]

        # At floor=0.30 nothing suppresses (both cos > 0.30)
        assert report.points[0][1] == 0.0
        assert report.points[0][2] == 0.0
        # At floor=0.50 off-corpus (0.40) suppressed; in-corpus (0.55) survives
        assert report.points[1][1] == 1.0
        assert report.points[1][2] == 0.0
        # At floor=0.70 both suppressed
        assert report.points[2][1] == 1.0
        assert report.points[2][2] == 1.0

    def test_sweep_recommendation_under_default_max_fnr(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()

        def _resp_for(q, floor):
            cos = 0.40 if q in _OFF_CORPUS_QUERIES else 0.55
            return _FakeResult(cos < floor, cos)

        class _Searcher:
            def search(self, *, query, cwd, relevance_floor):
                return _resp_for(query, relevance_floor)

        def _fake_make_searcher(*a, **kw):
            return (None, None, _Searcher(), "corpus", lambda: None)

        with patch.object(abst, "_make_searcher", _fake_make_searcher):
            report = abst.sweep_floor(corpus, floors=(0.30, 0.50, 0.70))
        # 0.50 has FNR=0; 0.70 has FNR=1.0. Highest qualifying = 0.50.
        assert report.recommended_floor == 0.50

    def test_sweep_invalid_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            abst.sweep_floor(tmp_path / "nope")

    def test_sweep_empty_floors_raises(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        with pytest.raises(ValueError):
            abst.sweep_floor(corpus, floors=())


# ════════════════════════════════════════════════ rendering


class TestRender:
    def test_render_abstention_includes_metrics(self):
        report = AbstentionReport(
            floor=0.5, n_off_corpus=30, n_in_corpus=12,
            suppression_rate=0.68, false_negative_rate=0.05,
        )
        s = render_abstention_report(report)
        assert "0.50" in s
        assert "30" in s
        assert "12" in s
        assert "68" in s  # 68%
        assert "5" in s  # 5%

    def test_render_sweep_marks_recommended(self):
        report = SweepReport(
            points=[
                (0.30, 0.42, 0.00),
                (0.50, 0.68, 0.05),
                (0.70, 0.89, 0.45),
            ],
            recommended_floor=0.50,
            max_fnr=0.05,
        )
        s = render_sweep_table(report)
        assert "recommended" in s
        # Specific marker on the recommended row
        rec_line = [ln for ln in s.splitlines() if "0.50" in ln and "←" in ln]
        assert len(rec_line) == 1

    def test_render_sweep_omits_marker_when_disabled(self):
        report = SweepReport(
            points=[(0.30, 0.42, 0.00), (0.50, 0.68, 0.05)],
            recommended_floor=0.50,
            max_fnr=0.05,
        )
        s = render_sweep_table(report, show_recommended=False)
        assert "←" not in s


# ════════════════════════════════════════════════ real-embedder smoke


@pytest.mark.skipif(
    os.environ.get("LONGCTX_RUN_REAL_EMBEDDER") != "1",
    reason="real-embedder smoke; gated by LONGCTX_RUN_REAL_EMBEDDER=1",
)
def test_smoke_real_embedder_synthetic_corpus(tmp_path):
    """Real-embedder smoke against the synthetic corpus.

    Confirms the full pipeline runs and produces sane metrics. The
    actual numbers depend on the embedder and BM25 tokenizer; we
    assert loose bounds so a small drift doesn't break the test."""
    from longctx_daemon.eval.messy_queries_real import _build_corpus

    corpus = tmp_path / "myapp"
    _build_corpus(corpus)

    report = abst.run_abstention(corpus, floor=0.50)
    # Loose bounds
    assert 0.0 <= report.suppression_rate <= 1.0
    assert 0.0 <= report.false_negative_rate <= 1.0
    # Spec: at floor=0.50, suppression should be solid majority.
    assert report.suppression_rate >= 0.5
    # And the in-corpus suite shouldn't be largely rejected
    assert report.false_negative_rate <= 0.3
