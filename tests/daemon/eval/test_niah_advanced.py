"""Unit tests for ``longctx_daemon.eval.niah_advanced``.

These tests cover the metric/aggregation/rendering surface area with
mocked search results so the suite stays fast (no real embedder, no
SQLite). End-to-end pipeline correctness is exercised by the smoke
run in the bench's CLI; that's not appropriate for the test suite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from unittest.mock import patch

import pytest

from longctx_daemon.eval import niah_advanced as ni


# ─────────────────────────────────────────────── lightweight fakes

@dataclass
class _FakeChunk:
    """Tiny stand-in for ``SearchChunk`` — only ``.text`` is read by the
    metric helpers so that's all we need."""
    text: str


@dataclass
class _FakeSearchResult:
    chunks: tuple[_FakeChunk, ...]


def _result_from_texts(texts: Sequence[str]) -> _FakeSearchResult:
    return _FakeSearchResult(chunks=tuple(_FakeChunk(t) for t in texts))


# ─────────────────────────────────────────────── _find_needle_rank

class TestFindNeedleRank:
    def test_first_position(self):
        r = _result_from_texts(["alpha NEEDLE_X end", "beta", "gamma"])
        assert ni._find_needle_rank(r.chunks, "NEEDLE_X") == 1

    def test_middle_position(self):
        r = _result_from_texts(["alpha", "beta NEEDLE_X", "gamma"])
        assert ni._find_needle_rank(r.chunks, "NEEDLE_X") == 2

    def test_missing(self):
        r = _result_from_texts(["alpha", "beta", "gamma"])
        assert ni._find_needle_rank(r.chunks, "NEEDLE_X") is None

    def test_first_match_wins_on_duplicates(self):
        """If the marker appears in two chunks, the first rank is reported."""
        r = _result_from_texts(["foo NEEDLE_X", "bar NEEDLE_X"])
        assert ni._find_needle_rank(r.chunks, "NEEDLE_X") == 1

    def test_empty_chunks(self):
        r = _result_from_texts([])
        assert ni._find_needle_rank(r.chunks, "NEEDLE_X") is None


# ─────────────────────────────────────────────── haystack builders

class TestSplitIntoFiles:
    def test_single_file_under_cap(self, tmp_path: Path):
        text = "hello world " * 100
        files = ni._split_into_files(text, tmp_path)
        assert len(files) == 1
        # round-trip through the file
        assert files[0][0].read_text() == text
        assert files[0][1] == 0
        assert files[0][2] == len(text)

    def test_multiple_files_over_cap(self, tmp_path: Path, monkeypatch):
        # Lower the cap so the test stays fast.
        monkeypatch.setattr(ni, "_FILE_BYTES_CAP", 1024)
        text = "a" * 5000
        files = ni._split_into_files(text, tmp_path)
        assert len(files) >= 4   # 5000 / 1024 ≈ 5
        # Ranges should be contiguous, non-overlapping.
        prev_end = 0
        total = 0
        for path, start, end in files:
            assert start == prev_end
            assert end > start
            total += end - start
            prev_end = end
        assert total == len(text)

    def test_empty_text_writes_one_file(self, tmp_path: Path):
        files = ni._split_into_files("", tmp_path)
        assert len(files) == 1
        # Non-empty placeholder so the indexer's zero-byte filter doesn't
        # drop the project entirely.
        assert files[0][0].read_text() != ""


class TestInsertNeedleAtOffset:
    def test_insert_in_first_file(self, tmp_path: Path):
        text = "hello " * 200
        files = ni._split_into_files(text, tmp_path)
        path, rel = ni._insert_needle_at_offset(files, 50, "MARKER_NEEDLE")
        # The file should now contain the marker.
        assert "MARKER_NEEDLE" in path.read_text()

    def test_insert_past_end_appends_to_last(
        self, tmp_path: Path, monkeypatch,
    ):
        monkeypatch.setattr(ni, "_FILE_BYTES_CAP", 200)
        text = "x" * 1000
        files = ni._split_into_files(text, tmp_path)
        # Past-end offset
        path, rel = ni._insert_needle_at_offset(files, 10**9, "TAIL_NEEDLE")
        assert path == files[-1][0]
        assert "TAIL_NEEDLE" in path.read_text()


class TestBuildHaystack:
    def test_single_needle_lands_in_corpus(self, tmp_path: Path):
        placements = ni._build_haystack(
            corpus_tokens=2_000, root=tmp_path, seed=42,
            needles=[(0.5, "ONLY_NEEDLE_TEXT")],
        )
        assert len(placements) == 1
        # The needle text should be retrievable from one of the files.
        all_text = "".join(
            p.read_text() for p in sorted(tmp_path.glob("*.txt"))
        )
        assert "ONLY_NEEDLE_TEXT" in all_text

    def test_multi_needles_all_present(self, tmp_path: Path):
        placements = ni._build_haystack(
            corpus_tokens=4_000, root=tmp_path, seed=7,
            needles=[
                (0.2, "NEEDLE_AAA"),
                (0.5, "NEEDLE_BBB"),
                (0.8, "NEEDLE_CCC"),
            ],
        )
        assert len(placements) == 3
        all_text = "".join(
            p.read_text() for p in sorted(tmp_path.glob("*.txt"))
        )
        for marker in ("NEEDLE_AAA", "NEEDLE_BBB", "NEEDLE_CCC"):
            assert marker in all_text

    def test_placements_match_input_order(self, tmp_path: Path):
        """Even though we plant in depth-sorted order, the returned list
        must follow the caller's input order (so the caller's indices
        line up with codes/queries)."""
        placements = ni._build_haystack(
            corpus_tokens=3_000, root=tmp_path, seed=1,
            # Deliberately supply needles out of depth order.
            needles=[
                (0.9, "ZZZ_LATE"),
                (0.1, "AAA_EARLY"),
                (0.5, "MMM_MID"),
            ],
        )
        # placements[0] should still correspond to "ZZZ_LATE".
        late_path = placements[0][0]
        assert "ZZZ_LATE" in late_path.read_text()


# ─────────────────────────────────────────────── multi-needle aggregation

class TestMultiNeedleAggregation:
    def _make_sample(self, hits, ranks=None, seed=1, wall=0.1):
        if ranks is None:
            ranks = tuple(i + 1 if h else None for i, h in enumerate(hits))
        return ni.MultiNeedleSample(
            seed=seed, per_fact_hits=tuple(hits),
            per_fact_ranks=tuple(ranks),
            all_of_n_hit=all(hits), wall_secs=wall,
        )

    def test_all_hits(self):
        report = ni.MultiNeedleReport(
            n_facts=3, corpus_tokens=100_000, n_samples=2, top_k=50,
            seed=0, embedder_id="fake", device="cpu",
        )
        report.samples.append(self._make_sample([True, True, True]))
        report.samples.append(self._make_sample([True, True, True]))
        # mimic the aggregation logic in run_multi_needle by recomputing
        # via the report fields after we've populated samples.
        total_facts = report.n_facts * len(report.samples)
        total_hits = sum(
            sum(1 for h in s.per_fact_hits if h) for s in report.samples
        )
        assert total_hits / total_facts == 1.0
        assert (
            sum(1 for s in report.samples if s.all_of_n_hit)
            / len(report.samples)
        ) == 1.0

    def test_partial_hits(self):
        s1 = self._make_sample([True, True, False])  # all-of-N FALSE
        s2 = self._make_sample([True, True, True])   # all-of-N TRUE
        report = ni.MultiNeedleReport(
            n_facts=3, corpus_tokens=100_000, n_samples=2, top_k=50,
            seed=0, embedder_id="fake", device="cpu",
            samples=[s1, s2],
        )
        # 5/6 facts hit, 1/2 samples all-of-N
        total_facts = 6
        total_hits = sum(
            sum(1 for h in s.per_fact_hits if h) for s in report.samples
        )
        assert total_hits == 5
        assert total_hits / total_facts == pytest.approx(5 / 6)
        all_of_n = sum(1 for s in report.samples if s.all_of_n_hit)
        assert all_of_n == 1
        assert all_of_n / len(report.samples) == 0.5

    def test_n_facts_validation(self):
        with pytest.raises(ValueError):
            ni.run_multi_needle(
                n_facts=0, corpus_tokens=100, n_samples=1,
                seed=0, embedder_id="fake", device="cpu",
            )
        with pytest.raises(ValueError):
            ni.run_multi_needle(
                n_facts=999, corpus_tokens=100, n_samples=1,
                seed=0, embedder_id="fake", device="cpu",
            )


# ─────────────────────────────────────────────── near-needle correctness

class TestNearNeedleCorrectness:
    def test_real_above_distractor_counts_as_win(self):
        s = ni.NearNeedleSample(
            seed=1, real_rank=2, distractor_rank=5,
            real_beats_distractor=True, wall_secs=0.0,
        )
        # The dataclass holds the bool — verify our scoring semantics
        # match the field.
        assert (
            s.real_rank is not None
            and (s.distractor_rank is None or s.real_rank < s.distractor_rank)
        )

    def test_distractor_above_real_loses(self):
        # real_rank > distractor_rank → loss
        real_rank, distractor_rank = 7, 3
        beats = (
            real_rank is not None
            and (distractor_rank is None or real_rank < distractor_rank)
        )
        assert beats is False

    def test_real_present_distractor_missing_wins(self):
        real_rank, distractor_rank = 2, None
        beats = (
            real_rank is not None
            and (distractor_rank is None or real_rank < distractor_rank)
        )
        assert beats is True

    def test_real_missing_loses_even_if_distractor_missing(self):
        real_rank, distractor_rank = None, None
        beats = (
            real_rank is not None
            and (distractor_rank is None or real_rank < distractor_rank)
        )
        assert beats is False

    def test_aggregate_rates(self):
        report = ni.NearNeedleReport(
            corpus_tokens=100_000, n_samples=4, top_k=50, seed=0,
            embedder_id="fake", device="cpu",
            samples=[
                ni.NearNeedleSample(
                    seed=1, real_rank=1, distractor_rank=2,
                    real_beats_distractor=True, wall_secs=0.1,
                ),
                ni.NearNeedleSample(
                    seed=2, real_rank=4, distractor_rank=3,
                    real_beats_distractor=False, wall_secs=0.1,
                ),
                ni.NearNeedleSample(
                    seed=3, real_rank=None, distractor_rank=None,
                    real_beats_distractor=False, wall_secs=0.1,
                ),
                ni.NearNeedleSample(
                    seed=4, real_rank=2, distractor_rank=None,
                    real_beats_distractor=True, wall_secs=0.1,
                ),
            ],
        )
        beats = (
            sum(1 for s in report.samples if s.real_beats_distractor)
            / len(report.samples)
        )
        recall = (
            sum(1 for s in report.samples if s.real_rank is not None)
            / len(report.samples)
        )
        assert beats == 0.5
        assert recall == 0.75


# ─────────────────────────────────────────────── depth sweep shape

class TestDepthSweepShape:
    def test_cell_grid_shape(self):
        """Cell list has depth × size entries in row-major order."""
        report = ni.DepthSweepReport(
            depths=(0.1, 0.5, 0.9),
            corpus_tokens_list=(1000, 5000),
            n_samples_per_cell=1, top_k=50, seed=0,
            embedder_id="fake", device="cpu",
            cells=[
                ni.DepthCellResult(0.1, 1000, 1, 1, 1.0, 1.0, 0.1),
                ni.DepthCellResult(0.1, 5000, 1, 1, 1.0, 1.0, 0.1),
                ni.DepthCellResult(0.5, 1000, 1, 1, 1.0, 1.0, 0.1),
                ni.DepthCellResult(0.5, 5000, 1, 0, 0.0, None, 0.1),
                ni.DepthCellResult(0.9, 1000, 1, 1, 1.0, 1.0, 0.1),
                ni.DepthCellResult(0.9, 5000, 1, 1, 1.0, 1.0, 0.1),
            ],
        )
        # 3 depths × 2 sizes = 6 cells.
        assert len(report.cells) == 6
        # Render produces a markdown matrix without crashing.
        out = ni.render_depth_sweep(report)
        assert "depth" in out
        assert "1,000" in out
        assert "5,000" in out
        # A miss cell shows 0.00.
        assert "0.00" in out


# ─────────────────────────────────────────────── rendering

class TestRendering:
    def test_render_multi_needle(self):
        report = ni.MultiNeedleReport(
            n_facts=3, corpus_tokens=100_000, n_samples=5, top_k=50,
            seed=0, embedder_id="fake", device="cpu",
            per_fact_recall=0.9, all_of_n_rate=0.7, mean_wall_secs=12.3,
        )
        out = ni.render_multi_needle(report)
        assert "0.90" in out
        assert "0.70" in out
        assert "100,000" in out

    def test_render_near_needle(self):
        report = ni.NearNeedleReport(
            corpus_tokens=100_000, n_samples=5, top_k=50, seed=0,
            embedder_id="fake", device="cpu",
            real_recall=0.95, real_beats_distractor_rate=0.80,
            mean_wall_secs=8.4,
        )
        out = ni.render_near_needle(report)
        assert "0.95" in out
        assert "0.80" in out

    def test_render_depth_sweep_missing_cell_shows_dash(self):
        # Cell list missing one entry — renderer should not crash.
        report = ni.DepthSweepReport(
            depths=(0.1, 0.5),
            corpus_tokens_list=(1000, 5000),
            n_samples_per_cell=1, top_k=50, seed=0,
            embedder_id="fake", device="cpu",
            cells=[
                ni.DepthCellResult(0.1, 1000, 1, 1, 1.0, 1.0, 0.1),
                # 0.1 × 5000 missing intentionally
                ni.DepthCellResult(0.5, 1000, 1, 1, 1.0, 1.0, 0.1),
                ni.DepthCellResult(0.5, 5000, 1, 1, 1.0, 1.0, 0.1),
            ],
        )
        out = ni.render_depth_sweep(report)
        assert "—" in out


# ─────────────────────────────────────────────── pipeline integration (mocked)

class _StubSearcher:
    """Tiny searcher stand-in. ``.search()`` returns a fixed result for
    each query in registration order; cycles when exhausted."""

    def __init__(self, results: list[_FakeSearchResult]) -> None:
        self.calls: list[str] = []
        self._results = results
        self._i = 0

    def search(self, query: str, **_: object) -> _FakeSearchResult:
        self.calls.append(query)
        r = self._results[self._i % len(self._results)]
        self._i += 1
        return r


def _stub_pipeline(results: list[_FakeSearchResult]):
    """Build a ``_build_pipeline`` stand-in returning the stub searcher.

    The real ``_build_pipeline`` returns a 6-tuple; we mimic the shape
    so the consumers don't need to branch on the test case.
    """
    searcher = _StubSearcher(results)

    class _NoopStore:
        def close(self) -> None:
            pass

    return None, searcher, _NoopStore(), _NoopStore(), None, "cpu"


class TestRunOneMultiNeedleMocked:
    def test_three_hits_marks_all_of_n(self, monkeypatch):
        # All three queries return chunks that contain their own code.
        # Codes in run_one_multi are formatted "{seed:03d}{i:03d}":
        # for sample_seed=42 → 042000, 042001, 042002
        results = [
            _result_from_texts(
                ["unrelated", f"hit chunk 042{i:03d}", "extra"],
            )
            for i in range(3)
        ]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        # Avoid hitting the filesystem in _build_haystack.
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)] * 3,
        )
        sample = ni._run_one_multi_needle(
            n_facts=3, corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert sample.all_of_n_hit is True
        assert all(sample.per_fact_hits)
        assert sample.per_fact_ranks == (2, 2, 2)

    def test_partial_hits_falsifies_all_of_n(self, monkeypatch):
        # Second query: the second fact's code (042001) is missing.
        results = [
            _result_from_texts(["hit 042000"]),
            _result_from_texts(["miss"]),  # 042001 not present
            _result_from_texts(["hit 042002"]),
        ]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)] * 3,
        )
        sample = ni._run_one_multi_needle(
            n_facts=3, corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert sample.all_of_n_hit is False
        assert sample.per_fact_hits == (True, False, True)


class TestRunOneNearNeedleMocked:
    def test_real_outranks_distractor(self, monkeypatch):
        # sample_seed=42 → REAL000042 / FAKE000042
        results = [_result_from_texts([
            "junk", "REAL000042 here", "junk2", "FAKE000042 here",
        ])]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)] * 2,
        )
        sample = ni._run_one_near_needle(
            corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert sample.real_beats_distractor is True
        assert sample.real_rank == 2
        assert sample.distractor_rank == 4

    def test_distractor_outranks_real(self, monkeypatch):
        results = [_result_from_texts([
            "FAKE000042 first", "junk", "REAL000042 second",
        ])]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)] * 2,
        )
        sample = ni._run_one_near_needle(
            corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert sample.real_beats_distractor is False
        assert sample.real_rank == 3
        assert sample.distractor_rank == 1


class TestRunOneDepthMocked:
    def test_hit(self, monkeypatch):
        # sample_seed=42 → DEPTH000042
        results = [_result_from_texts(["DEPTH000042 found"])]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)],
        )
        rank, wall = ni._run_one_depth(
            depth=0.5, corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert rank == 1
        assert wall >= 0.0

    def test_miss(self, monkeypatch):
        results = [_result_from_texts(["nothing here"])]
        monkeypatch.setattr(
            ni, "_build_pipeline",
            lambda **_: _stub_pipeline(results),
        )
        monkeypatch.setattr(
            ni, "_build_haystack",
            lambda **_: [(Path("/dev/null"), 0)],
        )
        rank, _wall = ni._run_one_depth(
            depth=0.5, corpus_tokens=100, sample_seed=42,
            embedder_id="fake", device="cpu", top_k=50,
        )
        assert rank is None


class TestRunMultiNeedleAggregation:
    def test_aggregation_via_mocked_runner(self, monkeypatch):
        """Patch ``_run_one_multi_needle`` so we control samples and
        verify that ``run_multi_needle`` aggregates per-fact recall and
        all-of-N correctly."""
        canned = [
            ni.MultiNeedleSample(
                seed=1, per_fact_hits=(True, True, True),
                per_fact_ranks=(1, 2, 3),
                all_of_n_hit=True, wall_secs=0.5,
            ),
            ni.MultiNeedleSample(
                seed=2, per_fact_hits=(True, True, False),
                per_fact_ranks=(1, 2, None),
                all_of_n_hit=False, wall_secs=0.7,
            ),
        ]
        it = iter(canned)
        monkeypatch.setattr(
            ni, "_run_one_multi_needle",
            lambda **_: next(it),
        )
        report = ni.run_multi_needle(
            n_facts=3, corpus_tokens=100, n_samples=2,
            seed=0, embedder_id="fake", device="cpu",
        )
        # 5/6 hits, 1/2 all-of-N, mean wall = 0.6
        assert report.per_fact_recall == pytest.approx(5 / 6)
        assert report.all_of_n_rate == 0.5
        assert report.mean_wall_secs == pytest.approx(0.6)


class TestRunDepthSweepAggregation:
    def test_cell_grid_walked_in_order(self, monkeypatch):
        """run_depth_sweep should produce one cell per (depth, size)
        combo, in row-major (depth-then-size) order."""

        def fake_run(*, depth, corpus_tokens, sample_seed, **_):
            # All hits except 0.9 × 5000 (last cell).
            if depth == 0.9 and corpus_tokens == 5000:
                return None, 0.05
            return 1, 0.05

        monkeypatch.setattr(ni, "_run_one_depth", fake_run)
        report = ni.run_depth_sweep(
            depths=(0.1, 0.5, 0.9),
            corpus_tokens_list=(1000, 5000),
            n_samples_per_cell=2, seed=0,
            embedder_id="fake", device="cpu",
        )
        assert len(report.cells) == 6
        # Walk order: (0.1,1000), (0.1,5000), (0.5,1000), (0.5,5000), ...
        assert report.cells[0].depth == 0.1
        assert report.cells[0].corpus_tokens == 1000
        assert report.cells[5].depth == 0.9
        assert report.cells[5].corpus_tokens == 5000
        assert report.cells[5].recall == 0.0
        # Every other cell should have recall=1.0.
        for cell in report.cells[:5]:
            assert cell.recall == 1.0


# ─────────────────────────────────────────────── parsers + json

class TestArgParsers:
    def test_tokens_list(self):
        assert ni._parse_tokens_list("100,200,300") == [100, 200, 300]
        assert ni._parse_tokens_list("100, 200 , 300") == [100, 200, 300]

    def test_depths(self):
        assert ni._parse_depths("0.1,0.5,0.9") == [0.1, 0.5, 0.9]


class TestJsonSnapshot:
    def test_dataclass_round_trip(self, tmp_path: Path, monkeypatch):
        """``_json_default`` should serialize nested dataclasses; the
        round-trip preserves the report fields we care about."""
        monkeypatch.setattr(
            ni, "_output_dir", lambda: tmp_path,
        )
        report = ni.MultiNeedleReport(
            n_facts=2, corpus_tokens=1000, n_samples=1, top_k=10,
            seed=0, embedder_id="fake", device="cpu",
            per_fact_recall=0.5, all_of_n_rate=0.0,
            samples=[
                ni.MultiNeedleSample(
                    seed=1, per_fact_hits=(True, False),
                    per_fact_ranks=(1, None),
                    all_of_n_hit=False, wall_secs=0.1,
                ),
            ],
        )
        path = ni._save_json_snapshot("multi_needle", report)
        data = json.loads(path.read_text())
        assert data["n_facts"] == 2
        assert data["per_fact_recall"] == 0.5
        assert data["samples"][0]["per_fact_hits"] == [True, False]
        assert data["samples"][0]["per_fact_ranks"] == [1, None]


# ─────────────────────────────────────────────── CLI smoke (no real calls)

class TestCliSmoke:
    def test_cmd_multi_needle_dispatch(self, monkeypatch, capsys):
        """Smoke-test the CLI dispatch path. We patch the runner so we
        never load a real embedder."""
        canned = ni.MultiNeedleReport(
            n_facts=3, corpus_tokens=1000, n_samples=1, top_k=10,
            seed=0, embedder_id="fake", device="cpu",
            per_fact_recall=1.0, all_of_n_rate=1.0, mean_wall_secs=0.1,
        )
        monkeypatch.setattr(ni, "run_multi_needle", lambda **_: canned)
        monkeypatch.setattr(
            "sys.argv",
            ["niah_advanced", "multi-needle", "--samples", "1",
             "--tokens", "1000"],
        )
        rc = ni.main()
        assert rc == 0
        captured = capsys.readouterr().out
        assert "1.00" in captured

    def test_cmd_near_needle_dispatch(self, monkeypatch, capsys):
        canned = ni.NearNeedleReport(
            corpus_tokens=1000, n_samples=1, top_k=10, seed=0,
            embedder_id="fake", device="cpu",
            real_recall=1.0, real_beats_distractor_rate=1.0,
            mean_wall_secs=0.1,
        )
        monkeypatch.setattr(
            ni, "run_needle_near_needle", lambda **_: canned,
        )
        monkeypatch.setattr(
            "sys.argv",
            ["niah_advanced", "near-needle", "--samples", "1",
             "--tokens", "1000"],
        )
        rc = ni.main()
        assert rc == 0

    def test_cmd_depth_excludes_12m_by_default(self, monkeypatch):
        called_with: dict = {}

        def fake_run_depth_sweep(*, corpus_tokens_list, **_):
            called_with["sizes"] = list(corpus_tokens_list)
            return ni.DepthSweepReport(
                depths=(0.1,), corpus_tokens_list=tuple(corpus_tokens_list),
                n_samples_per_cell=1, top_k=10, seed=0,
                embedder_id="fake", device="cpu",
            )

        monkeypatch.setattr(ni, "run_depth_sweep", fake_run_depth_sweep)
        monkeypatch.setattr(
            "sys.argv",
            ["niah_advanced", "depth", "--samples", "1",
             "--tokens", "100000,12000000",
             "--depths", "0.5"],
        )
        ni.main()
        # 12M filtered out without --include-12m
        assert 12_000_000 not in called_with["sizes"]
        assert 100_000 in called_with["sizes"]

    def test_cmd_depth_include_12m_keeps_it(self, monkeypatch):
        called_with: dict = {}

        def fake_run_depth_sweep(*, corpus_tokens_list, **_):
            called_with["sizes"] = list(corpus_tokens_list)
            return ni.DepthSweepReport(
                depths=(0.1,), corpus_tokens_list=tuple(corpus_tokens_list),
                n_samples_per_cell=1, top_k=10, seed=0,
                embedder_id="fake", device="cpu",
            )

        monkeypatch.setattr(ni, "run_depth_sweep", fake_run_depth_sweep)
        monkeypatch.setattr(
            "sys.argv",
            ["niah_advanced", "depth", "--samples", "1",
             "--tokens", "100000,12000000",
             "--depths", "0.5", "--include-12m"],
        )
        ni.main()
        assert 12_000_000 in called_with["sizes"]
