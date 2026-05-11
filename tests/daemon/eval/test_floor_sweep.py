"""Tests for floor-sweep recommendation logic.

Pure-function tests over hand-built sweep points. The corpus-indexing
half is covered in ``test_abstention.py`` — this file focuses on the
recommendation-rule edge cases."""
from __future__ import annotations

import pytest

from longctx_daemon.eval.abstention import (
    SweepReport,
    recommend_floor,
    render_sweep_table,
)


class TestRecommendFloorEdgeCases:
    """The recommend_floor logic is the most-likely-to-regress piece —
    cover the boundaries explicitly."""

    def test_classic_curve_picks_50(self):
        """The example curve in the spec docstring should pick 0.50."""
        points = [
            (0.30, 0.42, 0.00),
            (0.35, 0.48, 0.00),
            (0.40, 0.55, 0.02),
            (0.45, 0.62, 0.03),
            (0.50, 0.68, 0.05),
            (0.55, 0.74, 0.12),
            (0.60, 0.81, 0.20),
            (0.65, 0.85, 0.32),
            (0.70, 0.89, 0.45),
        ]
        assert recommend_floor(points, max_fnr=0.05) == 0.50

    def test_max_fnr_strictly_under_picks_45(self):
        points = [
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.02),
            (0.45, 0.62, 0.03),
            (0.50, 0.68, 0.05),
        ]
        # max_fnr=0.04 → 0.50 disqualified (0.05 > 0.04)
        assert recommend_floor(points, max_fnr=0.04) == 0.45

    def test_max_fnr_zero_picks_zero_fnr_only(self):
        points = [
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.00),
            (0.50, 0.68, 0.01),
        ]
        # Strict zero-FNR → only 0.30 / 0.40 qualify; pick 0.40
        assert recommend_floor(points, max_fnr=0.0) == 0.40

    def test_descending_fnr_curve(self):
        """Pathological case: FNR decreases with floor (shouldn't
        happen in practice, but the rule must still work)."""
        points = [
            (0.30, 0.40, 0.10),
            (0.50, 0.60, 0.05),
            (0.70, 0.80, 0.02),
        ]
        # max_fnr=0.05 → 0.50 and 0.70 qualify; highest = 0.70
        assert recommend_floor(points, max_fnr=0.05) == 0.70

    def test_only_one_point_qualifies(self):
        points = [
            (0.30, 0.42, 0.10),
            (0.50, 0.68, 0.04),
            (0.70, 0.89, 0.40),
        ]
        assert recommend_floor(points, max_fnr=0.05) == 0.50

    def test_extreme_max_fnr_picks_highest(self):
        points = [
            (0.30, 0.10, 0.00),
            (0.50, 0.20, 0.50),
            (0.70, 0.90, 0.99),
        ]
        # Max FNR = 1.0 → all qualify; pick 0.70
        assert recommend_floor(points, max_fnr=1.0) == 0.70

    def test_two_points_same_floor_does_not_crash(self):
        """Defensive: duplicate floors should still produce a
        deterministic answer."""
        points = [
            (0.50, 0.6, 0.04),
            (0.50, 0.6, 0.04),
        ]
        assert recommend_floor(points, max_fnr=0.05) == 0.50

    def test_floats_with_precision_drift(self):
        """Floors built via accumulator (start + step) can drift past
        floating-point exactness; the recommend rule must still pick
        the right one."""
        points = [
            (0.30000000001, 0.4, 0.0),
            (0.45000000001, 0.6, 0.02),
            (0.60000000001, 0.8, 0.20),
        ]
        result = recommend_floor(points, max_fnr=0.05)
        assert abs(result - 0.45) < 1e-6


class TestSweepReportShape:
    def test_dataclass_fields(self):
        report = SweepReport(
            points=[(0.5, 0.6, 0.04)],
            recommended_floor=0.5,
            max_fnr=0.05,
        )
        assert report.recommended_floor == 0.5
        assert report.max_fnr == 0.05
        assert len(report.points) == 1

    def test_render_includes_all_floors(self):
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
        assert "0.30" in s
        assert "0.50" in s
        assert "0.70" in s

    def test_render_shows_recommendation_marker_only_once(self):
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
        assert s.count("←") == 1
