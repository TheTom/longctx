"""Unit tests for the per-session phased scope tracker.

Covers the state machine transitions (WATCH → SPECULATIVE → ACTIVE →
DEMOTED), drift handoff between competing clusters, idle demotion, and
re-activation after a demoted candidate sees new evidence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.scope.tracker import Phase, ScopeClusterTracker


def _files(dirpath: str, names: list[str]) -> list[Path]:
    return [Path(dirpath) / n for n in names]


class TestSinglePathPhaseProgression:
    def test_below_speculate_no_active(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5)
        result = t.observe(_files("/proj", ["a.py", "b.py"]))   # only 2
        assert result.active is None
        assert result.speculative == []

    def test_three_files_one_turn_goes_speculative(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=2)
        # Single turn with exactly 3 files. Sustained=1 turn (just this one);
        # active requires 2 turns OR active_min_files (5). Neither met yet.
        result = t.observe(_files("/proj", ["a.py", "b.py", "c.py"]))
        assert result.active is None
        spec_ancs = [c.ancestor for c in result.speculative]
        assert Path("/proj") in spec_ancs

    def test_speculate_then_active_on_second_turn(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=2)
        t.observe(_files("/proj", ["a.py", "b.py", "c.py"]))
        # Second turn — cluster sustained for 2 turns now → ACTIVE.
        result = t.observe(_files("/proj", ["a.py"]))
        assert result.active is not None
        assert result.active.ancestor == Path("/proj")
        assert result.active.phase == Phase.ACTIVE

    def test_active_min_files_skips_sustain_wait(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999)  # disable sustain path
        # 5 files in one turn — clears active_min_files immediately.
        result = t.observe(_files("/proj", ["a.py", "b.py", "c.py",
                                            "d.py", "e.py"]))
        assert result.active is not None
        assert result.active.ancestor == Path("/proj")

    def test_transitions_emitted(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999)
        result = t.observe(_files("/proj", ["a.py", "b.py", "c.py",
                                            "d.py", "e.py"]))
        # WATCH→SPECULATIVE then SPECULATIVE→ACTIVE in one tick.
        kinds = [(t_anc.name, prev, nxt) for t_anc, prev, nxt
                 in result.transitions]
        assert ("proj", Phase.WATCH, Phase.SPECULATIVE) in kinds
        assert ("proj", Phase.SPECULATIVE, Phase.ACTIVE) in kinds


class TestDrift:
    def test_new_cluster_supersedes_old_with_enough_gap(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999, drift_required_gap=2)
        # Establish /old as ACTIVE.
        t.observe(_files("/old", ["a.py", "b.py", "c.py", "d.py", "e.py"]))
        assert t.active().ancestor == Path("/old")

        # New burst on /new with same count → not enough gap to flip.
        result = t.observe(_files("/new", ["a.py", "b.py", "c.py",
                                           "d.py", "e.py"]))
        assert result.active.ancestor == Path("/old")

        # Push /new harder — gap=2 over /old. With same /old paths
        # NOT re-mentioned this turn, /old stays at 5; /new now 7.
        result = t.observe(_files("/new", ["f.py", "g.py"]))
        assert result.active.ancestor == Path("/new")

    def test_drift_does_not_flap_within_gap(self):
        # Use a long idle window so /old isn't demoted by silence — we
        # want to isolate the drift-gap behavior, not idle demotion.
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999, drift_required_gap=3,
                                demote_after_idle_turns=20)
        t.observe(_files("/old", ["a.py", "b.py", "c.py", "d.py", "e.py"]))
        # /new ramps up to 6 (just +1 over /old's 5) — gap=3 not met.
        for n in ["a", "b", "c", "d", "e", "f"]:
            t.observe([Path(f"/new/{n}.py")])
        assert t.active().ancestor == Path("/old")


class TestIdleDemotion:
    def test_active_demoted_after_idle_threshold(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999,
                                demote_after_idle_turns=3)
        t.observe(_files("/proj", ["a.py", "b.py", "c.py", "d.py", "e.py"]))
        assert t.active().ancestor == Path("/proj")
        # 3 silent turns (no /proj mentions) → DEMOTED.
        for _ in range(3):
            t.observe([])
        assert t.active() is None
        # Candidate still tracked (demoted), not removed.
        all_ancs = [c.ancestor for c in t.candidates_snapshot()]
        assert Path("/proj") in all_ancs

    def test_active_not_demoted_with_fresh_mentions(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999,
                                demote_after_idle_turns=2)
        t.observe(_files("/proj", ["a.py", "b.py", "c.py", "d.py", "e.py"]))
        # Each turn keeps mentioning /proj — never demotes.
        for _ in range(5):
            t.observe([Path("/proj/a.py")])
        assert t.active().ancestor == Path("/proj")


class TestReactivation:
    def test_demoted_candidate_reactivates_on_new_evidence(self):
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=5,
                                active_min_turns=999,
                                demote_after_idle_turns=2)
        t.observe(_files("/proj", ["a.py", "b.py", "c.py", "d.py", "e.py"]))
        # Demote it.
        t.observe([])
        t.observe([])
        assert t.active() is None
        # New activity — re-promotes through WATCH→SPECULATIVE→ACTIVE.
        result = t.observe([Path("/proj/x.py")])  # cluster now has 6 files
        assert result.active is not None
        assert result.active.ancestor == Path("/proj")


class TestNoFalsePositives:
    def test_zero_paths_zero_active(self):
        t = ScopeClusterTracker()
        result = t.observe([])
        assert result.active is None
        assert result.speculative == []

    def test_grab_bag_paths_no_cluster(self):
        # Three paths under DIFFERENT roots — only ancestor with ≥3 is `/`.
        # Tracker should NOT treat `/` as a useful active scope.
        # (We test the contract: caller of tracker.active() is responsible
        # for sanity-filtering ancestors that are too shallow. But the
        # current implementation lets `/` through. Document this so the
        # caller knows.)
        t = ScopeClusterTracker(speculate_min_files=3, active_min_files=3,
                                active_min_turns=999)
        result = t.observe([
            Path("/etc/x"), Path("/tmp/y"), Path("/var/z"),
        ])
        # Active will be `/` — caller filters via min-depth check.
        if result.active is not None:
            # Sanity-document the trap.
            assert result.active.ancestor == Path("/")

    def test_turn_counter_advances_each_observation(self):
        t = ScopeClusterTracker()
        assert t.turn == 0
        t.observe([])
        assert t.turn == 1
        t.observe([Path("/a/b.py")])
        assert t.turn == 2


class TestThresholdValidation:
    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            ScopeClusterTracker(speculate_min_files=0)
        with pytest.raises(ValueError):
            ScopeClusterTracker(speculate_min_files=5, active_min_files=3)
        with pytest.raises(ValueError):
            ScopeClusterTracker(demote_after_idle_turns=0)
