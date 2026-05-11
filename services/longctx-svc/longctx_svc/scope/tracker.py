"""Per-session phased scope tracker.

Accumulates path mentions across turns and transitions a scope candidate
through four phases:

  watch        — Below speculative threshold. Track but don't act.
  speculative  — Enough unique files to start a background index. Don't
                 inject retrieval yet (could still be a one-shot mention).
  active       — High-confidence working scope. Inject on every turn.
  demoted      — Active scope went silent across N turns. Stop injecting,
                 keep index around in case it re-activates.

A new cluster can supersede an active one if it accumulates more evidence
("drift" — user switched projects mid-session). The previous active scope
demotes; the new winner promotes.

Thread-safe via a single ``RLock``. No I/O — the indexer / state layer
handles disk and embeddings off this signal.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from longctx_svc.scope.cluster import ClusterResult, cluster_paths


class Phase(str, Enum):
    WATCH = "watch"
    SPECULATIVE = "speculative"
    ACTIVE = "active"
    DEMOTED = "demoted"


@dataclass
class CandidateState:
    """Per-cluster bookkeeping."""
    ancestor: Path
    phase: Phase
    unique_files: int
    first_seen_turn: int
    last_seen_turn: int
    promoted_at: float | None = None      # wall-clock when entered ACTIVE
    demoted_at: float | None = None

    def is_alive(self) -> bool:
        return self.phase in (Phase.SPECULATIVE, Phase.ACTIVE)


@dataclass
class TickResult:
    """Snapshot returned by :meth:`ScopeClusterTracker.observe`."""
    active: CandidateState | None
    speculative: list[CandidateState] = field(default_factory=list)
    transitions: list[tuple[Path, Phase, Phase]] = field(default_factory=list)
    turn: int = 0

    def has_active(self) -> bool:
        return self.active is not None


class ScopeClusterTracker:
    """Phased scope detection across turns of one session.

    Thresholds (turn over to ctor kwargs for tunability, sane defaults below):

      speculate_min_files  — files-in-cluster to enter SPECULATIVE (default 3)
      active_min_files     — files-in-cluster to enter ACTIVE        (default 5)
      active_min_turns     — alt-path: turns of SPECULATIVE before
                              auto-promotion if speculate threshold sustained.
                              Default 2: one turn of "saw it" + one of "still
                              seeing it" = enough confirmation.
      demote_after_idle_turns — how many consecutive turns with zero
                              mentions of the active scope before it gets
                              demoted (default 5).
      drift_required_gap   — a new candidate must exceed the current
                              active's unique_files by this much to take
                              over. Avoids flapping. Default 2.

    All observations are SESSION-LOCAL. One tracker per session_id.
    """

    def __init__(
        self,
        *,
        speculate_min_files: int = 3,
        active_min_files: int = 5,
        active_min_turns: int = 2,
        demote_after_idle_turns: int = 5,
        drift_required_gap: int = 2,
    ) -> None:
        if speculate_min_files < 1:
            raise ValueError("speculate_min_files must be ≥ 1")
        if active_min_files < speculate_min_files:
            raise ValueError("active_min_files must be ≥ speculate_min_files")
        if demote_after_idle_turns < 1:
            raise ValueError("demote_after_idle_turns must be ≥ 1")
        self._spec_min = speculate_min_files
        self._active_min = active_min_files
        self._active_min_turns = active_min_turns
        self._demote_after = demote_after_idle_turns
        self._drift_gap = drift_required_gap

        self._lock = threading.RLock()
        # turn counter increments on every observe() call.
        self._turn = 0
        # ancestor -> CandidateState
        self._candidates: dict[Path, CandidateState] = {}
        # Cumulative deduped paths seen across the whole session.
        self._paths_seen: set[Path] = set()

    # ------------------------------------------------------------ accessors

    @property
    def turn(self) -> int:
        with self._lock:
            return self._turn

    def candidates_snapshot(self) -> list[CandidateState]:
        with self._lock:
            return list(self._candidates.values())

    def active(self) -> CandidateState | None:
        with self._lock:
            for c in self._candidates.values():
                if c.phase == Phase.ACTIVE:
                    return c
            return None

    def speculative(self) -> list[CandidateState]:
        with self._lock:
            return [c for c in self._candidates.values()
                    if c.phase == Phase.SPECULATIVE]

    # ------------------------------------------------------------ mutation

    def observe(self, paths_this_turn: list[Path]) -> TickResult:
        """Record paths from one turn. Advances the turn counter and runs
        the phase state machine.

        ``paths_this_turn`` should be the absolute paths extracted from
        this turn's messages (tool calls, tool results, user text). Caller
        is responsible for deduping if it wants — we dedupe internally for
        unique-file accounting.
        """
        with self._lock:
            self._turn += 1
            transitions: list[tuple[Path, Phase, Phase]] = []

            # Phase advancement is driven by FRESH evidence under a cluster
            # ancestor; demotion is driven by lack of it. We dedupe via the
            # cumulative path set (a path mentioned 10 times across the
            # session counts as 1 file's worth of unique-file evidence).
            for p in paths_this_turn:
                self._paths_seen.add(p)
            touched_this_turn = set(paths_this_turn)

            # Recompute clusters from all paths seen so far. Cheap — bounded
            # by session-cumulative path count, typically <1000.
            all_paths = list(self._paths_seen)
            top = cluster_paths(all_paths, min_files=self._spec_min)
            self._update_phases(top, touched_this_turn, transitions)

            # Demote stale actives — runs LAST so a re-activated candidate
            # this turn doesn't immediately fall back over on idle check.
            self._demote_idle(transitions)

            return TickResult(
                active=self.active(),
                speculative=self.speculative(),
                transitions=transitions,
                turn=self._turn,
            )

    # ------------------------------------------------------------ internals

    def _update_phases(
        self,
        top: ClusterResult | None,
        touched_this_turn: set[Path],
        transitions: list[tuple[Path, Phase, Phase]],
    ) -> None:
        """Apply the phase state machine given the current top cluster.

        Only fires phase advancement when the cluster received fresh
        mentions THIS TURN — a cluster that still exists purely from
        cumulative history but saw no activity should NOT count as
        evidence for promotion or for refreshing the idleness clock.

        The active-cluster drift check still fires on every turn, but the
        challenger must itself have fresh-this-turn evidence; otherwise an
        old cluster could keep its place at the top of the heap forever.
        """
        if top is None:
            return

        anc = top.ancestor

        # Did the top cluster see any fresh mention under it this turn?
        # `parents` gives the chain of dir ancestors; a path "under anc"
        # has anc in its parents.
        has_fresh = any(anc in p.parents for p in touched_this_turn)
        if not has_fresh:
            # Cluster persists due to cumulative history only. Skip phase
            # work; let _demote_idle handle staleness.
            return

        cand = self._candidates.get(anc)
        if cand is None:
            cand = CandidateState(
                ancestor=anc,
                phase=Phase.WATCH,
                unique_files=top.unique_files,
                first_seen_turn=self._turn,
                last_seen_turn=self._turn,
            )
            self._candidates[anc] = cand

        prev_phase = cand.phase
        cand.unique_files = top.unique_files
        cand.last_seen_turn = self._turn

        # If a previously-demoted candidate accumulates new evidence, treat
        # it like a fresh WATCH entry (lets re-activation re-fire the state
        # machine through SPECULATIVE → ACTIVE again).
        if cand.phase == Phase.DEMOTED:
            cand.phase = Phase.WATCH
            cand.first_seen_turn = self._turn

        # Drift check: another candidate (the one that's currently ACTIVE,
        # if any) may need to step aside in favor of this fresh challenger.
        active = self._active_locked()
        if active is not None and active.ancestor != anc:
            if cand.unique_files >= active.unique_files + self._drift_gap \
                    and cand.unique_files >= self._active_min:
                old = active.phase
                active.phase = Phase.DEMOTED
                active.demoted_at = time.time()
                transitions.append((active.ancestor, old, Phase.DEMOTED))

        # Promotion ladder.
        if cand.phase == Phase.WATCH:
            if cand.unique_files >= self._spec_min:
                cand.phase = Phase.SPECULATIVE
                transitions.append((anc, prev_phase, Phase.SPECULATIVE))

        if cand.phase == Phase.SPECULATIVE:
            sustained_turns = (self._turn - cand.first_seen_turn) + 1
            should_activate = (
                cand.unique_files >= self._active_min
                or sustained_turns >= self._active_min_turns
            )
            # Only activate if there is no active OR drift cleared it.
            other_active = self._active_locked()
            if should_activate and (other_active is None
                                    or other_active.ancestor == anc):
                prev_phase_2 = cand.phase
                cand.phase = Phase.ACTIVE
                cand.promoted_at = time.time()
                transitions.append((anc, prev_phase_2, Phase.ACTIVE))

    def _active_locked(self) -> CandidateState | None:
        # Caller already holds ``self._lock``.
        for c in self._candidates.values():
            if c.phase == Phase.ACTIVE:
                return c
        return None

    def _demote_idle(
        self,
        transitions: list[tuple[Path, Phase, Phase]],
    ) -> None:
        """Demote SPECULATIVE / ACTIVE candidates that haven't accumulated
        new evidence in ``self._demote_after`` turns.

        Demotion preserves the candidate (so re-activation can revive it)
        but flips its phase so we stop injecting / indexing speculatively.
        """
        for c in self._candidates.values():
            if c.phase not in (Phase.SPECULATIVE, Phase.ACTIVE):
                continue
            idle = self._turn - c.last_seen_turn
            if idle >= self._demote_after:
                prev = c.phase
                c.phase = Phase.DEMOTED
                c.demoted_at = time.time()
                transitions.append((c.ancestor, prev, Phase.DEMOTED))
