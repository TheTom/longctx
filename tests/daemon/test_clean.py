"""Tests for ``longctx_daemon.clean`` — manual cleanup sweeps."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from longctx_daemon.clean import (
    CleanupDecision,
    CleanupPlan,
    collect_idle,
    collect_missing,
    collect_orphans,
    collect_replay_shards,
    execute_plan,
    format_plan,
    parse_duration_seconds,
)
from longctx_daemon.types import Project


# ---------------------------------------------------------- duration parser

@pytest.mark.parametrize("s,seconds", [
    ("90d", 90 * 86400),
    ("2w", 2 * 604800),
    ("12h", 12 * 3600),
    ("30m", 30 * 60),
    ("1d", 86400),
])
def test_parse_duration_valid(s, seconds):
    assert parse_duration_seconds(s) == seconds


@pytest.mark.parametrize("bad", ["", "90", "9x", "1.5d", "h", "abc"])
def test_parse_duration_invalid(bad):
    with pytest.raises(ValueError):
        parse_duration_seconds(bad)


# ------------------------------------------------------------- collectors

class _FakeChunkStore:
    def __init__(self, projects=None):
        self._projects = list(projects or [])
        self._files: dict[str, list] = {}

    def list_projects(self):
        return tuple(self._projects)

    def list_files(self, project_name):
        return tuple(self._files.get(project_name, []))

    def get_chunks_by_file(self, fid):
        return ()

    def delete_project(self, name):
        self._projects = [p for p in self._projects if p.name != name]


def test_collect_idle_drops_stale_projects():
    now = time.time()
    store = _FakeChunkStore(projects=[
        Project(name="recent", root_path="/tmp/r",
                last_full_scan_at=int(now)),
        Project(name="stale", root_path="/tmp/s",
                last_full_scan_at=int(now - 100 * 86400)),
        Project(name="never", root_path="/tmp/n",
                last_full_scan_at=0),
    ])
    decisions = collect_idle(store, max_age_seconds=90 * 86400, now=now)
    targets = {d.target for d in decisions}
    assert targets == {"stale"}


def test_collect_orphans_finds_session_bound():
    store = _FakeChunkStore(projects=[
        Project(name="permanent", root_path="/tmp/p", session_id=None),
        Project(name="orphan", root_path="/tmp/o",
                session_id="ses_dead123"),
    ])
    decisions = collect_orphans(store)
    assert {d.target for d in decisions} == {"orphan"}
    assert decisions[0].kind == "session_orphan"


def test_collect_missing_drops_nonexistent_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    store = _FakeChunkStore(projects=[
        Project(name="exists", root_path=str(real)),
        Project(name="gone", root_path=str(tmp_path / "doesnotexist")),
    ])
    decisions = collect_missing(store)
    assert {d.target for d in decisions} == {"gone"}


def test_collect_replay_shards_drops_old(tmp_path):
    """Old gzipped shards drop; current jsonl is untouched."""
    cur = tmp_path / "interactions.jsonl"
    cur.write_text('{"ts": "now"}\n')
    old = tmp_path / "interactions.jsonl.2024-01-01.gz"
    old.write_bytes(b"compressed-content")
    # Manually backdate
    import os
    long_ago = time.time() - 60 * 86400
    os.utime(old, (long_ago, long_ago))

    decisions = collect_replay_shards(
        tmp_path, older_than_seconds=30 * 86400,
    )
    assert {d.target for d in decisions} == {old.name}


def test_collect_replay_shards_returns_empty_for_missing_dir(tmp_path):
    assert collect_replay_shards(tmp_path / "nope", 86400) == []


# ----------------------------------------------------------- format_plan

def test_format_plan_empty():
    out = format_plan(CleanupPlan(decisions=()))
    assert "nothing" in out


def test_format_plan_renders_each_decision():
    plan = CleanupPlan(decisions=(
        CleanupDecision(target="myapp", reason="idle 92d > 90d",
                        bytes_freed=1024 * 1024 * 5, kind="project"),
        CleanupDecision(target="ses_orphan", reason="session dead",
                        bytes_freed=512, kind="session_orphan"),
    ))
    out = format_plan(plan)
    assert "myapp" in out
    assert "ses_orphan" in out
    assert "5 MB" in out
    assert "total: 2 item(s)" in out


# ------------------------------------------------------------- execute_plan

def test_execute_plan_calls_delete():
    p1 = Project(name="a", root_path="/tmp/a", session_id="s")
    p2 = Project(name="b", root_path="/tmp/b")
    store = _FakeChunkStore(projects=[p1, p2])
    plan = CleanupPlan(decisions=(
        CleanupDecision(target="a", reason="orphan", bytes_freed=0,
                        kind="session_orphan"),
    ))
    n = execute_plan(plan, chunk_store=store, interactions_dir=Path("/tmp"))
    assert n == 1
    remaining = {p.name for p in store.list_projects()}
    assert remaining == {"b"}


def test_execute_plan_removes_replay_shards(tmp_path):
    shard = tmp_path / "interactions.jsonl.old.gz"
    shard.write_bytes(b"x")
    plan = CleanupPlan(decisions=(
        CleanupDecision(target=shard.name, reason="old",
                        bytes_freed=1, kind="replay_shard"),
    ))
    store = _FakeChunkStore()
    n = execute_plan(plan, chunk_store=store, interactions_dir=tmp_path)
    assert n == 1
    assert not shard.exists()
