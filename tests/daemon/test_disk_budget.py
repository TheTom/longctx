"""Tier 3 disk-budget LRU eviction tests (experimental opt-in)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from longctx_daemon.disk_budget import (
    EvictionPlan,
    cache_size_bytes,
    execute_eviction,
    maybe_evict,
    plan_eviction,
)
from longctx_daemon.types import FileRecord, Project


# --------------------------------------------------------------- fakes

@dataclass
class _FakeChunk:
    text: str


class _FakeChunkStore:
    """In-memory store that mimics the contract plan_eviction needs."""

    def __init__(self) -> None:
        self._projects: list[Project] = []
        self._files: dict[str, list[FileRecord]] = {}
        self._chunks: dict[int, list[_FakeChunk]] = {}
        self._fid = 0

    def add(self, name: str, last_full_scan_at: int = 0,
            n_chunks: int = 1, chars_per_chunk: int = 1024) -> None:
        self._projects.append(Project(
            name=name, root_path=f"/tmp/{name}",
            last_full_scan_at=last_full_scan_at,
        ))
        self._fid += 1
        f = FileRecord(
            id=self._fid, project=name, rel_path="x.py",
            mtime=last_full_scan_at, size_bytes=1, content_hash="z" * 64,
        )
        self._files[name] = [f]
        self._chunks[self._fid] = [
            _FakeChunk(text="x" * chars_per_chunk) for _ in range(n_chunks)
        ]

    def list_projects(self):
        return tuple(self._projects)

    def get_project(self, name: str):
        for p in self._projects:
            if p.name == name:
                return p
        return None

    def list_files(self, project: Optional[str] = None):
        if project is None:
            return tuple(f for fs in self._files.values() for f in fs)
        return tuple(self._files.get(project, []))

    def get_chunks_by_file(self, file_id: int):
        return tuple(self._chunks.get(file_id, []))

    def delete_project(self, name: str) -> None:
        self._projects = [p for p in self._projects if p.name != name]
        if name in self._files:
            for f in self._files[name]:
                self._chunks.pop(f.id, None)
            del self._files[name]


# --------------------------------------------------------------- size

def test_cache_size_walks_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 2048)
    assert cache_size_bytes(tmp_path) == 1024 + 2048


def test_cache_size_returns_zero_for_missing_dir(tmp_path):
    assert cache_size_bytes(tmp_path / "nope") == 0


def test_cache_size_skips_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.bin").write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(real)
    # Total includes real/a.bin once, not via symlink.
    n = cache_size_bytes(tmp_path)
    assert n == 3


# --------------------------------------------------------------- plan

def test_plan_no_op_when_budget_zero(tmp_path):
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=int(time.time()))
    plan = plan_eviction(tmp_path, store, budget_gb=0.0)
    assert plan.targets == ()


def test_plan_no_op_when_under_budget(tmp_path):
    """Budget far above current size → empty plan."""
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=int(time.time()))
    (tmp_path / "tiny.bin").write_bytes(b"x" * 1024)
    plan = plan_eviction(tmp_path, store, budget_gb=1.0)
    assert plan.targets == ()
    assert plan.budget_exceeded_by == 0


def test_plan_evicts_lru_first(tmp_path):
    """When over budget, oldest queried project evicts first."""
    store = _FakeChunkStore()
    store.add("recent", last_full_scan_at=2000, n_chunks=10,
              chars_per_chunk=2000)
    store.add("oldest", last_full_scan_at=100, n_chunks=10,
              chars_per_chunk=2000)
    store.add("middle", last_full_scan_at=1000, n_chunks=10,
              chars_per_chunk=2000)

    # Fake the on-disk size to exceed a tiny budget
    (tmp_path / "big.bin").write_bytes(b"x" * 10 * 1024 ** 2)   # 10 MB
    plan = plan_eviction(tmp_path, store, budget_gb=0.001)   # ~1 MB

    assert "oldest" in plan.targets
    # And LRU order: oldest comes before middle in target list
    if "middle" in plan.targets:
        assert plan.targets.index("oldest") < plan.targets.index("middle")


def test_plan_honors_pinned_projects(tmp_path):
    """Pinned projects never evict, even when LRU."""
    store = _FakeChunkStore()
    store.add("pinned", last_full_scan_at=100, n_chunks=10,
              chars_per_chunk=2000)
    store.add("not_pinned", last_full_scan_at=200, n_chunks=10,
              chars_per_chunk=2000)
    (tmp_path / "big.bin").write_bytes(b"x" * 10 * 1024 ** 2)
    plan = plan_eviction(
        tmp_path, store, budget_gb=0.001,
        pinned_projects=("pinned",),
    )
    assert "pinned" not in plan.targets


def test_plan_uses_last_query_overrides(tmp_path):
    """Watcher's live last_query_at takes precedence over scan time."""
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=2000, n_chunks=10,
              chars_per_chunk=2000)
    store.add("b", last_full_scan_at=100, n_chunks=10,
              chars_per_chunk=2000)
    # Override: a was actually queried recently; b idle for a long time
    overrides = {"a": 5000.0, "b": 50.0}
    (tmp_path / "big.bin").write_bytes(b"x" * 10 * 1024 ** 2)
    plan = plan_eviction(
        tmp_path, store, budget_gb=0.001,
        last_query_overrides=overrides,
    )
    # b is older per overrides → evicts first
    assert plan.targets and plan.targets[0] == "b"


# --------------------------------------------------------------- execute

def test_execute_calls_indexer_delete():
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=100)

    deleted: list[str] = []

    class _FakeIndexer:
        def delete_project(self, name: str) -> None:
            deleted.append(name)

    plan = EvictionPlan(
        targets=("a",), bytes_to_free=1, pre_eviction_bytes=0,
        budget_bytes=0, budget_exceeded_by=0,
    )
    n = execute_eviction(plan, indexer=_FakeIndexer(), chunk_store=store)
    assert n == 1
    assert deleted == ["a"]


def test_execute_falls_back_to_chunk_store():
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=100)
    plan = EvictionPlan(
        targets=("a",), bytes_to_free=1, pre_eviction_bytes=0,
        budget_bytes=0, budget_exceeded_by=0,
    )
    n = execute_eviction(plan, chunk_store=store)
    assert n == 1
    # Project gone via chunk_store cascade
    assert store.get_project("a") is None


def test_execute_continues_on_failure(caplog):
    """If one delete raises, the rest of the plan still runs."""
    deleted: list[str] = []

    class _PartialFail:
        def delete_project(self, name: str) -> None:
            if name == "a":
                raise RuntimeError("boom")
            deleted.append(name)

    plan = EvictionPlan(
        targets=("a", "b"), bytes_to_free=2,
        pre_eviction_bytes=0, budget_bytes=0, budget_exceeded_by=0,
    )
    n = execute_eviction(plan, indexer=_PartialFail())
    assert n == 1   # only "b" succeeded
    assert deleted == ["b"]


# --------------------------------------------------------------- maybe_evict

def test_maybe_evict_no_op_when_under_budget(tmp_path):
    store = _FakeChunkStore()
    store.add("a")
    plan = maybe_evict(tmp_path, store, budget_gb=10.0)
    assert plan.targets == ()


def test_maybe_evict_runs_when_over_budget(tmp_path):
    store = _FakeChunkStore()
    store.add("a", last_full_scan_at=100, n_chunks=10,
              chars_per_chunk=2000)
    (tmp_path / "big.bin").write_bytes(b"x" * 10 * 1024 ** 2)
    plan = maybe_evict(tmp_path, store, budget_gb=0.001)
    assert plan.targets == ("a",)
    assert store.get_project("a") is None
