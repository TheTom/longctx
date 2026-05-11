"""Tests for longctx_daemon.replay_log — JSONL append, rotation, gzip."""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from longctx_daemon.replay_log import ReplayLog, DEFAULT_MAX_BYTES


def test_writes_three_lines_as_valid_json(tmp_path: Path):
    log = ReplayLog(tmp_path / "interactions.jsonl")
    log.record({"a": 1})
    log.record({"b": 2})
    log.record({"c": 3})
    log.close()

    contents = (tmp_path / "interactions.jsonl").read_text()
    lines = [ln for ln in contents.splitlines() if ln]
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
    assert json.loads(lines[2]) == {"c": 3}


def test_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "log.jsonl"
    log = ReplayLog(target)
    log.record({"x": 1})
    log.close()
    assert target.exists()


def test_close_is_idempotent(tmp_path: Path):
    log = ReplayLog(tmp_path / "x.jsonl")
    log.close()
    log.close()  # should not raise


def test_record_after_close_raises(tmp_path: Path):
    log = ReplayLog(tmp_path / "x.jsonl")
    log.close()
    with pytest.raises(RuntimeError):
        log.record({"x": 1})


def test_context_manager_closes(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    with ReplayLog(path) as log:
        log.record({"x": 1})
    # close() invoked — re-open and verify content stable.
    assert json.loads(path.read_text().strip()) == {"x": 1}


def test_appends_across_constructions(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    log1 = ReplayLog(path)
    log1.record({"a": 1})
    log1.close()
    log2 = ReplayLog(path)
    log2.record({"b": 2})
    log2.close()
    lines = [ln for ln in path.read_text().splitlines() if ln]
    assert len(lines) == 2


# ============================================================ rotation
def test_rotation_kicks_in_when_size_exceeded(tmp_path: Path):
    """Rotation triggers when the *next* write would exceed max_bytes.
    Active file is replaced with a fresh empty one; rotated shard is
    gzip'd."""
    log = ReplayLog(tmp_path / "interactions.jsonl", max_bytes=200)
    payload = {"x": "a" * 50}
    # Each record line is ~60 bytes; ~3 records fill the threshold.
    for _ in range(10):
        log.record(payload)
    log.close()

    files = sorted(tmp_path.iterdir())
    # We expect at least one rotated .gz shard plus the active jsonl.
    gz_shards = [f for f in files if f.suffix == ".gz"]
    active_files = [f for f in files if f.name == "interactions.jsonl"]
    assert len(gz_shards) >= 1, files
    assert len(active_files) == 1

    # Each gz shard is valid gzip and contains JSON lines.
    for gz in gz_shards:
        with gzip.open(gz, "rt") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    json.loads(line)  # should not raise


def test_rotated_shard_filename_format(tmp_path: Path):
    log = ReplayLog(tmp_path / "interactions.jsonl", max_bytes=100)
    big = {"x": "a" * 200}
    log.record(big)
    log.record(big)
    log.close()

    gz_shards = list(tmp_path.glob("interactions-*.jsonl.gz"))
    assert len(gz_shards) >= 1


def test_rotation_disabled_when_max_bytes_zero(tmp_path: Path):
    log = ReplayLog(tmp_path / "x.jsonl", max_bytes=0)
    big = {"x": "a" * 1000}
    for _ in range(10):
        log.record(big)
    log.close()
    # Only the active file, no shards.
    assert list(tmp_path.glob("*.gz")) == []


def test_no_partial_lines_after_rotation(tmp_path: Path):
    """Every line in every shard + the active file must be valid JSON
    on its own — rotation must not split a record across files."""
    log = ReplayLog(tmp_path / "interactions.jsonl", max_bytes=300)
    for i in range(50):
        log.record({"i": i, "padding": "x" * 30})
    log.close()

    total_records = 0
    for f in tmp_path.iterdir():
        if f.suffix == ".gz":
            with gzip.open(f, "rt") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        json.loads(line)
                        total_records += 1
        elif f.name == "interactions.jsonl":
            for line in f.read_text().splitlines():
                if line:
                    json.loads(line)
                    total_records += 1
    assert total_records == 50


# ============================================================ encoder
def test_handles_dataclass_payload(tmp_path: Path):
    @dataclass
    class Payload:
        a: int
        b: str

    log = ReplayLog(tmp_path / "x.jsonl")
    log.record({"obj": Payload(a=1, b="z")})
    log.close()
    line = (tmp_path / "x.jsonl").read_text().strip()
    parsed = json.loads(line)
    assert parsed["obj"] == {"a": 1, "b": "z"}


def test_handles_tuple_payload(tmp_path: Path):
    log = ReplayLog(tmp_path / "x.jsonl")
    log.record({"items": (1, 2, 3)})
    log.close()
    parsed = json.loads((tmp_path / "x.jsonl").read_text().strip())
    assert parsed == {"items": [1, 2, 3]}


# ============================================================ thread safety
def test_concurrent_writes_dont_interleave(tmp_path: Path):
    """Two threads slamming record() must produce well-formed JSONL —
    no torn lines."""
    log = ReplayLog(tmp_path / "x.jsonl")

    def _writer(label: str):
        for i in range(100):
            log.record({"label": label, "i": i})

    t1 = threading.Thread(target=_writer, args=("A",))
    t2 = threading.Thread(target=_writer, args=("B",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    log.close()

    lines = [ln for ln in (tmp_path / "x.jsonl").read_text().splitlines() if ln]
    assert len(lines) == 200
    for line in lines:
        json.loads(line)  # well-formed


# ============================================================ default
def test_default_max_bytes_is_one_gb():
    assert DEFAULT_MAX_BYTES == 1024 * 1024 * 1024
