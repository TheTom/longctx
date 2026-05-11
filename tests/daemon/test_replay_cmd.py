"""Tests for ``longctx_daemon.replay`` — captured-call replay."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from longctx_daemon.replay import (
    ReplayCall,
    ReplayDelta,
    diff_top_k,
    iter_replay_log,
)


# ------------------------------------------------------------- iter

def test_iter_jsonl(tmp_path):
    p = tmp_path / "interactions.jsonl"
    records = [
        {"trace_id": "1", "tool": "search_codebase",
         "args": {"query": "foo"}, "result": {"chunks": []}},
        {"trace_id": "2", "tool": "list_projects", "args": {}, "result": {}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = list(iter_replay_log(p))
    assert len(out) == 2
    assert out[0].trace_id == "1"
    assert out[0].tool == "search_codebase"
    assert out[1].tool == "list_projects"


def test_iter_skips_blank_and_malformed(tmp_path, capsys):
    p = tmp_path / "interactions.jsonl"
    p.write_text(
        '{"trace_id":"1","tool":"x","args":{},"result":{}}\n'
        '\n'
        'not valid json\n'
        '{"trace_id":"2","tool":"y","args":{},"result":{}}\n'
    )
    out = list(iter_replay_log(p))
    captured = capsys.readouterr()
    assert {c.trace_id for c in out} == {"1", "2"}
    assert "skip line 3" in captured.err


def test_iter_handles_gzipped(tmp_path):
    p = tmp_path / "interactions.jsonl.gz"
    rec = {"trace_id": "g", "tool": "search_codebase",
           "args": {}, "result": {}}
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    out = list(iter_replay_log(p))
    assert len(out) == 1
    assert out[0].trace_id == "g"


# ----------------------------------------------------- diff_top_k

def test_diff_no_change():
    same = {
        "chunks": [
            {"file_path": "a.py", "start_line": 1, "end_line": 10,
             "relevance_score": 0.5},
        ],
        "_trace_id": "t",
        "query": "q",
    }
    delta = diff_top_k(same, same)
    assert delta.rank_changes == 0
    assert delta.score_drift == 0.0


def test_diff_rank_shifted():
    captured = {
        "chunks": [
            {"file_path": "a.py", "start_line": 1, "end_line": 10,
             "relevance_score": 0.5},
            {"file_path": "b.py", "start_line": 1, "end_line": 5,
             "relevance_score": 0.4},
        ],
        "_trace_id": "t", "query": "q",
    }
    current = {
        "chunks": [
            {"file_path": "b.py", "start_line": 1, "end_line": 5,
             "relevance_score": 0.6},
            {"file_path": "a.py", "start_line": 1, "end_line": 10,
             "relevance_score": 0.5},
        ],
    }
    delta = diff_top_k(captured, current)
    assert delta.rank_changes == 2
    # b.py moved from 0.4 → 0.6 → drift 0.2; a.py unchanged
    assert delta.score_drift == pytest.approx(0.2, abs=1e-6)


def test_diff_handles_empty():
    delta = diff_top_k({}, {})
    assert delta.rank_changes == 0
    assert delta.score_drift == 0.0
    assert delta.captured_top_files == ()
    assert delta.current_top_files == ()
