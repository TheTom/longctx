"""Tests for ``longctx_daemon.watch_cmd`` — live MCP-traffic tail."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from longctx_daemon.watch_cmd import (
    WatchFilter,
    _ClientPalette,
    _record_ts,
    iter_log_records,
    parse_filters,
    parse_since_seconds,
    render_record,
)


# --------------------------------------------------------------- duration

@pytest.mark.parametrize("s,seconds", [
    ("1h", 3600),
    ("30m", 30 * 60),
    ("5d", 5 * 86400),
    ("90s", 90),
    ("60", 60),   # bare integer = seconds
])
def test_parse_since_valid(s, seconds):
    assert parse_since_seconds(s) == seconds


def test_parse_since_invalid():
    with pytest.raises(ValueError):
        parse_since_seconds("9x")


# ------------------------------------------------------------- filter spec

def test_parse_filter_spec_full():
    f = parse_filters("client=opencode,tool=search_codebase,min_ms=50")
    assert f.client_name == "opencode"
    assert f.tool == "search_codebase"
    assert f.min_total_ms == 50.0


def test_parse_filter_spec_empty():
    f = parse_filters(None)
    assert f == WatchFilter()
    assert parse_filters("") == WatchFilter()


def test_filter_matches():
    f = WatchFilter(client_name="opencode", tool="search_codebase",
                    min_total_ms=50.0)
    rec_match = {
        "client": {"name": "opencode"}, "tool": "search_codebase",
        "latency_ms": {"total": 100.0},
    }
    rec_wrong_client = {
        "client": {"name": "claude"}, "tool": "search_codebase",
        "latency_ms": {"total": 100.0},
    }
    rec_too_fast = {
        "client": {"name": "opencode"}, "tool": "search_codebase",
        "latency_ms": {"total": 25.0},
    }
    assert f.matches(rec_match)
    assert not f.matches(rec_wrong_client)
    assert not f.matches(rec_too_fast)


def test_filter_only_errors():
    f = WatchFilter(only_errors=True)
    assert f.matches({"level": "ERROR"})
    assert f.matches({"level": "CRITICAL"})
    assert not f.matches({"level": "INFO"})


# --------------------------------------------------------------- render

def test_render_search_call_shows_query_and_files():
    palette = _ClientPalette()
    rec = {
        "ts": "2026-05-09T14:33:01Z",
        "component": "mcp",
        "client": {"name": "opencode", "version": "0.4.2"},
        "tool": "search_codebase",
        "args": {"query": "where is auth middleware"},
        "scope": {"primary_project": "myapp",
                  "primary_source": "cwd_walk_to_sentinel"},
        "result": {"chunk_count": 3, "is_fully_fresh": True,
                   "files": ["src/auth.py:1-20", "src/types.py:1-10"]},
        "latency_ms": {"total": 65.0},
    }
    out = render_record(rec, palette)
    assert "opencode/0.4.2" in out
    assert "search_codebase" in out
    assert "where is auth middleware" in out
    assert "scope=myapp" in out
    assert "3 chunks" in out
    assert "src/auth.py:1-20" in out
    assert "fresh" in out


def test_render_partial_freshness():
    palette = _ClientPalette()
    rec = {
        "ts": "2026-05-09T14:33:01Z",
        "component": "mcp",
        "client": {"name": "Pi", "version": "2.1.0"},
        "tool": "search_codebase",
        "args": {"query": "q"},
        "scope": {},
        "result": {"chunk_count": 0, "is_fully_fresh": False,
                   "pending_updates": 3},
        "latency_ms": {"total": 12.0},
    }
    out = render_record(rec, palette)
    assert "3 pending" in out


def test_render_verbose_shows_trace_and_latency():
    palette = _ClientPalette()
    rec = {
        "ts": "2026-05-09T14:33:01Z", "component": "mcp",
        "trace_id": "01H8XKY7M3...",
        "client": {"name": "x", "version": "1"},
        "tool": "search_codebase", "args": {"query": "q"},
        "scope": {},
        "result": {"chunk_count": 0, "is_fully_fresh": True,
                   "files": []},
        "latency_ms": {"total": 50, "embed_query": 10, "bm25_score": 5,
                       "dense_score": 25, "rrf_fuse": 5,
                       "fetch_chunks": 5},
    }
    out = render_record(rec, palette, verbose=True)
    assert "trace=01H8XKY7M3..." in out
    assert "embed=10" in out
    assert "dense=25" in out


def test_palette_is_stable_per_client():
    palette = _ClientPalette()
    a = palette.color_for("opencode")
    b = palette.color_for("Pi")
    a_again = palette.color_for("opencode")
    assert a == a_again
    assert a != b


# --------------------------------------------------------------- iter

def test_iter_log_no_follow(tmp_path):
    p = tmp_path / "longctx.log"
    records = [{"ts": "2026-05-09T14:33:01Z", "tool": "x"}]
    p.write_text(json.dumps(records[0]) + "\n")
    got = list(iter_log_records(p, follow=False))
    assert len(got) == 1


def test_record_ts_handles_z_suffix():
    """ISO with trailing Z parses correctly into epoch."""
    assert _record_ts({"ts": "2026-05-09T14:33:01Z"}) > 0
    assert _record_ts({}) == 0
    assert _record_ts({"ts": "garbage"}) == 0
