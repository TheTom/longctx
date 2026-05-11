"""Tests for the ``longctx status`` CLI command (PRD §10.2).

The command reads ``server.info`` (Agent F) and GETs ``/health``
(Agent D), then renders the §10.2 banner. We mock both inputs:

  * ``server.info``: write a tiny JSON file in tmp_path. ``read_server_info``
    validates the recorded PID via ``os.kill(pid, 0)`` so we use the
    current process's PID — guaranteed alive for the duration of the test.
  * ``/health``: monkeypatch the in-module HTTP fetcher.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from longctx_daemon import cli


# --------------------------------------------------- server.info helper
def _write_server_info(tmp_path: Path, **overrides) -> Path:
    """Drop a minimal server.info into tmp_path. Returns its path."""
    payload = {
        # Use os.getpid() so read_server_info's liveness check passes.
        "pid": os.getpid(),
        "started_at": "2026-05-09T10:00:00Z",
        "mcp_port": 8767,
        "mcp_transports": ["sse"],
        "status_port": 8766,
        "version": "0.2.0",
    }
    payload.update(overrides)
    info = tmp_path / "server.info"
    info.write_text(json.dumps(payload))
    return info


def _health_payload(**overrides) -> dict:
    base = {
        "status": "ready",
        "version": "0.2.0",
        "uptime_seconds": 3847,
        "projects": [
            {
                "name": "obsidian", "chunks": 762,
                "tokens_approx": 1_543_000,
                "last_updated": "2026-05-09T08:00:00Z",
            },
            {
                "name": "longctx", "chunks": 124,
                "tokens_approx": 248_000,
                "last_updated": "2026-05-09T13:50:00Z",
            },
        ],
        "watcher": {"queued": 0, "events_processed_total": 1284},
        "embedder": {
            "model": "bge-small-en-v1.5", "sha256": "a1b2c3d4e5f6",
        },
        "memory_rss_mb": 5872,
        "indexed_through": "2026-05-09T14:33:01Z",
    }
    base.update(overrides)
    return base


# --------------------------------------------------- not-running path
class TestNotRunning:
    def test_missing_server_info_prints_instructions(
        self, tmp_path, capsys, monkeypatch,
    ):
        rc = cli.main([
            "status", "--server-info", str(tmp_path / "no-such.info"),
        ])
        captured = capsys.readouterr()
        # rc != 0 so user knows daemon isn't running.
        assert rc == 1
        out = captured.out + captured.err
        assert "not running" in out.lower()

    def test_malformed_server_info_treated_as_missing(
        self, tmp_path, capsys, monkeypatch,
    ):
        bad = tmp_path / "server.info"
        bad.write_text("{not json")
        rc = cli.main(["status", "--server-info", str(bad)])
        captured = capsys.readouterr()
        assert rc == 1
        assert "not running" in (captured.out + captured.err).lower()


# --------------------------------------------------- running, /health up
class TestRunningHappyPath:
    def test_pretty_print_basic_layout(
        self, tmp_path, capsys, monkeypatch,
    ):
        info = _write_server_info(tmp_path)
        # Mock the HTTP fetcher.
        monkeypatch.setattr(
            cli, "_fetch_health", lambda host, port: _health_payload(),
        )
        rc = cli.main(["status", "--server-info", str(info)])
        out = capsys.readouterr().out
        assert rc == 0
        # PRD §10.2 banner header.
        assert "longctx daemon: running" in out
        assert f"PID {os.getpid()}" in out
        # Per-project rows.
        assert "obsidian" in out
        assert "longctx" in out
        # Watcher line.
        assert "Watcher" in out
        assert "1,284 events processed" in out
        # MCP transport line.
        assert "MCP server: sse" in out
        # Embedder line.
        assert "bge-small-en-v1.5" in out
        # Memory line — 5872 MB → ~5.7 GB
        assert "Memory" in out

    def test_json_mode(self, tmp_path, capsys, monkeypatch):
        info = _write_server_info(tmp_path)
        monkeypatch.setattr(
            cli, "_fetch_health", lambda host, port: _health_payload(),
        )
        rc = cli.main(["status", "--server-info", str(info), "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert "server_info" in payload
        assert "health" in payload
        assert payload["server_info"]["pid"] == os.getpid()
        assert payload["health"]["projects"][0]["name"] == "obsidian"

    def test_uptime_format(self, tmp_path, capsys, monkeypatch):
        info = _write_server_info(tmp_path)
        # 1h 4m = 3840s
        monkeypatch.setattr(
            cli, "_fetch_health",
            lambda host, port: _health_payload(uptime_seconds=3840),
        )
        cli.main(["status", "--server-info", str(info)])
        out = capsys.readouterr().out
        assert "uptime 1h 4m" in out

    def test_memory_under_1gb_shows_mb(self, tmp_path, capsys, monkeypatch):
        info = _write_server_info(tmp_path)
        monkeypatch.setattr(
            cli, "_fetch_health",
            lambda host, port: _health_payload(memory_rss_mb=512),
        )
        cli.main(["status", "--server-info", str(info)])
        out = capsys.readouterr().out
        assert "512 MB RSS" in out

    def test_no_projects(self, tmp_path, capsys, monkeypatch):
        info = _write_server_info(tmp_path)
        monkeypatch.setattr(
            cli, "_fetch_health",
            lambda host, port: _health_payload(projects=[]),
        )
        rc = cli.main(["status", "--server-info", str(info)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "(none)" in out


# --------------------------------------------------- running, /health down
class TestHealthUnreachable:
    def test_falls_back_to_server_info_summary(
        self, tmp_path, capsys, monkeypatch,
    ):
        info = _write_server_info(tmp_path)
        # Health fetch returns None → fallback path.
        monkeypatch.setattr(cli, "_fetch_health", lambda h, p: None)
        rc = cli.main(["status", "--server-info", str(info)])
        out = capsys.readouterr().out
        assert rc == 0  # not a hard failure
        assert "longctx daemon: running" in out
        assert "unreachable" in out


# --------------------------------------------------- helpers
class TestHelperFunctions:
    @pytest.mark.parametrize("seconds, expected", [
        (5, "5s"),
        (60, "1m"),
        (125, "2m"),
        (3700, "1h 1m"),
        (90000, "1d 1h"),
    ])
    def test_format_uptime(self, seconds, expected):
        assert cli._format_uptime(seconds) == expected

    @pytest.mark.parametrize("n, expected", [
        (5, "~5 tokens"),
        (1500, "~2K tokens"),
        (1_543_000, "~1.5M tokens"),
    ])
    def test_format_tokens(self, n, expected):
        assert cli._format_tokens(n) == expected

    def test_format_count_thousands(self):
        assert cli._format_count(1284) == "1,284"
        assert cli._format_count(0) == "0"

    def test_format_age_never_for_zero(self):
        assert cli._format_age(None) == "never"
        assert cli._format_age(0) == "never"

    def test_format_age_just_now_for_recent_unix(self):
        import time as _time
        assert cli._format_age(int(_time.time())) == "just now"

    def test_format_age_days_for_old_unix(self):
        import time as _time
        # 3 days ago.
        assert cli._format_age(int(_time.time()) - 3 * 86400) == "3d ago"

    def test_format_age_iso(self):
        import time as _time
        # Use a real recent ISO ts.
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # "just now" because it's NOW
        assert cli._format_age(ts) == "just now"


# --------------------------------------------------- HTTP fetcher
class TestFetchHealth:
    def test_returns_none_on_unreachable(self):
        # Pick a closed port (high number, unlikely to be in use)
        result = cli._fetch_health("127.0.0.1", 1)
        assert result is None
