"""CLI tests for Phase 2.1 daemon-aware subcommands.

We don't stand up a real daemon — instead we write a fake
``server.info`` and assert each subcommand reads it correctly.

Covers:
  * ``longctx port mcp`` / ``longctx port status``
  * ``longctx status``
  * ``longctx reload``  (sends SIGHUP via ``os.kill``; mocked)
  * ``longctx stop``    (sends SIGTERM via ``os.kill``; mocked)
  * ``longctx mcp-stdio --check`` (verifies daemon reachable)
  * ``longctx serve --transports`` arg parsing
  * ``longctx serve --daemon`` arg parsing

The mcp-stdio bridge itself spins up an asyncio loop with the SDK; we
exercise the *resolution* of server.info (the failure path is the most
common; the success path goes via --check which doesn't bridge).
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from longctx_daemon import cli
from longctx_daemon.server_info import ServerInfo, now_iso, write_server_info


def _write_fake_info(
    tmp_path: Path,
    *,
    pid: int = None,
    mcp_port: int = 8765,
    status_port: int = 8766,
    transports: tuple[str, ...] = ("sse", "streamable-http"),
    version: str = "0.2.0",
) -> Path:
    info = ServerInfo(
        pid=pid if pid is not None else os.getpid(),
        started_at=now_iso(),
        mcp_port=mcp_port,
        mcp_transports=transports,
        status_port=status_port,
        version=version,
    )
    info_path = tmp_path / "server.info"
    write_server_info(info, info_path)
    return info_path


# ============================================================ port
def test_port_no_daemon(tmp_path, capsys):
    info_path = tmp_path / "server.info"
    rc = cli.main(["port", "mcp", "--server-info", str(info_path)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "no running" in err.lower()


def test_port_mcp_default(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path, mcp_port=12345)
    rc = cli.main(["port", "--server-info", str(info_path)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == "12345"


def test_port_mcp_explicit(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path, mcp_port=12345)
    rc = cli.main(["port", "mcp", "--server-info", str(info_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "12345"


def test_port_status(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path, status_port=99999)
    rc = cli.main(["port", "status", "--server-info", str(info_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "99999"


def test_port_invalid_choice(tmp_path):
    info_path = _write_fake_info(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["port", "garbage", "--server-info", str(info_path)])


# ============================================================ status
def test_status_no_daemon(tmp_path, capsys):
    info_path = tmp_path / "server.info"
    rc = cli.main(["status", "--server-info", str(info_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "not running" in out.lower()


def test_status_running(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path, mcp_port=8765)
    rc = cli.main(["status", "--server-info", str(info_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "running" in out.lower()
    assert "8765" in out


def test_status_shows_transports(tmp_path, capsys):
    info_path = _write_fake_info(
        tmp_path, transports=("sse", "streamable-http"),
    )
    cli.main(["status", "--server-info", str(info_path)])
    out = capsys.readouterr().out
    assert "sse" in out
    assert "streamable-http" in out


def test_status_shows_pid(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path)
    cli.main(["status", "--server-info", str(info_path)])
    out = capsys.readouterr().out
    assert str(os.getpid()) in out


# ============================================================ reload
def test_reload_no_daemon(tmp_path, capsys):
    rc = cli.main(["reload", "--server-info", str(tmp_path / "no.info")])
    assert rc == 3


def test_reload_sends_sighup(tmp_path, capsys, monkeypatch):
    info_path = _write_fake_info(tmp_path)
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        # Filter out the liveness probe (sig=0) used by ``_pid_alive``.
        if sig != 0:
            sent.append((pid, sig))

    monkeypatch.setattr("longctx_daemon.cli.os.kill", fake_kill)
    monkeypatch.setattr(
        "longctx_daemon.server_info.os.kill", fake_kill,
    )
    rc = cli.main(["reload", "--server-info", str(info_path)])
    assert rc == 0
    assert sent == [(os.getpid(), signal.SIGHUP)]


def test_reload_handles_dead_pid(tmp_path, capsys, monkeypatch):
    info_path = _write_fake_info(tmp_path)

    def fake_kill(pid, sig):
        # First call: liveness check returns OK (raised by ProcessLookupError
        # would mean dead — we let it pass). Second call: actual kill fails.
        raise ProcessLookupError()

    # Monkey-patch only the module-level os.kill used in _cmd_reload.
    monkeypatch.setattr(
        "longctx_daemon.cli.os.kill", fake_kill,
    )
    # We also need read_server_info to think it's alive — patch _pid_alive.
    monkeypatch.setattr(
        "longctx_daemon.server_info._pid_alive", lambda pid: True,
    )
    rc = cli.main(["reload", "--server-info", str(info_path)])
    assert rc == 3


# ============================================================ stop
def test_stop_no_daemon(tmp_path):
    rc = cli.main(["stop", "--server-info", str(tmp_path / "no.info")])
    assert rc == 3


def test_stop_sends_sigterm(tmp_path, capsys, monkeypatch):
    info_path = _write_fake_info(tmp_path)
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        if sig != 0:
            sent.append((pid, sig))

    monkeypatch.setattr("longctx_daemon.cli.os.kill", fake_kill)
    monkeypatch.setattr(
        "longctx_daemon.server_info.os.kill", fake_kill,
    )
    rc = cli.main(["stop", "--server-info", str(info_path)])
    assert rc == 0
    assert sent == [(os.getpid(), signal.SIGTERM)]


def test_stop_permission_denied(tmp_path, monkeypatch):
    info_path = _write_fake_info(tmp_path)

    def fake_kill(pid, sig):
        raise PermissionError("nope")

    monkeypatch.setattr("longctx_daemon.cli.os.kill", fake_kill)
    monkeypatch.setattr(
        "longctx_daemon.server_info._pid_alive", lambda pid: True,
    )
    rc = cli.main(["stop", "--server-info", str(info_path)])
    assert rc == 3


# ============================================================ mcp-stdio
def test_mcp_stdio_check_no_daemon(tmp_path, capsys):
    rc = cli.main([
        "mcp-stdio", "--check",
        "--server-info", str(tmp_path / "no.info"),
    ])
    assert rc == 3


def test_mcp_stdio_check_succeeds(tmp_path, capsys):
    info_path = _write_fake_info(tmp_path, mcp_port=12345)
    rc = cli.main([
        "mcp-stdio", "--check", "--server-info", str(info_path),
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "12345" in err


def test_mcp_stdio_rejects_daemon_without_sse(tmp_path, capsys):
    """If the running daemon only exposes streamable-http we can't bridge."""
    info_path = _write_fake_info(
        tmp_path, transports=("streamable-http",),
    )
    rc = cli.main([
        "mcp-stdio", "--server-info", str(info_path),
    ])
    assert rc == 4
    err = capsys.readouterr().err
    assert "sse" in err.lower()


# ============================================================ serve --transports
def test_serve_parser_accepts_transports():
    parser = cli.build_parser()
    args = parser.parse_args([
        "serve", "--corpus-dir", "/tmp/foo",
        "--transports", "sse,streamable-http",
    ])
    assert args.transports == "sse,streamable-http"


def test_serve_parser_accepts_daemon_flag():
    parser = cli.build_parser()
    args = parser.parse_args([
        "serve", "--corpus-dir", "/tmp/foo", "--daemon",
    ])
    assert args.daemon is True


def test_serve_parser_accepts_port():
    parser = cli.build_parser()
    args = parser.parse_args([
        "serve", "--corpus-dir", "/tmp/foo", "--port", "9876",
    ])
    assert args.port == 9876


def test_serve_parser_accepts_host():
    parser = cli.build_parser()
    args = parser.parse_args([
        "serve", "--corpus-dir", "/tmp/foo", "--host", "0.0.0.0",
    ])
    assert args.host == "0.0.0.0"


def test_serve_parser_status_port():
    parser = cli.build_parser()
    args = parser.parse_args([
        "serve", "--corpus-dir", "/tmp/foo", "--status-port", "9999",
    ])
    assert args.status_port == 9999


def test_serve_parser_default_daemon_false():
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "--corpus-dir", "/tmp/foo"])
    assert args.daemon is False


# ============================================================ subcommand registry
def test_all_daemon_subcommands_registered():
    parser = cli.build_parser()
    actions = [
        a for a in parser._actions
        if hasattr(a, "choices") and a.choices is not None
    ]
    sub_choices = set(actions[0].choices)
    assert "port" in sub_choices
    assert "status" in sub_choices
    assert "reload" in sub_choices
    assert "stop" in sub_choices
    assert "mcp-stdio" in sub_choices
