"""Tests for the shared sidecar manager.

Engine adapters (vllm-swift, vllm-turboquant, llama-server-longctx
wrapper) all call into this when --enable-longctx is set. If this
breaks, the one-command UX breaks for all three.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request

import pytest

from longctx_svc.sidecar import (
    PortInUseError,
    Sidecar,
    _free_port,
    _wait_healthy,
    is_port_in_use,
    managed_sidecar,
    pick_port,
    spawn_sidecar,
)


def test_free_port_returns_int_in_range():
    p = _free_port(start=12000)
    assert 12000 <= p <= 65535


def test_wait_healthy_timeout_when_dead():
    assert _wait_healthy("http://127.0.0.1:1", timeout=1.0) is False


def test_spawn_and_stop_roundtrip(tmp_path):
    log_path = tmp_path / "sidecar.log"
    sc = spawn_sidecar(
        cache_dir=str(tmp_path / "cache"),
        log_path=str(log_path),
        boot_timeout=20.0,
        extra_env={"LONGCTX_NO_JANITOR": "1"},
    )
    try:
        assert isinstance(sc, Sidecar)
        # /healthz reachable
        with urllib.request.urlopen(f"{sc.url}/healthz", timeout=3.0) as r:
            data = json.loads(r.read())
        assert data["status"] == "ok"
        # cache dir env was honored — status text reports it
        req = urllib.request.Request(
            f"{sc.url}/longctx/status",
            headers={"accept": "text/plain"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as r:
            text = r.read().decode()
        assert str(tmp_path / "cache") in text
    finally:
        sc.stop()
    # After stop, port should be free again (give the kernel a moment)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{sc.url}/healthz", timeout=0.5)
        except Exception:  # noqa: BLE001
            return
        time.sleep(0.1)
    pytest.fail("sidecar still answering after stop()")


def test_stop_is_idempotent(tmp_path):
    sc = spawn_sidecar(
        cache_dir=str(tmp_path / "c"),
        boot_timeout=20.0,
        extra_env={"LONGCTX_NO_JANITOR": "1"},
    )
    sc.stop()
    sc.stop()  # must not raise


def test_managed_sidecar_context(tmp_path):
    """Ensures cleanup runs even if the body raises."""
    captured_url = ""
    raised = False
    try:
        with managed_sidecar(
            cache_dir=str(tmp_path / "c"),
            boot_timeout=20.0,
            extra_env={"LONGCTX_NO_JANITOR": "1"},
        ) as sc:
            captured_url = sc.url
            # Sanity: alive
            with urllib.request.urlopen(f"{sc.url}/healthz",
                                        timeout=3.0) as r:
                assert r.status == 200
            raise RuntimeError("simulated engine crash")
    except RuntimeError:
        raised = True
    assert raised
    assert captured_url
    # Verify cleanup actually happened
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{captured_url}/healthz", timeout=0.5)
        except Exception:  # noqa: BLE001
            return
        time.sleep(0.1)
    pytest.fail("managed_sidecar did not stop the subprocess on exception")


def test_spawn_picks_free_port_when_default_taken(tmp_path):
    """Run two sidecars: the second must land on a different port."""
    sc1 = spawn_sidecar(
        cache_dir=str(tmp_path / "c1"),
        boot_timeout=20.0,
        extra_env={"LONGCTX_NO_JANITOR": "1"},
    )
    sc2 = spawn_sidecar(
        cache_dir=str(tmp_path / "c2"),
        boot_timeout=20.0,
        extra_env={"LONGCTX_NO_JANITOR": "1"},
    )
    try:
        assert sc1.port != sc2.port
    finally:
        sc1.stop()
        sc2.stop()


def test_spawn_failure_raises_runtime_error(tmp_path):
    """Force boot timeout by handing in a python that exits immediately."""
    fake_python = tmp_path / "fake-python.sh"
    fake_python.write_text("#!/bin/sh\nexit 1\n")
    fake_python.chmod(0o755)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        spawn_sidecar(
            python=str(fake_python),
            cache_dir=str(tmp_path / "c"),
            boot_timeout=2.0,
            extra_env={"LONGCTX_NO_JANITOR": "1"},
        )


# ---------------------------------------------------------------------------
# Port collision detection (Tom: "check if port is in use")
# ---------------------------------------------------------------------------

def _bind_port_blocker(port: int):
    """Bind a socket to `port` and return the socket so it stays bound."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def test_is_port_in_use_false_for_free_port():
    p = _free_port(start=33000)
    assert is_port_in_use(p) is False


def test_is_port_in_use_true_for_bound_port():
    s = _bind_port_blocker(0)
    port = s.getsockname()[1]
    try:
        assert is_port_in_use(port) is True
    finally:
        s.close()


def test_pick_port_returns_preferred_when_free():
    # Find a definitely-free port, then ask pick_port to take it.
    p = _free_port(start=33500)
    assert pick_port(preferred=p) == p


def test_pick_port_raises_when_preferred_taken():
    s = _bind_port_blocker(0)
    port = s.getsockname()[1]
    try:
        with pytest.raises(PortInUseError) as ei:
            pick_port(preferred=port)
        # Error message must guide the user.
        msg = str(ei.value)
        assert str(port) in msg
        assert "in use" in msg
        assert "lsof" in msg or "--port" in msg
    finally:
        s.close()


def test_pick_port_auto_when_preferred_none():
    p = pick_port(preferred=None, start=34000)
    assert p >= 34000


def test_spawn_sidecar_raises_port_in_use_immediately(tmp_path):
    """Pre-flight check fires BEFORE the long subprocess boot, so we
    don't burn 30s on a doomed health check."""
    s = _bind_port_blocker(0)
    port = s.getsockname()[1]
    try:
        with pytest.raises(PortInUseError):
            spawn_sidecar(
                port=port,
                cache_dir=str(tmp_path / "c"),
                extra_env={"LONGCTX_NO_JANITOR": "1"},
            )
    finally:
        s.close()
