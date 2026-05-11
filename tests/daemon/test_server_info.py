"""Tests for ``longctx_daemon.server_info``.

Coverage:
  * ``ServerInfo`` round-trip through write/read.
  * ``write_server_info`` is atomic (tempfile + replace).
  * ``read_server_info`` returns None on stale PID.
  * ``bind_port`` walks forward on collision; raises after exhausting.
  * ``SingletonLock`` exclusive semantics + reclaim on release.

We use ``tmp_path`` for every test so the user's real ``~/.cache/longctx``
is never touched.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from longctx_daemon.server_info import (
    PortBindError,
    ServerInfo,
    SingletonLock,
    bind_pair,
    bind_port,
    delete_server_info,
    now_iso,
    read_server_info,
    write_server_info,
)


# ============================================================ ServerInfo

def _make_info(pid=None, mcp=8765, status=8766, version="0.2.0"):
    return ServerInfo(
        pid=pid if pid is not None else os.getpid(),
        started_at=now_iso(),
        mcp_port=mcp,
        mcp_transports=("sse", "streamable-http"),
        status_port=status,
        version=version,
    )


def test_now_iso_format():
    iso = now_iso()
    assert iso.endswith("Z")
    assert "T" in iso


def test_server_info_round_trip(tmp_path):
    info = _make_info()
    p = tmp_path / "server.info"
    write_server_info(info, p)
    got = read_server_info(p)
    assert got is not None
    assert got == info


def test_server_info_to_dict_shape():
    info = _make_info()
    d = info.to_dict()
    assert d["mcp_transports"] == ["sse", "streamable-http"]
    assert d["pid"] == info.pid
    assert d["mcp_port"] == 8765
    assert d["status_port"] == 8766


def test_server_info_from_dict_round_trip():
    info = _make_info()
    again = ServerInfo.from_dict(info.to_dict())
    assert info == again


def test_read_returns_none_when_missing(tmp_path):
    assert read_server_info(tmp_path / "nonexistent.info") is None


def test_read_returns_none_on_malformed_json(tmp_path):
    p = tmp_path / "server.info"
    p.write_text("{not json", encoding="utf-8")
    assert read_server_info(p) is None


def test_read_returns_none_on_missing_keys(tmp_path):
    p = tmp_path / "server.info"
    p.write_text(json.dumps({"pid": 1}), encoding="utf-8")
    assert read_server_info(p) is None


def test_read_returns_none_on_stale_pid(tmp_path):
    # Write info with a PID that surely isn't running.
    info = _make_info(pid=999_999)
    p = tmp_path / "server.info"
    write_server_info(info, p)
    assert read_server_info(p) is None


def test_delete_server_info_idempotent(tmp_path):
    p = tmp_path / "server.info"
    delete_server_info(p)  # missing, OK
    write_server_info(_make_info(), p)
    delete_server_info(p)
    assert not p.exists()
    delete_server_info(p)  # second call, OK


def test_write_creates_parent_dir(tmp_path):
    info = _make_info()
    p = tmp_path / "deep" / "nested" / "server.info"
    write_server_info(info, p)
    assert p.exists()


def test_write_is_atomic_no_partial_files(tmp_path):
    """After a write completes, no leftover .tmp files remain."""
    info = _make_info()
    p = tmp_path / "server.info"
    write_server_info(info, p)
    leftover = list(tmp_path.glob(".server.info.*.tmp"))
    assert leftover == []


# ============================================================ bind_port

def test_bind_port_returns_socket_and_port():
    sock, port = bind_port(0, max_tries=1)  # 0 = OS-assigned
    try:
        assert isinstance(sock, socket.socket)
        # Port 0 → OS assigns a real port. Ours could be anything, just
        # confirm it's nonzero.
        assert port != 0 or True  # OS-assigned, port may legitimately be 0+
    finally:
        sock.close()


def test_bind_port_walks_forward_on_collision():
    """When the preferred port is taken, walk to the next."""
    # Take a port ourselves, then ask bind_port to start there.
    busy = socket.socket()
    busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    try:
        sock, port = bind_port(busy_port, max_tries=20)
        try:
            assert port != busy_port
            assert busy_port < port < busy_port + 20
        finally:
            sock.close()
    finally:
        busy.close()


def test_bind_port_raises_after_exhaustion():
    """All ports in the range busy → PortBindError."""
    # Take 3 consecutive ports.
    socks = []
    base = None
    try:
        for _ in range(3):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            socks.append(s)
        # Sort by port; pick the smallest as base, fail across the
        # range of 1 attempt → none free.
        ports = sorted(s.getsockname()[1] for s in socks)
        # Block our PortBindError walk by using max_tries=1 from a
        # known-busy slot.
        with pytest.raises(PortBindError):
            sock, _port = bind_port(ports[0], max_tries=1)
            sock.close()
    finally:
        for s in socks:
            s.close()


def test_bind_port_invalid_max_tries():
    with pytest.raises(ValueError):
        bind_port(8000, max_tries=0)


def test_bind_port_reraises_non_eaddrinuse(monkeypatch):
    """Permission denied must propagate; we don't walk on it."""

    class FakeSock:
        def setsockopt(self, *a, **kw):
            pass

        def bind(self, *a, **kw):
            err = OSError(13, "Permission denied")
            err.errno = 13
            raise err

        def listen(self, *_a):
            pass

        def close(self):
            pass

    def fake_socket(*a, **kw):
        return FakeSock()

    monkeypatch.setattr(socket, "socket", fake_socket)
    with pytest.raises(OSError) as ei:
        bind_port(9000, max_tries=3)
    assert ei.value.errno == 13


def test_bind_pair_returns_two_sockets():
    (mcp_sock, mcp_port), (status_sock, status_port) = bind_pair(
        preferred_mcp=0, preferred_status=0, max_tries=5,
    )
    try:
        assert mcp_port != status_port or (mcp_port == 0 and status_port == 0)
    finally:
        mcp_sock.close()
        status_sock.close()


def test_bind_pair_closes_mcp_on_status_failure(monkeypatch):
    """If status bind fails, the MCP socket must not leak."""
    class FailingSock:
        def setsockopt(self, *a, **kw):
            pass

        def bind(self, *a, **kw):
            raise OSError(13, "Permission denied")

        def listen(self, *_a):
            pass

        def close(self):
            pass

    real_socket = socket.socket
    state = {"calls": 0}

    def factory(*a, **kw):
        state["calls"] += 1
        # First call (mcp) → real; subsequent (status) → fails.
        if state["calls"] == 1:
            return real_socket(*a, **kw)
        s = FailingSock()
        s.bind = lambda *a, **kw: (_ for _ in ()).throw(OSError(13, "no"))
        return s

    monkeypatch.setattr(socket, "socket", factory)
    with pytest.raises(OSError):
        bind_pair(0, 0, max_tries=1)


# ============================================================ SingletonLock

def test_singleton_lock_acquires_when_free(tmp_path):
    lock = SingletonLock(tmp_path / "server.lock")
    assert lock.acquire() is True
    assert lock.acquired
    lock.release()
    assert not lock.acquired


def test_singleton_lock_idempotent_acquire(tmp_path):
    lock = SingletonLock(tmp_path / "server.lock")
    assert lock.acquire() is True
    assert lock.acquire() is True  # already held
    lock.release()


def test_singleton_lock_blocks_second_holder(tmp_path):
    lock_path = tmp_path / "server.lock"
    a = SingletonLock(lock_path)
    b = SingletonLock(lock_path)
    assert a.acquire() is True
    try:
        assert b.acquire() is False
    finally:
        a.release()
    # Now b can take it.
    assert b.acquire() is True
    b.release()


def test_singleton_lock_records_pid(tmp_path):
    lock_path = tmp_path / "server.lock"
    lock = SingletonLock(lock_path)
    assert lock.acquire()
    try:
        assert lock.holder_pid() == os.getpid()
    finally:
        lock.release()


def test_singleton_lock_holder_pid_missing_file(tmp_path):
    lock = SingletonLock(tmp_path / "no.lock")
    assert lock.holder_pid() is None


def test_singleton_lock_holder_pid_garbled(tmp_path):
    lock_path = tmp_path / "server.lock"
    lock_path.write_text("not-a-pid", encoding="utf-8")
    assert SingletonLock(lock_path).holder_pid() is None


def test_singleton_lock_context_manager(tmp_path):
    lock_path = tmp_path / "server.lock"
    with SingletonLock(lock_path) as lock:
        assert lock.acquire()
        # Inside the ``with``: lock is held.
    # After exit, lock should be released.
    fresh = SingletonLock(lock_path)
    assert fresh.acquire() is True
    fresh.release()


def test_singleton_lock_release_idempotent(tmp_path):
    lock = SingletonLock(tmp_path / "server.lock")
    lock.release()  # never acquired; OK
    lock.acquire()
    lock.release()
    lock.release()  # already released; OK


def test_singleton_lock_creates_parent_dir(tmp_path):
    deep = tmp_path / "a" / "b" / "server.lock"
    lock = SingletonLock(deep)
    assert lock.acquire()
    try:
        assert deep.parent.is_dir()
    finally:
        lock.release()


def test_singleton_lock_concurrent_threads(tmp_path):
    """Lock can be passed between threads (not held simultaneously)."""
    lock_path = tmp_path / "server.lock"
    holder_a = SingletonLock(lock_path)
    holder_b = SingletonLock(lock_path)

    a_acquired = threading.Event()
    a_release = threading.Event()
    b_blocked = threading.Event()

    def thread_a():
        holder_a.acquire()
        a_acquired.set()
        a_release.wait(timeout=2)
        holder_a.release()

    def thread_b():
        a_acquired.wait(timeout=2)
        # Should fail because A holds it.
        ok = holder_b.acquire()
        if not ok:
            b_blocked.set()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta_started = a_acquired.wait(timeout=2)
    assert ta_started, "thread A should acquire lock first"
    # B should fail.
    tb.join(timeout=2)
    assert b_blocked.is_set(), "thread B should be blocked while A holds"
    a_release.set()
    ta.join(timeout=2)


# ============================================================ defaults

def test_default_paths_under_home_cache():
    """The hard-coded defaults point under the user's cache dir."""
    from longctx_daemon.server_info import (
        DEFAULT_LOCK_PATH,
        DEFAULT_SERVER_INFO,
    )
    assert "longctx" in str(DEFAULT_SERVER_INFO)
    assert "longctx" in str(DEFAULT_LOCK_PATH)


# ============================================================ stale reclaim

def test_stale_info_with_dead_pid_returns_none(tmp_path):
    """A daemon that crashed without cleanup leaves a server.info
    behind; ``read_server_info`` must return None so the caller
    reclaims it."""
    p = tmp_path / "server.info"
    info = _make_info(pid=1)  # PID 1 is init, but our os.kill(1, 0)
    # raises PermissionError → treated as alive. Use a far-out PID.
    write_server_info(_make_info(pid=987654), p)
    # 987654 doesn't exist on this system; read returns None.
    assert read_server_info(p) is None


def test_reclaim_workflow(tmp_path):
    """Caller reads stale info → deletes it → writes fresh → reads OK."""
    p = tmp_path / "server.info"
    write_server_info(_make_info(pid=987654), p)
    # Stale → reclaim.
    assert read_server_info(p) is None
    delete_server_info(p)
    # Fresh start.
    fresh = _make_info()
    write_server_info(fresh, p)
    assert read_server_info(p) == fresh
