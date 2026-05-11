"""Tests for ``longctx_daemon.daemon``.

Coverage:
  * ``DaemonConfig`` validation.
  * ``Daemon.run`` writes server.info, takes the lock, runs transports,
    cleans up on shutdown.
  * Reload + reindex hooks fire on the corresponding event flags.
  * Singleton refusal: a second Daemon under the same lock fails fast.
  * Cleanup-on-error: if the HTTP runner crashes, lock + server.info
    are still cleaned up.

We avoid driving real signals (pytest's main thread already has
handlers installed); instead we drive ``request_shutdown`` /
``request_reload`` directly.
"""
from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from longctx_daemon.daemon import (
    DEFAULT_MCP_PORT,
    DEFAULT_STATUS_PORT,
    Daemon,
    DaemonConfig,
)
from longctx_daemon.mcp_server import MCPServer
from longctx_daemon.server_info import (
    SingletonLock,
    read_server_info,
)


# ============================================================ fakes
@dataclass
class _FakeSearcher:
    def search(self, **kwargs):
        from longctx_daemon.types import (
            LatencyBreakdown,
            ScopeDecision,
            SearchFreshness,
            SearchResult,
        )
        return SearchResult(
            chunks=(),
            scope_decision=ScopeDecision(
                primary_project=None,
                primary_source="none",
                fanout_projects=(),
                fanout_reason="none",
                ambiguity_score=0.0,
            ),
            freshness=SearchFreshness(
                is_fully_fresh=True, stale_files=(), pending_updates=0,
                indexed_through="2026-05-09T00:00:00Z",
            ),
            latency_ms=LatencyBreakdown(
                total=1.0, embed_query=0.0, bm25_score=0.0,
                dense_score=0.0, rrf_fuse=0.0, fetch_chunks=0.0,
            ),
        )


@dataclass
class _FakeIndexer:
    def status(self):
        from longctx_daemon.types import IndexStatus
        return IndexStatus(
            status="ready", total_chunks=0, pending_updates=0,
            embedder_model="fake", embedder_sha256="fake",
            last_full_scan="2026-05-09T00:00:00Z", projects=(),
        )


class _FakeChunkStore:
    def list_projects(self):
        return ()

    def list_files(self, project=None):
        return ()

    def get_chunks_by_file(self, file_id):
        return ()

    def close(self):
        pass


def _make_server() -> MCPServer:
    import inspect
    sig = inspect.signature(MCPServer.__init__)
    kwargs = dict(
        searcher=_FakeSearcher(), indexer=_FakeIndexer(),
        chunk_store=_FakeChunkStore(),
    )
    if "embed_store" in sig.parameters:
        kwargs["embed_store"] = None
    return MCPServer(**kwargs)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _make_config(tmp_path: Path, transports=("sse",)) -> DaemonConfig:
    return DaemonConfig(
        host="127.0.0.1",
        mcp_port=_free_port(),
        status_port=_free_port(),
        transports=transports,
        lock_path=tmp_path / "server.lock",
        server_info_path=tmp_path / "server.info",
        shutdown_grace_seconds=2.0,
    )


# ============================================================ DaemonConfig
def test_daemon_config_defaults_are_localhost():
    cfg = DaemonConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.mcp_port == DEFAULT_MCP_PORT
    assert cfg.status_port == DEFAULT_STATUS_PORT
    assert "sse" in cfg.transports


def test_daemon_config_rejects_unknown_transport():
    with pytest.raises(ValueError):
        DaemonConfig(transports=("nope",))


def test_daemon_config_accepts_all_three_transports():
    cfg = DaemonConfig(transports=("sse", "streamable-http", "stdio"))
    assert "stdio" in cfg.transports


# ============================================================ Daemon lifecycle
@pytest.mark.asyncio
async def test_daemon_run_writes_server_info_and_releases(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    server = _make_server()
    daemon = Daemon(config=cfg, mcp_server=server)

    # Run in background; trigger shutdown after server.info appears.
    task = asyncio.create_task(daemon.run())
    try:
        # Wait for server.info to be written.
        for _ in range(200):
            await asyncio.sleep(0.05)
            if cfg.server_info_path.exists():
                break
        else:
            pytest.fail("server.info never written")

        info = read_server_info(cfg.server_info_path)
        assert info is not None
        assert info.pid == os.getpid()
        assert info.mcp_port == cfg.mcp_port
        assert info.status_port == cfg.status_port

        daemon.request_shutdown()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # Cleanup must have run.
    assert not cfg.server_info_path.exists()
    # Lock should be released.
    fresh = SingletonLock(cfg.lock_path)
    assert fresh.acquire() is True
    fresh.release()


@pytest.mark.asyncio
async def test_daemon_refuses_concurrent_instance(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    # Take the lock in another holder before Daemon.run.
    blocker = SingletonLock(cfg.lock_path)
    assert blocker.acquire()
    try:
        daemon = Daemon(config=cfg, mcp_server=_make_server())
        with pytest.raises(RuntimeError, match="already running"):
            await daemon.run()
    finally:
        blocker.release()


@pytest.mark.asyncio
async def test_daemon_request_shutdown_idempotent(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    daemon = Daemon(config=cfg, mcp_server=_make_server())
    daemon.request_shutdown()
    daemon.request_shutdown()  # noop
    assert daemon._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_daemon_request_reload_fires_callback(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    server = _make_server()

    reload_calls = []

    async def _on_reload():
        reload_calls.append(True)

    daemon = Daemon(
        config=cfg, mcp_server=server, on_reload=_on_reload,
    )
    task = asyncio.create_task(daemon.run())
    try:
        # Wait for ready
        for _ in range(200):
            await asyncio.sleep(0.05)
            if cfg.server_info_path.exists():
                break
        # Trigger reload.
        daemon.request_reload()
        # Give the watcher a chance to run.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if reload_calls:
                break
        assert reload_calls, "on_reload callback should have fired"
    finally:
        daemon.request_shutdown()
        with patch.object(daemon, "_on_reload", None):
            pass
        try:
            await asyncio.wait_for(task, timeout=10)
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_daemon_request_reindex_fires_callback(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    server = _make_server()

    reindex_calls = []

    async def _on_reindex():
        reindex_calls.append(True)

    daemon = Daemon(
        config=cfg, mcp_server=server, on_reindex=_on_reindex,
    )
    task = asyncio.create_task(daemon.run())
    try:
        for _ in range(200):
            await asyncio.sleep(0.05)
            if cfg.server_info_path.exists():
                break
        daemon.request_reindex()
        for _ in range(50):
            await asyncio.sleep(0.05)
            if reindex_calls:
                break
        assert reindex_calls
    finally:
        daemon.request_shutdown()
        try:
            await asyncio.wait_for(task, timeout=10)
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_daemon_cleanup_on_error_path(tmp_path):
    """If startup raises after binding, the lock + server.info must still
    get cleaned up.

    We poison the bind step so run() exits before any transport task
    starts, then check that a fresh daemon can take the lock.
    """
    cfg = _make_config(tmp_path, transports=("sse",))
    server = _make_server()
    daemon = Daemon(config=cfg, mcp_server=server)

    # Force ``write_server_info`` to throw; the AsyncExitStack must
    # still release the lock + close the bound sockets.
    with patch(
        "longctx_daemon.daemon.write_server_info",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await daemon.run()

    # Lock must be releasable now.
    fresh = SingletonLock(cfg.lock_path)
    assert fresh.acquire() is True
    fresh.release()


@pytest.mark.asyncio
async def test_daemon_server_info_property(tmp_path):
    cfg = _make_config(tmp_path, transports=("sse",))
    daemon = Daemon(config=cfg, mcp_server=_make_server())
    assert daemon.server_info is None
    assert not daemon.is_running

    task = asyncio.create_task(daemon.run())
    try:
        for _ in range(200):
            await asyncio.sleep(0.05)
            if daemon.server_info is not None:
                break
        assert daemon.server_info is not None
        assert daemon.is_running
    finally:
        daemon.request_shutdown()
        try:
            await asyncio.wait_for(task, timeout=10)
        except (asyncio.CancelledError, Exception):
            pass


# ============================================================ ports
@pytest.mark.asyncio
async def test_daemon_records_actual_ports_from_walk(tmp_path):
    """When the preferred MCP port is busy, Daemon walks forward and
    server.info reflects the *actual* port."""
    busy = socket.socket()
    busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    try:
        cfg = DaemonConfig(
            host="127.0.0.1",
            mcp_port=busy_port,
            status_port=_free_port(),
            transports=("sse",),
            lock_path=tmp_path / "server.lock",
            server_info_path=tmp_path / "server.info",
            shutdown_grace_seconds=2.0,
            max_port_walk=20,
        )
        daemon = Daemon(config=cfg, mcp_server=_make_server())
        task = asyncio.create_task(daemon.run())
        try:
            for _ in range(200):
                await asyncio.sleep(0.05)
                if cfg.server_info_path.exists():
                    break
            info = read_server_info(cfg.server_info_path)
            assert info is not None
            assert info.mcp_port != busy_port
            assert busy_port < info.mcp_port < busy_port + cfg.max_port_walk
        finally:
            daemon.request_shutdown()
            try:
                await asyncio.wait_for(task, timeout=10)
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        busy.close()


# ============================================================ DaemonConfig
def test_daemon_config_post_init_validates_lock_path(tmp_path):
    """Path types are accepted; conversion happens lazily via SingletonLock."""
    cfg = DaemonConfig(
        lock_path=tmp_path / "lock",
        server_info_path=tmp_path / "info",
    )
    assert isinstance(cfg.lock_path, Path)


@pytest.mark.asyncio
async def test_daemon_no_transports_raises(tmp_path):
    """A DaemonConfig with transports=() is rejected at construction."""
    with pytest.raises(ValueError):
        DaemonConfig(transports=())


# ============================================================ pytest config
@pytest.fixture(scope="module")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
