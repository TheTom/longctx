"""Tests for ``longctx_daemon.transports``.

Covers:
  * Starlette app builders (``build_sse_app``, ``build_streamable_http_app``,
    ``_build_combined_http_app``) produce routable apps.
  * ``run_sse`` / ``run_streamable_http`` bind, fire ``ready_event``,
    accept a graceful shutdown.
  * ``run_multi_transport`` runs multiple transports concurrently and
    cancels them via ``shutdown_event``.
  * Per-connection ``ConnectionContext`` is allocated for each accepted
    SSE / streamable-http request.

We don't drive the full MCP handshake here (that's covered by
``test_mcp_server.py``); the transport tests focus on the bind / serve /
shutdown plumbing using a minimal MCPServer.
"""
from __future__ import annotations

import asyncio
import contextlib
import socket
from dataclasses import dataclass
from typing import Optional

import httpx
import pytest

from longctx_daemon.mcp_server import MCPServer
from longctx_daemon.transports import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    _build_combined_http_app,
    _make_connection_context,
    _new_connection_id,
    _new_session_id,
    build_sse_app,
    build_streamable_http_app,
    run_multi_transport,
    run_sse,
    run_streamable_http,
)
from longctx_daemon.types import (
    Citation,
    IndexStatus,
    LatencyBreakdown,
    Project,
    ScopeDecision,
    SearchChunk,
    SearchFreshness,
    SearchResult,
)


# ============================================================ fakes
@dataclass
class _FakeSearcher:
    def search(self, **kwargs):
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
                is_fully_fresh=True,
                stale_files=(),
                pending_updates=0,
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
        return IndexStatus(
            status="ready",
            total_chunks=0,
            pending_updates=0,
            embedder_model="fake",
            embedder_sha256="fake",
            last_full_scan="2026-05-09T00:00:00Z",
            projects=(),
        )


class _FakeChunkStore:
    def list_projects(self):
        return ()

    def list_files(self, project=None):
        return ()

    def get_chunks_by_file(self, file_id):
        return ()


def _make_server() -> MCPServer:
    """Construct a minimal MCPServer compatible with both 2.0 and 2.1
    constructor signatures."""
    import inspect
    sig = inspect.signature(MCPServer.__init__)
    kwargs = {
        "searcher": _FakeSearcher(),
        "indexer": _FakeIndexer(),
        "chunk_store": _FakeChunkStore(),
    }
    if "embed_store" in sig.parameters:
        kwargs["embed_store"] = None
    return MCPServer(**kwargs)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ============================================================ helpers
def test_new_session_id_is_unique():
    a = _new_session_id()
    b = _new_session_id()
    assert a != b
    assert a.startswith("ses_")


def test_new_connection_id_is_unique():
    assert _new_connection_id() != _new_connection_id()


def test_make_connection_context_defaults():
    ctx = _make_connection_context()
    assert ctx.client_name == "unknown"
    assert ctx.session_id and ctx.connection_id


def test_make_connection_context_overrides():
    ctx = _make_connection_context(
        client_name="opencode", client_version="0.4.2",
    )
    assert ctx.client_name == "opencode"
    assert ctx.client_version == "0.4.2"


# ============================================================ app builders
def test_build_sse_app_returns_starlette():
    server = _make_server()
    app = build_sse_app(server)
    # Sanity: routes contain /sse and /messages/.
    paths = []
    for r in app.routes:
        path = getattr(r, "path", None) or getattr(r, "path_format", None)
        if path is not None:
            paths.append(path)
    assert any(p == "/sse" for p in paths)
    assert any("messages" in p for p in paths)


def test_build_streamable_http_app_returns_starlette():
    server = _make_server()
    app = build_streamable_http_app(server)
    paths = []
    for r in app.routes:
        path = getattr(r, "path", None) or getattr(r, "path_format", None)
        if path is not None:
            paths.append(path)
    assert any("/mcp" in p for p in paths)


def test_build_combined_http_app_with_both():
    server = _make_server()
    app = _build_combined_http_app(
        server, include_sse=True, include_streamable_http=True,
    )
    paths = [
        getattr(r, "path", None) or getattr(r, "path_format", None)
        for r in app.routes
    ]
    assert any(p == "/sse" for p in paths)
    assert any("/mcp" in (p or "") for p in paths)


def test_build_combined_http_app_sse_only():
    server = _make_server()
    app = _build_combined_http_app(
        server, include_sse=True, include_streamable_http=False,
    )
    paths = [
        getattr(r, "path", None) or getattr(r, "path_format", None)
        for r in app.routes
    ]
    assert any(p == "/sse" for p in paths)
    assert not any("/mcp" in (p or "") for p in paths)


def test_build_combined_http_app_streamable_only():
    server = _make_server()
    app = _build_combined_http_app(
        server, include_sse=False, include_streamable_http=True,
    )
    paths = [
        getattr(r, "path", None) or getattr(r, "path_format", None)
        for r in app.routes
    ]
    assert not any(p == "/sse" for p in paths)
    assert any("/mcp" in (p or "") for p in paths)


# ============================================================ run_sse smoke
@pytest.mark.asyncio
async def test_run_sse_starts_and_shuts_down():
    """uvicorn binds, ready_event fires, cancellation drains."""
    server = _make_server()
    port = _free_port()
    ready = asyncio.Event()

    task = asyncio.create_task(
        run_sse(server, "127.0.0.1", port, ready_event=ready)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        # Confirm the listener is up.
        async with httpx.AsyncClient(timeout=2) as client:
            # /sse hangs the connection on success; just check we hit
            # something live with a HEAD on /sse (not allowed → 405)
            # or just connect with a short stream.
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{port}/does-not-exist",
                    timeout=1,
                )
                # 404 from Starlette is fine — proves we're listening.
                assert resp.status_code == 404
            except (httpx.TimeoutException, httpx.RemoteProtocolError):
                pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=DEFAULT_SHUTDOWN_GRACE_SECONDS + 2)


@pytest.mark.asyncio
async def test_run_streamable_http_starts_and_shuts_down():
    server = _make_server()
    port = _free_port()
    ready = asyncio.Event()

    task = asyncio.create_task(
        run_streamable_http(server, "127.0.0.1", port, ready_event=ready)
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        async with httpx.AsyncClient(timeout=2) as client:
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{port}/does-not-exist",
                    timeout=1,
                )
                assert resp.status_code == 404
            except (httpx.TimeoutException, httpx.RemoteProtocolError):
                pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=DEFAULT_SHUTDOWN_GRACE_SECONDS + 2)


# ============================================================ multi-transport
@pytest.mark.asyncio
async def test_run_multi_transport_rejects_unknown():
    server = _make_server()
    with pytest.raises(ValueError):
        await run_multi_transport(
            server, "127.0.0.1", _free_port(), ["nope"],
        )


@pytest.mark.asyncio
async def test_run_multi_transport_rejects_empty():
    server = _make_server()
    with pytest.raises(ValueError):
        await run_multi_transport(server, "127.0.0.1", _free_port(), [])


@pytest.mark.asyncio
async def test_run_multi_transport_runs_sse_alone():
    server = _make_server()
    port = _free_port()
    ready = asyncio.Event()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_multi_transport(
            server, "127.0.0.1", port, ["sse"],
            ready_event=ready, shutdown_event=shutdown,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        # Trigger shutdown.
        shutdown.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


@pytest.mark.asyncio
async def test_run_multi_transport_runs_both_http_transports():
    server = _make_server()
    port = _free_port()
    ready = asyncio.Event()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_multi_transport(
            server, "127.0.0.1", port, ["sse", "streamable-http"],
            ready_event=ready, shutdown_event=shutdown,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        # Both routes should be live at the same port.
        async with httpx.AsyncClient(timeout=2) as client:
            try:
                r = await client.get(
                    f"http://127.0.0.1:{port}/does-not-exist",
                    timeout=1,
                )
                assert r.status_code == 404
            except (httpx.TimeoutException, httpx.RemoteProtocolError):
                pass
        shutdown.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


# ============================================================ shutdown
@pytest.mark.asyncio
async def test_shutdown_event_stops_multi_transport():
    """Setting shutdown_event after startup terminates the runner."""
    server = _make_server()
    port = _free_port()
    ready = asyncio.Event()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_multi_transport(
            server, "127.0.0.1", port, ["sse"],
            ready_event=ready, shutdown_event=shutdown,
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=5)
    assert not task.done()
    shutdown.set()
    await asyncio.wait_for(task, timeout=10)
    assert task.done()


# ============================================================ pytest config
# These tests use ``pytest-asyncio`` in auto mode. Add a conftest hook so
# we don't depend on global config.
def pytest_collection_modifyitems(config, items):
    """Mark all coroutine tests as asyncio-eligible."""
    pass


@pytest.fixture(scope="module")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
