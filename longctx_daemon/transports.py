"""SSE + streamable-http MCP transports for the longctx daemon — Phase 2.1.

The daemon exposes all three common MCP transports concurrently:

  * stdio              — already shipped in 2.0 via ``MCPServer.run_stdio``
  * sse                — HTTP SSE (older spec, well-supported)
  * streamable-http    — newer MCP spec; some clients only speak this

Each HTTP transport is a Starlette ASGI app served by uvicorn. We wrap
``mcp.server.sse.SseServerTransport`` for SSE and
``mcp.server.streamable_http_manager.StreamableHTTPSessionManager`` for
streamable-http; we own the Starlette routing + lifespan glue so both
can run side-by-side in the same daemon.

Per-connection state:
  * Each accepted connection allocates a fresh ``ConnectionContext``
    with a unique session_id + connection_id.
  * The context is stashed in an asyncio contextvar
    (``mcp_server._active_context_var``) for the duration of the
    connection. Tool dispatches inside that connection see it.

Concurrency:
  * ``run_multi_transport`` launches one asyncio task per transport
    plus stdio if requested.
  * Each task can be cancelled cleanly; the task group exits when all
    tasks have completed or were cancelled.
  * Graceful shutdown signal is wired through a shared
    ``shutdown_event`` so the orchestrator can flip it from a signal
    handler.

Test surface:
  * ``run_sse`` / ``run_streamable_http`` accept an optional
    ``ready_event`` so callers (the daemon) know when listeners are up
    and the server.info file is safe to write.
  * ``build_sse_app`` / ``build_streamable_http_app`` are factored out
    so unit tests can exercise the Starlette wiring without standing
    up uvicorn.

Limitations / TODOs:
  * No auth on the HTTP transports yet. PRD §13.4 requires auth for
    non-localhost binds; we default to 127.0.0.1 to avoid that
    headache.
  * The DNS-rebinding-protection middleware shipped with the SDK is
    enabled with default settings; a future tunable exposes the
    allowed hosts list.
  * Pending requests get up to ``shutdown_grace_seconds`` to drain.
    The mechanism is uvicorn's graceful shutdown — fine for SSE, but
    streamable-http SSE responses may be force-closed on the deadline.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

import uvicorn
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from longctx_daemon.mcp_server import (
    ConnectionContext,
    MCPServer,
    reset_active_connection,
    set_active_connection,
)

# Default shutdown grace period for in-flight requests on SIGTERM.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 10.0

# Default uvicorn log level — we keep it at WARNING so the operational
# §14.4 trace log isn't drowned out by ASGI noise.
DEFAULT_UVICORN_LOG_LEVEL = "warning"


# ============================================================ helpers
def _new_session_id() -> str:
    """ULID-shaped 26-char ID for diagnostics + log correlation.

    We don't use ``mcp_server.new_trace_id`` here so the trace log can
    distinguish "session" (long-lived MCP connection) from "trace"
    (single tool call).
    """
    # 26 hex chars; we just want stable + unique per process.
    return f"ses_{uuid4().hex[:22]}"


def _new_connection_id() -> str:
    return f"conn_{uuid4().hex[:20]}"


def _make_connection_context(
    *,
    client_name: str = "unknown",
    client_version: str = "unknown",
) -> ConnectionContext:
    return ConnectionContext(
        session_id=_new_session_id(),
        connection_id=_new_connection_id(),
        client_name=client_name,
        client_version=client_version,
    )


# ============================================================ SSE
def build_sse_app(server: MCPServer, *, endpoint: str = "/messages/") -> Starlette:
    """Build a Starlette app exposing the SSE MCP transport.

    Routes:
      * ``GET  /sse``      — opens an SSE stream; SDK sends an
        ``endpoint`` event pointing at the messages POST URL.
      * ``POST /messages/`` — clients POST JSON-RPC messages here; the
        SDK matches the session_id query param and forwards.

    Each accepted SSE connection gets its own ``ConnectionContext``
    pushed onto the contextvar for the life of that
    ``server.run(...)`` call.
    """
    sse = SseServerTransport(endpoint)

    async def handle_sse(request: Request) -> Response:
        ctx = _make_connection_context(client_name="sse-client")
        token = set_active_connection(ctx)
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send,
            ) as (read, write):
                await server.server.run(
                    read, write, server.initialization_options(),
                )
        finally:
            reset_active_connection(token)
        # connect_sse hijacks the connection; Starlette doesn't actually
        # need the response, but we have to return something.
        return Response()

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount(endpoint, app=sse.handle_post_message),
    ]
    return Starlette(routes=routes)


async def run_sse(
    server: MCPServer,
    host: str,
    port: int,
    *,
    ready_event: Optional[asyncio.Event] = None,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    log_level: str = DEFAULT_UVICORN_LOG_LEVEL,
    bound_socket: Any = None,
) -> None:
    """Bind + serve the SSE transport on ``host:port``.

    Set ``ready_event`` once uvicorn reports it is listening. The caller
    (daemon orchestrator) writes ``server.info`` only after both
    transports' ready events fire so external readers don't see a
    half-bound daemon.

    ``bound_socket`` (optional): a pre-bound socket from
    ``server_info.bind_port``. uvicorn then skips its own bind step,
    which lets the daemon walk the port forward atomically.
    """
    app = build_sse_app(server)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=int(shutdown_grace_seconds),
    )
    server_obj = uvicorn.Server(config)
    if bound_socket is not None:
        # uvicorn 0.27+ honours ``server.serve(sockets=[...])`` — we
        # hand off the pre-bound socket so we keep ownership of it
        # across the bind/serve gap (no race window).
        await _serve_with_ready(server_obj, ready_event, sockets=[bound_socket])
    else:
        await _serve_with_ready(server_obj, ready_event)


# ============================================================ streamable-http
def build_streamable_http_app(
    server: MCPServer,
    *,
    json_response: bool = False,
    stateless: bool = False,
) -> Starlette:
    """Build a Starlette app exposing the streamable-http MCP transport.

    All requests land on a single ``/mcp`` endpoint; the SDK's session
    manager routes by session-id header.
    """
    manager = StreamableHTTPSessionManager(
        app=server.server,
        json_response=json_response,
        stateless=stateless,
    )

    async def handle_mcp(scope, receive, send) -> None:
        # Per-connection context. Streamable-http's "session" is shorter
        # than SSE's; we still allocate a context for trace-log
        # correlation on each request. A more sophisticated
        # implementation would keep a session_id-keyed dict and reuse
        # contexts across requests with the same MCP session-id; that
        # lives behind a TODO until we have multi-request session state
        # (sticky active project) that needs to persist.
        ctx = _make_connection_context(client_name="streamable-http-client")
        token = set_active_connection(ctx)
        try:
            await manager.handle_request(scope, receive, send)
        finally:
            reset_active_connection(token)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    routes = [Mount("/mcp", app=handle_mcp)]
    return Starlette(routes=routes, lifespan=lifespan)


async def run_streamable_http(
    server: MCPServer,
    host: str,
    port: int,
    *,
    ready_event: Optional[asyncio.Event] = None,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    log_level: str = DEFAULT_UVICORN_LOG_LEVEL,
    bound_socket: Any = None,
    json_response: bool = False,
    stateless: bool = False,
) -> None:
    """Bind + serve the streamable-http transport on ``host:port``."""
    app = build_streamable_http_app(
        server, json_response=json_response, stateless=stateless,
    )
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=int(shutdown_grace_seconds),
    )
    server_obj = uvicorn.Server(config)
    if bound_socket is not None:
        await _serve_with_ready(server_obj, ready_event, sockets=[bound_socket])
    else:
        await _serve_with_ready(server_obj, ready_event)


# ============================================================ multi-transport
async def run_multi_transport(
    server: MCPServer,
    host: str,
    mcp_port: int,
    transports: Sequence[str],
    *,
    ready_event: Optional[asyncio.Event] = None,
    shutdown_event: Optional[asyncio.Event] = None,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> None:
    """Run any subset of {sse, streamable-http, stdio} concurrently.

    All HTTP transports share ``host:mcp_port``. SSE and streamable-http
    can both be present because they live on different routes (``/sse``
    vs ``/mcp``). When both are requested we mount them in the same
    Starlette app + uvicorn instance so we don't need two ports.

    stdio is launched as its own task. Daemon-mode callers should
    almost never enable stdio (it consumes the daemon's own stdin); the
    foreground ``longctx serve`` does enable stdio when no HTTP
    transports are requested.

    Shutdown:
      * On ``shutdown_event.set()`` we cancel the inner task group and
        let uvicorn drain.
      * If a transport task crashes, the others are cancelled and the
        exception propagates.
    """
    transports = tuple(transports)
    valid = {"sse", "streamable-http", "stdio"}
    bad = [t for t in transports if t not in valid]
    if bad:
        raise ValueError(f"unknown transport(s): {bad} (valid: {sorted(valid)})")
    if not transports:
        raise ValueError("at least one transport must be requested")

    # Build a unified ASGI app when both HTTP transports are wanted.
    http_transports = [t for t in transports if t in ("sse", "streamable-http")]
    has_stdio = "stdio" in transports

    # Holder for the uvicorn server so the shutdown watcher can flip
    # ``should_exit``. ``serve()`` reacts to that flag every 100ms; raw
    # task cancellation won't cleanly drain Starlette's lifespan queue.
    uvicorn_server: list[uvicorn.Server] = []

    async def _run_http() -> None:
        if not http_transports:
            return
        app = _build_combined_http_app(
            server,
            include_sse="sse" in http_transports,
            include_streamable_http="streamable-http" in http_transports,
        )
        config = uvicorn.Config(
            app,
            host=host,
            port=mcp_port,
            log_level=DEFAULT_UVICORN_LOG_LEVEL,
            timeout_graceful_shutdown=int(shutdown_grace_seconds),
        )
        server_obj = uvicorn.Server(config)
        uvicorn_server.append(server_obj)
        await _serve_with_ready(server_obj, ready_event)

    async def _run_stdio() -> None:
        if not has_stdio:
            return
        if ready_event is not None:
            ready_event.set()
        await server.run_stdio()

    async def _run_shutdown_watcher() -> None:
        if shutdown_event is None:
            return
        await shutdown_event.wait()
        # Tell uvicorn to drain. We don't raise CancelledError here
        # because TaskGroup-on-cancel during a Starlette lifespan can
        # leave the lifespan queue dangling with an open
        # ``await receive()``.
        for srv in uvicorn_server:
            srv.should_exit = True

    tasks: list[asyncio.Task] = []
    try:
        async with asyncio.TaskGroup() as tg:
            t_http = tg.create_task(_run_http(), name="longctx-http-transports")
            tasks.append(t_http)
            if has_stdio:
                t_stdio = tg.create_task(
                    _run_stdio(), name="longctx-stdio-transport",
                )
                tasks.append(t_stdio)
            if shutdown_event is not None:
                t_watch = tg.create_task(
                    _run_shutdown_watcher(), name="longctx-shutdown",
                )
                tasks.append(t_watch)
    except* asyncio.CancelledError:
        # External cancellation: forward to uvicorn cleanly.
        for srv in uvicorn_server:
            srv.should_exit = True
        raise


def _build_combined_http_app(
    server: MCPServer,
    *,
    include_sse: bool,
    include_streamable_http: bool,
) -> Starlette:
    """Build one Starlette app that hosts both HTTP transports.

    Routes are non-overlapping (``/sse`` + ``/messages/`` for SSE,
    ``/mcp`` for streamable-http) so we can mount them side by side.
    """
    routes: list = []
    lifespans: list[Callable[[Starlette], Any]] = []

    if include_sse:
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> Response:
            ctx = _make_connection_context(client_name="sse-client")
            token = set_active_connection(ctx)
            try:
                async with sse.connect_sse(
                    request.scope, request.receive, request._send,
                ) as (read, write):
                    await server.server.run(
                        read, write, server.initialization_options(),
                    )
            finally:
                reset_active_connection(token)
            return Response()

        routes.append(Route("/sse", endpoint=handle_sse, methods=["GET"]))
        routes.append(Mount("/messages/", app=sse.handle_post_message))

    if include_streamable_http:
        manager = StreamableHTTPSessionManager(app=server.server)

        async def handle_mcp(scope, receive, send) -> None:
            ctx = _make_connection_context(
                client_name="streamable-http-client"
            )
            token = set_active_connection(ctx)
            try:
                await manager.handle_request(scope, receive, send)
            finally:
                reset_active_connection(token)

        routes.append(Mount("/mcp", app=handle_mcp))

        # streamable-http needs lifespan integration for its session
        # manager.
        @contextlib.asynccontextmanager
        async def streamable_lifespan(_app: Starlette):
            async with manager.run():
                yield

        lifespans.append(streamable_lifespan)

    if not lifespans:
        return Starlette(routes=routes)

    @contextlib.asynccontextmanager
    async def combined_lifespan(app: Starlette):
        async with contextlib.AsyncExitStack() as stack:
            for ls in lifespans:
                await stack.enter_async_context(ls(app))
            yield

    return Starlette(routes=routes, lifespan=combined_lifespan)


# ============================================================ helpers
async def _serve_with_ready(
    server_obj: uvicorn.Server,
    ready_event: Optional[asyncio.Event],
    *,
    sockets: Any = None,
) -> None:
    """Run uvicorn and signal ``ready_event`` once it reports started.

    uvicorn's ``Server.startup`` flips ``self.started`` once the
    listener is up; we poll for that and set the event. This lets the
    daemon orchestrator wait for "listening" before publishing
    ``server.info``.

    We also disable uvicorn's own signal handlers — the daemon manages
    SIGTERM/SIGHUP and pipes shutdown into our own ``shutdown_event``.
    """
    server_obj.config.install_signal_handlers = False  # type: ignore[attr-defined]

    if ready_event is None:
        if sockets is not None:
            await server_obj.serve(sockets=sockets)
        else:
            await server_obj.serve()
        return

    async def _watch_ready() -> None:
        while not server_obj.started:
            await asyncio.sleep(0.01)
        ready_event.set()

    watcher = asyncio.create_task(_watch_ready(), name="uvicorn-ready-watcher")
    try:
        if sockets is not None:
            await server_obj.serve(sockets=sockets)
        else:
            await server_obj.serve()
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watcher


__all__ = [
    "DEFAULT_SHUTDOWN_GRACE_SECONDS",
    "DEFAULT_UVICORN_LOG_LEVEL",
    "build_sse_app",
    "build_streamable_http_app",
    "run_multi_transport",
    "run_sse",
    "run_streamable_http",
]
