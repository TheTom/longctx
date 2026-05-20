"""Daemon orchestrator — Phase 2.1.

Owns the lifecycle of a long-running ``longctx`` process:

  1. Take the ``server.lock`` singleton.
  2. Bind ``mcp_port`` + ``status_port`` (port walking on collisions).
  3. Write ``server.info`` once HTTP transports are listening.
  4. Spawn the requested MCP transports.
  5. Install signal handlers for SIGTERM / SIGINT / SIGHUP / SIGUSR1.
  6. On graceful shutdown: cancel transports → wait up to N seconds for
     in-flight retrievals → release lock → delete server.info.

The Daemon class deliberately does NOT own the watcher / config-reload
plumbing; those belong to Agents G and E. Hooks are exposed so the
daemon's signal handlers can call into them (``request_reload`` calls
``config.reconcile`` if available; ``request_shutdown`` is the public
shutdown hook).

Detach-to-background mode (``--daemon``) wraps ``Daemon.run`` in a
``python_daemon.DaemonContext``. We don't double-fork in tests; the
test suite only exercises the foreground path.

Signal model (PRD §9.2):

| Signal   | Effect                                             |
| -------- | -------------------------------------------------- |
| SIGTERM  | graceful shutdown                                  |
| SIGINT   | graceful shutdown                                  |
| SIGHUP   | reconcile config without dropping connections      |
| SIGUSR1  | trigger best-effort full reindex                   |

The handlers run on the asyncio loop's signal handler queue so we
don't block the running event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from longctx_daemon import __version__
from longctx_daemon.server_info import (
    DEFAULT_LOCK_PATH,
    DEFAULT_SERVER_INFO,
    PortBindError,
    ServerInfo,
    SingletonLock,
    bind_pair,
    delete_server_info,
    now_iso,
    write_server_info,
)
from longctx_daemon.transports import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    _build_combined_http_app,
)


# Defaults match PRD §8.1.
DEFAULT_MCP_PORT = 8765
DEFAULT_STATUS_PORT = 8766


# ============================================================ DaemonConfig
@dataclass
class DaemonConfig:
    """Knobs that the orchestrator reads — split out from the larger
    user-facing config (Agent E) so the daemon module has no coupling
    to that file's schema.
    """
    host: str = "127.0.0.1"
    mcp_port: int = DEFAULT_MCP_PORT
    status_port: int = DEFAULT_STATUS_PORT
    transports: tuple[str, ...] = ("sse", "streamable-http")
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS
    lock_path: Path = DEFAULT_LOCK_PATH
    server_info_path: Path = DEFAULT_SERVER_INFO
    max_port_walk: int = 20

    def __post_init__(self) -> None:
        # Validate transports up front so a typo fails fast instead of
        # at the asyncio task boundary.
        valid = {"sse", "streamable-http", "stdio"}
        if not self.transports:
            raise ValueError("at least one transport must be configured")
        bad = [t for t in self.transports if t not in valid]
        if bad:
            raise ValueError(f"invalid transports {bad}; valid={sorted(valid)}")


# ============================================================ Daemon
class Daemon:
    """Lifecycle orchestrator for a longctx daemon process."""

    def __init__(
        self,
        config: DaemonConfig,
        mcp_server: Any,
        *,
        chunk_store: Any = None,
        embed_store: Any = None,
        indexer: Any = None,
        searcher: Any = None,
        watcher: Any = None,
        on_reload: Optional[Callable[[], Awaitable[None]]] = None,
        on_reindex: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Args:

        config:    DaemonConfig (host, ports, transports, paths).
        mcp_server: an ``MCPServer`` instance (Agent H built it). The
            daemon doesn't construct it — the CLI wires the searcher,
            indexer and chunk_store + replay log + per-corpus dirs.
        chunk_store / embed_store / indexer / searcher: passed through
            for cleanup on shutdown (close()) and so the watcher /
            reload paths can find them.
        on_reload: optional async callback invoked from SIGHUP. The
            production wiring resolves this from the config module
            (``longctx_daemon.config.reconcile``) at run-time so this
            module has no hard import.
        on_reindex: optional async callback invoked from SIGUSR1. Best
            effort.
        """
        self.config = config
        self.mcp_server = mcp_server
        self.chunk_store = chunk_store
        self.embed_store = embed_store
        self.indexer = indexer
        self.searcher = searcher
        self.watcher = watcher
        self._on_reload = on_reload
        self._on_reindex = on_reindex

        # Mutable lifecycle state.
        self._shutdown_event = asyncio.Event()
        self._reload_event = asyncio.Event()
        self._reindex_event = asyncio.Event()
        self._lock: Optional[SingletonLock] = None
        self._server_info: Optional[ServerInfo] = None
        self._mcp_socket: Optional[socket.socket] = None
        self._status_socket: Optional[socket.socket] = None
        self._actual_mcp_port: Optional[int] = None
        self._actual_status_port: Optional[int] = None
        self._signal_tokens: list[Any] = []

    # ------------------------------------------------------ public lifecycle
    async def run(self) -> None:
        """Take the singleton lock, bind ports, run transports.

        Always cleans up on exit:
          * cancels transport tasks,
          * deletes server.info,
          * releases the singleton lock,
          * closes any chunk/embed stores we own.

        Raises:
            RuntimeError if another daemon already holds the lock.
            PortBindError if the configured ports + 20 walk forward all fail.
        """
        async with contextlib.AsyncExitStack() as stack:
            # ------ 1. singleton
            lock = SingletonLock(self.config.lock_path)
            if not lock.acquire():
                holder = lock.holder_pid()
                raise RuntimeError(
                    f"longctx daemon already running (pid={holder}); "
                    f"see {self.config.lock_path}"
                )
            self._lock = lock
            stack.callback(self._release_lock)

            # ------ 2. ports (walk on collision; both walk independently)
            try:
                (mcp_sock, mcp_port), (status_sock, status_port) = bind_pair(
                    preferred_mcp=self.config.mcp_port,
                    preferred_status=self.config.status_port,
                    max_tries=self.config.max_port_walk,
                    host=self.config.host,
                )
            except PortBindError:
                raise

            self._mcp_socket = mcp_sock
            self._status_socket = status_sock
            self._actual_mcp_port = mcp_port
            self._actual_status_port = status_port
            stack.callback(self._close_sockets)

            # ------ 3. signal handlers
            self._install_signal_handlers()
            stack.callback(self._uninstall_signal_handlers)

            # ------ 4. server.info (atomic write)
            info = ServerInfo(
                pid=os.getpid(),
                started_at=now_iso(),
                mcp_port=mcp_port,
                mcp_transports=tuple(
                    t for t in self.config.transports if t != "stdio"
                ),
                status_port=status_port,
                version=__version__,
            )
            write_server_info(info, self.config.server_info_path)
            self._server_info = info
            stack.callback(self._delete_server_info)

            # ------ 5. spawn transport tasks. We take the pre-bound MCP
            # socket and hand it to uvicorn so port-walking + listening
            # happen in the same caller-controlled step.
            await self._run_until_shutdown(mcp_sock)

    # ------------------------------------------------------ shutdown / reload
    def request_shutdown(self) -> None:
        """Signal-safe shutdown trigger. Idempotent."""
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

    def request_reload(self) -> None:
        """Signal-safe reload trigger. Idempotent."""
        if not self._reload_event.is_set():
            self._reload_event.set()

    def request_reindex(self) -> None:
        """Signal-safe reindex trigger. Idempotent."""
        if not self._reindex_event.is_set():
            self._reindex_event.set()

    @property
    def server_info(self) -> Optional[ServerInfo]:
        return self._server_info

    @property
    def is_running(self) -> bool:
        return (
            self._server_info is not None and not self._shutdown_event.is_set()
        )

    # ============================================================ internals
    async def _run_until_shutdown(self, mcp_sock: socket.socket) -> None:
        """Run all transport tasks + the reload/reindex watchers until
        ``_shutdown_event`` flips.

        Uses ``asyncio.TaskGroup`` so any task error cancels the rest.
        We wrap the inner serve in a try/except so a transport crash
        triggers a clean shutdown instead of leaking sockets.
        """
        # Build the unified ASGI app with the transports we want.
        include_sse = "sse" in self.config.transports
        include_streamable_http = "streamable-http" in self.config.transports
        include_stdio = "stdio" in self.config.transports

        if not (include_sse or include_streamable_http or include_stdio):
            raise ValueError("daemon needs at least one transport configured")

        app = None
        if include_sse or include_streamable_http:
            app = _build_combined_http_app(
                self.mcp_server,
                include_sse=include_sse,
                include_streamable_http=include_streamable_http,
            )

        # uvicorn: lazy import so unit tests that don't drive the
        # transport loop don't pay the import cost.
        import uvicorn  # noqa: WPS433

        async def _http_runner() -> None:
            if app is None:
                return
            cfg = uvicorn.Config(
                app,
                log_level="warning",
                timeout_graceful_shutdown=int(
                    self.config.shutdown_grace_seconds
                ),
            )
            srv = uvicorn.Server(cfg)
            srv.config.install_signal_handlers = False  # we own signals
            try:
                await srv.serve(sockets=[mcp_sock])
            except asyncio.CancelledError:
                # Trigger uvicorn's graceful shutdown if we were
                # cancelled mid-serve. ``force_exit`` is ALSO set so
                # long-lived SSE connections (Claude Code MCP bridge,
                # etc.) don't block the shutdown indefinitely — they
                # get closed instead of waited on. Without this the
                # daemon would hang for the full grace timeout every
                # SIGTERM and leave stale server.info / server.lock
                # behind (observed 2026-05-19).
                srv.should_exit = True
                srv.force_exit = True
                raise

        async def _stdio_runner() -> None:
            if not include_stdio:
                return
            await self.mcp_server.run_stdio()

        async def _shutdown_watcher() -> None:
            await self._shutdown_event.wait()
            # Cancel the task group via raising CancelledError.
            raise asyncio.CancelledError()

        async def _reload_watcher() -> None:
            while True:
                await self._reload_event.wait()
                self._reload_event.clear()
                if self._on_reload is not None:
                    try:
                        await self._on_reload()
                    except Exception as e:
                        # Reload failures must not bring down the daemon.
                        sys.stderr.write(
                            f"[longctx daemon] reload failed: {e!r}\n"
                        )

        async def _reindex_watcher() -> None:
            while True:
                await self._reindex_event.wait()
                self._reindex_event.clear()
                if self._on_reindex is not None:
                    try:
                        await self._on_reindex()
                    except Exception as e:
                        sys.stderr.write(
                            f"[longctx daemon] reindex failed: {e!r}\n"
                        )

        async def _fs_watcher_runner() -> None:
            # File-system change watcher. Drives live re-indexing so the
            # daemon picks up new / modified / deleted files without a
            # restart. Optional — if no watcher was constructed (test
            # harnesses, ask-mode, etc.), this task no-ops cleanly.
            if self.watcher is None:
                return
            try:
                await self.watcher.run()
            except asyncio.CancelledError:
                # Shutdown path: signal the watcher to stop so it cancels
                # its per-project tasks before the TaskGroup tears down.
                try:
                    self.watcher.stop()
                except Exception:  # noqa: BLE001
                    pass
                raise
            except Exception as e:  # noqa: BLE001
                # Watcher crash must not bring down the daemon — log and
                # exit this task. Indexer stays usable; only live updates
                # stop working until a restart.
                sys.stderr.write(
                    f"[longctx daemon] fs watcher crashed: {e!r}\n"
                )

        # Run everything under one task group. CancelledError from the
        # shutdown watcher cancels the group; we catch the resulting
        # ExceptionGroup at the boundary.
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_http_runner(), name="longctx-http")
                tg.create_task(_stdio_runner(), name="longctx-stdio")
                tg.create_task(_reload_watcher(), name="longctx-reload")
                tg.create_task(_reindex_watcher(), name="longctx-reindex")
                tg.create_task(_fs_watcher_runner(), name="longctx-fs-watcher")
                tg.create_task(_shutdown_watcher(), name="longctx-shutdown")
        except* asyncio.CancelledError:
            # Expected — shutdown_watcher fired or transports were cancelled.
            pass

    # ---------------------------------------------------------- signals
    def _install_signal_handlers(self) -> None:
        """Wire SIGTERM / SIGINT / SIGHUP / SIGUSR1 into the loop.

        On Windows, ``loop.add_signal_handler`` raises NotImplementedError
        for everything except SIGINT; we fall back to ``signal.signal``
        with a thread-safe ``call_soon_threadsafe`` shim.
        """
        loop = asyncio.get_running_loop()

        def _term(*_a) -> None:
            self.request_shutdown()

        def _hup(*_a) -> None:
            self.request_reload()

        def _usr1(*_a) -> None:
            self.request_reindex()

        for sig, handler in (
            (signal.SIGTERM, _term),
            (signal.SIGINT, _term),
            (getattr(signal, "SIGHUP", None), _hup),
            (getattr(signal, "SIGUSR1", None), _usr1),
        ):
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, handler)
                self._signal_tokens.append(sig)
            except (NotImplementedError, RuntimeError):
                # Windows or non-main-thread fallback.
                try:
                    prev = signal.signal(
                        sig,
                        lambda s, f, h=handler: loop.call_soon_threadsafe(h),
                    )
                    self._signal_tokens.append((sig, prev))
                except (OSError, ValueError):
                    # Some signals can't be installed in some contexts
                    # (e.g. running inside a non-main thread under
                    # pytest); silently skip — tests can still drive
                    # the event flags directly.
                    pass

    def _uninstall_signal_handlers(self) -> None:
        loop_running = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop_running = False
            loop = None
        for token in self._signal_tokens:
            if isinstance(token, tuple):
                sig, prev = token
                try:
                    signal.signal(sig, prev)
                except (OSError, ValueError):
                    pass
            else:
                if loop_running and loop is not None:
                    try:
                        loop.remove_signal_handler(token)
                    except (NotImplementedError, ValueError, RuntimeError):
                        pass
        self._signal_tokens.clear()

    # ---------------------------------------------------------- cleanup
    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _delete_server_info(self) -> None:
        if self._server_info is not None:
            delete_server_info(self.config.server_info_path)
            self._server_info = None

    def _close_sockets(self) -> None:
        for s in (self._mcp_socket, self._status_socket):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._mcp_socket = None
        self._status_socket = None


# ============================================================ detach helper
def detach_and_run(run_fn: Callable[[], int], *, log_dir: Path) -> int:
    """Detach to background via ``python-daemon`` then call ``run_fn``.

    Used by ``longctx serve --daemon``. The function runs in the
    detached child process; the parent returns immediately with exit
    code 0.

    On macOS, when invoked from a terminal, ``python-daemon`` does the
    standard double-fork + setsid + redirect. Under launchd, we already
    run as a daemonized child, so we skip the detach step (PRD §9.3
    documents that ``service install`` wraps launchd which does its own
    detach).

    NOTE: ``python-daemon`` is an optional dep; if it's not installed
    we fall back to a no-op (the caller still runs in the foreground)
    and emit a warning.
    """
    try:
        import daemon as python_daemon  # type: ignore[import-untyped]
    except ImportError:
        sys.stderr.write(
            "warning: python-daemon not installed; --daemon falls back to "
            "foreground. Install with: pip install longctx[daemon]\n"
        )
        return run_fn()

    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = open(log_dir / "stdout.log", "a", encoding="utf-8")
    stderr_log = open(log_dir / "stderr.log", "a", encoding="utf-8")

    ctx = python_daemon.DaemonContext(
        stdout=stdout_log,
        stderr=stderr_log,
        # Don't close all FDs — we may have stores already open in the
        # parent before the detach (CLI smoke path constructs them
        # before deciding to detach). Production wiring constructs
        # stores AFTER the detach, in which case files_preserve is
        # irrelevant.
        files_preserve=[stdout_log, stderr_log],
        detach_process=True,
    )
    with ctx:
        rc = run_fn()
    return rc or 0


__all__ = [
    "DEFAULT_MCP_PORT",
    "DEFAULT_STATUS_PORT",
    "Daemon",
    "DaemonConfig",
    "detach_and_run",
]
