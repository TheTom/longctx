"""HTTP status server for the longctx daemon (PRD §10.1).

Tiny aiohttp server bound to a status port (default 8766). Exposes:

  GET /health       — JSON dump of daemon state
  GET /             — same (legacy alias)

The response shape is exactly what spec §10.1 calls for. The
``StatusServer`` is duck-typed against a ``DaemonState`` (see protocol
below); production wires the real daemon, tests pass a fake.

Why aiohttp instead of a 30-line socket loop:
  * Already a transitive dep of the MCP SDK in the SSE path
  * Handles HTTP/1.1 framing, keep-alive, and graceful shutdown
  * Easy to extend with /metrics, /debug/* later without rewriting
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from aiohttp import web


# ----------------------------------------------------------- duck types

class DaemonState(Protocol):
    """Minimum surface the StatusServer reads off the daemon.

    Production passes the real daemon (Agent F); tests pass a
    ``_FakeDaemon`` defined inline in the test module. Each property is
    cheap so the /health endpoint can be hit aggressively (k8s
    liveness probes, etc.) without driving expensive index walks.
    """

    @property
    def status(self) -> str:
        """One of 'ready', 'indexing', 'error'."""
        ...

    @property
    def started_at(self) -> float:
        """Unix timestamp from ``time.time()`` at daemon boot."""
        ...

    @property
    def projects(self) -> tuple[Any, ...]:
        """Per-project stats. Each entry is a dict with at minimum
        ``name``, ``chunks``, ``tokens_approx``, ``last_updated``."""
        ...

    @property
    def watcher(self) -> dict[str, Any]:
        """``{"queued": int, "events_processed_total": int}``."""
        ...

    @property
    def embedder(self) -> dict[str, Any]:
        """``{"model": str, "sha256": str}``."""
        ...

    @property
    def indexed_through(self) -> str:
        """ISO timestamp of last completed index update."""
        ...


# ----------------------------------------------------------- helpers

def _now_iso() -> str:
    """ISO-8601 UTC with trailing Z (matches §6.3 / §10.1 examples)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _process_rss_mb() -> int:
    """Resident set size in MB. Falls back to 0 if psutil unavailable
    (avoid blowing up health probes on a misconfigured install)."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return 0
    try:
        rss = psutil.Process().memory_info().rss
    except Exception:  # pragma: no cover
        return 0
    return int(rss / (1024 * 1024))


# ----------------------------------------------------------- payload

def build_health_payload(daemon: DaemonState, version: str) -> dict[str, Any]:
    """Render the §10.1 JSON payload from a DaemonState.

    This is split out so the CLI ``longctx status`` command can call it
    over the daemon when running in-process (or against an HTTP
    response when running standalone).
    """
    started = daemon.started_at
    uptime = max(0, int(time.time() - started))

    # Per-project stats: pass through whatever shape the daemon
    # exposes, but also normalize keys so consumers don't have to
    # remember "is it chunk_count or chunks?".
    projects: list[dict[str, Any]] = []
    for proj in daemon.projects:
        if isinstance(proj, dict):
            d = proj
        else:
            # dataclass / object with attributes — best-effort flatten
            d = {
                "name": getattr(proj, "name", None),
                "chunks": getattr(proj, "chunks", None),
                "tokens_approx": getattr(proj, "tokens_approx", None),
                "last_updated": getattr(proj, "last_updated", None),
            }
        # Normalize: PRD §10.1 uses "chunks", but our IndexStatus uses
        # "chunk_count" — accept either at input, emit "chunks" at
        # output so the contract matches the spec example.
        if "chunks" not in d and "chunk_count" in d:
            d["chunks"] = d["chunk_count"]
        if "tokens_approx" not in d and "token_count" in d:
            d["tokens_approx"] = d["token_count"]
        projects.append({
            "name": d.get("name"),
            "chunks": d.get("chunks", 0),
            "tokens_approx": d.get("tokens_approx", 0),
            "last_updated": d.get("last_updated"),
        })

    return {
        "status": daemon.status,
        "version": version,
        "uptime_seconds": uptime,
        "projects": projects,
        "watcher": dict(daemon.watcher),
        "embedder": dict(daemon.embedder),
        "memory_rss_mb": _process_rss_mb(),
        "indexed_through": daemon.indexed_through,
    }


# ----------------------------------------------------------- server

class StatusServer:
    """aiohttp-based HTTP status server.

    Two endpoints, both serve the same payload:
      * ``GET /health`` — canonical PRD §10.1 path
      * ``GET /`` — legacy alias so old probes don't break

    Lifecycle:
      * ``await server.run()`` blocks; respects loop cancellation.
      * ``server.stop()`` triggers a graceful shutdown.
    """

    def __init__(
        self,
        daemon: DaemonState,
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        version: Optional[str] = None,
    ) -> None:
        from longctx_daemon import __version__ as pkg_version

        self._daemon = daemon
        self._host = host
        self._port = port
        self._version = version or pkg_version
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._stop_evt: Optional[Any] = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def app(self) -> web.Application:
        """Build (or reuse) the aiohttp Application.

        Exposed publicly so tests can drive the handlers via
        ``aiohttp.test_utils`` without binding a real port.
        """
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/", self._handle_health)
        return app

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Render the §10.1 payload as application/json.

        TODO(phase 2.2): add ETag + If-None-Match for cheaper
        polling (k8s probe rate × number of operators adds up).
        """
        payload = build_health_payload(self._daemon, self._version)
        return web.json_response(payload)

    async def run(self) -> None:
        """Bind the port and serve until ``stop()`` fires.

        Uses aiohttp's AppRunner instead of ``web.run_app`` so the
        daemon can supervise it as one task among many (watcher, MCP
        SSE, etc.) inside the same event loop.
        """
        import asyncio
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._stop_evt = asyncio.Event()
        try:
            await self._stop_evt.wait()
        finally:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._stop_evt = None

    def stop(self) -> None:
        """Signal the ``run()`` task to shut down gracefully.

        Idempotent: calling stop() before run() is a no-op.
        """
        if self._stop_evt is not None:
            try:
                self._stop_evt.set()
            except Exception:  # pragma: no cover
                pass


__all__ = [
    "DaemonState",
    "StatusServer",
    "build_health_payload",
]
