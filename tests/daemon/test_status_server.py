"""Tests for ``longctx_daemon.status_server``.

Covers /health JSON shape, uptime, memory_rss_mb, and the optional /
alias. Uses ``aiohttp.test_utils`` to drive the handlers without
binding a real port.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from longctx_daemon.status_server import (
    StatusServer,
    build_health_payload,
)


# -------------------------------------------------------- fake daemon
@dataclass
class _FakeDaemon:
    """Minimum DaemonState for the StatusServer tests."""
    status: str = "ready"
    started_at: float = field(default_factory=time.time)
    projects: tuple = ()
    watcher: dict = field(default_factory=lambda: {
        "queued": 0,
        "events_processed_total": 0,
    })
    embedder: dict = field(default_factory=lambda: {
        "model": "BAAI/bge-small-en-v1.5",
        "sha256": "deadbeef" * 8,
    })
    indexed_through: str = "2026-05-09T12:00:00Z"


# -------------------------------------------------------- payload tests

class TestBuildHealthPayload:
    def test_required_keys_present(self):
        d = _FakeDaemon()
        out = build_health_payload(d, version="0.2.0")
        for key in (
            "status", "version", "uptime_seconds", "projects",
            "watcher", "embedder", "memory_rss_mb", "indexed_through",
        ):
            assert key in out

    def test_version_passthrough(self):
        d = _FakeDaemon()
        out = build_health_payload(d, version="9.9.9-rc1")
        assert out["version"] == "9.9.9-rc1"

    def test_uptime_counts_up(self):
        # Daemon started 100s ago.
        d = _FakeDaemon(started_at=time.time() - 100)
        out = build_health_payload(d, version="0.2.0")
        assert out["uptime_seconds"] >= 99
        # Don't be picky about ceiling — scheduling jitter is fine.

    def test_uptime_floor_zero(self):
        # Future-dated start (clock skew) → uptime is 0, not negative.
        d = _FakeDaemon(started_at=time.time() + 60)
        out = build_health_payload(d, version="0.2.0")
        assert out["uptime_seconds"] == 0

    def test_memory_rss_mb_is_positive_int(self):
        d = _FakeDaemon()
        out = build_health_payload(d, version="0.2.0")
        # psutil reports a real number on this test host.
        assert isinstance(out["memory_rss_mb"], int)
        assert out["memory_rss_mb"] >= 0

    def test_projects_pass_through_dicts(self):
        d = _FakeDaemon(projects=(
            {"name": "obs", "chunks": 762, "tokens_approx": 1543000,
             "last_updated": "2026-05-09T11:00:00Z"},
        ))
        out = build_health_payload(d, version="0.2.0")
        assert out["projects"] == [{
            "name": "obs",
            "chunks": 762,
            "tokens_approx": 1543000,
            "last_updated": "2026-05-09T11:00:00Z",
        }]

    def test_projects_normalize_chunk_count_alias(self):
        # IndexStatus uses chunk_count + token_count; we accept both.
        d = _FakeDaemon(projects=(
            {"name": "p", "chunk_count": 50, "token_count": 9999,
             "last_updated": 1234},
        ))
        out = build_health_payload(d, version="0.2.0")
        assert out["projects"][0]["chunks"] == 50
        assert out["projects"][0]["tokens_approx"] == 9999

    def test_watcher_round_trips(self):
        d = _FakeDaemon(watcher={"queued": 3, "events_processed_total": 1284})
        out = build_health_payload(d, version="0.2.0")
        assert out["watcher"] == {"queued": 3, "events_processed_total": 1284}

    def test_embedder_round_trips(self):
        d = _FakeDaemon(embedder={"model": "test", "sha256": "abc"})
        out = build_health_payload(d, version="0.2.0")
        assert out["embedder"] == {"model": "test", "sha256": "abc"}

    def test_object_projects_flattened(self):
        # Pass a dataclass-like with attributes instead of a dict.
        @dataclass
        class _P:
            name: str = "x"
            chunks: int = 10
            tokens_approx: int = 100
            last_updated: str = "2026-01-01T00:00:00Z"

        d = _FakeDaemon(projects=(_P(),))
        out = build_health_payload(d, version="0.2.0")
        assert out["projects"][0]["name"] == "x"
        assert out["projects"][0]["chunks"] == 10


# -------------------------------------------------------- HTTP tests
class TestStatusServerEndpoint(AioHTTPTestCase):
    """Drives the StatusServer's ``app`` through aiohttp's test
    client so we don't actually bind a port."""

    async def get_application(self) -> web.Application:
        self._daemon = _FakeDaemon(
            projects=(
                {"name": "obsidian", "chunks": 762,
                 "tokens_approx": 1_543_000,
                 "last_updated": "2026-05-09T11:00:00Z"},
            ),
            watcher={"queued": 0, "events_processed_total": 1284},
        )
        srv = StatusServer(self._daemon)
        return srv.app

    async def test_health_returns_200(self):
        async with self.client.get("/health") as resp:
            assert resp.status == 200
            body = await resp.json()
        assert body["status"] == "ready"
        assert body["version"]  # non-empty
        assert "uptime_seconds" in body

    async def test_root_alias_returns_same_payload(self):
        async with self.client.get("/health") as resp:
            health = await resp.json()
        async with self.client.get("/") as resp:
            root = await resp.json()
        # uptime + memory may drift by a tiny amount between calls;
        # compare the static keys.
        for key in ("status", "version", "projects", "watcher",
                    "embedder", "indexed_through"):
            assert health[key] == root[key]

    async def test_health_content_type_is_json(self):
        async with self.client.get("/health") as resp:
            assert resp.content_type == "application/json"

    async def test_health_carries_memory_rss(self):
        async with self.client.get("/health") as resp:
            body = await resp.json()
        assert "memory_rss_mb" in body

    async def test_unknown_path_404(self):
        async with self.client.get("/no-such") as resp:
            assert resp.status == 404


# -------------------------------------------------------- run lifecycle
class TestStatusServerRunLifecycle:
    def test_run_then_stop(self):
        """``run()`` blocks; ``stop()`` releases. Combined: one tick."""
        d = _FakeDaemon()
        # Walk to a likely-free port: pick a high random port to avoid
        # collisions on busy CI hosts.
        srv = StatusServer(d, host="127.0.0.1", port=18766)

        async def _run_and_stop():
            run_task = asyncio.create_task(srv.run())
            # Give the server a beat to bind, then stop.
            await asyncio.sleep(0.05)
            srv.stop()
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except asyncio.TimeoutError:
                pytest.fail("server did not stop after stop()")

        asyncio.run(_run_and_stop())

    def test_stop_before_run_is_noop(self):
        d = _FakeDaemon()
        srv = StatusServer(d)
        # Should not raise.
        srv.stop()

    def test_health_endpoint_accessible_when_bound(self):
        """End-to-end: bind a real port, hit it via stdlib urllib."""
        import urllib.request

        d = _FakeDaemon()
        srv = StatusServer(d, host="127.0.0.1", port=18767)

        async def _scenario():
            run_task = asyncio.create_task(srv.run())
            try:
                await asyncio.sleep(0.1)  # let the bind settle
                # Use loop.run_in_executor since urlopen is blocking.
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(
                        "http://127.0.0.1:18767/health", timeout=2.0,
                    ),
                )
                body = resp.read().decode("utf-8")
                resp.close()
                return body
            finally:
                srv.stop()
                try:
                    await asyncio.wait_for(run_task, timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        body = asyncio.run(_scenario())
        import json
        payload = json.loads(body)
        assert payload["status"] == "ready"
