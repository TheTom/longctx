"""Tests for the live ``wait_for_quiescence`` MCP tool (PRD §6.4).

Two paths:
  * No watcher attached → return ``{pending_updates: 0, indexed_through: now}``
    immediately. Daemon mode wires the real watcher (Agent G);
    stdio path passes None.
  * Watcher attached → ``await watcher.wait_for_quiescence(project,
    timeout_ms)``. Errors from the watcher surface as a non-fatal
    error response (search still works; just log the breakage).
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import MagicMock

import pytest

from longctx_daemon.mcp_server import MCPServer


# ---------------------------------------------------------- fakes

class _FakeWatcher:
    """Mock watcher with an awaitable ``wait_for_quiescence``."""

    def __init__(self, *, pending: int = 0,
                 indexed_through: str = "2026-05-09T12:00:00Z",
                 raise_on_call: bool = False) -> None:
        self.pending = pending
        self.indexed_through = indexed_through
        self.raise_on_call = raise_on_call
        self.calls: list[tuple[Optional[str], int]] = []

    async def wait_for_quiescence(
        self, project: Optional[str], timeout_ms: int,
    ) -> tuple[int, str]:
        self.calls.append((project, timeout_ms))
        if self.raise_on_call:
            raise RuntimeError("watcher unhappy")
        return self.pending, self.indexed_through


class _FakeBlockingWatcher:
    """Watcher whose wait_for_quiescence simulates draining a queue:
    blocks for ``delay_s`` then reports 0 pending."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls: list[tuple[Optional[str], int]] = []

    async def wait_for_quiescence(self, project, timeout_ms):
        self.calls.append((project, timeout_ms))
        await asyncio.sleep(self.delay_s)
        return 0, "2026-05-09T12:00:01Z"


def _make_server(*, watcher=None) -> MCPServer:
    return MCPServer(
        searcher=MagicMock(),
        indexer=MagicMock(),
        chunk_store=MagicMock(),
        watcher=watcher,
    )


# ---------------------------------------------------------- tests
class TestNoWatcherAttached:
    def test_returns_quiescent_immediately(self):
        server = _make_server()
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert out["pending_updates"] == 0
        assert "indexed_through" in out
        # ISO Z timestamp.
        assert out["indexed_through"].endswith("Z")

    def test_default_timeout_2000(self):
        # No watcher → timeout doesn't matter; we just verify the
        # response shape doesn't carry any leftover sentinel.
        server = _make_server()
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert "pending_updates" in out
        assert "error" not in out

    def test_with_project_filter(self):
        server = _make_server()
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {"project": "myapp"},
        ))
        assert out["pending_updates"] == 0


class TestWatcherAttached:
    def test_already_quiescent_returns_zero(self):
        w = _FakeWatcher(pending=0)
        server = _make_server(watcher=w)
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert out["pending_updates"] == 0
        assert out["indexed_through"] == "2026-05-09T12:00:00Z"
        assert w.calls == [(None, 2000)]

    def test_pending_count_propagated(self):
        # Watcher reports 5 still queued — that's not a server-side
        # error, just a freshness signal.
        w = _FakeWatcher(pending=5)
        server = _make_server(watcher=w)
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert out["pending_updates"] == 5

    def test_project_arg_forwarded(self):
        w = _FakeWatcher()
        server = _make_server(watcher=w)
        asyncio.run(server._dispatch_tool(
            "wait_for_quiescence",
            {"project": "myapp", "timeout_ms": 1500},
        ))
        assert w.calls == [("myapp", 1500)]

    def test_timeout_default_2000(self):
        w = _FakeWatcher()
        server = _make_server(watcher=w)
        asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert w.calls[0][1] == 2000

    def test_blocks_until_drained(self):
        # Watcher takes ~50ms to drain. Make sure we await it.
        w = _FakeBlockingWatcher(delay_s=0.05)
        server = _make_server(watcher=w)

        async def _run():
            t0 = asyncio.get_event_loop().time()
            await server._dispatch_tool(
                "wait_for_quiescence", {"timeout_ms": 1000},
            )
            return asyncio.get_event_loop().time() - t0

        elapsed = asyncio.run(_run())
        assert elapsed >= 0.04  # slightly under to absorb scheduler jitter

    def test_watcher_exception_surfaces_as_error(self):
        w = _FakeWatcher(raise_on_call=True)
        server = _make_server(watcher=w)
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert "error" in out
        assert "watcher unhappy" in out["error"]

    def test_watcher_exception_does_not_crash_dispatch(self):
        """The server must not let watcher hiccups break the MCP
        connection — search should still work right after."""
        w = _FakeWatcher(raise_on_call=True)
        server = _make_server(watcher=w)
        out = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert "error" in out
        # Subsequent call still goes through.
        w.raise_on_call = False
        out2 = asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {},
        ))
        assert out2["pending_updates"] == 0


class TestArgCoercion:
    def test_timeout_str_int(self):
        # timeout_ms could come in as a string from a sloppy client;
        # we coerce to int.
        w = _FakeWatcher()
        server = _make_server(watcher=w)
        asyncio.run(server._dispatch_tool(
            "wait_for_quiescence", {"timeout_ms": "750"},
        ))
        assert w.calls[0][1] == 750
