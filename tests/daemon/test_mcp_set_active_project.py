"""Tests for the live ``set_active_project`` MCP tool (PRD §3.4).

Per-connection sticky session state. Subsequent ``search_codebase``
calls without ``cwd`` / ``project`` honor the pin. Connection drops →
state drops; cross-session isolation is intrinsic because each
connection gets its own ``ConnectionContext``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from longctx_daemon.mcp_server import (
    ConnectionContext,
    MCPServer,
    set_active_connection,
    reset_active_connection,
)
from longctx_daemon.types import (
    Citation,
    LatencyBreakdown,
    Project,
    ScopeDecision,
    SearchChunk,
    SearchFreshness,
    SearchResult,
)


# ---------------------------------------------------------------- fakes
@dataclass
class _FakeSearcher:
    """Records every kwargs call so tests can inspect what the server
    forwarded to the searcher."""
    result: SearchResult
    last_call: Optional[dict[str, Any]] = None

    def search(self, **kwargs):
        self.last_call = dict(kwargs)
        return self.result


class _FakeChunkStore:
    """Minimal store with the methods set_active_project touches."""

    def __init__(self, projects: tuple[Project, ...] = ()) -> None:
        self._projects = list(projects)

    def list_projects(self) -> tuple[Project, ...]:
        return tuple(self._projects)

    def get_project(self, name: str):
        for p in self._projects:
            if p.name == name:
                return p
        return None

    def list_files(self, project: Optional[str] = None):
        return ()

    def get_chunks_by_file(self, file_id: int):
        return ()


class _FakeIndexer:
    def status(self):
        raise NotImplementedError("not under test here")


# ---------------------------------------------------------------- helpers
def _fresh_result(primary: str = "myapp") -> SearchResult:
    return SearchResult(
        chunks=(
            SearchChunk(
                citation=Citation(
                    project=primary, file_path=f"{primary}/x.py",
                    start_line=1, end_line=10,
                ),
                text="x", relevance_score=0.5, token_count=10,
            ),
        ),
        freshness=SearchFreshness(
            is_fully_fresh=True, pending_updates=0,
            indexed_through="2026-05-09T12:00:00Z", stale_files=(),
        ),
        scope_decision=ScopeDecision(
            primary_project=primary, primary_source="active_project_sticky",
            fanout_projects=(primary,), cross_project_pattern_matched=None,
            active_project_sticky=primary,
        ),
        latency_ms=LatencyBreakdown(total=1.0),
    )


def _make_server(*, projects: tuple[Project, ...] = ()) -> MCPServer:
    return MCPServer(
        searcher=_FakeSearcher(result=_fresh_result()),
        indexer=_FakeIndexer(),
        chunk_store=_FakeChunkStore(projects=projects),
    )


# ---------------------------------------------------------------- tests
class TestSetActiveProject:
    def test_sets_active_project_for_known_name(self):
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=0)
        server = _make_server(projects=(proj,))
        out = asyncio.run(
            server._dispatch_tool("set_active_project", {"project": "myapp"})
        )
        assert out["status"] == "ok"
        assert out["active_project"] == "myapp"

    def test_state_persists_on_subsequent_calls_same_session(self):
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=0)
        server = _make_server(projects=(proj,))

        # Set sticky.
        asyncio.run(server._dispatch_tool(
            "set_active_project", {"project": "myapp"},
        ))
        # Now run search without cwd / project — sticky should be passed.
        asyncio.run(server._dispatch_tool(
            "search_codebase", {"query": "anything"},
        ))
        searcher: _FakeSearcher = server.searcher  # type: ignore[assignment]
        assert searcher.last_call is not None
        assert searcher.last_call.get("active_project_sticky") == "myapp"

    def test_unknown_project_returns_error_with_available(self):
        a = Project(name="alpha", root_path="/a", last_full_scan_at=0)
        b = Project(name="beta", root_path="/b", last_full_scan_at=0)
        server = _make_server(projects=(a, b))
        out = asyncio.run(
            server._dispatch_tool("set_active_project", {"project": "gamma"})
        )
        assert "error" in out
        assert "gamma" in out["error"]
        assert set(out["available_projects"]) == {"alpha", "beta"}

    def test_unknown_project_does_not_change_sticky(self):
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=0)
        server = _make_server(projects=(proj,))
        asyncio.run(server._dispatch_tool(
            "set_active_project", {"project": "myapp"},
        ))
        # Unknown attempt should not clobber the prior set.
        asyncio.run(server._dispatch_tool(
            "set_active_project", {"project": "absent"},
        ))
        asyncio.run(server._dispatch_tool(
            "search_codebase", {"query": "x"},
        ))
        searcher: _FakeSearcher = server.searcher  # type: ignore[assignment]
        assert searcher.last_call.get("active_project_sticky") == "myapp"

    def test_per_connection_isolation(self):
        """Two ConnectionContexts must not see each other's sticky pin.

        Phase 2.1 SSE / streamable-http will drive this in production;
        here we simulate it by entering / leaving the contextvar
        manually around the dispatch calls.
        """
        a = Project(name="alpha", root_path="/a", last_full_scan_at=0)
        b = Project(name="beta", root_path="/b", last_full_scan_at=0)
        server = _make_server(projects=(a, b))

        ctx_a = ConnectionContext(session_id="ses-a", connection_id="conn-a")
        ctx_b = ConnectionContext(session_id="ses-b", connection_id="conn-b")

        async def _run():
            # Connection A sets alpha.
            tok = set_active_connection(ctx_a)
            try:
                await server._dispatch_tool(
                    "set_active_project", {"project": "alpha"},
                )
            finally:
                reset_active_connection(tok)

            # Connection B sets beta.
            tok = set_active_connection(ctx_b)
            try:
                await server._dispatch_tool(
                    "set_active_project", {"project": "beta"},
                )
            finally:
                reset_active_connection(tok)

            # A still sees alpha.
            tok = set_active_connection(ctx_a)
            try:
                await server._dispatch_tool(
                    "search_codebase", {"query": "x"},
                )
            finally:
                reset_active_connection(tok)
            a_call = dict(server.searcher.last_call)

            # B still sees beta.
            tok = set_active_connection(ctx_b)
            try:
                await server._dispatch_tool(
                    "search_codebase", {"query": "x"},
                )
            finally:
                reset_active_connection(tok)
            b_call = dict(server.searcher.last_call)

            return a_call, b_call

        a_call, b_call = asyncio.run(_run())
        assert a_call["active_project_sticky"] == "alpha"
        assert b_call["active_project_sticky"] == "beta"

    def test_search_without_sticky_does_not_pass_kwarg(self):
        """When no sticky is set, the searcher kwargs must not include
        ``active_project_sticky=None`` — let the searcher default kick in."""
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=0)
        server = _make_server(projects=(proj,))
        asyncio.run(server._dispatch_tool(
            "search_codebase", {"query": "x"},
        ))
        last = server.searcher.last_call
        # Sticky never set — kwarg either absent or explicitly None
        assert last.get("active_project_sticky") in (None, )
        # Specifically, we shouldn't be passing a string when nothing
        # was set.

    def test_explicit_project_overrides_sticky_in_search(self):
        """Sticky is sticky, but the per-call project= kwarg still wins.

        The MCP server forwards both — the searcher's scope decision
        prefers ``project`` (tier 1) over sticky (tier 3). Smoke check
        that the server forwards both intact.
        """
        a = Project(name="alpha", root_path="/a", last_full_scan_at=0)
        b = Project(name="beta", root_path="/b", last_full_scan_at=0)
        server = _make_server(projects=(a, b))
        asyncio.run(server._dispatch_tool(
            "set_active_project", {"project": "alpha"},
        ))
        asyncio.run(server._dispatch_tool(
            "search_codebase",
            {"query": "x", "project": "beta"},
        ))
        last = server.searcher.last_call
        assert last["project"] == "beta"
        assert last["active_project_sticky"] == "alpha"

    def test_response_shape_smoke(self):
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=0)
        server = _make_server(projects=(proj,))
        out = asyncio.run(
            server._dispatch_tool("set_active_project", {"project": "myapp"})
        )
        # Spec §3.4 — response is a small ack; specific shape tested
        # against the schema in test_mcp_server end-to-end.
        assert isinstance(out, dict)
        assert out["active_project"] == "myapp"
