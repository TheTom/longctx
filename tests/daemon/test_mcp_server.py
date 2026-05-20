"""Tests for longctx_daemon.mcp_server.

Two flavors of test:
  1. Direct: drive ``MCPServer._dispatch_tool`` in-process. No transport.
  2. End-to-end: drive the full server via the MCP SDK's in-memory
     ``create_connected_server_and_client_session`` so the initialize
     handshake + clientInfo capture is exercised.

Both flavors share fakes for searcher / indexer / chunk_store. The
fakes implement only what the live tools touch — duck typing is enough
because the production protocols haven't been written yet (Phase 2.0
ships these alongside).
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
from loguru import logger as _loguru_logger
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import Implementation

from longctx_daemon import logging as lcl
from longctx_daemon.mcp_server import MCPServer
from longctx_daemon.replay_log import ReplayLog
from longctx_daemon.types import (
    Citation,
    FileRecord,
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
class FakeSearcher:
    result: SearchResult
    last_call: dict[str, Any] = None  # type: ignore[assignment]

    def search(self, **kwargs):
        self.last_call = dict(kwargs)
        return self.result


@dataclass
class FakeIndexer:
    status_value: IndexStatus

    def status(self):
        return self.status_value


class FakeChunkStore:
    def __init__(self, projects=(), files_by_project=None, chunks_by_file=None):
        self._projects = list(projects)
        self._files = files_by_project or {}
        self._chunks = chunks_by_file or {}

    def list_projects(self):
        return tuple(self._projects)

    def list_files(self, project: Optional[str] = None):
        return tuple(self._files.get(project, ()))

    def get_chunks_by_file(self, file_id: int):
        return tuple(self._chunks.get(file_id, ()))

    def get_project(self, name: str):
        for p in self._projects:
            if p.name == name:
                return p
        return None

    def upsert_project(self, project) -> None:
        for i, existing in enumerate(self._projects):
            if existing.name == project.name:
                self._projects[i] = project
                return
        self._projects.append(project)

    def get_file(self, project: str, rel_path: str):
        for rec in self._files.get(project, ()):
            if rec.rel_path == rel_path:
                return rec
        return None

    def get_file_by_id(self, file_id: int):
        for files in self._files.values():
            for rec in files:
                if rec.id == file_id:
                    return rec
        return None

    def get_chunks_by_id(self, ids):
        ids_set = set(ids)
        out = []
        for chunks in self._chunks.values():
            for c in chunks:
                if c.id in ids_set:
                    out.append(c)
        return tuple(out)


# ============================================================ builders
def _make_search_result(
    *,
    chunks: list[SearchChunk] | None = None,
    is_fresh: bool = True,
    pending: int = 0,
    stale: tuple[str, ...] = (),
    primary_project: str = "myapp",
) -> SearchResult:
    if chunks is None:
        chunks = [
            SearchChunk(
                citation=Citation(
                    project="myapp",
                    file_path="src/auth/middleware.ts",
                    start_line=42,
                    end_line=89,
                ),
                text="export function authMiddleware() { ... }",
                relevance_score=0.91,
                token_count=120,
            ),
            SearchChunk(
                citation=Citation(
                    project="myapp",
                    file_path="src/auth/types.ts",
                    start_line=1,
                    end_line=30,
                ),
                text="export interface AuthState { ... }",
                relevance_score=0.74,
                token_count=80,
            ),
        ]
    return SearchResult(
        chunks=tuple(chunks),
        freshness=SearchFreshness(
            is_fully_fresh=is_fresh,
            pending_updates=pending,
            indexed_through="2026-05-09T14:33:00.500Z",
            stale_files=stale,
        ),
        scope_decision=ScopeDecision(
            primary_project=primary_project,
            primary_source="cwd_walk_to_sentinel",
            fanout_projects=(primary_project,),
            cross_project_pattern_matched=None,
            active_project_sticky=None,
        ),
        latency_ms=LatencyBreakdown(
            wait_quiescence=12.0,
            embed_query=18.0,
            bm25_score=7.0,
            dense_score=24.0,
            rrf_fuse=1.0,
            fetch_chunks=3.0,
            total=65.0,
        ),
    )


def _make_index_status(**overrides) -> IndexStatus:
    base = dict(
        status="ready",
        total_chunks=42,
        pending_updates=0,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        embedder_sha256="deadbeef" * 8,
        last_full_scan="2026-05-09T14:00:00Z",
        error=None,
        projects=(
            Project(name="myapp", root_path="/Users/tom/dev/myapp", last_full_scan_at=1234),
        ),
    )
    base.update(overrides)
    return IndexStatus(**base)


# ============================================================ fixtures
@pytest.fixture(autouse=True)
def _reset_logging():
    _loguru_logger.remove()
    lcl.clear_trace_context()
    yield
    lcl.clear_trace_context()
    _loguru_logger.remove()


@pytest.fixture
def captured_lines():
    """Install a memory sink that captures serialized JSON lines."""
    captured: list[str] = []

    def _sink(message):
        captured.append(message)

    _loguru_logger.add(_sink, level="DEBUG", serialize=True)
    return captured


def _make_server(
    *,
    searcher=None,
    indexer=None,
    chunk_store=None,
    replay_log=None,
) -> MCPServer:
    return MCPServer(
        searcher=searcher or FakeSearcher(result=_make_search_result()),
        indexer=indexer or FakeIndexer(status_value=_make_index_status()),
        chunk_store=chunk_store or FakeChunkStore(),
        replay_log=replay_log,
    )


# ============================================================ search_codebase
class TestSearchCodebase:
    def test_happy_path(self):
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool(
                "search_codebase",
                {"query": "where is the auth middleware", "cwd": "/Users/tom/dev/myapp"},
            )
        )
        assert out["is_fully_fresh"] is True
        assert out["pending_updates"] == 0
        assert len(out["chunks"]) == 2
        first = out["chunks"][0]
        assert first["project"] == "myapp"
        assert first["file_path"] == "src/auth/middleware.ts"
        assert first["start_line"] == 42
        assert first["end_line"] == 89
        assert "text" in first
        assert "relevance_score" in first

    def test_scope_decision_in_response(self):
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("search_codebase", {"query": "x"})
        )
        sd = out["scope_decision"]
        assert sd["primary_project"] == "myapp"
        assert sd["primary_source"] == "cwd_walk_to_sentinel"
        assert sd["fanout_projects"] == ["myapp"]

    def test_max_tokens_caps_chunks(self):
        # Two chunks: 120 + 80 tokens = 200 total. max_tokens=150 keeps
        # only the first.
        searcher = FakeSearcher(result=_make_search_result())
        server = _make_server(searcher=searcher)
        out = asyncio.run(
            server._dispatch_tool(
                "search_codebase",
                {"query": "x", "max_tokens": 150},
            )
        )
        assert len(out["chunks"]) == 1
        assert out["chunks"][0]["file_path"] == "src/auth/middleware.ts"

    def test_max_tokens_keeps_at_least_one_chunk(self):
        """Tiny budget shouldn't return empty when there's a hit."""
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool(
                "search_codebase",
                {"query": "x", "max_tokens": 1},
            )
        )
        assert len(out["chunks"]) == 1

    def test_max_results_caps_chunk_count(self):
        searcher = FakeSearcher(result=_make_search_result())
        server = _make_server(searcher=searcher)
        out = asyncio.run(
            server._dispatch_tool(
                "search_codebase",
                {"query": "x", "max_tokens": 100_000, "max_results": 1},
            )
        )
        assert len(out["chunks"]) == 1

    def test_args_forwarded_to_searcher(self):
        searcher = FakeSearcher(result=_make_search_result())
        server = _make_server(searcher=searcher)
        asyncio.run(
            server._dispatch_tool(
                "search_codebase",
                {
                    "query": "auth",
                    "cwd": "/x/y",
                    "project": "myapp",
                    "max_tokens": 500,
                    "max_results": 3,
                    "wait_for_quiescence_ms": 1000,
                },
            )
        )
        assert searcher.last_call["query"] == "auth"
        assert searcher.last_call["cwd"] == "/x/y"
        assert searcher.last_call["project"] == "myapp"
        assert searcher.last_call["max_tokens"] == 500
        assert searcher.last_call["max_results"] == 3
        assert searcher.last_call["wait_for_quiescence_ms"] == 1000

    def test_async_searcher_supported(self):
        class AsyncSearcher:
            def __init__(self, result):
                self._result = result
                self.last_call = None

            async def search(self, **kwargs):
                self.last_call = kwargs
                return self._result

        searcher = AsyncSearcher(_make_search_result())
        server = _make_server(searcher=searcher)
        out = asyncio.run(
            server._dispatch_tool("search_codebase", {"query": "x"})
        )
        assert len(out["chunks"]) == 2

    def test_stale_files_propagated(self):
        searcher = FakeSearcher(
            result=_make_search_result(
                is_fresh=False, pending=2, stale=("src/old.ts",)
            )
        )
        server = _make_server(searcher=searcher)
        out = asyncio.run(server._dispatch_tool("search_codebase", {"query": "x"}))
        assert out["is_fully_fresh"] is False
        assert out["pending_updates"] == 2
        assert out["stale_files"] == ["src/old.ts"]


# ============================================================ list_projects
class TestListProjects:
    def test_returns_per_project_stats(self):
        proj = Project(name="myapp", root_path="/x", last_full_scan_at=42)
        files = [FileRecord(id=1, project="myapp", rel_path="a.py",
                            mtime=0, size_bytes=0, content_hash="h")]
        # We need a Chunk for the token_count summing path. Importing
        # locally to keep the top-level imports clean.
        from longctx_daemon.types import Chunk
        chunks = [
            Chunk(id=1, file_id=1, chunk_index=0, start_offset=0, end_offset=10,
                  start_line=1, end_line=2, token_count=15, content_hash="h",
                  text="x", embedder_model="m", embedder_sha256="s",
                  embedding_row=0)
        ]
        store = FakeChunkStore(
            projects=[proj],
            files_by_project={"myapp": files},
            chunks_by_file={1: chunks},
        )
        server = _make_server(chunk_store=store)
        out = asyncio.run(server._dispatch_tool("list_projects", {}))
        assert "projects" in out
        p = out["projects"][0]
        assert p["name"] == "myapp"
        assert p["root_path"] == "/x"
        assert p["file_count"] == 1
        assert p["chunk_count"] == 1
        assert p["token_count"] == 15
        assert p["last_updated"] == 42

    def test_empty_project_list(self):
        server = _make_server(chunk_store=FakeChunkStore(projects=[]))
        out = asyncio.run(server._dispatch_tool("list_projects", {}))
        assert out["projects"] == []


# ============================================================ index_status
class TestIndexStatus:
    def test_returns_flattened_status(self):
        server = _make_server()
        out = asyncio.run(server._dispatch_tool("index_status", {}))
        assert out["status"] == "ready"
        assert out["total_chunks"] == 42
        assert out["pending_updates"] == 0
        assert "embedder_model" in out
        assert "embedder_sha256" in out
        # projects is flattened to list of dicts.
        assert isinstance(out["projects"], list)
        assert out["projects"][0]["name"] == "myapp"

    def test_error_state(self):
        indexer = FakeIndexer(
            status_value=_make_index_status(status="error", error="disk full")
        )
        server = _make_server(indexer=indexer)
        out = asyncio.run(server._dispatch_tool("index_status", {}))
        assert out["status"] == "error"
        assert out["error"] == "disk full"


# ============================================================ Phase 2.1 tools
class TestPhase21Tools:
    """Sanity smoke for the four Phase-2.1 tools that were stubs in 2.0.

    The full behavioral coverage lives in
    ``test_mcp_set_active_project.py``, ``test_mcp_add_project.py``,
    ``test_mcp_find_related.py``, and ``test_mcp_wait_for_quiescence.py``.
    Here we only check the dispatcher routes them away from the
    not_implemented sentinel.
    """

    def test_set_active_project_no_longer_stub(self):
        # Unknown project triggers the validation branch, but the
        # response is the live error shape, not the 2.0 stub sentinel.
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("set_active_project", {"project": "absent"})
        )
        assert "status" not in out or out.get("status") != "not_implemented_in_2_0"
        assert "error" in out

    def test_wait_for_quiescence_no_watcher_returns_quiescent(self):
        server = _make_server()
        out = asyncio.run(server._dispatch_tool("wait_for_quiescence", {}))
        assert out["pending_updates"] == 0
        assert "indexed_through" in out

    def test_find_related_without_embed_store_errors(self):
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("find_related", {"file_path": "x/y.py"})
        )
        assert "error" in out

    def test_add_project_rejects_nonexistent_path(self, tmp_path):
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool(
                "add_project",
                {"path": str(tmp_path / "definitely-not-here")},
            )
        )
        assert "error" in out


# ============================================================ trace logging
class TestTraceLogging:
    def test_call_emits_log_line_with_trace_id(self, captured_lines):
        server = _make_server()
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "hi"}))
        # Find the mcp_call line.
        mcp_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_call"
        ]
        assert len(mcp_lines) == 1
        extra = mcp_lines[0]["record"]["extra"]
        assert extra["trace_id"] is not None
        assert len(extra["trace_id"]) == 26
        assert extra["tool"] == "search_codebase"
        assert "scope" in extra
        assert "latency_ms" in extra

    def test_summary_omits_chunk_text(self, captured_lines):
        server = _make_server()
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "hi"}))
        mcp_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_call"
        ]
        result = mcp_lines[0]["record"]["extra"]["result"]
        # Summary contains files + counts + freshness, NOT chunk text.
        assert "chunk_count" in result
        assert "files" in result
        assert "is_fully_fresh" in result
        # text not echoed
        assert "text" not in json.dumps(result)

    def test_each_call_gets_unique_trace_id(self, captured_lines):
        server = _make_server()
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "a"}))
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "b"}))
        ids = [
            json.loads(c)["record"]["extra"]["trace_id"]
            for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_call"
        ]
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_unknown_client_when_no_session(self, captured_lines):
        server = _make_server()
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "x"}))
        mcp_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_call"
        ]
        client = mcp_lines[0]["record"]["extra"]["client"]
        # No MCP session active in the direct-dispatch path.
        assert client == {"name": "unknown", "version": "unknown"}


# ============================================================ error path
class TestErrorPath:
    def test_searcher_error_propagates(self, captured_lines):
        class BoomSearcher:
            def search(self, **kw):
                raise RuntimeError("boom")

        server = _make_server(searcher=BoomSearcher())
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(server._dispatch_tool("search_codebase", {"query": "x"}))

        # An ERROR-level mcp_error log line should appear.
        err_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_error"
        ]
        assert len(err_lines) == 1
        extra = err_lines[0]["record"]["extra"]
        assert extra["error"] == "boom"
        assert extra["tool"] == "search_codebase"
        assert extra["level"] == "ERROR"

    def test_unknown_tool_logs_error(self, captured_lines):
        server = _make_server()
        with pytest.raises(ValueError):
            asyncio.run(server._dispatch_tool("nonexistent", {}))
        err_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_error"
        ]
        assert len(err_lines) == 1


# ============================================================ replay log
class TestReplayLog:
    def test_full_payload_recorded(self, tmp_path: Path):
        replay = ReplayLog(tmp_path / "interactions.jsonl")
        server = _make_server(replay_log=replay)
        asyncio.run(server._dispatch_tool("search_codebase", {"query": "x"}))
        replay.close()

        line = (tmp_path / "interactions.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["tool"] == "search_codebase"
        # Full chunk payload is in result, including text.
        assert record["result"]["chunks"][0]["text"]
        assert "trace_id" in record

    def test_error_recorded_as_replay_entry(self, tmp_path: Path):
        class BoomSearcher:
            def search(self, **kw):
                raise RuntimeError("kaboom")

        replay = ReplayLog(tmp_path / "interactions.jsonl")
        server = _make_server(searcher=BoomSearcher(), replay_log=replay)
        with pytest.raises(RuntimeError):
            asyncio.run(server._dispatch_tool("search_codebase", {"query": "x"}))
        replay.close()

        line = (tmp_path / "interactions.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["error"] == "kaboom"
        assert record["tool"] == "search_codebase"


# ============================================================ end-to-end via SDK
@asynccontextmanager
async def _connected_session(server: MCPServer, *, client_info: Optional[Implementation] = None):
    """Drive the actual MCP server through its underlying low-level
    Server using the SDK's in-memory transport. Yields a
    ``mcp.client.session.ClientSession`` already initialized."""
    async with create_connected_server_and_client_session(
        server._server,
        client_info=client_info,
        raise_exceptions=True,
    ) as client:
        yield client


class TestEndToEnd:
    def test_list_tools_advertises_seven_tools(self):
        async def _run():
            server = _make_server()
            async with _connected_session(server) as client:
                resp = await client.list_tools()
                names = [t.name for t in resp.tools]
                return names

        names = asyncio.run(_run())
        assert set(names) == {
            "search_codebase",
            "list_projects",
            "index_status",
            "find_related",
            "set_active_project",
            "add_project",
            "wait_for_quiescence",
        }

    def test_search_codebase_returns_structured_content(self):
        async def _run():
            server = _make_server()
            async with _connected_session(server) as client:
                resp = await client.call_tool(
                    "search_codebase",
                    arguments={"query": "auth"},
                )
                return resp

        resp = asyncio.run(_run())
        assert resp.isError is False
        # The SDK wraps dict returns in structuredContent.
        assert resp.structuredContent is not None
        assert "chunks" in resp.structuredContent
        assert resp.structuredContent["is_fully_fresh"] is True

    def test_client_info_captured_in_log(self, captured_lines):
        async def _run():
            server = _make_server()
            async with _connected_session(
                server,
                client_info=Implementation(name="opencode", version="0.4.2"),
            ) as client:
                await client.call_tool("search_codebase", arguments={"query": "x"})

        asyncio.run(_run())
        mcp_lines = [
            json.loads(c) for c in captured_lines
            if json.loads(c)["record"].get("message") == "mcp_call"
        ]
        assert len(mcp_lines) >= 1
        client = mcp_lines[-1]["record"]["extra"]["client"]
        assert client == {"name": "opencode", "version": "0.4.2"}

    def test_phase21_tool_via_sdk(self):
        # set_active_project is now live — unknown project returns
        # an explicit error response (not a 2.0 sentinel).
        async def _run():
            server = _make_server()
            async with _connected_session(server) as client:
                resp = await client.call_tool("set_active_project",
                                              arguments={"project": "x"})
                return resp

        resp = asyncio.run(_run())
        assert resp.isError is False
        sc = resp.structuredContent
        assert "error" in sc
        assert sc["error"].startswith("unknown project")

    def test_search_with_input_validation(self):
        """Missing required ``query`` arg should be rejected by the SDK
        because the inputSchema marks query as required."""
        async def _run():
            server = _make_server()
            async with _connected_session(server) as client:
                resp = await client.call_tool("search_codebase", arguments={})
                return resp

        # raise_exceptions=True propagates the protocol error as a
        # CallToolResult with isError=True (input validation falls
        # back to a result error).
        resp = asyncio.run(_run())
        assert resp.isError is True

    def test_max_tokens_via_sdk(self):
        async def _run():
            server = _make_server()
            async with _connected_session(server) as client:
                resp = await client.call_tool(
                    "search_codebase",
                    arguments={"query": "x", "max_tokens": 100},
                )
                return resp

        resp = asyncio.run(_run())
        assert resp.isError is False
        # 100 tokens < 120 of first chunk → keeps 1 (always-at-least-one).
        assert len(resp.structuredContent["chunks"]) == 1


# -------------------------- suggested_followup hint helper

def test_suggested_followup_no_relevant_results_suggests_prior_context():
    from longctx_daemon.mcp_server import _build_suggested_followup
    hint = _build_suggested_followup(
        current_query="anything",
        current_chunk_ids=(),
        current_quality="abstain",
        no_relevant_results=True,
        recent_searches=[],
    )
    assert hint is not None
    assert hint["action"] == "prior_context"
    assert "prior_context" in hint["reason"]


def test_suggested_followup_recurring_chunk_suggests_suppress_ids():
    from longctx_daemon.mcp_server import _build_suggested_followup
    # Earlier query returned chunk 42; current query returns it again.
    hint = _build_suggested_followup(
        current_query="round 2",
        current_chunk_ids=(42, 99),
        current_quality="medium",
        no_relevant_results=False,
        recent_searches=[("round 1", (42, 7))],
    )
    assert hint is not None
    assert hint["action"] == "suppress_ids"
    assert 42 in hint["values"]
    assert 99 in hint["values"]


def test_suggested_followup_silent_on_clean_first_shot():
    """High-quality first-shot with no recurring chunks → no hint
    (agent has what it needs; don't nag)."""
    from longctx_daemon.mcp_server import _build_suggested_followup
    hint = _build_suggested_followup(
        current_query="clean query",
        current_chunk_ids=(1, 2, 3),
        current_quality="high",
        no_relevant_results=False,
        recent_searches=[("prior", (99, 100))],
    )
    assert hint is None


def test_suggested_followup_silent_on_medium_quality_no_recurring():
    """Medium quality but no recurring chunks → no hint."""
    from longctx_daemon.mcp_server import _build_suggested_followup
    hint = _build_suggested_followup(
        current_query="fresh angle",
        current_chunk_ids=(10, 20),
        current_quality="medium",
        no_relevant_results=False,
        recent_searches=[("earlier", (50, 60))],
    )
    assert hint is None
