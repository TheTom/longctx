"""MCP server (stdio transport) for the longctx daemon — Phase 2.0.

Phase 2.0 ships ONLY the stdio transport. SSE / streamable-http land in
2.1 (PRD §6.1).

Tools live: search_codebase, list_projects, index_status.
Tools registered as not-yet-implemented stubs so the spec surface is
visible to MCP clients but obviously inactive:
  set_active_project, add_project, find_related, wait_for_quiescence

Each tool invocation:
  1. Generates a fresh trace_id (ULID).
  2. Captures clientInfo from the initialize request (§14.6).
  3. Calls ``set_trace_context`` so downstream sub-events get the
     same trace_id (§14.5).
  4. Runs the tool body.
  5. Emits a §14.4 per-call log line via ``log_mcp_call`` and (if the
     ReplayLog is wired in) a full payload line via ``ReplayLog.record``.
  6. On exception, emits ``log_mcp_error`` and propagates.

The class takes a duck-typed ``searcher``, ``indexer`` and
``chunk_store``: in tests we pass mocks; in production the daemon
wires them up. The contract is:
  * searcher.search(query, *, cwd, project, max_tokens, max_results,
    wait_for_quiescence_ms) -> SearchResult
  * indexer.status() -> IndexStatus
  * chunk_store.list_projects() -> tuple[Project, ...]
  * chunk_store.list_files(project=name) -> tuple[FileRecord, ...]
  * chunk_store.get_chunks_by_file(file_id) -> tuple[Chunk, ...] (used
    only to compute the per-project token_count when we don't already
    have a cheaper way).
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from time import perf_counter
from typing import Any, Optional

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from longctx_daemon.logging import (
    log_mcp_call,
    log_mcp_error,
    new_trace_id,
    set_trace_context,
)
from longctx_daemon.replay_log import ReplayLog
from longctx_daemon.types import (
    IndexStatus,
    SearchResult,
)


# ----------------------------------------------------------- tool docstrings
# Spec §6.4 — copy verbatim, don't paraphrase. Agents read these.

_SEARCH_CODEBASE_DOC = (
    "Search the indexed codebase for code, comments, or documentation\n"
    "relevant to the query. Returns top-matching chunks with file paths,\n"
    "line numbers, and relevance scores.\n"
    "\n"
    "Args:\n"
    "    query: Natural language description of what to find. A question,\n"
    "        a code-pattern description, a feature name, or an identifier.\n"
    "    cwd: Optional path the agent is currently working in. Used to bias\n"
    "        results toward the active project. If None, falls back to the\n"
    "        session's sticky active project (set via set_active_project).\n"
    "    project: Optional project name to restrict search to. Available\n"
    "        projects can be enumerated via list_projects.\n"
    "    max_tokens: Cap the total token count of returned chunks. Default\n"
    "        4096. Agents with smaller context windows pass less.\n"
    "    max_results: Optional cap on chunk count. Use max_tokens preferentially;\n"
    "        this is a fallback for callers that prefer count-based limits.\n"
    "    wait_for_quiescence_ms: Block up to N ms waiting for in-flight index\n"
    "        updates to drain before searching. Default 500. Pass 0 for\n"
    "        zero-wait; pass 2000 if the agent just wrote many files.\n"
    "\n"
    "Returns:\n"
    "    SearchResponse with:\n"
    "        chunks: list of {project, file_path, start_line, end_line,\n"
    "                          text, relevance_score}\n"
    "        stale_files: list of files whose mtime > last_indexed_at\n"
    "        pending_updates: queue size at response time (0 = fully fresh)\n"
    "        indexed_through: ISO timestamp of last completed index update\n"
)

_FIND_RELATED_DOC = (
    "Find chunks semantically similar to the chunk at file_path:line\n"
    "(or to the whole file if line is None).\n"
    "\n"
    "Phase 2 implementation: dense embedding similarity only. The chunk at\n"
    "file_path:line is embedded; the top-K nearest other chunks are\n"
    "returned. Useful for \"show me similar implementations in other parts\n"
    "of the codebase\" or \"what else looks like this\".\n"
    "\n"
    "NOT a static-analysis tool — it does not parse callers/callees,\n"
    "follow imports, or resolve symbol references. Static-analysis-aware\n"
    "variants (callers_of, callees_of, references_to) are planned for\n"
    "Phase 3+ and will be separate tools so the agent never has to guess\n"
    "which kind of \"related\" it asked for.\n"
    "\n"
    "(ships in 2.1)"
)

_LIST_PROJECTS_DOC = (
    "List all projects currently indexed, with stats:\n"
    "name, root_path, file_count, chunk_count, token_count, last_updated."
)

_SET_ACTIVE_PROJECT_DOC = (
    "Set the sticky active project for this MCP session. Subsequent\n"
    "search_codebase calls without cwd or project args use this scope.\n"
    "\n"
    "(ships in 2.1)"
)

_ADD_PROJECT_DOC = (
    "Index a new directory. persist=True writes to config (survives\n"
    "daemon restart); persist=False indexes for this session only.\n"
    "\n"
    "(ships in 2.1)"
)

_WAIT_FOR_QUIESCENCE_DOC = (
    "Block until the index has no pending updates for the given project\n"
    "(or all projects). Returns when queue is empty or timeout fires.\n"
    "Useful between 'I just wrote N files' and 'now search for them'.\n"
    "\n"
    "(ships in 2.1)"
)

_INDEX_STATUS_DOC = (
    "Current daemon state: status, total_chunks, pending_updates,\n"
    "embedder_model + sha256, last_full_scan, per-project stats."
)


# --------------------------------------------------------- input schemas
_SEARCH_CODEBASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "cwd": {"type": ["string", "null"]},
        "project": {"type": ["string", "null"]},
        "max_tokens": {"type": "integer", "default": 4096},
        "max_results": {"type": ["integer", "null"]},
        "wait_for_quiescence_ms": {"type": ["integer", "null"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_LIST_PROJECTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_INDEX_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_FIND_RELATED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "line": {"type": ["integer", "null"]},
        "max_results": {"type": "integer", "default": 5},
    },
    "required": ["file_path"],
    "additionalProperties": False,
}

_SET_ACTIVE_PROJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"project": {"type": "string"}},
    "required": ["project"],
    "additionalProperties": False,
}

_ADD_PROJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "persist": {"type": "boolean", "default": False},
    },
    "required": ["path"],
    "additionalProperties": False,
}

_WAIT_FOR_QUIESCENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "project": {"type": ["string", "null"]},
        "timeout_ms": {"type": "integer", "default": 2000},
    },
    "additionalProperties": False,
}


# Sentinel for not-yet-implemented stubs.
_STUB_RESPONSE: dict[str, Any] = {"status": "not_implemented_in_2_0"}


# ============================================================ MCPServer
class MCPServer:
    """Wraps a low-level ``mcp.server.Server`` with longctx-specific
    tool handlers and the §14.4 trace-logging wrapper.

    Pass mocks for ``searcher`` / ``indexer`` / ``chunk_store`` in tests.
    """

    def __init__(
        self,
        searcher: Any,
        indexer: Any,
        chunk_store: Any,
        *,
        server_name: str = "longctx",
        server_version: str = "0.2.0",
        replay_log: Optional[ReplayLog] = None,
        connection_id: Optional[str] = None,
    ) -> None:
        self.searcher = searcher
        self.indexer = indexer
        self.chunk_store = chunk_store
        self.server_name = server_name
        self.server_version = server_version
        self.replay_log = replay_log
        # One MCPServer instance corresponds to one stdio MCP connection,
        # so the connection_id is set at construction. For SSE / streamable
        # this'll move to per-connection state in 2.1.
        self.connection_id = connection_id or new_trace_id()

        self._server: Server = Server(server_name, version=server_version)
        self._register_handlers()

    # ----------------------------------------------------- public entrypoint
    async def run_stdio(self) -> None:
        """Block on the stdio MCP transport.

        Returns when the client closes the streams. The MCP SDK handles
        the initialize / capabilities handshake; we just hook our tool
        handlers and let it run.
        """
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    # --------------------------------------------------- handler registration
    def _register_handlers(self) -> None:
        """Register list_tools and call_tool with the underlying SDK
        server. We inject the trace-logging wrapper at the call_tool
        layer so every tool gets it for free."""

        @self._server.list_tools()
        async def _list_tools() -> list[Tool]:
            return [
                Tool(
                    name="search_codebase",
                    description=_SEARCH_CODEBASE_DOC,
                    inputSchema=_SEARCH_CODEBASE_SCHEMA,
                ),
                Tool(
                    name="list_projects",
                    description=_LIST_PROJECTS_DOC,
                    inputSchema=_LIST_PROJECTS_SCHEMA,
                ),
                Tool(
                    name="index_status",
                    description=_INDEX_STATUS_DOC,
                    inputSchema=_INDEX_STATUS_SCHEMA,
                ),
                Tool(
                    name="find_related",
                    description=_FIND_RELATED_DOC,
                    inputSchema=_FIND_RELATED_SCHEMA,
                ),
                Tool(
                    name="set_active_project",
                    description=_SET_ACTIVE_PROJECT_DOC,
                    inputSchema=_SET_ACTIVE_PROJECT_SCHEMA,
                ),
                Tool(
                    name="add_project",
                    description=_ADD_PROJECT_DOC,
                    inputSchema=_ADD_PROJECT_SCHEMA,
                ),
                Tool(
                    name="wait_for_quiescence",
                    description=_WAIT_FOR_QUIESCENCE_DOC,
                    inputSchema=_WAIT_FOR_QUIESCENCE_SCHEMA,
                ),
            ]

        @self._server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await self._dispatch_tool(name, arguments)

    # -------------------------------------------------------- tool dispatch
    async def _dispatch_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Wrap a tool invocation in trace context + structured logging.

        We separate this from the @call_tool handler so tests can call
        it directly without driving the whole MCP transport.
        """
        # Capture clientInfo from the active session, if available.
        client_name, client_version = self._extract_client_info()

        trace_id = new_trace_id()
        session_id = self._extract_session_id()
        set_trace_context(
            trace_id=trace_id,
            session_id=session_id,
            connection_id=self.connection_id,
            client_name=client_name,
            client_version=client_version,
        )

        t0 = perf_counter()
        try:
            handler = self._handler_for(name)
            if handler is None:
                # Unknown tool. Raise — the outer except clause does the
                # error-line emission so we don't double-log.
                raise ValueError(f"unknown tool: {name}")

            result, scope_dict, latency_dict, summary = await handler(arguments)

            # Operational log: short summary only.
            log_mcp_call(
                tool=name,
                args=arguments,
                scope=scope_dict,
                latency_ms=latency_dict,
                result=summary,
            )

            # Replay log: full request + full response (§14.8).
            if self.replay_log is not None:
                self.replay_log.record(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "connection_id": self.connection_id,
                        "client": {"name": client_name, "version": client_version},
                        "tool": name,
                        "args": arguments,
                        "scope": scope_dict,
                        "latency_ms": latency_dict,
                        "result": result,
                    }
                )
            return result
        except Exception as e:
            elapsed = (perf_counter() - t0) * 1000.0
            log_mcp_error(tool=name, args=arguments, error=str(e))
            if self.replay_log is not None:
                self.replay_log.record(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "connection_id": self.connection_id,
                        "client": {"name": client_name, "version": client_version},
                        "tool": name,
                        "args": arguments,
                        "error": str(e),
                        "latency_ms": {"total": elapsed},
                    }
                )
            raise

    def _handler_for(self, name: str):
        """Resolve a tool name → coroutine (args) → (result, scope,
        latency, summary). Returns None for unknown names."""
        return {
            "search_codebase": self._handle_search_codebase,
            "list_projects": self._handle_list_projects,
            "index_status": self._handle_index_status,
            "find_related": self._handle_stub("find_related"),
            "set_active_project": self._handle_stub("set_active_project"),
            "add_project": self._handle_stub("add_project"),
            "wait_for_quiescence": self._handle_stub("wait_for_quiescence"),
        }.get(name)

    # ============================================================ live tools
    async def _handle_search_codebase(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Implements PRD §6.2 search_codebase + §6.3 response invariants."""
        query = args["query"]
        cwd = args.get("cwd")
        project = args.get("project")
        max_tokens = args.get("max_tokens", 4096)
        max_results = args.get("max_results")
        wait_ms = args.get("wait_for_quiescence_ms")

        # The searcher may be sync or async; await accordingly.
        result = self.searcher.search(
            query=query,
            cwd=cwd,
            project=project,
            max_tokens=max_tokens,
            max_results=max_results,
            wait_for_quiescence_ms=wait_ms,
        )
        if asyncio.iscoroutine(result):
            result = await result

        if not isinstance(result, SearchResult):
            raise TypeError(
                f"searcher.search returned {type(result).__name__}, "
                "expected SearchResult"
            )

        # Trim chunks to max_tokens. We honor the caller's budget by
        # truncating in rank order; an explicit max_results overrides
        # the count cap. The original SearchResult is left untouched
        # so the replay log keeps the full payload.
        kept_chunks: list[dict[str, Any]] = []
        running_tokens = 0
        for sc in result.chunks:
            if max_results is not None and len(kept_chunks) >= max_results:
                break
            if running_tokens + sc.token_count > max_tokens and kept_chunks:
                # Stop once adding this chunk would push us over budget,
                # but always keep at least one chunk so a user with a
                # tiny budget gets the top hit instead of an empty list.
                break
            kept_chunks.append(_search_chunk_to_dict(sc))
            running_tokens += sc.token_count

        response: dict[str, Any] = {
            "chunks": kept_chunks,
            "is_fully_fresh": result.freshness.is_fully_fresh,
            "stale_files": list(result.freshness.stale_files),
            "pending_updates": result.freshness.pending_updates,
            "indexed_through": result.freshness.indexed_through,
            "scope_decision": _dataclass_to_flat_dict(result.scope_decision),
        }

        scope_dict = response["scope_decision"]
        latency_dict = _dataclass_to_flat_dict(result.latency_ms)

        # Operational summary (no chunk text — that's in the replay log).
        summary = {
            "chunk_count": len(kept_chunks),
            "files": [
                f"{c['file_path']}:{c['start_line']}-{c['end_line']}"
                for c in kept_chunks
            ],
            "is_fully_fresh": response["is_fully_fresh"],
            "pending_updates": response["pending_updates"],
            "indexed_through": response["indexed_through"],
        }
        return response, scope_dict, latency_dict, summary

    async def _handle_list_projects(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Returns {"projects": [...]} with per-project stats."""
        t0 = perf_counter()
        projects = self.chunk_store.list_projects()
        out: list[dict[str, Any]] = []
        for proj in projects:
            files = self.chunk_store.list_files(project=proj.name)
            file_count = len(files)
            chunk_count = 0
            token_count = 0
            for f in files:
                chunks = self.chunk_store.get_chunks_by_file(f.id)
                chunk_count += len(chunks)
                token_count += sum(c.token_count for c in chunks)
            out.append(
                {
                    "name": proj.name,
                    "root_path": proj.root_path,
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "token_count": token_count,
                    "last_updated": proj.last_full_scan_at,
                }
            )
        elapsed = (perf_counter() - t0) * 1000.0
        return (
            {"projects": out},
            None,
            {"total": elapsed},
            {"project_count": len(out)},
        )

    async def _handle_index_status(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Flatten ``IndexStatus`` to a JSON-friendly dict."""
        t0 = perf_counter()
        status = self.indexer.status()
        if not isinstance(status, IndexStatus):
            raise TypeError(
                f"indexer.status returned {type(status).__name__}, "
                "expected IndexStatus"
            )
        flat = _dataclass_to_flat_dict(status)
        # ``projects`` is a tuple[Project, ...] — flatten each.
        flat["projects"] = [_dataclass_to_flat_dict(p) for p in status.projects]
        elapsed = (perf_counter() - t0) * 1000.0
        summary = {
            "status": status.status,
            "total_chunks": status.total_chunks,
            "pending_updates": status.pending_updates,
            "project_count": len(status.projects),
        }
        return flat, None, {"total": elapsed}, summary

    # ============================================================ stubs
    def _handle_stub(self, tool_name: str):
        """Build a stub coroutine that records the spec surface but
        returns ``not_implemented_in_2_0`` cleanly."""

        async def _stub(args: dict[str, Any]):
            response = dict(_STUB_RESPONSE)
            response["tool"] = tool_name
            response["ships_in"] = "2.1"
            return response, None, {"total": 0.0}, response

        return _stub

    # ============================================================ helpers
    def _extract_client_info(self) -> tuple[str, str]:
        """Read ``clientInfo`` off the active MCP session, if any.

        Falls back to ("unknown", "unknown") when called outside a
        request context (eg unit tests that drive _dispatch_tool
        directly) or when the client used an older protocol that
        didn't send ``clientInfo`` (§14.6).
        """
        try:
            ctx = self._server.request_context
        except LookupError:
            return ("unknown", "unknown")
        try:
            params = ctx.session._client_params  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            return ("unknown", "unknown")
        if params is None or params.clientInfo is None:
            return ("unknown", "unknown")
        info = params.clientInfo
        return (info.name or "unknown", info.version or "unknown")

    def _extract_session_id(self) -> str:
        """Stable session ID for the current MCP connection.

        We use ``id(session)`` as a process-local fingerprint so all
        calls on the same session share an ID. Outside a request
        context (tests), we synthesize a stub.
        """
        try:
            ctx = self._server.request_context
        except LookupError:
            return f"ses_{self.connection_id[:12]}"
        return f"ses_{id(ctx.session):x}"


# ----------------------------------------------------------- shape helpers
def _search_chunk_to_dict(sc) -> dict[str, Any]:
    """SearchChunk → flat dict matching §6.3."""
    return {
        "project": sc.citation.project,
        "file_path": sc.citation.file_path,
        "start_line": sc.citation.start_line,
        "end_line": sc.citation.end_line,
        "text": sc.text,
        "relevance_score": sc.relevance_score,
    }


def _dataclass_to_flat_dict(obj: Any) -> dict[str, Any]:
    """Best-effort flattening for dataclass instances + nested tuples.

    Frozen dataclasses (which all our types are) work with asdict;
    we additionally coerce tuple fields to lists for JSON cleanliness.
    """
    if not is_dataclass(obj):
        raise TypeError(f"_dataclass_to_flat_dict: not a dataclass: {obj!r}")
    raw = asdict(obj)
    return {k: _jsonify(v) for k, v in raw.items()}


def _jsonify(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonify(x) for x in v]
    if isinstance(v, list):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    return v


__all__ = ["MCPServer"]
