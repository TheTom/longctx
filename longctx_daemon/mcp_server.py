"""MCP server (stdio + SSE transports) for the longctx daemon.

Phase 2.0 shipped stdio + the four "live" tools (search_codebase,
list_projects, index_status). Phase 2.1 makes the four spec-stubs real:
``set_active_project``, ``add_project``, ``find_related``, and
``wait_for_quiescence`` — see PRD §3.4, §3.8, §6.2, §6.4.

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
    wait_for_quiescence_ms, active_project_sticky) -> SearchResult
  * indexer.add_project(name, root_path) -> Project
  * indexer.full_scan(project) -> ScanResult
  * indexer.status() -> IndexStatus
  * indexer.config: IndexerConfig (for the forbidden_dirs check)
  * chunk_store.list_projects() -> tuple[Project, ...]
  * chunk_store.list_files(project=name) -> tuple[FileRecord, ...]
  * chunk_store.get_chunks_by_file(file_id) -> tuple[Chunk, ...]
  * chunk_store.upsert_project(Project) -> None  (used to set/clear
    session_id on session-bound projects)
  * embed_store.get(row) / embed_store.search_dense(...)  (find_related)
  * watcher.wait_for_quiescence(project, timeout_ms) -> (pending, iso)
    (optional; daemon mode wires it, stdio passes None)
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

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
    "    prior_context: Optional text (string or list of strings) capturing\n"
    "        what you already know about the problem — prior error traces,\n"
    "        previous query phrasings, observations from earlier turns.\n"
    "        Mixed into the query embedding so the next round drifts\n"
    "        toward what you've learned. AutoCodeRover-style retry pattern.\n"
    "    prior_context_weight: How heavily to mix prior_context into the\n"
    "        query. 0.0 = ignore prior; 0.3 = light bias (default); 0.5+\n"
    "        = strong bias toward prior direction.\n"
    "    suppress_ids: List of chunk_id ints already shown to the agent\n"
    "        (read from prior response chunks). Filtered out so the next\n"
    "        round returns DIFFERENT chunks. Use this to surface new\n"
    "        evidence after a first round didn't help.\n"
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
    "which kind of \"related\" it asked for."
)

_LIST_PROJECTS_DOC = (
    "List all projects currently indexed, with stats:\n"
    "name, root_path, file_count, chunk_count, token_count, last_updated."
)

_SET_ACTIVE_PROJECT_DOC = (
    "Set the sticky active project for this MCP session. Subsequent\n"
    "search_codebase calls without cwd or project args use this scope."
)

_ADD_PROJECT_DOC = (
    "Index a new directory. persist=True writes to config (survives\n"
    "daemon restart); persist=False indexes for this session only."
)

_WAIT_FOR_QUIESCENCE_DOC = (
    "Block until the index has no pending updates for the given project\n"
    "(or all projects). Returns when queue is empty or timeout fires.\n"
    "Useful between 'I just wrote N files' and 'now search for them'."
)

_INDEX_STATUS_DOC = (
    "Current daemon state: status, total_chunks, pending_updates,\n"
    "embedder_model + sha256, last_full_scan, per-project stats."
)


# --------------------------------------------------------- input schemas
_SEARCH_CODEBASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Phase 2.0.1: query accepts string OR array. When array, each
        # element runs independently and the response shape becomes
        # {"groups": [{"query": ..., "chunks": [...]}, ...]} so the
        # caller knows which chunks came from which sub-question.
        "query": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
            ],
        },
        "cwd": {"type": ["string", "null"]},
        "project": {"type": ["string", "null"]},
        "max_tokens": {"type": "integer", "default": 4096},
        "max_results": {"type": ["integer", "null"]},
        "wait_for_quiescence_ms": {"type": ["integer", "null"]},
        # Phase 2.0.1: per-call override of the dense-cosine
        # relevance floor. Top-1 cosine below this returns empty
        # chunks + no_relevant_results=True. 0 disables.
        "relevance_floor": {"type": ["number", "null"]},
        # Phase 3: opt into context-size + query-shape adaptive
        # routing. When True, the searcher detects query shape +
        # estimates corpus size, looks up the policy table, and
        # overrides retrieval weights for THIS call. Surfaces
        # rationale + embedder hint on the response.
        "auto_policy": {"type": "boolean", "default": False},
        # 2026-05-19: iterative retrieval (AutoCodeRover-style retry).
        # prior_context mixes prior text/observations into the query
        # embedding; suppress_ids drops chunks the agent has already
        # been shown. See SearcherConfig docs + commit bb88867.
        "prior_context": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ],
        },
        "prior_context_weight": {"type": "number", "default": 0.3},
        "suppress_ids": {
            "oneOf": [
                {"type": "array", "items": {"type": "integer"}},
                {"type": "null"},
            ],
        },
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


# Sentinel for not-yet-implemented stubs (kept for backwards compat
# with any external code that still inspects the marker).
_STUB_RESPONSE: dict[str, Any] = {"status": "not_implemented_in_2_0"}


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z`` (per spec §6.3)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _source_chunk_summary(project: str, rel_path: str, chunk) -> dict[str, Any]:
    """Compact identifier for the find_related source chunk."""
    return {
        "project": project,
        "file_path": f"{project}/{rel_path}",
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


# ====================================================== ConnectionContext
@dataclass
class ConnectionContext:
    """Per-connection state for an MCP session.

    Phase 2.0 only had a single connection per process (stdio). 2.1's
    SSE + streamable-http multiplex many sessions through one daemon, so
    each session gets its own context with:

    * ``session_id`` — stable for the life of the SSE/HTTP session;
      surfaced in the §14.4 trace log so multi-session noise is
      separable in a tail.
    * ``connection_id`` — the same value the existing trace context
      uses; a new connection = a new ID.
    * ``cwd_history`` — last N cwd values seen via ``search_codebase``.
      Used by the scope inference layer (Agent G/H wire-up).
    * ``active_project_sticky`` — set via ``set_active_project``;
      used as the fallback scope when no cwd is provided. Not persisted
      across daemon restarts.

    Transports build a fresh context per accepted connection and stash
    it in ``_active_context_var`` for the duration of any
    ``_dispatch_tool`` call so the dispatcher sees the right state.
    """
    session_id: str
    connection_id: str
    cwd_history: list[str] = field(default_factory=list)
    active_project_sticky: Optional[str] = None
    client_name: str = "unknown"
    client_version: str = "unknown"


# Per-task active connection. Transports set this in their accept-loop
# before invoking the SDK; the dispatcher reads it under any active
# request. Falls back to MCPServer's default connection_id when None.
_active_context_var: ContextVar[Optional[ConnectionContext]] = ContextVar(
    "longctx_active_connection", default=None,
)


def set_active_connection(ctx: Optional[ConnectionContext]) -> Any:
    """Set the active ``ConnectionContext`` for the current task.

    Returns the contextvar token so the caller can ``reset`` later.
    """
    return _active_context_var.set(ctx)


def get_active_connection() -> Optional[ConnectionContext]:
    """Return the active connection context, or None outside one."""
    return _active_context_var.get()


def reset_active_connection(token: Any) -> None:
    """Reset the contextvar via the token returned by ``set_active_connection``."""
    _active_context_var.reset(token)


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
        embed_store: Any = None,
        watcher: Any = None,
        config_path: Optional[Path] = None,
        background_runner: Optional[Callable[[Callable[[], Any]], Any]] = None,
        server_name: str = "longctx",
        server_version: str = "0.2.0",
        replay_log: Optional[ReplayLog] = None,
        connection_id: Optional[str] = None,
    ) -> None:
        """Wire up the server.

        Args:
            searcher / indexer / chunk_store: required core services.
            embed_store: optional EmbedStore. Required only by
                ``find_related`` — passing ``None`` means find_related
                returns an error response.
            watcher: optional watcher exposing
                ``async wait_for_quiescence(project, timeout_ms)``.
                When ``None`` the wait_for_quiescence tool returns
                "no watcher attached, queue is empty" semantics.
            config_path: optional path to the daemon config TOML.
                ``add_project(persist=True)`` appends a project entry.
                Phase 2.1 adds best-effort write; Agent E owns the
                full schema.
            background_runner: callable that schedules a function in
                the daemon's worker pool. Used to kick off the
                ``add_project`` full_scan without blocking the MCP
                response. Defaults to ``asyncio.to_thread`` via the
                running loop.
            replay_log: optional ReplayLog for full-payload tracing.
            connection_id: stdio path's connection identity. Ignored
                when SSE/streamable-http transports set a fresh
                ``ConnectionContext`` per accept.
        """
        self.searcher = searcher
        self.indexer = indexer
        self.chunk_store = chunk_store
        self.embed_store = embed_store
        self._watcher = watcher
        self._config_path = config_path
        self._background_runner = background_runner
        self.server_name = server_name
        self.server_version = server_version
        self.replay_log = replay_log
        # One MCPServer instance corresponds to one stdio MCP connection,
        # so the connection_id is set at construction. For SSE / streamable
        # transports, per-connection state lives in ``ConnectionContext``
        # set via ``set_active_connection`` in the transport's accept loop.
        self.connection_id = connection_id or new_trace_id()
        # Stdio fallback session state. Used when no ConnectionContext
        # is active (i.e. unit tests + the stdio transport which has
        # exactly one session per process).
        self._fallback_context = ConnectionContext(
            session_id=f"ses_{self.connection_id[:12]}",
            connection_id=self.connection_id,
        )

        self._server: Server = Server(server_name, version=server_version)
        self._register_handlers()

    # ---------------------------------------------------------- accessors
    @property
    def server(self) -> Server:
        """Underlying low-level ``mcp.server.lowlevel.Server`` instance.

        Exposed so transports (SSE / streamable-http) can call
        ``server.run(read, write, init_options)`` without poking at the
        private ``_server`` attribute. The dispatcher and tool handlers
        are already registered.
        """
        return self._server

    def initialization_options(self):
        """Convenience proxy to the SDK's create_initialization_options."""
        return self._server.create_initialization_options()

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
        # Per-connection context (set by SSE/streamable-http transports)
        # takes precedence over instance-level state. Stdio transport
        # doesn't set one, so we fall back to ``self.connection_id``.
        active_ctx = get_active_connection()
        if active_ctx is not None:
            connection_id = active_ctx.connection_id
            session_id = active_ctx.session_id
            client_name = active_ctx.client_name
            client_version = active_ctx.client_version
        else:
            client_name, client_version = self._extract_client_info()
            connection_id = self.connection_id
            session_id = self._extract_session_id()

        trace_id = new_trace_id()
        set_trace_context(
            trace_id=trace_id,
            session_id=session_id,
            connection_id=connection_id,
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
                        "connection_id": connection_id,
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
                        "connection_id": connection_id,
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
            "find_related": self._handle_find_related,
            "set_active_project": self._handle_set_active_project,
            "add_project": self._handle_add_project,
            "wait_for_quiescence": self._handle_wait_for_quiescence,
        }.get(name)

    # -------------------------------------------------- session-state helper
    def _session_state(self) -> ConnectionContext:
        """Return the active per-connection state.

        Phase 2.1 transports (SSE / streamable-http) push a fresh
        ``ConnectionContext`` via ``set_active_connection`` for each
        accepted session. Stdio + tests fall back to
        ``self._fallback_context`` so the API surface is uniform.
        """
        active = get_active_connection()
        if active is not None:
            return active
        return self._fallback_context

    # ============================================================ live tools
    async def _handle_search_codebase(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Implements PRD §6.2 search_codebase + §6.3 response invariants.

        Phase 2.0.1: ``query`` accepts ``str | list[str]``. When a list
        is passed, each sub-query runs through the full pipeline
        independently and the response shape becomes
        ``{"groups": [<per-query SearchResult dict>, ...]}`` instead
        of a flat top-level chunks list. No merging or de-duplication
        across groups — the caller needs to know which chunks came from
        which sub-question.
        """
        query = args["query"]
        cwd = args.get("cwd")
        project = args.get("project")
        max_tokens = args.get("max_tokens", 4096)
        max_results = args.get("max_results")
        wait_ms = args.get("wait_for_quiescence_ms")
        relevance_floor = args.get("relevance_floor")
        auto_policy = bool(args.get("auto_policy", False))
        prior_context = args.get("prior_context")
        prior_context_weight = args.get("prior_context_weight", 0.3)
        suppress_ids = args.get("suppress_ids")

        # Sticky session: when caller didn't pass project= or cwd, use
        # the session's set_active_project value (PRD §3.4 / §3.9). The
        # searcher's scope-decision tier 3 (active_project_sticky) does
        # the actual routing — we just thread the value through.
        sticky = self._session_state().active_project_sticky

        # ---------- multi-question dispatch ----------
        if isinstance(query, list):
            return await self._handle_search_codebase_multi(
                queries=tuple(query),
                cwd=cwd, project=project,
                max_tokens=max_tokens, max_results=max_results,
                wait_ms=wait_ms,
                relevance_floor=relevance_floor,
                sticky=sticky,
                auto_policy=auto_policy,
                prior_context=prior_context,
                prior_context_weight=prior_context_weight,
                suppress_ids=suppress_ids,
            )

        # ---------- single-string path (Phase 2.0 contract) ----------
        kwargs: dict[str, Any] = {
            "query": query,
            "cwd": cwd,
            "project": project,
            "max_tokens": max_tokens,
            "max_results": max_results,
            "wait_for_quiescence_ms": wait_ms,
        }
        if sticky is not None:
            kwargs["active_project_sticky"] = sticky
        if relevance_floor is not None:
            kwargs["relevance_floor"] = relevance_floor
        if auto_policy:
            kwargs["auto_policy"] = True
        if prior_context is not None:
            kwargs["prior_context"] = prior_context
            kwargs["prior_context_weight"] = float(prior_context_weight)
        if suppress_ids:
            kwargs["suppress_ids"] = list(suppress_ids)
        try:
            result = self.searcher.search(**kwargs)
        except TypeError:
            # Older fakes don't take new kwargs — drop them progressively.
            kwargs.pop("suppress_ids", None)
            kwargs.pop("prior_context", None)
            kwargs.pop("prior_context_weight", None)
            kwargs.pop("active_project_sticky", None)
            kwargs.pop("relevance_floor", None)
            kwargs.pop("auto_policy", None)
            result = self.searcher.search(**kwargs)
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
            # Phase 2.0.1 honest-retrieval signals:
            "no_relevant_results": getattr(
                result, "no_relevant_results", False,
            ),
            "top1_dense_cosine": getattr(
                result, "top1_dense_cosine", 0.0,
            ),
            "query_type": getattr(
                result, "query_type", "natural_language",
            ),
            "confidence_gap": getattr(result, "confidence_gap", 0.0),
            # Phase 3 honest-retrieval signals (auto-policy +
            # retrieval_quality):
            "query_shape": getattr(result, "query_shape", "unknown"),
            "applied_policy_rationale": getattr(
                result, "applied_policy_rationale", "",
            ),
            "embedder_hint": getattr(result, "embedder_hint", ""),
            "retrieval_quality": getattr(
                result, "retrieval_quality", "unknown",
            ),
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
            "no_relevant_results": response["no_relevant_results"],
            "top1_dense_cosine": response["top1_dense_cosine"],
            "query_type": response["query_type"],
        }
        return response, scope_dict, latency_dict, summary

    async def _handle_search_codebase_multi(
        self,
        *,
        queries: tuple[str, ...],
        cwd: Optional[str],
        project: Optional[str],
        max_tokens: int,
        max_results: Optional[int],
        wait_ms: Optional[int],
        relevance_floor: Optional[float],
        sticky: Optional[str],
        auto_policy: bool = False,
        prior_context: Any = None,
        prior_context_weight: float = 0.3,
        suppress_ids: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Phase 2.0.1 multi-question path.

        Each sub-query runs through ``searcher.search`` independently;
        results are returned as ``groups: [<per-query SearchResult>]``
        with NO merging or de-duplication — the caller needs to know
        which chunk answered which sub-question.

        Falls back to ``searcher.search_multi`` when the searcher
        exposes it (Phase 2.0.1+); otherwise loops manually so older
        ``FakeSearcher`` test fixtures keep working.
        """
        # Try the rich search_multi entry point first.
        kwargs: dict[str, Any] = {
            "queries": queries,
            "cwd": cwd,
            "project": project,
            "max_tokens": max_tokens,
            "max_results": max_results,
            "wait_for_quiescence_ms": wait_ms,
        }
        if sticky is not None:
            kwargs["active_project_sticky"] = sticky
        if relevance_floor is not None:
            kwargs["relevance_floor"] = relevance_floor
        if auto_policy:
            kwargs["auto_policy"] = True
        if prior_context is not None:
            kwargs["prior_context"] = prior_context
            kwargs["prior_context_weight"] = float(prior_context_weight)
        if suppress_ids:
            kwargs["suppress_ids"] = list(suppress_ids)

        multi: Optional[Any] = None
        if hasattr(self.searcher, "search_multi"):
            try:
                multi = self.searcher.search_multi(**kwargs)
            except TypeError:
                kwargs.pop("suppress_ids", None)
                kwargs.pop("prior_context", None)
                kwargs.pop("prior_context_weight", None)
                kwargs.pop("auto_policy", None)
                kwargs.pop("relevance_floor", None)
                kwargs.pop("active_project_sticky", None)
                try:
                    multi = self.searcher.search_multi(**kwargs)
                except TypeError:
                    multi = None
            if asyncio.iscoroutine(multi):
                multi = await multi

        if multi is None:
            # Fallback: loop ``search`` per sub-query.
            groups: list[SearchResult] = []
            for q in queries:
                kw: dict[str, Any] = {
                    "query": q,
                    "cwd": cwd,
                    "project": project,
                    "max_tokens": max_tokens,
                    "max_results": max_results,
                    "wait_for_quiescence_ms": wait_ms,
                }
                if sticky is not None:
                    kw["active_project_sticky"] = sticky
                if relevance_floor is not None:
                    kw["relevance_floor"] = relevance_floor
                if auto_policy:
                    kw["auto_policy"] = True
                if prior_context is not None:
                    kw["prior_context"] = prior_context
                    kw["prior_context_weight"] = float(prior_context_weight)
                if suppress_ids:
                    kw["suppress_ids"] = list(suppress_ids)
                try:
                    r = self.searcher.search(**kw)
                except TypeError:
                    kw.pop("suppress_ids", None)
                    kw.pop("prior_context", None)
                    kw.pop("prior_context_weight", None)
                    kw.pop("active_project_sticky", None)
                    kw.pop("relevance_floor", None)
                    kw.pop("auto_policy", None)
                    r = self.searcher.search(**kw)
                if asyncio.iscoroutine(r):
                    r = await r
                groups.append(r)
        else:
            groups = list(multi.groups)

        # Build a per-group response dict by reusing _search_chunk_to_dict
        # logic. We keep the response shape symmetric with the single-
        # query path — each group has the same fields the agent would
        # see for a standalone search_codebase call.
        group_dicts: list[dict[str, Any]] = []
        for q, g in zip(queries, groups):
            chunks_d = [_search_chunk_to_dict(sc) for sc in g.chunks]
            group_dicts.append({
                "query": q,
                "chunks": chunks_d,
                "no_relevant_results": getattr(
                    g, "no_relevant_results", False,
                ),
                "top1_dense_cosine": getattr(
                    g, "top1_dense_cosine", 0.0,
                ),
                "query_type": getattr(g, "query_type", "natural_language"),
                "confidence_gap": getattr(g, "confidence_gap", 0.0),
                "query_shape": getattr(g, "query_shape", "unknown"),
                "applied_policy_rationale": getattr(
                    g, "applied_policy_rationale", "",
                ),
                "embedder_hint": getattr(g, "embedder_hint", ""),
                "retrieval_quality": getattr(
                    g, "retrieval_quality", "unknown",
                ),
            })

        # Shared fields: take the first group's scope; worst-case
        # freshness across groups; sum latencies.
        first = groups[0] if groups else None
        if first is not None:
            scope_dict = _dataclass_to_flat_dict(first.scope_decision)
        else:
            scope_dict = {}

        any_stale = any(
            (not g.freshness.is_fully_fresh) for g in groups
        )
        max_pending = max(
            (g.freshness.pending_updates for g in groups), default=0,
        )
        oldest_indexed = min(
            (g.freshness.indexed_through for g in groups),
            default=_now_iso(),
        )
        stale_union: set[str] = set()
        for g in groups:
            stale_union.update(g.freshness.stale_files)

        response: dict[str, Any] = {
            "groups": group_dicts,
            "is_fully_fresh": (not any_stale and max_pending == 0),
            "pending_updates": max_pending,
            "indexed_through": oldest_indexed,
            "stale_files": sorted(stale_union),
            "scope_decision": scope_dict,
        }

        latency_dict = {
            "wait_quiescence": sum(
                g.latency_ms.wait_quiescence for g in groups
            ),
            "embed_query": sum(g.latency_ms.embed_query for g in groups),
            "bm25_score": sum(g.latency_ms.bm25_score for g in groups),
            "dense_score": sum(g.latency_ms.dense_score for g in groups),
            "rrf_fuse": sum(g.latency_ms.rrf_fuse for g in groups),
            "fetch_chunks": sum(g.latency_ms.fetch_chunks for g in groups),
            "total": sum(g.latency_ms.total for g in groups),
        }

        summary = {
            "n_groups": len(group_dicts),
            "queries": list(queries),
            "chunk_counts": [len(g["chunks"]) for g in group_dicts],
            "no_relevant_results": [
                g["no_relevant_results"] for g in group_dicts
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

    # ============================================================ Phase 2.1 tools

    async def _handle_set_active_project(
        self, args: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """PRD §3.4 — pin a project for the current MCP session.

        Subsequent ``search_codebase`` calls without ``cwd`` / ``project``
        use this scope. State lives on the per-connection
        ``ConnectionContext`` so three terminals = three independent
        contexts; connection drops drop the state.
        """
        t0 = perf_counter()
        project = args["project"]
        # Validate the project exists. We list everything indexed so the
        # error response can include the available names — tiny hint that
        # saves an extra round-trip in the agent.
        proj_obj = self.chunk_store.get_project(project)
        if proj_obj is None:
            available = [p.name for p in self.chunk_store.list_projects()]
            response = {
                "error": f"unknown project: {project}",
                "available_projects": available,
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # Update session-scoped state.
        self._session_state().active_project_sticky = project
        elapsed = (perf_counter() - t0) * 1000.0
        response = {"status": "ok", "active_project": project}
        return response, None, {"total": elapsed}, response

    async def _handle_add_project(
        self, args: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """PRD §3.8 — runtime project addition.

        - Resolves ``path`` (~ expansion + absolute).
        - Verifies it exists + is a directory.
        - Refuses if basename matches ``forbidden_dirs``.
        - Refuses if a project with the same name (basename) is already
          indexed (returns informative error with the existing root).
        - ``persist=True`` writes a config entry; ``persist=False``
          binds the project to the current session via ``session_id``.
        - Kicks off a background ``full_scan`` and returns immediately
          with ``{"status": "indexing"}``. Caller polls
          ``wait_for_quiescence`` (or ``index_status``) to know when
          it's ready.
        """
        t0 = perf_counter()
        raw_path = args["path"]
        persist = bool(args.get("persist", False))

        try:
            abs_path = Path(raw_path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            response = {"error": f"invalid path: {raw_path} ({exc})"}
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        if not abs_path.exists():
            response = {"error": f"path does not exist: {abs_path}"}
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response
        if not abs_path.is_dir():
            response = {"error": f"path is not a directory: {abs_path}"}
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # Forbidden-dirs check. The indexer enforces this too via
        # ``ForbiddenProjectError`` at add_project time, but we surface
        # the error as a clean response rather than letting it bubble
        # into the per-call trace as an unhandled exception. We pull
        # the set off the indexer's config so behavior stays in sync.
        forbidden = self._forbidden_dirs()
        if abs_path.name in forbidden:
            response = {
                "error": (
                    f"refusing to index forbidden_dirs root: {abs_path.name}"
                ),
                "forbidden_dirs": sorted(forbidden),
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # Already indexed?
        name = abs_path.name
        existing = self.chunk_store.get_project(name)
        if existing is not None:
            response = {
                "error": "project already indexed",
                "name": name,
                "root_path": existing.root_path,
                "session_bound": existing.session_id is not None,
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # Run the indexer's add_project (raises ForbiddenProjectError
        # for nested-forbidden roots; we already filtered the basename).
        try:
            project_obj = self.indexer.add_project(name=name, root_path=abs_path)
        except Exception as exc:
            response = {"error": f"indexer.add_project failed: {exc}"}
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # If non-persistent, stamp the project row with the session_id
        # so the GC-on-disconnect path can pick it up. If persistent,
        # ensure session_id is None and (best-effort) write the config.
        session_id = self._session_state().session_id
        if not persist:
            self.chunk_store.upsert_project(
                replace(project_obj, session_id=session_id),
            )
        else:
            # Make sure session_id is cleared on persist.
            if project_obj.session_id is not None:
                self.chunk_store.upsert_project(
                    replace(project_obj, session_id=None),
                )
            self._persist_project_to_config(name, str(abs_path))

        # Kick off the full_scan asynchronously — return immediately
        # with status=indexing. Agent calls wait_for_quiescence to
        # know when it's done.
        self._launch_full_scan(name)

        response = {
            "status": "indexing",
            "project": name,
            "root_path": str(abs_path),
            "persist": persist,
        }
        if not persist:
            response["session_id"] = session_id
        elapsed = (perf_counter() - t0) * 1000.0
        summary = {"project": name, "persist": persist}
        return response, None, {"total": elapsed}, summary

    async def _handle_find_related(
        self, args: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """PRD §6.2 (revised) — semantic similarity only.

        - Locate the chunk containing ``file_path``:``line``. With
          ``line=None`` the first chunk of the file wins; otherwise
          the chunk whose [start_line, end_line] interval contains
          ``line`` (falls back to the first chunk if no chunk covers
          the line — which happens when the file has been re-chunked
          after the agent's last view).
        - Embedding lookup via ``embed_store.get(chunk.embedding_row)``.
        - Brute-force ``embed_store.search_dense`` over all chunks
          excluding the source chunk's own ID.
        - Returns top-K with the same shape as ``search_codebase``.

        ``file_path`` accepts either ``project/rel/path.py`` (the form
        ``search_codebase`` returns) or the bare ``rel/path.py``. The
        former is preferred — the bare form requires walking
        ``list_files`` across all projects.
        """
        t0 = perf_counter()
        file_path = args["file_path"]
        line = args.get("line")
        max_results = int(args.get("max_results", 5))

        if self.embed_store is None:
            response = {
                "error": "find_related unavailable: no embed_store wired",
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        located = self._locate_file_record(file_path)
        if located is None:
            response = {"error": "file not indexed", "file_path": file_path}
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response
        file_rec, _matched_form = located

        chunks = self.chunk_store.get_chunks_by_file(file_rec.id)
        if not chunks:
            response = {
                "error": "file has no indexed chunks",
                "file_path": file_path,
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        source_chunk = self._pick_chunk_for_line(chunks, line)
        if source_chunk.embedding_row is None:
            response = {
                "error": "source chunk has no embedding",
                "file_path": file_path,
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        # Pull the source embedding and run dense search.
        query_emb = self.embed_store.get(source_chunk.embedding_row)
        # search_dense doesn't have a "exclude these chunk_ids" param,
        # so we ask for K+1 then filter the source chunk out client-side.
        hits = self.embed_store.search_dense(
            query_emb,
            max_results + 1,
            None,
        )
        kept_ids: list[int] = []
        kept_scores: dict[int, float] = {}
        for hit in hits:
            if hit.chunk_id == source_chunk.id:
                continue
            kept_ids.append(hit.chunk_id)
            kept_scores[hit.chunk_id] = float(hit.score)
            if len(kept_ids) >= max_results:
                break

        if not kept_ids:
            response = {
                "chunks": [],
                "source": _source_chunk_summary(
                    file_rec.project, file_rec.rel_path, source_chunk,
                ),
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        chunks_by_id = {
            c.id: c for c in self.chunk_store.get_chunks_by_id(kept_ids)
        }
        # Materialize chunks in score order (the Hit list is already
        # ranked).
        out_chunks: list[dict[str, Any]] = []
        for cid in kept_ids:
            chunk = chunks_by_id.get(cid)
            if chunk is None:
                # Chunk vanished between search_dense and get_chunks_by_id
                # (race with an indexer write). Skip rather than blow up.
                continue
            file_for_chunk = self.chunk_store.get_file_by_id(chunk.file_id)
            if file_for_chunk is None:
                continue
            out_chunks.append({
                "project": file_for_chunk.project,
                "file_path": f"{file_for_chunk.project}/{file_for_chunk.rel_path}",
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "relevance_score": kept_scores[cid],
            })

        response = {
            "chunks": out_chunks,
            "source": _source_chunk_summary(
                file_rec.project, file_rec.rel_path, source_chunk,
            ),
        }
        elapsed = (perf_counter() - t0) * 1000.0
        summary = {
            "chunk_count": len(out_chunks),
            "files": [
                f"{c['file_path']}:{c['start_line']}-{c['end_line']}"
                for c in out_chunks
            ],
        }
        return response, None, {"total": elapsed}, summary

    async def _handle_wait_for_quiescence(
        self, args: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """PRD §6.4 — block on the watcher queue until drained or
        ``timeout_ms`` elapses.

        With no watcher attached (stdio path / unit tests), returns the
        always-quiescent response immediately. Daemon mode wires the
        real watcher (Agent G) and we just await its callback.
        """
        t0 = perf_counter()
        project = args.get("project")
        timeout_ms = int(args.get("timeout_ms", 2000))

        if self._watcher is None:
            response = {
                "pending_updates": 0,
                "indexed_through": _now_iso(),
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        try:
            pending, indexed_through = await self._watcher.wait_for_quiescence(
                project, timeout_ms,
            )
        except Exception as exc:
            response = {
                "error": f"watcher.wait_for_quiescence failed: {exc}",
                "pending_updates": -1,
                "indexed_through": _now_iso(),
            }
            elapsed = (perf_counter() - t0) * 1000.0
            return response, None, {"total": elapsed}, response

        response = {
            "pending_updates": int(pending),
            "indexed_through": indexed_through,
        }
        elapsed = (perf_counter() - t0) * 1000.0
        return response, None, {"total": elapsed}, response

    # ----------------------------------------------- helpers for live tools

    def _forbidden_dirs(self) -> frozenset[str]:
        """Pull ``forbidden_dirs`` off the indexer's config when present.

        Test fakes don't always have a ``config`` attribute; we fall
        back to the sensible default set so misuse on test paths is
        still rejected.
        """
        cfg = getattr(self.indexer, "config", None)
        if cfg is not None and hasattr(cfg, "forbidden_dirs"):
            return frozenset(cfg.forbidden_dirs)
        # Last-resort default mirrors indexer.DEFAULT_FORBIDDEN_DIRS.
        return frozenset({"secrets", "credentials", ".aws", ".ssh", ".gnupg"})

    def _launch_full_scan(self, project: str) -> None:
        """Schedule an indexer.full_scan in the background.

        The daemon (Agent F) will plumb a worker pool via
        ``background_runner``; in test paths we synthesize the call
        with ``asyncio.create_task`` if a running loop is available,
        otherwise we run it inline (synchronous) and swallow exceptions
        — the indexer's own logging is the user-visible audit trail.
        """
        runner = self._background_runner

        def _scan() -> Any:
            try:
                return self.indexer.full_scan(project)
            except Exception:  # pragma: no cover — logged inside indexer
                # Don't propagate: the scan happens AFTER the response
                # is already on the wire.
                return None

        if runner is not None:
            try:
                runner(_scan)
                return
            except Exception:  # pragma: no cover
                pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop — fall back to inline (test path with a manual
            # event loop). We've already returned the response shape,
            # so a slow inline scan only delays the dispatch_tool's
            # post-handler bookkeeping; acceptable for tests.
            _scan()
            return
        loop.create_task(asyncio.to_thread(_scan))

    def _persist_project_to_config(self, name: str, root_path: str) -> None:
        """Best-effort append of a ``[[projects]]`` block to the daemon
        config TOML.

        Phase 2.1 — Agent E owns the canonical schema. We append to a
        well-known marker section so the live add survives restart;
        when Agent E ships the structured writer, this naive append
        will get folded in.
        """
        path = self._config_path
        if path is None:
            return
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            block = (
                "\n# added at runtime via add_project(persist=True)\n"
                "[[projects]]\n"
                f'name = "{name}"\n'
                f'root_path = "{root_path}"\n'
            )
            with path.open("a", encoding="utf-8") as f:
                f.write(block)
        except OSError:  # pragma: no cover — best-effort
            return

    def _locate_file_record(self, file_path: str):
        """Resolve ``file_path`` to a ``FileRecord``.

        Acceptable inputs:
          * ``"projectname/rel/path.py"`` — the canonical form
            (matches what ``search_codebase`` returns in citations)
          * ``"rel/path.py"`` — bare relative form; we walk every
            project's files looking for a unique match.

        Returns ``(FileRecord, matched_form)`` or ``None``.
        """
        # Canonical form first.
        if "/" in file_path:
            head, tail = file_path.split("/", 1)
            rec = self.chunk_store.get_file(head, tail)
            if rec is not None:
                return rec, "project/rel"
        # Bare-rel-path fallback. Match across all projects; if more
        # than one project has the same rel_path we pick the first
        # (deterministic by project name order).
        for proj in self.chunk_store.list_projects():
            rec = self.chunk_store.get_file(proj.name, file_path)
            if rec is not None:
                return rec, "rel-only"
        return None

    def _pick_chunk_for_line(self, chunks, line: Optional[int]):
        """Return the chunk whose [start_line, end_line] covers ``line``.

        ``line=None`` selects the first chunk in chunk_index order.
        Falls back to the first chunk if no chunk covers the requested
        line (which happens when the file has been re-chunked since
        the agent last saw it).
        """
        ordered = sorted(chunks, key=lambda c: c.chunk_index)
        if line is None:
            return ordered[0]
        for c in ordered:
            if c.start_line <= line <= c.end_line:
                return c
        return ordered[0]

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
    """SearchChunk → flat dict matching §6.3.

    Includes ``dense_cosine`` (Phase 2.0.1) — the per-chunk pre-fusion
    semantic match anchor — so the agent can act per-chunk on
    confidence rather than only on the response-level top-1 number.
    Tolerates older SearchChunk instances (test fakes) that don't have
    the field via getattr fallback.
    """
    out = {
        "project": sc.citation.project,
        "file_path": sc.citation.file_path,
        "start_line": sc.citation.start_line,
        "end_line": sc.citation.end_line,
        "text": sc.text,
        "relevance_score": sc.relevance_score,
    }
    cos = getattr(sc, "dense_cosine", None)
    if cos is not None:
        out["dense_cosine"] = float(cos)
    cid = getattr(sc, "chunk_id", None)
    if cid is not None:
        out["chunk_id"] = int(cid)
    return out


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


__all__ = [
    "ConnectionContext",
    "MCPServer",
    "get_active_connection",
    "reset_active_connection",
    "set_active_connection",
]
