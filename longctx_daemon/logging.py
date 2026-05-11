"""Structured trace logging for the longctx daemon.

Backed by ``loguru``. Two output sinks:
  * stderr (always — keeps stdio MCP traffic clean on stdout)
  * optional rotating file sink

The single most useful artifact this module emits is the per-MCP-call
JSON line described in PRD-phase2-mcp-daemon.md §14.4. ``log_mcp_call``
emits exactly that schema, with the exact key names the spec specifies.

Trace context (trace_id, session_id, connection_id, client name/version)
is propagated via ``contextvars.ContextVar`` so that any sub-event logged
during the same async task automatically carries the same trace_id.
``grep <trace_id> longctx.log`` then yields the full causal chain
(see §14.5).

Usage:
    setup_logging(level="INFO", file_path="~/.cache/longctx/longctx.log")
    tid = new_trace_id()
    set_trace_context(trace_id=tid, session_id="ses_x",
                      connection_id="conn_y",
                      client_name="opencode", client_version="0.4.2")
    log_mcp_call(tool="search_codebase", args={"query": "..."},
                 scope=scope_dict, latency_ms=latency_dict, result=result)

TODO: time-based file rotation knobs (currently relies on loguru's
default daily rotation when configured). §14.8 replay-log retention is
implemented separately in ``replay_log.py``.
"""
from __future__ import annotations

import os
import secrets
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------- contextvars
# Per-task trace context. Default to None so a log call outside an MCP
# request still works (just without trace_id / client info attached).
_trace_id_var: ContextVar[Optional[str]] = ContextVar("longctx_trace_id", default=None)
_session_id_var: ContextVar[Optional[str]] = ContextVar(
    "longctx_session_id", default=None
)
_connection_id_var: ContextVar[Optional[str]] = ContextVar(
    "longctx_connection_id", default=None
)
_client_name_var: ContextVar[Optional[str]] = ContextVar(
    "longctx_client_name", default=None
)
_client_version_var: ContextVar[Optional[str]] = ContextVar(
    "longctx_client_version", default=None
)


# ------------------------------------------------------------------- ULID gen
# Crockford base32 alphabet (no I, L, O, U).
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_trace_id() -> str:
    """Generate a fresh ULID-style 26-char Crockford base32 trace ID.

    First 10 chars encode the millisecond timestamp (sortable by time);
    last 16 chars are random. We don't bother with a real ulid library
    for the daemon — this gives the same lexicographic-time ordering
    properties and only takes a few lines.

    Example: ``01J0K3GW8M7TQXC2VBYHE5N4ZF``
    """
    # 48-bit ms timestamp.
    ts_ms = int(time.time() * 1000)
    ts_part = _encode_base32(ts_ms, 10)
    # 80 bits of randomness encoded into 16 chars (16 * 5 = 80 bits).
    rand_int = secrets.randbits(80)
    rand_part = _encode_base32(rand_int, 16)
    return ts_part + rand_part


def _encode_base32(value: int, length: int) -> str:
    """Encode a non-negative int into a fixed-length Crockford base32
    string. High-order zeros are padded on the left."""
    chars: list[str] = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


# --------------------------------------------------------------- setup sinks
_logging_configured: bool = False


def setup_logging(
    level: str = "INFO",
    file_path: Optional[str | os.PathLike[str]] = None,
    redact_query_text: bool = False,
    redact_cwd: bool = False,
) -> None:
    """Configure the global loguru sinks.

    Idempotent: calling twice replaces the previous configuration rather
    than stacking duplicate sinks. We always remove all sinks first so
    repeat calls don't double-log.

    Args:
        level: minimum severity to emit (e.g. "DEBUG", "INFO", "WARN").
        file_path: optional file sink path. If None, only stderr is used.
            Parent directories are created on demand.
        redact_query_text: stored as a module-level default that
            ``log_mcp_call`` honors when its own ``redact_query`` flag
            is False (ie config wins unless caller explicitly opts in).
        redact_cwd: same semantics for cwd redaction.
    """
    global _logging_configured, _default_redact_query, _default_redact_cwd
    _default_redact_query = redact_query_text
    _default_redact_cwd = redact_cwd

    # Remove every previously-installed sink. loguru adds a default stderr
    # sink at import time; we want clean control here.
    logger.remove()

    # Stderr sink — JSON lines so log shipping is uniform across
    # transports. stdio MCP traffic is on stdout, so emitting to stderr
    # keeps the protocol stream clean (§14.1).
    logger.add(
        sys.stderr,
        level=level,
        serialize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    # Optional rotating file sink. loguru handles rotation natively.
    if file_path is not None:
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=level,
            serialize=True,
            rotation="1 day",      # §14.3: daily rotation
            retention="7 days",    # §14.3: keep 7 days
            compression="gz",      # §14.3: gzip older
            backtrace=False,
            diagnose=False,
            enqueue=False,
        )

    _logging_configured = True


# Module-level redaction defaults set by setup_logging.
_default_redact_query: bool = False
_default_redact_cwd: bool = False


# ------------------------------------------------------------ trace context
def set_trace_context(
    *,
    trace_id: str,
    session_id: str,
    connection_id: str,
    client_name: str,
    client_version: str,
) -> None:
    """Bind the trace context for subsequent log calls in this task.

    Uses ``contextvars`` so async children inherit. Each MCP tool
    invocation calls this once at the start of the handler so all
    sub-events (embedder, BM25 lookup, atomic-update transaction) get
    the same trace_id (§14.5).
    """
    _trace_id_var.set(trace_id)
    _session_id_var.set(session_id)
    _connection_id_var.set(connection_id)
    _client_name_var.set(client_name)
    _client_version_var.set(client_version)


def clear_trace_context() -> None:
    """Reset trace context to None. Useful between unit tests."""
    _trace_id_var.set(None)
    _session_id_var.set(None)
    _connection_id_var.set(None)
    _client_name_var.set(None)
    _client_version_var.set(None)


def current_trace_id() -> Optional[str]:
    """Read the active trace_id, if any. Returns None outside a request."""
    return _trace_id_var.get()


def _trace_context_dict() -> dict[str, Any]:
    """Snapshot of the trace ContextVars for inclusion in a log payload."""
    return {
        "trace_id": _trace_id_var.get(),
        "session_id": _session_id_var.get(),
        "connection_id": _connection_id_var.get(),
        "client": {
            "name": _client_name_var.get() or "unknown",
            "version": _client_version_var.get() or "unknown",
        },
    }


# --------------------------------------------------------------- ts helper
def _now_iso() -> str:
    """ISO8601 UTC timestamp with millisecond precision and trailing Z."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# ------------------------------------------------------------ MCP trace API
def log_mcp_call(
    *,
    tool: str,
    args: dict[str, Any],
    scope: Optional[dict[str, Any]],
    latency_ms: dict[str, Any],
    result: dict[str, Any],
    redact_query: bool = False,
    redact_cwd: bool = False,
) -> None:
    """Emit one structured MCP-call log line at INFO.

    Schema matches PRD §14.4 exactly:
        ts, level, component, trace_id, session_id, connection_id,
        client {name, version}, tool, args, scope, latency_ms, result

    ``args`` is a shallow copy with redaction applied; the caller's
    dict is not mutated. ``scope`` may be None for non-search tools
    (omitted from the payload entirely).

    Redaction precedence: explicit param wins over module default
    (set by setup_logging), so a tool can opt INTO redaction even when
    the daemon defaults to plain logging.
    """
    safe_args = _redact_args(
        args,
        redact_query=redact_query or _default_redact_query,
        redact_cwd=redact_cwd or _default_redact_cwd,
    )

    payload: dict[str, Any] = {
        "ts": _now_iso(),
        "level": "INFO",
        "component": "mcp",
        **_trace_context_dict(),
        "tool": tool,
        "args": safe_args,
        "latency_ms": latency_ms,
        "result": result,
    }
    if scope is not None:
        # Insert scope between args and latency to match the §14.4
        # example field order. Python dict preserves insertion order
        # since 3.7 so this is reliable.
        ordered: dict[str, Any] = {}
        for k, v in payload.items():
            ordered[k] = v
            if k == "args":
                ordered["scope"] = scope
        payload = ordered

    # bind() puts the dict in `extra` so loguru's serialize=True picks
    # it up. We also embed it into the rendered message for the
    # plain-text human-tail case.
    logger.bind(**payload).info("mcp_call")


def log_mcp_error(
    *,
    tool: str,
    args: dict[str, Any],
    error: str,
    redact_query: bool = False,
    redact_cwd: bool = False,
) -> None:
    """Emit one structured MCP-error log line at ERROR.

    Same skeleton as log_mcp_call (so per-trace correlation still works)
    but without latency/result and with an ``error`` string field. The
    exception itself is expected to propagate to the MCP client; this
    just records it operationally.
    """
    safe_args = _redact_args(
        args,
        redact_query=redact_query or _default_redact_query,
        redact_cwd=redact_cwd or _default_redact_cwd,
    )
    payload: dict[str, Any] = {
        "ts": _now_iso(),
        "level": "ERROR",
        "component": "mcp",
        **_trace_context_dict(),
        "tool": tool,
        "args": safe_args,
        "error": error,
    }
    logger.bind(**payload).error("mcp_error")


def _redact_args(
    args: dict[str, Any],
    *,
    redact_query: bool,
    redact_cwd: bool,
) -> dict[str, Any]:
    """Apply redaction to the args dict per §14.9 knobs."""
    safe = dict(args)
    if redact_query and "query" in safe:
        q = safe["query"]
        # Length is preserved so log analysis can spot unusually large
        # payloads even with redaction on.
        if isinstance(q, str):
            safe["query"] = f"<query of {len(q)} chars>"
        else:
            safe["query"] = "<query>"
    if redact_cwd and "cwd" in safe and safe["cwd"] is not None:
        safe["cwd"] = "<cwd>"
    return safe


__all__ = [
    "setup_logging",
    "new_trace_id",
    "set_trace_context",
    "clear_trace_context",
    "current_trace_id",
    "log_mcp_call",
    "log_mcp_error",
]
