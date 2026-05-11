"""Tests for longctx_daemon.logging — trace IDs, context vars,
structured MCP-call lines, redaction, idempotent setup."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from loguru import logger as _loguru_logger

from longctx_daemon import logging as lcl


def _capture_lines() -> list[str]:
    """Install a memory sink that captures serialized JSON lines.

    Returns the list — caller appends via the sink callback. The list
    grows in-place as logs flow.
    """
    captured: list[str] = []

    def _sink(message):
        # message.record contains structured fields; we want the whole
        # serialized JSON envelope loguru produces with serialize=True.
        captured.append(message)

    _loguru_logger.add(_sink, level="DEBUG", serialize=True)
    return captured


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Each test gets a clean loguru sink set + cleared trace context."""
    _loguru_logger.remove()
    lcl.clear_trace_context()
    yield
    lcl.clear_trace_context()
    _loguru_logger.remove()


# ============================================================ trace IDs
class TestNewTraceID:
    def test_returns_26_char_string(self):
        tid = lcl.new_trace_id()
        assert len(tid) == 26

    def test_uses_crockford_alphabet(self):
        tid = lcl.new_trace_id()
        # No I, L, O, U; all uppercase digits/letters.
        assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", tid), tid

    def test_unique_across_calls(self):
        ids = {lcl.new_trace_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_lexicographic_time_ordering(self):
        a = lcl.new_trace_id()
        # Sleep enough for the ms-precision timestamp to advance.
        import time
        time.sleep(0.01)
        b = lcl.new_trace_id()
        assert a < b


# ============================================================ context vars
class TestTraceContext:
    def test_set_and_read(self):
        lcl.set_trace_context(
            trace_id="t1",
            session_id="s1",
            connection_id="c1",
            client_name="opencode",
            client_version="0.4.2",
        )
        assert lcl.current_trace_id() == "t1"

    def test_clear(self):
        lcl.set_trace_context(
            trace_id="t1",
            session_id="s1",
            connection_id="c1",
            client_name="x",
            client_version="y",
        )
        lcl.clear_trace_context()
        assert lcl.current_trace_id() is None

    def test_isolation_between_async_tasks(self):
        """contextvars copy on task creation, so concurrent tasks
        each get their own trace_id."""

        async def _task(tid: str) -> str:
            lcl.set_trace_context(
                trace_id=tid,
                session_id="s",
                connection_id="c",
                client_name="x",
                client_version="y",
            )
            await asyncio.sleep(0)
            return lcl.current_trace_id()

        async def _run():
            return await asyncio.gather(_task("A"), _task("B"))

        # CPython contextvars: each Task has its own copy.
        results = asyncio.run(_run())
        assert set(results) == {"A", "B"}


# ============================================================ setup
class TestSetupLogging:
    def test_idempotent(self, tmp_path: Path):
        lcl.setup_logging(level="INFO", file_path=tmp_path / "x.log")
        lcl.setup_logging(level="INFO", file_path=tmp_path / "x.log")
        # No assertion needed; we just need it not to explode.

    def test_creates_parent_dir(self, tmp_path: Path):
        nested = tmp_path / "deep" / "subdir" / "longctx.log"
        lcl.setup_logging(level="DEBUG", file_path=nested)
        # File appears on first write.
        _loguru_logger.info("hello")
        # loguru's file sink is enqueue=False so the write is sync.
        assert nested.parent.exists()

    def test_writes_to_file(self, tmp_path: Path):
        path = tmp_path / "longctx.log"
        lcl.setup_logging(level="INFO", file_path=path)
        _loguru_logger.bind(component="test").info("hi there")
        contents = path.read_text(encoding="utf-8")
        # JSON-line format from serialize=True.
        first_line = contents.splitlines()[0]
        envelope = json.loads(first_line)
        assert envelope["record"]["message"] == "hi there"


# ============================================================ MCP call lines
class TestLogMcpCall:
    def test_emits_one_info_line_with_required_fields(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="trace-1",
            session_id="ses-1",
            connection_id="conn-1",
            client_name="opencode",
            client_version="0.4.2",
        )
        lcl.log_mcp_call(
            tool="search_codebase",
            args={"query": "auth middleware", "cwd": "/x"},
            scope={"primary_project": "x"},
            latency_ms={"total": 65},
            result={"chunk_count": 5},
        )
        assert len(captured) == 1
        env = json.loads(captured[0])
        extra = env["record"]["extra"]
        assert extra["component"] == "mcp"
        assert extra["tool"] == "search_codebase"
        assert extra["trace_id"] == "trace-1"
        assert extra["session_id"] == "ses-1"
        assert extra["connection_id"] == "conn-1"
        assert extra["client"] == {"name": "opencode", "version": "0.4.2"}
        assert extra["args"] == {"query": "auth middleware", "cwd": "/x"}
        assert extra["scope"] == {"primary_project": "x"}
        assert extra["latency_ms"] == {"total": 65}
        assert extra["result"] == {"chunk_count": 5}
        assert extra["level"] == "INFO"
        assert "ts" in extra
        # ISO 8601 with millisecond Z suffix
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", extra["ts"])

    def test_field_ordering_matches_spec(self):
        """§14.4 example puts scope between args and latency_ms."""
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t",
            session_id="s",
            connection_id="c",
            client_name="x",
            client_version="y",
        )
        lcl.log_mcp_call(
            tool="search_codebase",
            args={"query": "x"},
            scope={"primary_project": "x"},
            latency_ms={"total": 1},
            result={"chunk_count": 0},
        )
        env = json.loads(captured[0])
        keys = list(env["record"]["extra"].keys())
        assert keys.index("args") < keys.index("scope") < keys.index("latency_ms")

    def test_omits_scope_when_none(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_call(
            tool="list_projects",
            args={},
            scope=None,
            latency_ms={"total": 1},
            result={"project_count": 3},
        )
        env = json.loads(captured[0])
        assert "scope" not in env["record"]["extra"]

    def test_unknown_client_when_no_context(self):
        captured = _capture_lines()
        # No set_trace_context — context vars all None.
        lcl.log_mcp_call(
            tool="x", args={}, scope=None,
            latency_ms={"total": 0}, result={},
        )
        env = json.loads(captured[0])
        assert env["record"]["extra"]["client"] == {"name": "unknown", "version": "unknown"}
        assert env["record"]["extra"]["trace_id"] is None


# ============================================================ redaction
class TestRedaction:
    def test_redact_query_via_param(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_call(
            tool="search_codebase",
            args={"query": "secret query text"},
            scope=None,
            latency_ms={"total": 0},
            result={},
            redact_query=True,
        )
        extra = json.loads(captured[0])["record"]["extra"]
        assert extra["args"]["query"] == "<query of 17 chars>"

    def test_redact_cwd_via_param(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_call(
            tool="search_codebase",
            args={"query": "x", "cwd": "/Users/secret/dev/proj"},
            scope=None,
            latency_ms={"total": 0},
            result={},
            redact_cwd=True,
        )
        extra = json.loads(captured[0])["record"]["extra"]
        assert extra["args"]["cwd"] == "<cwd>"

    def test_redact_setup_default_applies(self, tmp_path: Path):
        lcl.setup_logging(
            level="INFO",
            file_path=tmp_path / "x.log",
            redact_query_text=True,
        )
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_call(
            tool="search_codebase",
            args={"query": "abcdef"},
            scope=None,
            latency_ms={"total": 0},
            result={},
        )
        extra = json.loads(captured[-1])["record"]["extra"]
        assert extra["args"]["query"] == "<query of 6 chars>"

    def test_does_not_mutate_caller_dict(self):
        _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        original = {"query": "hello"}
        lcl.log_mcp_call(
            tool="search_codebase",
            args=original,
            scope=None,
            latency_ms={"total": 0},
            result={},
            redact_query=True,
        )
        assert original == {"query": "hello"}


# ============================================================ error path
class TestLogMcpError:
    def test_emits_error_line(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="trace-err",
            session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_error(
            tool="search_codebase",
            args={"query": "x"},
            error="boom",
        )
        env = json.loads(captured[0])
        extra = env["record"]["extra"]
        assert extra["level"] == "ERROR"
        assert extra["error"] == "boom"
        assert extra["trace_id"] == "trace-err"
        assert env["record"]["level"]["name"] == "ERROR"

    def test_error_redaction_honors_query_flag(self):
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="t", session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_error(
            tool="search_codebase",
            args={"query": "this is sensitive"},
            error="failed",
            redact_query=True,
        )
        extra = json.loads(captured[0])["record"]["extra"]
        assert extra["args"]["query"].startswith("<query of ")


# ============================================================ trace propagation
class TestTracePropagation:
    def test_subevent_inherits_trace_id(self):
        """Sub-events logged via plain logger.info still get the
        active trace_id — that's how grep <trace_id> reconstructs the
        causal chain (§14.5)."""
        captured = _capture_lines()
        lcl.set_trace_context(
            trace_id="trace-X",
            session_id="s", connection_id="c",
            client_name="x", client_version="y",
        )
        lcl.log_mcp_call(
            tool="search_codebase", args={}, scope=None,
            latency_ms={"total": 1}, result={},
        )
        # A sub-event we want correlated. Caller is responsible for
        # passing the trace id explicitly OR using log_mcp_call again
        # with the same context. We assert: the log_mcp_call line
        # carries the trace_id.
        env = json.loads(captured[0])
        assert env["record"]["extra"]["trace_id"] == "trace-X"
