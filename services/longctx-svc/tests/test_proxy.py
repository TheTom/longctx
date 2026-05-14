# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `longctx_svc.proxy`.

Covers the pure helpers (message extraction, splicing, dump) plus the
503 "proxy disabled" guard on every endpoint. Heavy retrieve + upstream
forward paths are exercised by integration tests against a live engine;
unit-testing them with full httpx mocks adds little signal for the
churn cost.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from longctx_svc.proxy import (
    _debug_dump_dir,
    _dump,
    _flatten_messages_to_prefill,
    _format_chunks_block,
    _last_user_query,
    _splice_into_messages,
    _upstream,
    router,
)


# ----------------------------------------------------- env helpers


def test_upstream_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("LONGCTX_UPSTREAM", raising=False)
    assert _upstream() is None


def test_upstream_returns_env_value(monkeypatch):
    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://engine:9000")
    assert _upstream() == "http://engine:9000"


def test_debug_dump_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("LONGCTX_DEBUG_DUMP", raising=False)
    assert _debug_dump_dir() is None


def test_debug_dump_dir_creates_when_set(tmp_path, monkeypatch):
    target = tmp_path / "dumps"
    monkeypatch.setenv("LONGCTX_DEBUG_DUMP", str(target))
    p = _debug_dump_dir()
    assert p == target
    assert target.is_dir()


def test_dump_noop_when_env_unset(tmp_path, monkeypatch):
    """No env → silent return, never writes."""
    monkeypatch.delenv("LONGCTX_DEBUG_DUMP", raising=False)
    _dump("anything", {"x": 1})  # must not raise
    assert list(tmp_path.iterdir()) == []


def test_dump_writes_file_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGCTX_DEBUG_DUMP", str(tmp_path))
    _dump("chat-completions", {"messages": [{"role": "user", "content": "hi"}]})
    files = list(tmp_path.glob("*-chat-completions.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["messages"][0]["content"] == "hi"


def test_dump_swallows_serialization_failure(tmp_path, monkeypatch):
    """Non-JSON-serializable body should NOT crash the request path."""
    monkeypatch.setenv("LONGCTX_DEBUG_DUMP", str(tmp_path))

    class Unserializable:
        pass

    # Should silently swallow the TypeError from json.dumps.
    _dump("label", {"weird": Unserializable()})


# ----------------------------------------------------- _last_user_query


def test_last_user_query_string_content():
    msgs = [{"role": "user", "content": "Hello world"}]
    assert _last_user_query(msgs) == "Hello world"


def test_last_user_query_returns_most_recent_user():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_query(msgs) == "second"


def test_last_user_query_list_content():
    """OpenAI multi-modal: content is a list of `{type, text}` parts."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "describe this"},
            ],
        }
    ]
    assert _last_user_query(msgs) == "describe this"


def test_last_user_query_empty_when_no_user_messages():
    msgs = [{"role": "system", "content": "you are a bot"}]
    assert _last_user_query(msgs) == ""


# ----------------------------------------------------- _flatten_messages_to_prefill


def test_flatten_with_string_content():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    out = _flatten_messages_to_prefill(msgs)
    assert "[system]" in out
    assert "[user]" in out
    assert "S" in out and "U" in out


def test_flatten_with_list_content():
    """Multi-modal content blocks must be flattened to plain text."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    out = _flatten_messages_to_prefill(msgs)
    assert "hello" in out
    assert "world" in out


# ----------------------------------------------------- _splice_into_messages


def test_splice_into_string_system_message():
    msgs = [
        {"role": "system", "content": "original"},
        {"role": "user", "content": "hi"},
    ]
    out = _splice_into_messages(msgs, "CHUNK BLOCK\n")
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("CHUNK BLOCK")
    assert "original" in out[0]["content"]
    # User message untouched.
    assert out[1] == {"role": "user", "content": "hi"}


def test_splice_into_list_system_message():
    msgs = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "original"}],
        },
        {"role": "user", "content": "hi"},
    ]
    out = _splice_into_messages(msgs, "CHUNK\n")
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["text"] == "CHUNK\n"
    assert out[0]["content"][1]["text"] == "original"


def test_splice_inserts_system_when_missing():
    msgs = [{"role": "user", "content": "hi"}]
    out = _splice_into_messages(msgs, "CHUNK\n")
    assert out[0]["role"] == "system"
    assert "CHUNK" in out[0]["content"]
    assert out[1]["role"] == "user"


# ----------------------------------------------------- _format_chunks_block


class _C:
    """Duck-typed chunk for tests — supports both attr and dict access."""

    def __init__(self, file_path, start_line, end_line, text):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.text = text


def test_format_chunks_block_basic():
    chunks = [_C("/r/a.py", 1, 3, "def foo(): pass")]
    out = _format_chunks_block(chunks)
    assert out.startswith("## Retrieved code context")
    assert "// /r/a.py:1-3" in out
    assert "def foo()" in out
    assert "```" in out


def test_format_chunks_block_accepts_dict_chunks():
    chunks = [
        {
            "file_path": "/r/b.py",
            "start_line": 5,
            "end_line": 7,
            "text": "x = 1",
        }
    ]
    out = _format_chunks_block(chunks)
    assert "// /r/b.py:5-7" in out
    assert "x = 1" in out


def test_format_chunks_block_truncates_oversize_chunk():
    """A single chunk larger than the budget gets truncated with a
    [truncated] marker rather than dropped."""
    # 100KB text far exceeds default 16KB budget.
    big = "X" * 100_000
    chunks = [_C("/r/big.py", 1, 1, big)]
    out = _format_chunks_block(chunks)
    assert "[truncated]" in out


def test_format_chunks_block_drops_late_chunks_when_budget_exhausted():
    """Once the budget is essentially used, subsequent chunks are
    skipped via the `remaining <= 200: break` short-circuit."""
    big = "Y" * 100_000
    chunks = [
        _C("/r/big.py", 1, 1, big),
        _C("/r/late.py", 2, 2, "should be dropped"),
    ]
    out = _format_chunks_block(chunks)
    # First chunk lands (truncated), second is dropped entirely.
    assert "/r/big.py" in out
    assert "/r/late.py" not in out


# ----------------------------------------------------- 503 guards


def _client_without_upstream(monkeypatch) -> TestClient:
    """Build a FastAPI app with just the proxy router, no upstream set."""
    monkeypatch.delenv("LONGCTX_UPSTREAM", raising=False)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def test_list_models_503_when_no_upstream(monkeypatch):
    c = _client_without_upstream(monkeypatch)
    r = c.get("/v1/models")
    assert r.status_code == 503
    assert "proxy disabled" in r.json()["detail"]


def test_chat_completions_503_when_no_upstream(monkeypatch):
    c = _client_without_upstream(monkeypatch)
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503


def test_completions_503_when_no_upstream(monkeypatch):
    c = _client_without_upstream(monkeypatch)
    r = c.post("/v1/completions", json={"prompt": "hi"})
    assert r.status_code == 503


# ----------------------------------------------------- /v1/models passthrough


def test_list_models_forwards_to_upstream(monkeypatch):
    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://up:1")
    app = FastAPI()
    app.include_router(router)

    captured = {}

    class FakeResp:
        status_code = 200
        content = b'{"data":[{"id":"m1"}]}'
        headers = {"content-type": "application/json"}

    class FakeAC:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, target, headers):
            captured["target"] = target
            captured["headers"] = headers
            return FakeResp()

    with patch("longctx_svc.proxy.httpx.AsyncClient", FakeAC):
        c = TestClient(app)
        r = c.get("/v1/models", headers={"authorization": "Bearer abc"})

    assert r.status_code == 200
    assert captured["target"] == "http://up:1/v1/models"
    # Authorization header forwarded.
    assert captured["headers"].get("authorization") == "Bearer abc"
    assert b'"id":"m1"' in r.content
