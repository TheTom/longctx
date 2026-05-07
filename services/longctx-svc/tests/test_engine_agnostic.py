"""Engine-agnostic surface tests.

Covers the two integration paths intended for vllm-swift, llama.cpp
(TheTom/llama-cpp-turboquant), and AMD vLLM (TheTom/vllm
feature/turboquant-amd-noautotune):

  1. POST /retrieve via the LongctxClient — the optional, embedded
     integration. Engine code calls it explicitly when configured.
  2. /v1/chat/completions and /v1/completions OpenAI-compat proxy —
     drop-in passthrough; engine is unmodified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# LongctxClient (engine-side optional helper)
# ---------------------------------------------------------------------------

def test_client_from_env_returns_none_without_endpoint(monkeypatch):
    """Tool optional: no endpoint → None → engine takes the no-retrieval
    fall-through path."""
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    from longctx_svc.client import LongctxClient
    assert LongctxClient.from_env() is None


def test_client_from_env_constructs(monkeypatch):
    monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:8765")
    from longctx_svc.client import LongctxClient
    c = LongctxClient.from_env()
    assert c is not None
    assert c.endpoint == "http://localhost:8765"


def test_client_retrieve_swallows_network_errors():
    """Optional tool: network failure must not break the engine."""
    from longctx_svc.client import LongctxClient
    c = LongctxClient("http://127.0.0.1:1")  # nothing listening
    c.timeout = 0.5
    res = c.retrieve("see /Users/x/foo.py", "what?", session_id="s")
    assert res.chunks == []
    assert res.session_id == "s"


def test_client_splice_no_chunks_passthrough():
    from longctx_svc.client import LongctxClient, RetrieveResult
    out = LongctxClient.splice("hello", RetrieveResult())
    assert out == "hello"


def test_client_splice_format_includes_path_lines():
    from longctx_svc.client import (
        LongctxClient, RetrieveResult, RetrievedChunk,
    )
    res = RetrieveResult(chunks=[
        RetrievedChunk(
            text="x = 1\n", file_path="/p/a.py",
            start_line=1, end_line=1, file_type="code", score=0.9,
        ),
    ])
    out = LongctxClient.splice("user prompt", res)
    assert "/p/a.py:1-1" in out
    assert "x = 1" in out
    assert out.endswith("user prompt")


def test_client_against_real_service(client, project_dir):
    """Drive the real FastAPI app via TestClient with the LongctxClient,
    routing httpx through the test client transport."""
    from longctx_svc.client import LongctxClient

    real_request = httpx.request

    def transport_request(method, url, **kw):
        # Rewrite to test client's app
        path = httpx.URL(url).raw_path.decode()
        if method == "GET":
            return client.get(path, **{
                k: v for k, v in kw.items() if k in ("headers", "params")
            })
        if method == "POST":
            return client.post(path, **{
                k: v for k, v in kw.items() if k in ("headers", "json", "data")
            })
        return real_request(method, url, **kw)

    with patch("httpx.post",
               side_effect=lambda url, **kw: transport_request("POST",
                                                                url, **kw)):
        c = LongctxClient("http://testserver")
        res = c.retrieve(
            prefill_text=f"see {project_dir}/src/auth.ts",
            query="auth",
            session_id="engine-1",
            top_k=4,
        )
    assert res.session_id == "engine-1"
    # scope was detectable, so status should be ready or empty (not no-scope)
    assert res.scope_status in ("ready", "empty")


# ---------------------------------------------------------------------------
# OpenAI-compat proxy (drop-in for any engine)
# ---------------------------------------------------------------------------

@pytest.fixture
def proxy_client(client, monkeypatch):
    """Same TestClient as the Sarah-journey fixture, just keeps env hygiene."""
    monkeypatch.delenv("LONGCTX_UPSTREAM", raising=False)
    return client


def test_proxy_disabled_when_upstream_unset(proxy_client, monkeypatch):
    """Tool optional: without LONGCTX_UPSTREAM the proxy returns 503,
    making it explicit that the proxy mode is opt-in."""
    monkeypatch.delenv("LONGCTX_UPSTREAM", raising=False)
    r = proxy_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert "proxy disabled" in r.text


def test_proxy_chat_splices_chunks_into_system(client, project_dir,
                                                monkeypatch):
    """Verify the proxy detects scope, retrieves, and prepends a system
    message containing the chunk block before forwarding."""
    captured: dict = {}

    class _MockResp:
        status_code = 200
        content = json.dumps({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "ok"
            }, "finish_reason": "stop"}],
        }).encode()
        headers = {"content-type": "application/json"}

    async def fake_post(self, url, json=None, headers=None):  # noqa: ARG001
        captured["url"] = url
        captured["body"] = json
        return _MockResp()

    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://upstream:8080")
    with patch("httpx.AsyncClient.post", new=fake_post):
        r = client.post(
            "/v1/chat/completions",
            headers={"x-session-affinity": "proxy-A"},
            json={
                "model": "qwen2.5-32b",
                "messages": [
                    {"role": "user", "content":
                     f"why does {project_dir}/src/auth.ts fail?"},
                ],
            },
        )
    assert r.status_code == 200
    assert captured["url"] == "http://upstream:8080/v1/chat/completions"
    msgs = captured["body"]["messages"]
    # First message must be the spliced system message
    assert msgs[0]["role"] == "system"
    assert "Retrieved code context" in msgs[0]["content"]
    # Original user message preserved
    assert any(m["role"] == "user" for m in msgs)
    # Headers carry the debug info
    assert r.headers["x-longctx-session"] == "proxy-A"
    assert "x-longctx-chunks-used" in r.headers


def test_proxy_chat_no_scope_passes_through_unmodified(client, monkeypatch):
    """No path in user message → no retrieval → body forwarded as-is.
    Critical for the 'optional' guarantee: scope detection mustn't break
    plain chat."""
    captured: dict = {}

    class _MockResp:
        status_code = 200
        content = b'{"choices":[]}'
        headers = {"content-type": "application/json"}

    async def fake_post(self, url, json=None, headers=None):  # noqa: ARG001
        captured["body"] = json
        return _MockResp()

    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://upstream:8080")
    with patch("httpx.AsyncClient.post", new=fake_post):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "what is 2+2?"},
                ],
            },
        )
    assert r.status_code == 200
    msgs = captured["body"]["messages"]
    # No chunk block prepended
    assert msgs[0]["content"] == "be helpful"
    assert "Retrieved code context" not in msgs[0]["content"]
    assert r.headers["x-longctx-scope-status"] == "no-scope"


def test_proxy_completions_splices_into_prompt(client, project_dir,
                                                monkeypatch):
    """Legacy /v1/completions endpoint (used by some llama.cpp clients,
    older OpenCode setups)."""
    captured: dict = {}

    class _MockResp:
        status_code = 200
        content = b'{"choices":[{"text":"ok"}]}'
        headers = {"content-type": "application/json"}

    async def fake_post(self, url, json=None, headers=None):  # noqa: ARG001
        captured["body"] = json
        return _MockResp()

    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://up:9000")
    with patch("httpx.AsyncClient.post", new=fake_post):
        r = client.post(
            "/v1/completions",
            json={
                "model": "x",
                "prompt": f"in {project_dir}/src/auth.ts please explain auth",
            },
        )
    assert r.status_code == 200
    assert "Retrieved code context" in captured["body"]["prompt"]


def test_proxy_forwards_authorization_header(client, monkeypatch):
    """Auth tokens must reach the upstream engine."""
    captured: dict = {}

    class _MockResp:
        status_code = 200
        content = b'{}'
        headers = {"content-type": "application/json"}

    async def fake_post(self, url, json=None, headers=None):
        captured["headers"] = dict(headers or {})
        return _MockResp()

    monkeypatch.setenv("LONGCTX_UPSTREAM", "http://up:9000")
    with patch("httpx.AsyncClient.post", new=fake_post):
        r = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer abc"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert captured["headers"].get("authorization") == "Bearer abc"
