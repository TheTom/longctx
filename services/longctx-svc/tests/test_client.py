# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `longctx_svc.client.LongctxClient`.

The client is the engine-agnostic seam consumers (vllm-swift, llama.cpp,
vLLM CUDA) call to hit /retrieve. These tests exercise the HTTP error
paths (which degrade silently to empty results), the sync and async
retrieve paths, env-based construction, and the splice helper.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from longctx_svc.client import (
    LongctxClient,
    RetrievedChunk,
    RetrieveResult,
)


# ----------------------------------------------------- construction


def test_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    assert LongctxClient.from_env() is None


def test_from_env_constructs_when_set(monkeypatch):
    monkeypatch.setenv("LONGCTX_ENDPOINT", "http://example:9999/")
    cli = LongctxClient.from_env()
    assert isinstance(cli, LongctxClient)
    # Trailing slash stripped.
    assert cli.endpoint == "http://example:9999"


def test_from_env_honors_custom_var(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    monkeypatch.setenv("MY_RETRIEVAL_URL", "http://other:1234")
    cli = LongctxClient.from_env("MY_RETRIEVAL_URL")
    assert cli is not None
    assert cli.endpoint == "http://other:1234"


def test_endpoint_strips_trailing_slash():
    cli = LongctxClient("http://x:1/")
    assert cli.endpoint == "http://x:1"


# ----------------------------------------------------- healthz


def test_healthz_true_on_200():
    cli = LongctxClient("http://x:1")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    with patch("longctx_svc.client.httpx.get", return_value=fake_resp):
        assert cli.healthz() is True


def test_healthz_false_on_non_200():
    cli = LongctxClient("http://x:1")
    fake_resp = MagicMock()
    fake_resp.status_code = 500
    with patch("longctx_svc.client.httpx.get", return_value=fake_resp):
        assert cli.healthz() is False


def test_healthz_false_on_http_error():
    cli = LongctxClient("http://x:1")
    with patch(
        "longctx_svc.client.httpx.get",
        side_effect=httpx.ConnectError("boom"),
    ):
        assert cli.healthz() is False


# ----------------------------------------------------- retrieve (sync)


def _fake_post_response(payload: dict) -> MagicMock:
    """Build a MagicMock that quacks like httpx.Response."""
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = payload
    return r


def test_retrieve_returns_empty_on_http_error():
    cli = LongctxClient("http://x:1")
    with patch(
        "longctx_svc.client.httpx.post",
        side_effect=httpx.ConnectError("down"),
    ):
        result = cli.retrieve(prefill_text="p", query="q", session_id="sid-x")
    assert isinstance(result, RetrieveResult)
    assert result.chunks == []
    # Caller still gets their session_id back so the engine can keep
    # correlating turns even when retrieval failed.
    assert result.session_id == "sid-x"


def test_retrieve_returns_empty_on_raise_for_status():
    cli = LongctxClient("http://x:1")
    bad_resp = MagicMock()
    bad_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        ),
    )
    with patch("longctx_svc.client.httpx.post", return_value=bad_resp):
        result = cli.retrieve(prefill_text="p", query="q")
    assert result.chunks == []


def test_retrieve_passes_explicit_scope_in_body():
    """When explicit_scope is provided, it MUST land in the JSON body so
    the server bypasses scope detection."""
    cli = LongctxClient("http://x:1")
    fake_resp = _fake_post_response({"chunks": []})
    with patch(
        "longctx_svc.client.httpx.post", return_value=fake_resp
    ) as mock_post:
        cli.retrieve(
            prefill_text="p",
            query="q",
            explicit_scope="/Users/me/myrepo",
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["explicit_scope"] == "/Users/me/myrepo"


def test_retrieve_omits_explicit_scope_when_not_set():
    cli = LongctxClient("http://x:1")
    fake_resp = _fake_post_response({"chunks": []})
    with patch(
        "longctx_svc.client.httpx.post", return_value=fake_resp
    ) as mock_post:
        cli.retrieve(prefill_text="p", query="q")
    body = mock_post.call_args.kwargs["json"]
    assert "explicit_scope" not in body


def test_retrieve_sets_session_affinity_header():
    cli = LongctxClient("http://x:1")
    fake_resp = _fake_post_response({"chunks": []})
    with patch(
        "longctx_svc.client.httpx.post", return_value=fake_resp
    ) as mock_post:
        cli.retrieve(prefill_text="p", query="q", session_id="sess-42")
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["x-session-affinity"] == "sess-42"


def test_retrieve_parses_full_payload():
    cli = LongctxClient("http://x:1")
    payload = {
        "chunks": [
            {
                "text": "def foo(): pass",
                "file_path": "/r/a.py",
                "start_line": 1,
                "end_line": 1,
                "file_type": "python",
                "score": 0.91,
            },
        ],
        "scope_path": "/r",
        "scope_status": "ready",
        "scope_sentinel": ".git",
        "scope_hash": "abc123",
        "session_id": "s",
        "used_rerank": True,
        "paraphrases_count": 3,
    }
    with patch(
        "longctx_svc.client.httpx.post",
        return_value=_fake_post_response(payload),
    ):
        r = cli.retrieve(prefill_text="p", query="q")
    assert len(r.chunks) == 1
    assert r.chunks[0].file_path == "/r/a.py"
    assert r.chunks[0].score == pytest.approx(0.91)
    assert r.scope_path == "/r"
    assert r.scope_status == "ready"
    assert r.used_rerank is True
    assert r.paraphrases_count == 3


# ----------------------------------------------------- retrieve (async)


@pytest.mark.asyncio
async def test_aretrieve_returns_empty_on_http_error():
    cli = LongctxClient("http://x:1")

    async def fake_post(*args, **kwargs):
        raise httpx.ConnectError("down")

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))

    with patch(
        "longctx_svc.client.httpx.AsyncClient", return_value=fake_client
    ):
        r = await cli.aretrieve(prefill_text="p", query="q", session_id="sid")
    assert r.chunks == []
    assert r.session_id == "sid"


@pytest.mark.asyncio
async def test_aretrieve_parses_response():
    cli = LongctxClient("http://x:1")

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "chunks": [
            {
                "text": "hi",
                "file_path": "/p/x.md",
                "start_line": 2,
                "end_line": 4,
                "file_type": "markdown",
                "score": 0.5,
            }
        ],
        "scope_path": "/p",
    }
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with patch(
        "longctx_svc.client.httpx.AsyncClient", return_value=fake_client
    ):
        r = await cli.aretrieve(
            prefill_text="p", query="q", explicit_scope="/p"
        )
    assert len(r.chunks) == 1
    assert r.chunks[0].file_path == "/p/x.md"
    # explicit_scope should have been sent in the body.
    sent_body = fake_client.post.call_args.kwargs["json"]
    assert sent_body["explicit_scope"] == "/p"


# ----------------------------------------------------- splice


def test_splice_noop_when_no_chunks():
    cli = LongctxClient("http://x:1")
    out = cli.splice("the prompt", RetrieveResult(chunks=[]))
    assert out == "the prompt"


def test_splice_prepends_header_and_fenced_chunks():
    cli = LongctxClient("http://x:1")
    chunks = [
        RetrievedChunk(
            text="def foo(): pass",
            file_path="/r/a.py",
            start_line=1,
            end_line=1,
            file_type="python",
            score=0.5,
        ),
    ]
    out = cli.splice("USER PROMPT", RetrieveResult(chunks=chunks))
    assert out.endswith("USER PROMPT")
    assert "## Retrieved code context" in out
    assert "// /r/a.py:1-1" in out
    assert "```" in out
    assert "def foo(): pass" in out


def test_retrieved_chunk_header_format():
    c = RetrievedChunk(
        text="x", file_path="/r/b.py", start_line=10, end_line=20,
        file_type="python", score=0.0,
    )
    assert c.header() == "// /r/b.py:10-20"
