"""Tests for LongCtxClient."""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_client_constructs_with_defaults(fake_pipeline):
    from longctx import LongCtxClient

    c = LongCtxClient(pipeline=fake_pipeline, model="test-model")
    assert c.model == "test-model"
    assert c.pipeline is fake_pipeline
    assert c.system_prompt
    assert "verbatim" in c.system_prompt.lower()


def test_client_ask_returns_response(fake_pipeline, mock_post_factory):
    from longctx import LongCtxClient

    c = LongCtxClient(pipeline=fake_pipeline, model="test-model")
    candidates = ["alpha message", "beta message", "gamma message"]

    with patch("longctx.rag.client.requests.post",
               return_value=mock_post_factory(content="alpha message")):
        result = c.ask(query="something", candidates=candidates, top_k=2)

    assert result.content == "alpha message"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2
    assert result.retrieved_indices == [0, 1]
    assert result.latency_s >= 0.0


def test_client_ask_payload_contains_candidates(
    fake_pipeline, mock_post_factory
):
    """Verify the user message contains every retrieved candidate."""
    from longctx import LongCtxClient

    c = LongCtxClient(pipeline=fake_pipeline, model="m")

    captured = {}

    def _capture(*args, **kwargs):
        captured["json"] = kwargs.get("json", {})
        return mock_post_factory()

    with patch("longctx.rag.client.requests.post", side_effect=_capture):
        c.ask("q", ["alpha", "beta", "gamma"], top_k=2)

    user_msg = captured["json"]["messages"][1]["content"]
    assert "alpha" in user_msg
    assert "beta" in user_msg
    assert "USER FINAL QUESTION" in user_msg
    assert captured["json"]["temperature"] == 0.0


def test_client_ask_raises_on_http_error(fake_pipeline, mock_post_factory):
    """LongCtxClient propagates HTTP errors via raise_for_status()."""
    from longctx import LongCtxClient
    from unittest.mock import MagicMock
    import requests as _requests

    c = LongCtxClient(pipeline=fake_pipeline, model="m")
    bad = mock_post_factory(status_code=500)
    bad.raise_for_status = MagicMock(
        side_effect=_requests.HTTPError("500"))

    with patch("longctx.rag.client.requests.post", return_value=bad):
        with pytest.raises(_requests.HTTPError):
            c.ask("q", ["a", "b"], top_k=1)


def test_client_default_system_prompt_has_unconditional_prefix():
    """Regression test for the prompt-fix commit (bed08f0).

    The library originally said "If the user provides a prefix string to
    prepend, prepend it" which the model interpreted as optional, dropping
    prefix on borderline samples and lowering MRCR scores. The fix made
    the prefix instruction unconditional. Don't regress.
    """
    from longctx.rag.client import DEFAULT_SYSTEM_PROMPT

    assert "If the user provides a prefix" not in DEFAULT_SYSTEM_PROMPT
    assert "prepending the prefix" in DEFAULT_SYSTEM_PROMPT.lower() or \
           "prefix string the user provides" in DEFAULT_SYSTEM_PROMPT
