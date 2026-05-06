"""Tests for longctx.eval.bench (canonical bench script).

Mocks LongCtxClient and MRCRRunner so main() can be exercised without
loading a real model or hitting a server.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch


def _make_summary(n=2, avg_score=0.7, prefix_pass_rate=1.0,
                  avg_prompt_tokens=12345, total_time_s=1.2):
    s = MagicMock()
    s.n = n
    s.avg_score = avg_score
    s.prefix_pass_rate = prefix_pass_rate
    s.avg_prompt_tokens = avg_prompt_tokens
    s.total_time_s = total_time_s
    return s


def _patches(summary=None):
    """Patch LongCtxClient and MRCRRunner where bench.py imports them."""
    summary = summary or _make_summary()
    runner = MagicMock()
    runner.run = MagicMock(return_value=summary)
    client = MagicMock()
    client.model = "fake-model"
    client.server = "http://x"
    client.system_prompt = "sys"
    client.timeout = 30
    client.pipeline = MagicMock()
    return (
        patch("longctx.eval.bench.LongCtxClient", return_value=client),
        patch("longctx.eval.bench.MRCRRunner", return_value=runner),
        client, runner, summary,
    )


def test_run_rag_calls_runner_run():
    from longctx.eval.bench import run_rag

    runner = MagicMock()
    runner.run = MagicMock(return_value=_make_summary())
    client = MagicMock()
    client.model = "m"

    out = run_rag(client, runner, "8k", n=5, top_k=8)
    runner.run.assert_called_once()
    kwargs = runner.run.call_args.kwargs
    assert kwargs["bin_name"] == "8k"
    assert kwargs["n"] == 5
    assert kwargs["top_k"] == 8
    assert out.avg_score == 0.7


def test_run_dense_passes_through_all_candidates():
    """run_dense wraps client into _DenseClient that uses top_k=len(cands)."""
    from longctx.eval.bench import run_dense
    from longctx.rag.client import LongCtxResponse

    inner = MagicMock()
    inner.model = "m"
    inner.ask = MagicMock(return_value=LongCtxResponse(
        content="x", retrieved_indices=[0, 1, 2],
        prompt_tokens=10, completion_tokens=2, latency_s=0.1,
    ))
    runner = MagicMock()

    captured_client = {}

    def _runner_run(passed_client, **kw):
        captured_client["c"] = passed_client
        # Exercise the dense client's ask() so we hit those branches
        passed_client.ask("q", ["a", "b", "c"], top_k=99)
        return _make_summary()

    runner.run = _runner_run
    out = run_dense(inner, runner, "8k", n=3)

    assert out.avg_score == 0.7
    # Inner.ask called with top_k = number of candidates (3), not the 99 passed
    assert inner.ask.call_args.kwargs["top_k"] == 3
    assert captured_client["c"].model == "m"


def test_run_chunked_rag_uses_chunked_retrieval():
    """run_chunked_rag wraps client; on ask() it calls retrieve_chunked + http."""
    from longctx.eval.bench import run_chunked_rag

    retrieval = MagicMock()
    retrieval.indices = [0, 1]
    retrieval.candidates = ["c0", "c1"]

    inner = MagicMock()
    inner.model = "m"
    inner.server = "http://x"
    inner.system_prompt = "sys"
    inner.timeout = 30
    inner.pipeline = MagicMock()
    inner.pipeline.retrieve_chunked = MagicMock(return_value=retrieval)

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "choices": [{"message": {"content": "out"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    })

    runner = MagicMock()

    def _runner_run(passed_client, **kw):
        with patch("requests.post", return_value=fake_response):
            resp = passed_client.ask("q", ["a", "b", "c"], top_k=2)
        assert resp.content == "out"
        assert resp.prompt_tokens == 11
        assert resp.retrieved_indices == [0, 1]
        return _make_summary()

    runner.run = _runner_run
    out = run_chunked_rag(inner, runner, "8k", n=3, top_k=2, chunk_size=100)
    assert out.avg_score == 0.7
    inner.pipeline.retrieve_chunked.assert_called_once()


def test_main_runs_rag_only(monkeypatch, capsys):
    """Default main path: no dense, no chunked, just RAG over given bins."""
    p_client, p_runner, client, runner, summary = _patches()
    args = [
        "longctx-bench",
        "--data-dir", "/tmp/data",
        "--model", "fake-model",
        "--bins", "8k", "32k",
        "--n", "2",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with p_client, p_runner:
        from longctx.eval.bench import main
        main()

    # Runner.run called twice (one per bin)
    assert runner.run.call_count == 2
    out = capsys.readouterr().out
    assert "longctx bench" in out
    assert "rag" in out
    assert "8k" in out


def test_main_with_dense_and_chunked_and_out(monkeypatch, tmp_path, capsys):
    """--include-dense + --include-chunked + --out exercises all branches."""
    p_client, p_runner, client, runner, summary = _patches(
        _make_summary(avg_score=0.80)  # > 0.659 SubQ
    )
    out_file = tmp_path / "results.json"
    args = [
        "longctx-bench",
        "--data-dir", "/tmp/data",
        "--model", "fake-model",
        "--bins", "8k",
        "--n", "1",
        "--include-dense",
        "--include-chunked",
        "--out", str(out_file),
    ]
    monkeypatch.setattr(sys, "argv", args)

    # _DenseClient uses inner.ask (which is client.ask), so make it work
    from longctx.rag.client import LongCtxResponse
    client.ask = MagicMock(return_value=LongCtxResponse(
        content="x", retrieved_indices=[0],
        prompt_tokens=1, completion_tokens=1, latency_s=0.1,
    ))

    # Make the runner.run actually use the passed client's ask once
    # so dense/chunked branches get hit.
    def _runner_run(passed_client, **kw):
        return summary
    runner.run = _runner_run

    with p_client, p_runner:
        from longctx.eval.bench import main
        main()

    text = out_file.read_text()
    data = json.loads(text)
    # 3 cells: dense + rag + chunked
    assert len(data) == 3
    pipelines = {c["pipeline_name"] for c in data}
    assert pipelines == {"dense", "rag", "chunked-rag"}

    out = capsys.readouterr().out
    # Above-SubQ summary line printed
    assert "exceeding SubQ" in out
