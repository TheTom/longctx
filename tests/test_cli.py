"""Tests for longctx-eval CLI.

Mocks RetrievalPipeline, LongCtxClient, and MRCRRunner so the CLI argparse
+ main flow can be exercised without loading sentence-transformers or
hitting a real server.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _summary():
    s = MagicMock()
    s.bin = "8k"
    s.n = 1
    s.avg_score = 0.7
    s.prefix_pass_rate = 1.0
    s.results = []
    return s


def _patches(summary=None):
    summary = summary or _summary()
    runner = MagicMock()
    runner.run = MagicMock(return_value=summary)
    client = MagicMock()
    pipeline = MagicMock()
    return (
        patch("longctx.RetrievalPipeline", return_value=pipeline),
        patch("longctx.eval.cli.LongCtxClient", return_value=client),
        patch("longctx.eval.cli.MRCRRunner", return_value=runner),
        client, runner, summary, pipeline,
    )


def test_cli_main_basic(monkeypatch):
    p_pipe, p_client, p_runner, client, runner, summary, pipeline = _patches()
    args = [
        "longctx-eval",
        "--bin", "8k",
        "--n", "5",
        "--model", "fake-model",
        "--data-dir", "/tmp/data",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with p_pipe, p_client, p_runner:
        from longctx.eval.cli import main
        main()

    runner.run.assert_called_once()
    kwargs = runner.run.call_args.kwargs
    assert kwargs["bin_name"] == "8k"
    assert kwargs["n"] == 5
    assert kwargs["top_k"] == 8


def test_cli_main_writes_summary(monkeypatch, tmp_path, capsys):
    p_pipe, p_client, p_runner, client, runner, summary, pipeline = _patches()
    out = tmp_path / "summary.json"
    args = [
        "longctx-eval",
        "--bin", "32k",
        "--n", "3",
        "--top-k", "4",
        "--max-tokens", "256",
        "--model", "fake-model",
        "--data-dir", "/tmp/data",
        "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", args)

    with p_pipe, p_client, p_runner, patch(
        "longctx.eval.cli.MRCRRunner.write_summary"
    ) as wsum:
        from longctx.eval.cli import main
        main()

    wsum.assert_called_once_with(summary, str(out))
    err = capsys.readouterr().err
    assert str(out) in err


def test_cli_main_with_reranker(monkeypatch):
    p_pipe, p_client, p_runner, client, runner, summary, pipeline = _patches()
    args = [
        "longctx-eval",
        "--bin", "8k",
        "--model", "fake-model",
        "--data-dir", "/tmp/data",
        "--reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with patch(
        "longctx.RetrievalPipeline", return_value=pipeline
    ) as ppi, p_client, p_runner:
        from longctx.eval.cli import main
        main()

    # RetrievalPipeline ctor received the reranker kw
    ppi.assert_called_once()
    kw = ppi.call_args.kwargs
    assert kw["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_cli_rejects_unknown_bin(monkeypatch):
    args = [
        "longctx-eval",
        "--bin", "nonsense",
        "--model", "fake-model",
        "--data-dir", "/tmp/data",
    ]
    monkeypatch.setattr(sys, "argv", args)

    from longctx.eval.cli import main
    with pytest.raises(SystemExit):
        main()


def test_cli_requires_data_dir(monkeypatch):
    args = [
        "longctx-eval",
        "--bin", "8k",
        "--model", "fake-model",
    ]
    monkeypatch.setattr(sys, "argv", args)

    from longctx.eval.cli import main
    with pytest.raises(SystemExit):
        main()
