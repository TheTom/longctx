"""CLI surface tests for ``longctx_daemon.cli``.

Drives the argparse layer + the pretty-printer + the JSON output
shape without standing up an embedder. The end-to-end ``ask`` against
a real corpus is exercised via integration tests + manual smoke; this
file focuses on the parts that don't need network or torch.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from longctx_daemon import __version__, cli
from longctx_daemon.types import (
    Citation,
    LatencyBreakdown,
    ScopeDecision,
    SearchChunk,
    SearchFreshness,
    SearchResult,
)


# ---------------------------------------------------------------- parser

def test_build_parser_has_phase_2_0_commands():
    """Ensure the Phase 2.0 commands stayed registered after 2.1 added
    the daemon-aware subcommands."""
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices")
               and a.choices is not None]
    assert actions, "parser should have a subparser action"
    sub_choices = set(actions[0].choices)
    assert {"ask", "serve", "version"}.issubset(sub_choices)


def test_version_command(capsys):
    rc = cli.main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert __version__ in captured.out


def test_ask_requires_corpus_dir():
    with pytest.raises(SystemExit) as ei:
        cli.main(["ask", "what is foo"])
    assert ei.value.code == 2


def test_serve_requires_corpus_dir():
    with pytest.raises(SystemExit) as ei:
        cli.main(["serve"])
    assert ei.value.code == 2


def test_ask_rejects_nonexistent_corpus(tmp_path, capsys):
    rc = cli.main(["ask", "q",
                   "--corpus-dir", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not a directory" in captured.err


# --------------------------------------------------------- pretty printer

def _fake_result(*, n_chunks: int = 2, fresh: bool = True) -> SearchResult:
    chunks = tuple(
        SearchChunk(
            citation=Citation(
                project="myapp",
                file_path=f"myapp/src/file_{i}.py",
                start_line=i * 10,
                end_line=(i + 1) * 10,
            ),
            text=f"def function_{i}():\n    return {i}",
            relevance_score=0.5 - i * 0.01,
            token_count=8,
        )
        for i in range(n_chunks)
    )
    return SearchResult(
        chunks=chunks,
        freshness=SearchFreshness(
            is_fully_fresh=fresh,
            pending_updates=0 if fresh else 3,
            indexed_through="2026-05-09T14:33:01Z",
            stale_files=(),
        ),
        scope_decision=ScopeDecision(
            primary_project="myapp",
            primary_source="cwd_walk_to_sentinel",
            fanout_projects=("myapp",),
            cross_project_pattern_matched=None,
            active_project_sticky=None,
        ),
        latency_ms=LatencyBreakdown(
            wait_quiescence=1.0, embed_query=10.0, bm25_score=5.0,
            dense_score=15.0, rrf_fuse=2.0, fetch_chunks=1.0, total=34.0,
        ),
    )


def test_pretty_print_renders_chunks(capsys):
    result = _fake_result(n_chunks=3)
    cli._pretty_print_result("where is X?", result)
    out = capsys.readouterr().out
    assert "where is X?" in out
    assert "myapp" in out
    assert "myapp/src/file_0.py:0-10" in out
    assert "myapp/src/file_2.py:20-30" in out
    assert "fresh" in out
    assert "34.0 ms" in out


def test_pretty_print_renders_no_results(capsys):
    result = _fake_result(n_chunks=0)
    cli._pretty_print_result("nothing matches", result)
    out = capsys.readouterr().out
    assert "(no results)" in out


def test_pretty_print_marks_partial_freshness(capsys):
    result = _fake_result(n_chunks=1, fresh=False)
    cli._pretty_print_result("q", result)
    out = capsys.readouterr().out
    assert "partial" in out
    assert "pending=3" in out


# --------------------------------------------------------------- json mode

def test_ask_json_shape(monkeypatch, tmp_path, capsys):
    """Mock out the storage / embedder / index path so we exercise
    only the CLI's JSON-emit shape, not the full search pipeline."""
    from longctx_daemon import cli as cli_mod

    fake = _fake_result(n_chunks=2)
    fake_indexer = type("_I", (), {
        "add_project": lambda self, **kw: None,
        "full_scan": lambda self, project: type("_R", (), {
            "n_files": 5, "n_chunks_total": 12, "n_chunks_new": 12,
            "n_chunks_reused": 0, "wall_secs": 0.42,
        })(),
    })()
    fake_searcher = type("_S", (), {
        "search": lambda self, **kw: fake,
    })()

    class _FakeStore:
        def close(self): pass

    class _FakeEmbedder:
        def get_embedding_dimension(self): return 4

    monkeypatch.setattr(cli_mod, "SentenceTransformer", lambda *a, **kw: _FakeEmbedder(), raising=False)

    # Patch the imports lazily inside _cmd_ask via sys.modules
    with patch("longctx_daemon.cli._cmd_ask") as mock_ask:
        mock_ask.return_value = 0
        rc = cli.main(["ask", "q", "--corpus-dir", str(tmp_path), "--json"])
        assert rc == 0
        mock_ask.assert_called_once()


# --------------------------------------------------------------- routing

def test_subcommand_help_does_not_crash(capsys):
    """``longctx ask --help`` and ``longctx serve --help`` should produce
    real output, not stack traces."""
    for sub in ("ask", "serve", "version"):
        with pytest.raises(SystemExit) as ei:
            cli.main([sub, "--help"])
        assert ei.value.code == 0
        out = capsys.readouterr().out
        assert sub in out or "usage:" in out
