"""Unit tests for longctx.rag.symbol_augment.

Shared module exercised by both ``longctx_svc.retrieve.pipeline`` and
``longctx_daemon.searcher``. These tests stay focused on the pure-Python
helpers (no embedder, no chunk store). Pipeline-level integration is
covered in services/longctx-svc/tests/test_symbol_augment_pipeline.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from longctx.rag.symbol_augment import (
    extract_symbols,
    file_type_weight,
    has_code_signal,
    query_features,
    symbol_grep_repo,
)


# ---------------------------------------------------------- extract_symbols

def test_extract_camelcase():
    """CamelCase class names get extracted as a single token."""
    assert "FilePathField" in extract_symbols(
        "FilePathField raises TypeError when path missing",
    )


def test_extract_backticked():
    """Markdown-style `backtick` identifiers are surfaced."""
    syms = extract_symbols("the `resolve_redirects` helper drops headers")
    assert "resolve_redirects" in syms


def test_extract_class_def():
    """`class Foo` patterns inside the query are extracted."""
    syms = extract_symbols("class Session has a bug")
    assert "Session" in syms


def test_extract_def():
    """`def foo` patterns inside the query are extracted."""
    syms = extract_symbols("def format_cursor_data returns wrong")
    assert "format_cursor_data" in syms


def test_extract_snake_case():
    """snake_case with at least two underscores is extracted."""
    syms = extract_symbols("the iter_content_chunks helper is broken")
    assert "iter_content_chunks" in syms


def test_extract_qualified_splits():
    """`X.y` qualified names get split into both components."""
    syms = extract_symbols("Session.resolve_redirects misbehaves")
    assert "Session" in syms
    assert "resolve_redirects" in syms


def test_extract_drops_stopwords():
    """English words long enough to match a regex but in the stopword
    list are dropped."""
    syms = extract_symbols("This Field These Models")
    assert "this" not in {s.lower() for s in syms}
    assert "field" not in {s.lower() for s in syms}
    assert "models" not in {s.lower() for s in syms}


def test_extract_drops_short():
    """Anything under length 4 is dropped — those are too noisy to grep."""
    syms = extract_symbols("Foo Bar Baz qux")  # Foo/Bar/Baz are len 3
    assert "Foo" not in syms
    assert "Bar" not in syms


# ------------------------------------------------------- symbol_grep_repo

def _make_repo(tmp_path: Path) -> Path:
    """A tiny throw-away repo to grep against."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "models.py").write_text(
        "class FilePathField(Field):\n    pass\n"
    )
    (repo / "pkg" / "helpers.py").write_text(
        "def resolve_redirects(req):\n    return req\n"
    )
    (repo / "pkg" / "tests").mkdir()
    (repo / "pkg" / "tests" / "test_thing.py").write_text(
        "def resolve_redirects():\n    pass\n"
    )
    (repo / "pkg" / "test_other.py").write_text(
        "class FilePathField:\n    pass\n"
    )
    (repo / "README.md").write_text(
        "## class FilePathField\nDescribes the field.\n"
    )
    return repo


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_grep_finds_class_def(tmp_path):
    repo = _make_repo(tmp_path)
    hits = symbol_grep_repo({"FilePathField"}, repo)
    assert any("models.py" in h for h in hits)


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_grep_finds_def(tmp_path):
    repo = _make_repo(tmp_path)
    hits = symbol_grep_repo({"resolve_redirects"}, repo)
    assert any("helpers.py" in h for h in hits)


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_grep_excludes_tests_dir(tmp_path):
    """`tests/` and `test_*.py` are filtered out — source under test is
    what we want, not the test file itself."""
    repo = _make_repo(tmp_path)
    hits = symbol_grep_repo({"resolve_redirects", "FilePathField"}, repo)
    assert not any("/tests/" in h for h in hits)
    assert not any("test_other.py" in h for h in hits)


def test_grep_empty_input_returns_empty():
    """No symbols → no rg call → empty list."""
    assert symbol_grep_repo(set(), Path("/nonexistent")) == []


def test_grep_missing_repo_returns_empty():
    """Bad root → empty list, no crash."""
    assert symbol_grep_repo({"Foo"}, Path("/definitely/not/here")) == []


# ----------------------------------------------------------- query features

def test_features_counts_traceback_lines():
    qf = query_features('Traceback... File "x.py", line 1 ... File "y.py", line 2')
    assert qf["n_tracebacks"] == 2


def test_features_counts_error_types():
    qf = query_features("raises ValueError and TypeError but not RuntimeError twice")
    # ValueError, TypeError, RuntimeError = 3
    assert qf["n_error_types"] == 3


def test_features_counts_symbol_defs():
    qf = query_features("def foo and class Bar are broken")
    assert qf["n_symbol_defs"] == 2


def test_features_zero_for_prose():
    """Prose without code markers → all zero."""
    qf = query_features("Please describe how the feature should work in plain English.")
    assert qf == {"n_tracebacks": 0, "n_symbol_defs": 0, "n_error_types": 0}


# ----------------------------------------------------------- has_code_signal

def test_signal_traceback_only():
    assert has_code_signal({"n_tracebacks": 1, "n_symbol_defs": 0, "n_error_types": 0})


def test_signal_error_only():
    assert has_code_signal({"n_tracebacks": 0, "n_symbol_defs": 0, "n_error_types": 1})


def test_signal_def_only():
    assert has_code_signal({"n_tracebacks": 0, "n_symbol_defs": 1, "n_error_types": 0})


def test_signal_none():
    assert not has_code_signal({"n_tracebacks": 0, "n_symbol_defs": 0, "n_error_types": 0})


# ----------------------------------------------------------- file_type_weight

def test_weight_no_signal_is_neutral():
    """Without code signal, everything weighs 1.0 — feature-add queries
    shouldn't be biased toward .py."""
    qf = {"n_tracebacks": 0, "n_symbol_defs": 0, "n_error_types": 0}
    assert file_type_weight("docs/index.rst", qf) == 1.0
    assert file_type_weight("src/x.py", qf) == 1.0


def test_weight_py_boost_under_signal():
    qf = {"n_tracebacks": 1, "n_symbol_defs": 0, "n_error_types": 0}
    assert file_type_weight("src/x.py", qf) == 1.5


def test_weight_doc_demote_under_signal():
    qf = {"n_tracebacks": 1, "n_symbol_defs": 0, "n_error_types": 0}
    for ext in (".rst", ".md", ".txt", ".yaml", ".yml", ".cff"):
        assert file_type_weight(f"docs/anything{ext}", qf) == 0.6


def test_weight_test_files_slightly_demoted():
    qf = {"n_tracebacks": 1, "n_symbol_defs": 0, "n_error_types": 0}
    assert file_type_weight("pkg/tests/test_thing.scala", qf) == 0.9


# Coverage gap: file_type_weight fallback when no extension family
# matches AND query has code signal — neutral 1.0 weight.
def test_weight_unknown_extension_neutral_under_signal():
    qf = {"n_tracebacks": 1, "n_symbol_defs": 0, "n_error_types": 0}
    # Binary-like extension not in code/doc/test buckets → neutral.
    assert file_type_weight("data/blob.parquet", qf) == 1.0


# Coverage gap: dotted-symbol decomposition keeps the parts long
# enough to clear the 4-char floor and drops the original.
def test_extract_decomposes_long_dotted_symbol():
    """A.B → drop A.B, keep A and B individually when each >= 4 chars.

    Verifies the split-and-readd branch in extract_symbols (lines 67-72).
    """
    syms = extract_symbols("ClassOne.MethodAlpha raises ValueError")
    assert "ClassOne" in syms
    assert "MethodAlpha" in syms
    # Original dotted form gone after decomposition.
    assert "ClassOne.MethodAlpha" not in syms


def test_extract_drops_short_part_of_dotted_symbol():
    """A.b where part is < 4 chars: that part is dropped, the long part
    survives. Confirms the `if len(p) >= 4` filter inside the split."""
    syms = extract_symbols("LongClassName.ab raises")
    assert "LongClassName" in syms
    # "ab" (< 4 chars) must NOT survive.
    assert "ab" not in syms


# Coverage gap: ripgrep call paths that exit non-zero / raise.


def test_grep_handles_ripgrep_exception(tmp_path, monkeypatch):
    """If `subprocess.run` raises (rg missing, timeout, etc.) for a
    symbol, that symbol contributes zero hits and we keep going.
    Covers the `except Exception: continue` swallow at line 103-104.
    """
    import subprocess as _subprocess

    real_run = _subprocess.run
    call_count = {"n": 0}

    def flaky_run(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _subprocess.TimeoutExpired(cmd="rg", timeout=8)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(_subprocess, "run", flaky_run)

    # Two symbols — first run() raises, second succeeds (returns nothing
    # in an empty tmp dir, but exercising the swallow + recovery path).
    hits = symbol_grep_repo(["AlphaName", "BetaName"], tmp_path)
    assert hits == []  # nothing found, but no exception bubbled up


def test_grep_dedupes_repeated_lines(tmp_path):
    """If the same line appears for two different symbols, it's only
    emitted once. Covers `if not line or line in seen: continue`
    at line 108-109."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    # Single file mentioning two symbols → ripgrep returns the same line
    # for both queries when symbols co-occur on a line.
    (tmp_path / "src.py").write_text("AlphaName and BetaName are siblings\n")
    hits = symbol_grep_repo(["AlphaName", "BetaName"], tmp_path)
    # Both should NOT add separate entries for the same matched line.
    unique = set(hits)
    assert len(hits) == len(unique), f"duplicate lines in hits: {hits}"
