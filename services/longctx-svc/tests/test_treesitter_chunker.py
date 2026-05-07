"""Tree-sitter chunker tests. PRD §6.1 / v0.3.1.

Validates that:
  1. Code splits at top-level definitions when a parser is available
  2. Falls back gracefully (returns None) when not
  3. Big functions degrade to line-window inside themselves
  4. Preamble (imports) and trailing tail are emitted as separate chunks
  5. The opt-in flag (use_treesitter) actually routes through the new path
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _ts_available(ext: str) -> bool:
    from longctx_svc.indexer.treesitter_chunker import has_parser_for
    return has_parser_for(Path(f"x{ext}"))


# ---------------------------------------------------------------------------
# Direct chunker tests (call chunk_code_treesitter)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ts_available(".py"),
                    reason="tree_sitter_python not installed")
def test_python_splits_at_def_and_class():
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    src = (
        "import os\n"
        "import sys\n"
        "\n"
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "class Bar:\n"
        "    def method(self):\n"
        "        return 2\n"
        "\n"
        "def gamma():\n"
        "    return 3\n"
    )
    chunks = chunk_code_treesitter(src, "/p/x.py", ".py", max_lines=50)
    assert chunks is not None
    # Expect: imports preamble, alpha, Bar, gamma → 4 chunks
    texts = [c.text for c in chunks]
    assert any("import os" in t for t in texts), texts
    assert any("def alpha" in t for t in texts), texts
    assert any("class Bar" in t for t in texts), texts
    assert any("def gamma" in t for t in texts), texts


@pytest.mark.skipif(not _ts_available(".py"),
                    reason="tree_sitter_python not installed")
def test_python_chunk_line_numbers_correct():
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    chunks = chunk_code_treesitter(src, "/p/x.py", ".py", max_lines=50)
    assert chunks is not None
    a = next(c for c in chunks if "def a" in c.text)
    b = next(c for c in chunks if "def b" in c.text)
    assert a.start_line == 1
    assert a.end_line == 2
    assert b.start_line == 4
    assert b.end_line == 5


@pytest.mark.skipif(not _ts_available(".py"),
                    reason="tree_sitter_python not installed")
def test_python_huge_function_falls_back_to_window():
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    body = "    pass\n" * 100
    src = f"def big():\n{body}"
    chunks = chunk_code_treesitter(src, "/p/x.py", ".py", max_lines=20)
    assert chunks is not None
    # Body has 101 lines; 20-line windows → at least 5 chunks
    assert len(chunks) >= 5
    # All chunks belong to the function — line numbers monotonic & overlap
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)


@pytest.mark.skipif(not _ts_available(".ts"),
                    reason="tree_sitter_typescript not installed")
def test_typescript_splits_at_function_and_class():
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    src = (
        "import { x } from 'y';\n"
        "\n"
        "export function authMiddleware() {\n"
        "  return true;\n"
        "}\n"
        "\n"
        "export class BillingService {\n"
        "  charge() { return 1; }\n"
        "}\n"
    )
    chunks = chunk_code_treesitter(src, "/p/x.ts", ".ts", max_lines=50)
    assert chunks is not None
    texts = [c.text for c in chunks]
    assert any("authMiddleware" in t for t in texts)
    assert any("BillingService" in t for t in texts)


def test_unsupported_extension_returns_none():
    """File types we don't have a parser for must return None so the
    caller falls back to line-window."""
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    chunks = chunk_code_treesitter("hello\n", "/p/x.cobol", ".cobol",
                                   max_lines=50)
    assert chunks is None


def test_no_top_level_defs_returns_none():
    """A script with only statements has no top-level defs to split on
    → caller falls back to line-window."""
    from longctx_svc.indexer.treesitter_chunker import chunk_code_treesitter
    src = "x = 1\ny = 2\nz = x + y\nprint(z)\n"
    chunks = chunk_code_treesitter(src, "/p/x.py", ".py", max_lines=50)
    # Either None (no top-level defs) OR returns a single chunk for the
    # tail. Both are valid; we only require we don't lose data.
    if chunks is not None:
        joined = "".join(c.text for c in chunks)
        assert "x = 1" in joined
        assert "print(z)" in joined


def test_missing_parser_module_returns_none():
    """If the tree-sitter language module isn't importable, return None."""
    from longctx_svc.indexer import treesitter_chunker
    saved = dict(treesitter_chunker._PARSER_CACHE)
    treesitter_chunker._PARSER_CACHE.clear()
    try:
        with patch.object(treesitter_chunker, "_get_parser",
                          return_value=None):
            chunks = treesitter_chunker.chunk_code_treesitter(
                "def x(): pass\n", "/p/x.py", ".py", max_lines=50,
            )
        assert chunks is None
    finally:
        treesitter_chunker._PARSER_CACHE.clear()
        treesitter_chunker._PARSER_CACHE.update(saved)


# ---------------------------------------------------------------------------
# Integration via chunk_text() with the use_treesitter config flag
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ts_available(".py"),
                    reason="tree_sitter_python not installed")
def test_chunk_text_uses_treesitter_when_enabled():
    from longctx_svc.config import ServiceConfig, set_config
    from longctx_svc.indexer.chunker import chunk_text
    set_config(ServiceConfig(use_treesitter=True))
    try:
        chunks = chunk_text(
            "def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
            "/p/x.py",
        )
        # Two top-level defs → two distinct chunks
        assert len(chunks) == 2
        assert any("alpha" in c.text for c in chunks)
        assert any("beta" in c.text for c in chunks)
    finally:
        set_config(ServiceConfig())


def test_chunk_text_default_off_uses_line_window():
    """Default (use_treesitter=False) preserves v0.3.0 behavior."""
    from longctx_svc.config import ServiceConfig, set_config
    from longctx_svc.indexer.chunker import chunk_text
    set_config(ServiceConfig(use_treesitter=False))
    try:
        src = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        chunks = chunk_text(src, "/p/x.py")
        # Single short file → line-window emits one chunk
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
    finally:
        set_config(ServiceConfig())
