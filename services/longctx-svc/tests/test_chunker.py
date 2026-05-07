"""Chunker tests. PRD §5.4 — code/prose/config/other strategies."""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.indexer.chunker import Chunk, chunk_file, chunk_text


def test_chunk_short_code_one_chunk():
    text = "def foo():\n    return 1\n"
    chunks = chunk_text(text, "/foo.py", file_type="code", lines_per_chunk=50)
    assert len(chunks) == 1
    assert chunks[0].file_type == "code"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[0].text == text


def test_chunk_long_code_multiple_chunks():
    lines = [f"line_{i}\n" for i in range(120)]
    chunks = chunk_text("".join(lines), "/foo.py", file_type="code",
                        lines_per_chunk=50, overlap=5)
    # 120 lines, 50/chunk, overlap 5 → step 45 → starts at 0,45,90 ⇒ 3 chunks
    assert len(chunks) == 3
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 50
    assert chunks[1].start_line == 46           # overlap
    assert chunks[-1].end_line == 120


def test_chunk_overlap_correct():
    lines = [f"l{i}\n" for i in range(20)]
    chunks = chunk_text("".join(lines), "/x.py", file_type="code",
                        lines_per_chunk=10, overlap=3)
    assert chunks[0].end_line == 10
    assert chunks[1].start_line == 8           # 10-3+1
    assert chunks[1].end_line in (17, 18, 19, 20)


def test_chunk_prose_paragraph_aware():
    text = ("First paragraph about alpha.\n\n"
            "Second paragraph about beta.\n\n"
            "Third paragraph about gamma.\n")
    chunks = chunk_text(text, "/r.md", file_type="prose",
                        lines_per_chunk=50)
    # Three paragraphs fit in single chunk under cap
    assert len(chunks) == 1
    assert chunks[0].file_type == "prose"
    assert "alpha" in chunks[0].text


def test_chunk_prose_splits_on_budget():
    paragraphs = []
    for i in range(20):
        paragraphs.append(f"paragraph_{i} line a\nparagraph_{i} line b\n\n")
    text = "".join(paragraphs)
    chunks = chunk_text(text, "/r.md", file_type="prose",
                        lines_per_chunk=10)
    assert len(chunks) > 1


def test_chunk_config_whole_if_small():
    text = '{\n  "name": "myapp"\n}\n'
    chunks = chunk_text(text, "/p.json", file_type="config",
                        lines_per_chunk=50)
    assert len(chunks) == 1
    assert chunks[0].file_type == "config"


def test_chunk_config_split_if_large():
    lines = [f'  "key_{i}": "value",\n' for i in range(80)]
    text = "{\n" + "".join(lines) + "}\n"
    chunks = chunk_text(text, "/big.json", file_type="config",
                        lines_per_chunk=20, overlap=2)
    assert len(chunks) > 1


def test_chunk_unknown_extension_falls_back_to_other():
    # A file like LICENSE has no extension; chunk_text auto-classifies via path.
    chunks = chunk_text("foo\nbar\n", "/LICENSE")
    # Other extensions get treated as code-like (line-window).
    assert len(chunks) == 1


def test_chunk_file_reads_and_chunks(project_dir: Path):
    auth = project_dir / "src" / "auth.ts"
    chunks = chunk_file(auth)
    assert len(chunks) >= 1
    assert all(c.file_path == str(auth) for c in chunks)
    assert chunks[0].file_type == "code"


def test_chunk_file_handles_missing(tmp_path: Path):
    nonexistent = tmp_path / "nope.txt"
    assert chunk_file(nonexistent) == []


def test_chunk_file_handles_empty(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert chunk_file(empty) == []


def test_chunk_display_format():
    c = Chunk(text="x", file_path="/a.py", start_line=10, end_line=20,
              file_type="code")
    assert c.display() == "/a.py:10-20"
