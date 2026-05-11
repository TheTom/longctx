"""Pipeline-level tests for the symbol-aware retrieval augment.

Unit tests for the helpers themselves live in the root
``tests/test_symbol_augment.py``. These tests exercise the augment as it
runs inside ``RetrievePipeline.retrieve`` — covering the scope_root /
no-scope_root paths, the file-type re-rank, and the
``used_symbol_augment`` flag on the result.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from longctx_svc.config import Limits, ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk
from longctx_svc.retrieve.pipeline import RetrievePipeline


def _index_with_chunks(chunks: list[Chunk], embedder,
                       scope_root: Path = Path("/tmp/fake")) -> ScopeIndex:
    embs = embedder.encode(
        [c.text for c in chunks],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return ScopeIndex(
        scope_root=scope_root, scope_hash="fake", chunks=chunks,
        embeddings=embs, embedder_name="fake",
    )


def _set_minimal_config(use_symbol_augment: bool = True):
    """Disable every other scale-aware lane so the augment shows in
    isolation. Reranker off, multi-query off, coarse-filter off."""
    set_config(ServiceConfig(
        embedder_model="fake", reranker_model=None, use_multi_query=False,
        use_coarse_filter=False, use_symbol_augment=use_symbol_augment,
        limits=Limits(rerank_min_chunks=10_000,
                      multiquery_min_chunks=10_000,
                      coarse_filter_min_chunks=100_000),
    ))


def _make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two-file repo: a docs page and a source file. Returns
    (repo, doc_path, source_path)."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = repo / "src" / "fields.py"
    source.write_text(
        "class FilePathField:\n"
        "    \"\"\"alpha alpha alpha\"\"\"\n"
        "    pass\n"
    )
    doc = repo / "CHANGES.rst"
    doc.write_text("FilePathField was changed in 1.0\nalpha alpha alpha\n")
    return repo, doc, source


# ------------------------------------------------------ result flag

def test_no_scope_root_means_no_augment(fake_embedder):
    """Without scope_root, augment is skipped — used_symbol_augment
    must be False even when query has identifiers."""
    _set_minimal_config()
    chunks = [
        Chunk(text="class FilePathField pass", file_path="/tmp/x.py",
              start_line=1, end_line=1, file_type="code"),
    ]
    idx = _index_with_chunks(chunks, fake_embedder)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None)
    result = pipe.retrieve("FilePathField raises TypeError", idx, top_k=3)
    assert result.used_symbol_augment is False


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_symbol_augment_engages_when_grep_matches(fake_embedder, tmp_path):
    """With a scope_root containing a grep-able definition site for an
    identifier mentioned in the query, used_symbol_augment is True."""
    _set_minimal_config()
    repo, doc, source = _make_repo(tmp_path)
    chunks = [
        Chunk(text="alpha alpha alpha", file_path=str(doc),
              start_line=1, end_line=2, file_type="prose"),
        Chunk(text="class FilePathField pass", file_path=str(source),
              start_line=1, end_line=3, file_type="code"),
    ]
    idx = _index_with_chunks(chunks, fake_embedder, scope_root=repo)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None)
    result = pipe.retrieve(
        "FilePathField raises TypeError when path is None",
        idx, top_k=2, scope_root=repo,
    )
    assert result.used_symbol_augment is True


# ------------------------------------------------------ rerank effect

@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_py_source_beats_rst_doc_with_code_signal(fake_embedder, tmp_path):
    """The fake embedder gives the .rst chunk a higher dense score (its
    text contains 'alpha alpha alpha' twice). Without augment, the doc
    wins; with augment + code signal, the .py source wins."""
    _set_minimal_config()
    repo, doc, source = _make_repo(tmp_path)
    chunks = [
        # Doc chunk: stronger dense match (more 'alpha' repetition)
        Chunk(text="alpha alpha alpha alpha alpha",
              file_path=str(doc),
              start_line=1, end_line=2, file_type="prose"),
        # Source chunk: weaker dense match
        Chunk(text="class FilePathField pass",
              file_path=str(source),
              start_line=1, end_line=3, file_type="code"),
    ]
    idx = _index_with_chunks(chunks, fake_embedder, scope_root=repo)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None)
    # Code-signal query: traceback + symbol mention. file-type prior
    # should kick in and demote .rst.
    query = (
        'Traceback File "x.py" FilePathField raises TypeError '
        'when path is None'
    )
    result = pipe.retrieve(query, idx, top_k=2, scope_root=repo)
    assert result.used_symbol_augment is True
    # Top result must be the .py source, not the .rst.
    assert result.chunks[0].file_path == str(source)


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_disable_flag_skips_augment(fake_embedder, tmp_path):
    """use_symbol_augment=False at the pipeline-call level skips augment
    even with scope_root + identifiers + matching files."""
    _set_minimal_config()
    repo, doc, source = _make_repo(tmp_path)
    chunks = [
        Chunk(text="class FilePathField pass", file_path=str(source),
              start_line=1, end_line=3, file_type="code"),
    ]
    idx = _index_with_chunks(chunks, fake_embedder, scope_root=repo)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None)
    result = pipe.retrieve(
        "FilePathField raises TypeError",
        idx, top_k=2, scope_root=repo, use_symbol_augment=False,
    )
    assert result.used_symbol_augment is False


def test_config_default_off_disables_augment(fake_embedder, tmp_path):
    """When ServiceConfig has use_symbol_augment=False, the augment
    doesn't fire even with scope_root."""
    _set_minimal_config(use_symbol_augment=False)
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    chunks = [
        Chunk(text="text", file_path="/tmp/x.py",
              start_line=1, end_line=1, file_type="code"),
    ]
    idx = _index_with_chunks(chunks, fake_embedder, scope_root=repo)
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=None)
    result = pipe.retrieve("FilePathField", idx, top_k=1, scope_root=repo)
    assert result.used_symbol_augment is False
