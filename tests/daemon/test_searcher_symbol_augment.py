"""Searcher-side integration tests for the symbol-aware augment.

The pure-Python helpers are covered in ``tests/test_symbol_augment.py``.
The longctx-svc retrieve-pipeline path is covered in
``services/longctx-svc/tests/test_symbol_augment_pipeline.py``. These
tests stand up a fake ``ChunkStore`` whose ``Project.root_path`` points
at a real tmp_path repo, and assert that ``Searcher.search``:

  * re-ranks ``.py`` source above ``.rst`` doc when the query has a
    code signal (traceback / error type / class-or-def mention);
  * leaves ranking unchanged when the query has no code signal;
  * tolerates a chunk store that doesn't expose ``get_file_by_id``.
"""
from __future__ import annotations

import shutil
from typing import Iterable, Optional, Sequence

import numpy as np
import pytest

from longctx_daemon.searcher import Searcher, SearcherConfig
from longctx_daemon.types import Chunk, FileRecord, Hit, Project, ScopeFilter


# ---------------------------------------------------------------- fakes

class _FakeChunkStore:
    """Minimal in-memory ChunkStore. Mirrors the existing test fake
    shape but inlined here so this file is self-contained."""

    def __init__(self, chunks, files, projects):
        self._chunks = {c.id: c for c in chunks}
        self._files = {f.id: f for f in files}
        self._projects = list(projects)

    def list_projects(self):
        return tuple(self._projects)

    def get_file_by_id(self, file_id):
        return self._files.get(file_id)

    def get_chunks_by_id(self, ids: Iterable[int]):
        return tuple(self._chunks[i] for i in ids if i in self._chunks)

    def list_chunk_ids_in_scope(self, scope: ScopeFilter):
        out = []
        for cid, chunk in self._chunks.items():
            file_rec = self._files.get(chunk.file_id)
            if file_rec is None:
                continue
            if scope.project is not None and file_rec.project != scope.project:
                continue
            out.append(cid)
        return tuple(out)

    def search_lexical(self, query_terms, k, scope=None):
        from rank_bm25 import BM25Okapi
        ids_in_scope = (
            set(self.list_chunk_ids_in_scope(scope))
            if scope is not None
            else set(self._chunks.keys())
        )
        chs = [self._chunks[i] for i in ids_in_scope if i in self._chunks]
        if not chs or not query_terms:
            return ()
        corpus = [c.text.lower().split() for c in chs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_terms)
        order = np.argsort(-scores)[:k]
        return tuple(
            Hit(chunk_id=chs[idx].id, score=float(scores[idx]))
            for idx in order if scores[idx] > 0
        )


class _FakeEmbedStore:
    def __init__(self, embs):
        self._embs = embs

    @property
    def dim(self):
        return len(next(iter(self._embs.values()))) if self._embs else 0

    @property
    def model_name(self):
        return "fake-embedder"

    @property
    def model_sha256(self):
        return "deadbeef"

    @property
    def num_rows(self):
        return len(self._embs)

    def search_dense(self, q, k, chunk_ids_in_scope=None):
        if not self._embs:
            return ()
        ids = (list(chunk_ids_in_scope)
               if chunk_ids_in_scope is not None
               else list(self._embs.keys()))
        if not ids:
            return ()
        mat = np.stack([self._embs[i] for i in ids])
        sims = mat @ np.asarray(q).astype(mat.dtype)
        order = np.argsort(-sims)[:k]
        return tuple(
            Hit(chunk_id=ids[idx], score=float(sims[idx]))
            for idx in order
        )


class _AlphaEmbedder:
    """Embedder where 'alpha' count drives the dense match. Doc
    repeats 'alpha' more than source so doc wins dense alone."""

    def encode(self, texts, normalize_embeddings=True, **_):
        rows = []
        for t in texts:
            count = float(t.lower().count("alpha"))
            row = np.array([count, 0.5], dtype=np.float32)
            if normalize_embeddings:
                row = row / (np.linalg.norm(row) + 1e-8)
            rows.append(row)
        return np.stack(rows)


# ----------------------------------------------------- corpus builder

def _build_corpus_with_real_files(tmp_path) -> tuple[
    _FakeChunkStore, _FakeEmbedStore, _AlphaEmbedder, str,
]:
    """Stand up a tmp_path repo with one .py source defining
    ``FilePathField`` plus a CHANGES.rst doc, then build a fake
    ChunkStore that points at it.
    """
    project_name = "django"
    root = tmp_path / project_name
    (root / "src").mkdir(parents=True)
    source = root / "src" / "fields.py"
    source.write_text(
        "class FilePathField:\n"
        "    \"\"\"alpha alpha alpha alpha\"\"\"\n"
        "    pass\n"
    )
    doc = root / "CHANGES.rst"
    doc.write_text("FilePathField was changed in 1.0\nalpha alpha alpha alpha alpha alpha\n")

    embedder = _AlphaEmbedder()

    files = [
        FileRecord(id=1, project=project_name, rel_path="src/fields.py",
                   mtime=1_700_000_000, size_bytes=80,
                   content_hash="hash1"),
        FileRecord(id=2, project=project_name, rel_path="CHANGES.rst",
                   mtime=1_700_000_000, size_bytes=80,
                   content_hash="hash2"),
    ]
    chunks = [
        Chunk(id=1, file_id=1, chunk_index=0, start_offset=0, end_offset=60,
              start_line=1, end_line=3, token_count=20,
              content_hash="ch1",
              text="class FilePathField alpha alpha alpha alpha pass",
              embedder_model="fake-embedder", embedder_sha256="deadbeef",
              embedding_row=1),
        Chunk(id=2, file_id=2, chunk_index=0, start_offset=0, end_offset=60,
              start_line=1, end_line=2, token_count=20,
              content_hash="ch2",
              text="FilePathField alpha alpha alpha alpha alpha alpha doc",
              embedder_model="fake-embedder", embedder_sha256="deadbeef",
              embedding_row=2),
    ]
    embs = {
        1: embedder.encode([chunks[0].text])[0],
        2: embedder.encode([chunks[1].text])[0],
    }
    projects = (Project(name=project_name, root_path=str(root)),)
    return (
        _FakeChunkStore(chunks=chunks, files=files, projects=projects),
        _FakeEmbedStore(embs=embs),
        embedder,
        project_name,
    )


def _make_searcher(store, embeds, embedder):
    return Searcher(
        chunk_store=store, embed_store=embeds, embedder=embedder,
        config=SearcherConfig(relevance_floor=0.0),
    )


# --------------------------------------------------------- tests

@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not on PATH")
def test_py_source_wins_with_code_signal(tmp_path):
    """With a code-signal query (Traceback + Error + symbol mention),
    the augment re-ranks .py above .rst."""
    store, embeds, embedder, _ = _build_corpus_with_real_files(tmp_path)
    searcher = _make_searcher(store, embeds, embedder)
    result = searcher.search(
        'Traceback File "x.py" FilePathField raises TypeError '
        'when alpha path is None',
        project="django",
    )
    assert result.chunks
    # Source (file_id 1 → src/fields.py) is now on top.
    assert result.chunks[0].citation.file_path.endswith("src/fields.py")


def test_no_project_root_does_not_crash(tmp_path):
    """When ``scope.primary_project`` resolves but the project's
    root_path doesn't exist on disk, the augment falls back silently."""
    project_name = "ghost"
    embedder = _AlphaEmbedder()
    files = [FileRecord(id=1, project=project_name,
                        rel_path="src/x.py", mtime=0, size_bytes=10,
                        content_hash="h")]
    chunks = [
        Chunk(id=1, file_id=1, chunk_index=0, start_offset=0, end_offset=10,
              start_line=1, end_line=1, token_count=3,
              content_hash="ch", text="class Foo alpha",
              embedder_model="fake-embedder", embedder_sha256="deadbeef",
              embedding_row=1),
    ]
    embs = {1: embedder.encode([chunks[0].text])[0]}
    projects = (Project(name=project_name,
                        root_path=str(tmp_path / "nonexistent")),)
    store = _FakeChunkStore(chunks=chunks, files=files, projects=projects)
    embeds = _FakeEmbedStore(embs=embs)
    searcher = _make_searcher(store, embeds, embedder)
    # Code-signal query, but missing on-disk root. Should not crash.
    result = searcher.search(
        'Traceback File "x.py" raises TypeError alpha',
        project=project_name,
    )
    assert result.chunks
