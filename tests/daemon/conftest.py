"""Daemon-test fixtures.

Provides ``_FakeChunkStore`` + ``_FakeEmbedStore`` so the indexer tests
don't depend on the real SQLite/memmap impls (sibling agent's work).
The fakes implement just enough of the Protocol surface for the
indexer to drive them; they are NOT a reference implementation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
import pytest

from longctx_daemon.storage.protocol import ChunkStore, EmbedStore
from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
    ScopeFilter,
)


class _FakeChunkStore:
    """In-memory ``ChunkStore`` for tests.

    Matches the Protocol surface: dict-of-dicts keyed by project / file.
    No transactions, no thread safety — fine for the indexer's
    single-threaded tests.
    """

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        # file_id -> FileRecord
        self._files: dict[int, FileRecord] = {}
        self._next_file_id = 1
        # file_id -> list[Chunk]
        self._chunks_by_file: dict[int, list[Chunk]] = {}
        self._next_chunk_id = 1

    # ----- project lifecycle -------------------------------------------------

    def upsert_project(self, project: Project) -> None:
        self._projects[project.name] = project

    def get_project(self, name: str) -> Optional[Project]:
        return self._projects.get(name)

    def list_projects(self) -> tuple[Project, ...]:
        return tuple(self._projects.values())

    def delete_project(self, name: str) -> None:
        self._projects.pop(name, None)
        for fid, rec in list(self._files.items()):
            if rec.project == name:
                self._chunks_by_file.pop(fid, None)
                self._files.pop(fid, None)

    # ----- file lifecycle ----------------------------------------------------

    def upsert_file(self, file: FileRecord) -> int:
        # Lookup by (project, rel_path); preserve id on update.
        for fid, rec in self._files.items():
            if rec.project == file.project and rec.rel_path == file.rel_path:
                self._files[fid] = replace(file, id=fid)
                return fid
        fid = self._next_file_id
        self._next_file_id += 1
        self._files[fid] = replace(file, id=fid)
        self._chunks_by_file.setdefault(fid, [])
        return fid

    def get_file(self, project: str, rel_path: str) -> Optional[FileRecord]:
        for rec in self._files.values():
            if rec.project == project and rec.rel_path == rel_path:
                return rec
        return None

    def list_files(self, project: Optional[str] = None) -> tuple[FileRecord, ...]:
        if project is None:
            return tuple(self._files.values())
        return tuple(r for r in self._files.values() if r.project == project)

    def delete_file(self, file_id: int) -> None:
        self._files.pop(file_id, None)
        self._chunks_by_file.pop(file_id, None)

    # ----- chunk crud --------------------------------------------------------

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            row = self._chunks_by_file.setdefault(c.file_id, [])
            replaced = False
            for i, existing in enumerate(row):
                if existing.chunk_index == c.chunk_index:
                    row[i] = replace(c, id=existing.id)
                    replaced = True
                    break
            if not replaced:
                row.append(replace(c, id=self._next_chunk_id))
                self._next_chunk_id += 1

    def get_chunks_by_file(self, file_id: int) -> tuple[Chunk, ...]:
        return tuple(self._chunks_by_file.get(file_id, []))

    def get_chunks_by_id(self, ids: Iterable[int]) -> tuple[Chunk, ...]:
        ids_set = set(ids)
        out: list[Chunk] = []
        for chunks in self._chunks_by_file.values():
            for c in chunks:
                if c.id in ids_set:
                    out.append(c)
        return tuple(out)

    def get_chunks_by_content_hash(
        self, file_id: int, content_hashes: Iterable[str]
    ) -> tuple[Chunk, ...]:
        hashes = set(content_hashes)
        return tuple(
            c for c in self._chunks_by_file.get(file_id, [])
            if c.content_hash in hashes
        )

    def delete_chunks_by_file(self, file_id: int) -> None:
        self._chunks_by_file[file_id] = []

    def chunk_count(self, scope: Optional[ScopeFilter] = None) -> int:
        if scope is None:
            return sum(len(v) for v in self._chunks_by_file.values())
        if scope.project is not None:
            return sum(
                len(self._chunks_by_file.get(fid, []))
                for fid, rec in self._files.items()
                if rec.project == scope.project
            )
        if scope.project_in is not None:
            return sum(
                len(self._chunks_by_file.get(fid, []))
                for fid, rec in self._files.items()
                if rec.project in scope.project_in
            )
        return sum(len(v) for v in self._chunks_by_file.values())

    # ----- lexical (BM25) — unused by the indexer tests ----------------------

    def search_lexical(
        self, query_terms: list[str], k: int,
        scope: Optional[ScopeFilter] = None,
    ) -> tuple[Hit, ...]:
        return ()

    def invalidate_lexical(self, file_id: Optional[int] = None) -> None:
        return None

    # ----- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        return None


class _FakeEmbedStore:
    """In-memory ``EmbedStore`` for tests.

    Stores rows in a list; ``free_row`` zeros the row + adds it to a
    freelist that ``append`` reuses. Mirrors the real memmap's
    expected behavior so the indexer's bookkeeping is exercised.
    """

    def __init__(self, dim: int = 4, model_name: str = "fake",
                 model_sha: str = "fake-sha") -> None:
        self._dim = dim
        self._rows: list[np.ndarray] = []
        self._free: list[int] = []
        self._model_name = model_name
        self._model_sha = model_sha
        self.frees: list[int] = []  # test-visible audit trail

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_sha256(self) -> str:
        return self._model_sha

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    def append(self, embedding: np.ndarray) -> int:
        emb = np.asarray(embedding, dtype=np.float32)
        if self._free:
            row = self._free.pop()
            self._rows[row] = emb
            return row
        self._rows.append(emb)
        return len(self._rows) - 1

    def append_batch(self, embeddings: np.ndarray) -> tuple[int, ...]:
        embs = np.asarray(embeddings, dtype=np.float32)
        return tuple(self.append(embs[i]) for i in range(embs.shape[0]))

    def get(self, row: int) -> np.ndarray:
        return self._rows[row]

    def get_batch(self, rows: list[int]) -> np.ndarray:
        return np.stack([self._rows[r] for r in rows])

    def search_dense(
        self, query_embedding: np.ndarray, k: int,
        chunk_ids_in_scope: Optional[tuple[int, ...]] = None,
    ) -> tuple[Hit, ...]:
        return ()

    def free_row(self, row: int) -> None:
        if row < 0 or row >= len(self._rows):
            return
        self._rows[row] = np.zeros(self._dim, dtype=np.float32)
        self._free.append(row)
        self.frees.append(row)

    def close(self) -> None:
        return None


class _FakeEmbedder:
    """Deterministic stand-in for ``SentenceTransformer``.

    Returns a (n, dim) array where each row is a hash-of-text vector
    (so different texts get different embeddings, identical texts
    collide). L2-normalized to match the real embedder's contract.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        # Allow tests to swap the SHA mid-flight.
        self.fake_sha = "fakeembedsha-aaaaaaaaaaaa"
        self.model_name = "fake-embedder/v1"
        self.encode_calls: list[list[str]] = []
        self.max_seq_length: Optional[int] = None

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Track call shape for batch-size assertions in tests.
        self.encode_calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            if normalize_embeddings:
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n
            out[i] = v
        return out


# ----- fixtures -------------------------------------------------------------

@pytest.fixture
def fake_chunk_store() -> _FakeChunkStore:
    return _FakeChunkStore()


@pytest.fixture
def fake_embed_store() -> _FakeEmbedStore:
    return _FakeEmbedStore(dim=4)


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder(dim=4)
