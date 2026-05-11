"""ChunkStore + EmbedStore protocols — the contract every storage
backend implements. Phase 2.0 ships SqliteChunkStore + MemmapEmbedStore
behind these; Phase 3+ can drop in LanceDB or Qdrant without touching
the indexer/searcher.
"""
from __future__ import annotations

from typing import Iterable, Optional, Protocol

import numpy as np

from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
    ScopeFilter,
)


class ChunkStore(Protocol):
    """Persistent store of projects, files, and chunks. Backends are
    responsible for atomic upsert/delete semantics — concurrent readers
    must always see a consistent snapshot.

    Methods are grouped:
      * project lifecycle
      * file lifecycle
      * chunk read/write (without embeddings — those are EmbedStore's job)
      * lexical search (BM25 lives close to chunk text)

    Dense search lives on EmbedStore so the embedding columns/files
    can swap independently of the chunk metadata.
    """

    # ------------------------------------------------------ project lifecycle

    def upsert_project(self, project: Project) -> None: ...

    def get_project(self, name: str) -> Optional[Project]: ...

    def list_projects(self) -> tuple[Project, ...]: ...

    def delete_project(self, name: str) -> None:
        """Cascades: deletes all files + chunks for the project."""
        ...

    # --------------------------------------------------------- file lifecycle

    def upsert_file(self, file: FileRecord) -> int:
        """Insert or update by (project, rel_path). Returns file_id."""
        ...

    def get_file(self, project: str, rel_path: str) -> Optional[FileRecord]: ...

    def get_file_by_id(self, file_id: int) -> Optional[FileRecord]:
        """Lookup by primary key. Used by the searcher to materialize
        citations from chunk hits (chunks know their file_id but
        project + rel_path live on the file row)."""
        ...

    def list_files(self, project: Optional[str] = None) -> tuple[FileRecord, ...]: ...

    def delete_file(self, file_id: int) -> None:
        """Cascades to chunks; frees their embedding rows via EmbedStore."""
        ...

    def rename_file(
        self, file_id: int, new_rel_path: str, *, mtime: int,
    ) -> None:
        """Cheap rename: update files.rel_path + files.mtime in-place.
        Chunks + embeddings preserved (text didn't change). Used by
        the watcher's RENAME path so a `git mv` of 50 files doesn't
        re-embed any of them."""
        ...

    # ----------------------------------------------------------- chunk crud

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Atomic batch insert/update. Implementation must be transactional
        so concurrent reads never see partial state."""
        ...

    def get_chunks_by_file(self, file_id: int) -> tuple[Chunk, ...]: ...

    def get_chunks_by_id(self, ids: Iterable[int]) -> tuple[Chunk, ...]:
        """Used by the searcher to materialize hits → text for response."""
        ...

    def get_chunks_by_content_hash(
        self, file_id: int, content_hashes: Iterable[str]
    ) -> tuple[Chunk, ...]:
        """For incremental update: find which existing chunks of a file
        match the new chunks by content. Lets the indexer reuse cached
        embeddings when only some chunks changed."""
        ...

    def delete_chunks_by_file(self, file_id: int) -> None:
        """Drop all chunks for a file_id without touching the file
        record itself. Used by the indexer's incremental-update path
        before re-upserting the new chunk set so orphaned chunks (from
        a shrinking file) don't linger.

        Embedding rows are NOT freed by this call — the indexer freed
        them already in the diff phase to keep the EmbedStore /
        ChunkStore split clean.
        """
        ...

    def chunk_count(self, scope: Optional[ScopeFilter] = None) -> int: ...

    def list_chunk_ids_in_scope(
        self, scope: Optional[ScopeFilter] = None,
    ) -> tuple[int, ...]:
        """All chunk IDs that match the scope filter (or every chunk if
        ``scope`` is None). Used by the searcher to bound dense + lexical
        searches to a project subset without loading every chunk row.
        Sorted by id for stability.
        """
        ...

    # ------------------------------------------------- lexical (BM25) search

    def search_lexical(
        self, query_terms: list[str], k: int, scope: Optional[ScopeFilter] = None
    ) -> tuple[Hit, ...]:
        """Top-K BM25 hits within the scope filter. Implementations may
        keep an in-memory BM25 index and rebuild lazily; ``invalidate_lexical``
        lets callers signal that the underlying chunks changed."""
        ...

    def invalidate_lexical(self, file_id: Optional[int] = None) -> None: ...

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None: ...


class EmbedStore(Protocol):
    """Persistent dense-embedding store. Backed by a numpy memmap in
    Phase 2.0. Each row corresponds to one Chunk; the chunk row's
    ``embedding_row`` field indexes into this store.

    Embeddings are L2-normalized at write time so dot-product == cosine.
    """

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    @property
    def num_rows(self) -> int:
        """Total appended rows. Includes freed rows (which are zeroed but
        still occupy space until compaction)."""
        ...

    def append(self, embedding: np.ndarray) -> int:
        """Append one row. Returns its index. Embedding must be (dim,)
        and L2-normalized. Storage layer asserts shape + norm in debug
        mode."""
        ...

    def append_batch(self, embeddings: np.ndarray) -> tuple[int, ...]:
        """Append N rows, return their indices. Embeddings shape (N, dim)."""
        ...

    def get(self, row: int) -> np.ndarray:
        """Read one row. Returns a view if possible (no copy)."""
        ...

    def get_batch(self, rows: list[int]) -> np.ndarray:
        """Gather multiple rows. Returns (len(rows), dim)."""
        ...

    def search_dense(
        self,
        query_embedding: np.ndarray,
        k: int,
        chunk_ids_in_scope: Optional[tuple[int, ...]] = None,
    ) -> tuple[Hit, ...]:
        """Top-K cosine hits. ``chunk_ids_in_scope`` filters to a known
        candidate set (for project-scoped search); None = global search.
        Implementations free to use brute-force or an ANN index — Phase 2.0
        ships brute-force (sufficient for ~50K chunks per project).
        """
        ...

    def free_row(self, row: int) -> None:
        """Mark a row as free (zero out + add to freelist). Reused on
        next append. Avoids unbounded memmap growth on heavy update loads."""
        ...

    def close(self) -> None: ...
