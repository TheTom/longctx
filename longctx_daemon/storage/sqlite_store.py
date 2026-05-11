"""SQLite-backed ``ChunkStore`` implementation for Phase 2.0.

Persists projects, files, and chunks under a single SQLite database with
WAL journaling and ``synchronous=NORMAL`` for low-latency commits while
keeping concurrent-read consistency. The BM25 lexical index is kept
in-memory (``rank_bm25.BM25Okapi``) and serialized as a sibling pickle
so a restart doesn't pay the rebuild cost; the pickle is validated
against the live ``chunks`` table on load and rebuilt on mismatch.

What's NOT here yet:
  * No vacuuming / GC of orphaned ``embedding_row`` indices — that's the
    embedder/indexer's job (it owns the memmap, see ``MemmapEmbedStore``).
  * No FTS5 path — the spec asks for ``rank_bm25`` BM25Okapi for now;
    swapping to FTS5 in Phase 3 is mechanical.
  * No multi-process write locking beyond what SQLite WAL gives us;
    Phase 2.0 ships single-writer.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
    ScopeFilter,
)

# ---------------------------------------------------------------- constants

#: Bumped whenever the on-disk schema changes incompatibly. Migrations
#: live in ``_MIGRATIONS`` below; a fresh DB applies all of them in
#: ascending order. An older daemon hitting a future schema raises
#: ``UnsupportedSchemaError`` so we never read bad pages.
SCHEMA_VERSION = 1

#: Minimum SQLite library version required for WAL semantics + ``ON DELETE
#: CASCADE`` foreign keys. 3.37 ships ``STRICT`` tables and the new
#: ``WITHOUT ROWID`` improvements but more importantly is the floor the
#: PRD §12.3 calls out.
MIN_SQLITE_VERSION = (3, 37, 0)

#: File mode for the SQLite DB and its sidecars. 0600 = owner-only RW.
INDEX_FILE_MODE = 0o600

#: Used to detect that a persisted BM25 pickle still describes the
#: current chunk corpus. Mismatch → rebuild from scratch.
@dataclass(frozen=True)
class _BM25Snapshot:
    """Validation header for the persisted BM25 pickle.

    ``corpus_digest`` is sha256 over the concatenated ``content_hash`` of
    every chunk in id-order; cheaper than rehashing chunk text and
    sufficient because content_hash is already stable identity.
    """
    chunk_count: int
    corpus_digest: str
    chunk_ids: tuple[int, ...]   # parallel to BM25 corpus rows


_LOG = logging.getLogger(__name__)


_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE projects (
            name        TEXT PRIMARY KEY,
            root_path   TEXT NOT NULL UNIQUE,
            last_full_scan_at  INTEGER NOT NULL,
            session_id  TEXT
        );

        CREATE TABLE files (
            id          INTEGER PRIMARY KEY,
            project     TEXT NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
            rel_path    TEXT NOT NULL,
            mtime       INTEGER NOT NULL,
            size_bytes  INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            UNIQUE(project, rel_path)
        );
        CREATE INDEX idx_files_project ON files(project);

        CREATE TABLE chunks (
            id              INTEGER PRIMARY KEY,
            file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_index     INTEGER NOT NULL,
            start_offset    INTEGER NOT NULL,
            end_offset      INTEGER NOT NULL,
            start_line      INTEGER NOT NULL,
            end_line        INTEGER NOT NULL,
            token_count     INTEGER NOT NULL,
            content_hash    TEXT NOT NULL,
            text            TEXT NOT NULL,
            embedder_model  TEXT NOT NULL,
            embedder_sha256 TEXT NOT NULL,
            embedding_row   INTEGER
        );
        CREATE INDEX idx_chunks_file ON chunks(file_id);
        CREATE INDEX idx_chunks_hash ON chunks(content_hash);

        CREATE TABLE schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    ),
)


class UnsupportedSchemaError(RuntimeError):
    """Raised when the on-disk DB advertises a schema version newer
    than this binary supports. Refusing to read is safer than
    speculatively migrating data we don't understand."""


class UnsupportedSqliteError(RuntimeError):
    """Raised when the linked SQLite library is older than
    :data:`MIN_SQLITE_VERSION`. Aborting at startup beats discovering
    silently-broken WAL semantics six hours into an embed pass."""


# ------------------------------------------------------------------ helpers


def _word_tokenize(text: str) -> list[str]:
    """Same lexical tokenizer as the Phase 1 coarse filter — keeps
    BM25 scores comparable across the daemon and the in-process
    pipeline."""
    import re

    return re.findall(r"\w+", text.lower())


def _digest_chunks(chunk_ids: list[int], content_hashes: list[str]) -> str:
    """Stable digest used to validate that a persisted BM25 pickle
    still corresponds to the SQLite corpus."""
    h = hashlib.sha256()
    for cid, ch in zip(chunk_ids, content_hashes):
        h.update(str(cid).encode())
        h.update(b"\0")
        h.update(ch.encode())
        h.update(b"\0")
    return h.hexdigest()


def _try_chmod(path: Path, mode: int) -> None:
    """``chmod`` that tolerates platforms / filesystems that don't
    support POSIX modes (Windows, FAT volumes). Best-effort security
    posture, never a hard failure."""
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        _LOG.debug("chmod(%s, %o) unsupported on this platform", path, mode)


# --------------------------------------------------------------- main class


class SqliteChunkStore:
    """Concrete ``ChunkStore`` backed by SQLite + an on-disk BM25 pickle.

    Thread-safe for concurrent reads; writes are serialized through an
    internal ``threading.RLock``. WAL gives us readers-don't-block-writers
    semantics inside SQLite, but the BM25 in-memory cache + pickle path
    is plain Python objects, hence the lock.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        check_same_thread: bool = False,
    ) -> None:
        """
        Args:
            db_path: location of the SQLite DB file. Created if missing.
                Sidecar BM25 pickle is stored at ``<db_path>.bm25.pkl``.
            check_same_thread: passed through to ``sqlite3.connect``.
                Defaults to ``False`` so the daemon can share the
                connection across worker threads. The internal RLock
                provides our own serialization.
        """
        self._verify_sqlite_version()

        self._db_path = Path(db_path)
        self._bm25_path = Path(str(self._db_path) + ".bm25.pkl")
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        new_db = not self._db_path.exists()

        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=check_same_thread,
            isolation_level=None,   # autocommit; we use explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")

        if new_db:
            _try_chmod(self._db_path, INDEX_FILE_MODE)

        self._init_schema()

        # In-memory BM25 state. Rebuilt lazily on first ``search_lexical``
        # if invalidated; persisted to disk on ``close()``.
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunk_ids: list[int] = []
        self._bm25_dirty = True
        self._load_bm25_if_valid()

    # -------------------------------------------------------- init helpers

    def _verify_sqlite_version(self) -> None:
        """Refuse to open a DB on too-old SQLite. WAL semantics +
        ``ON DELETE CASCADE`` foreign keys are part of the on-disk
        contract; older libs silently drop them."""
        cur = sqlite3.sqlite_version_info
        if cur < MIN_SQLITE_VERSION:
            raise UnsupportedSqliteError(
                f"sqlite {sqlite3.sqlite_version} too old; "
                f"need >= {'.'.join(str(p) for p in MIN_SQLITE_VERSION)}"
            )

    def _init_schema(self) -> None:
        """Apply pending migrations + verify the resulting schema is
        compatible with this binary. Single-writer, called from
        ``__init__``."""
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                "  id INTEGER PRIMARY KEY CHECK (id = 1),"
                "  version INTEGER NOT NULL"
                ")"
            )
            row = self._conn.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()
            current = int(row["version"]) if row else 0

            if current > SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"DB schema v{current} newer than supported v{SCHEMA_VERSION}; "
                    "upgrade longctx-daemon."
                )

            for version, ddl in _MIGRATIONS:
                if version > current:
                    self._conn.executescript(ddl)
                    current = version

            self._conn.execute(
                "INSERT INTO schema_version(id, version) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
                (current,),
            )

    # ------------------------------------------------------ project lifecycle

    def upsert_project(self, project: Project) -> None:
        """Idempotent insert/update keyed on ``name``."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO projects(name, root_path, last_full_scan_at, session_id) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "  root_path = excluded.root_path, "
                "  last_full_scan_at = excluded.last_full_scan_at, "
                "  session_id = excluded.session_id",
                (project.name, project.root_path,
                 project.last_full_scan_at, project.session_id),
            )

    def get_project(self, name: str) -> Optional[Project]:
        with self._lock:
            row = self._conn.execute(
                "SELECT name, root_path, last_full_scan_at, session_id "
                "FROM projects WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return Project(
            name=row["name"], root_path=row["root_path"],
            last_full_scan_at=row["last_full_scan_at"],
            session_id=row["session_id"],
        )

    def list_projects(self) -> tuple[Project, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, root_path, last_full_scan_at, session_id "
                "FROM projects ORDER BY name"
            ).fetchall()
        return tuple(
            Project(
                name=r["name"], root_path=r["root_path"],
                last_full_scan_at=r["last_full_scan_at"],
                session_id=r["session_id"],
            )
            for r in rows
        )

    def delete_project(self, name: str) -> None:
        """Cascades to files → chunks via FK ``ON DELETE CASCADE``.
        Lexical index is invalidated for the next search call."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        self._bm25_dirty = True

    # ---------------------------------------------------------- file lifecycle

    def upsert_file(self, file: FileRecord) -> int:
        """Insert or update by ``(project, rel_path)``. Returns the
        post-write ``file_id``. The ``id`` field on the input is honored
        when nonzero so callers can pre-allocate ids; pass ``0`` to let
        SQLite assign one."""
        with self._lock, self._conn:
            if file.id:
                self._conn.execute(
                    "INSERT INTO files(id, project, rel_path, mtime, size_bytes, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(project, rel_path) DO UPDATE SET "
                    "  mtime = excluded.mtime, "
                    "  size_bytes = excluded.size_bytes, "
                    "  content_hash = excluded.content_hash",
                    (file.id, file.project, file.rel_path, file.mtime,
                     file.size_bytes, file.content_hash),
                )
            else:
                self._conn.execute(
                    "INSERT INTO files(project, rel_path, mtime, size_bytes, content_hash) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(project, rel_path) DO UPDATE SET "
                    "  mtime = excluded.mtime, "
                    "  size_bytes = excluded.size_bytes, "
                    "  content_hash = excluded.content_hash",
                    (file.project, file.rel_path, file.mtime,
                     file.size_bytes, file.content_hash),
                )
            row = self._conn.execute(
                "SELECT id FROM files WHERE project = ? AND rel_path = ?",
                (file.project, file.rel_path),
            ).fetchone()
        return int(row["id"])

    def get_file(self, project: str, rel_path: str) -> Optional[FileRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, project, rel_path, mtime, size_bytes, content_hash "
                "FROM files WHERE project = ? AND rel_path = ?",
                (project, rel_path),
            ).fetchone()
        if row is None:
            return None
        return FileRecord(
            id=row["id"], project=row["project"], rel_path=row["rel_path"],
            mtime=row["mtime"], size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
        )

    def list_files(
        self, project: Optional[str] = None
    ) -> tuple[FileRecord, ...]:
        with self._lock:
            if project is None:
                rows = self._conn.execute(
                    "SELECT id, project, rel_path, mtime, size_bytes, content_hash "
                    "FROM files ORDER BY project, rel_path"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, project, rel_path, mtime, size_bytes, content_hash "
                    "FROM files WHERE project = ? ORDER BY rel_path",
                    (project,),
                ).fetchall()
        return tuple(
            FileRecord(
                id=r["id"], project=r["project"], rel_path=r["rel_path"],
                mtime=r["mtime"], size_bytes=r["size_bytes"],
                content_hash=r["content_hash"],
            )
            for r in rows
        )

    def delete_file(self, file_id: int) -> None:
        """Single transaction — chunk rows cascade. Caller is responsible
        for freeing memmap rows via ``EmbedStore.free_row``; this store
        only knows about chunk metadata."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self._bm25_dirty = True

    def rename_file(
        self, file_id: int, new_rel_path: str, *, mtime: int,
    ) -> None:
        """Cheap rename: chunks + embeddings preserved, only files
        row mutates. The watcher's RENAME path uses this so bulk
        ``git mv`` operations don't trigger re-embedding."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE files SET rel_path = ?, mtime = ? WHERE id = ?",
                (new_rel_path, mtime, file_id),
            )

    def get_file_by_id(self, file_id: int) -> Optional[FileRecord]:
        """Lookup by primary key. Used by the searcher when materializing
        citations from chunk hits — chunks know their file_id but the
        full project + rel_path needs the files row."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,),
            ).fetchone()
        if row is None:
            return None
        return FileRecord(
            id=row["id"], project=row["project"], rel_path=row["rel_path"],
            mtime=row["mtime"], size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
        )

    # ------------------------------------------------------------- chunk crud

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Atomic batch insert/update. Concurrent searches see either
        the pre-batch state or the post-batch state, never partial."""
        chunks = list(chunks)
        if not chunks:
            return

        with self._lock, self._conn:
            params = [
                (
                    c.id or None,   # SQLite assigns when None
                    c.file_id,
                    c.chunk_index,
                    c.start_offset,
                    c.end_offset,
                    c.start_line,
                    c.end_line,
                    c.token_count,
                    c.content_hash,
                    c.text,
                    c.embedder_model,
                    c.embedder_sha256,
                    c.embedding_row,
                )
                for c in chunks
            ]
            self._conn.executemany(
                "INSERT INTO chunks("
                "  id, file_id, chunk_index, start_offset, end_offset,"
                "  start_line, end_line, token_count, content_hash,"
                "  text, embedder_model, embedder_sha256, embedding_row"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  file_id = excluded.file_id, "
                "  chunk_index = excluded.chunk_index, "
                "  start_offset = excluded.start_offset, "
                "  end_offset = excluded.end_offset, "
                "  start_line = excluded.start_line, "
                "  end_line = excluded.end_line, "
                "  token_count = excluded.token_count, "
                "  content_hash = excluded.content_hash, "
                "  text = excluded.text, "
                "  embedder_model = excluded.embedder_model, "
                "  embedder_sha256 = excluded.embedder_sha256, "
                "  embedding_row = excluded.embedding_row",
                params,
            )
        self._bm25_dirty = True

    def get_chunks_by_file(self, file_id: int) -> tuple[Chunk, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE file_id = ? ORDER BY chunk_index",
                (file_id,),
            ).fetchall()
        return tuple(self._row_to_chunk(r) for r in rows)

    def get_chunks_by_id(self, ids: Iterable[int]) -> tuple[Chunk, ...]:
        ids = list(ids)
        if not ids:
            return ()
        # SQLite's parameter limit is high but finite; chunk-batch via IN clause.
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        # Preserve caller-requested ordering (handy for ranked-hit fetch).
        by_id = {r["id"]: r for r in rows}
        return tuple(
            self._row_to_chunk(by_id[i]) for i in ids if i in by_id
        )

    def get_chunks_by_content_hash(
        self, file_id: int, content_hashes: Iterable[str]
    ) -> tuple[Chunk, ...]:
        """Used by the indexer's incremental update path (PRD §4.5):
        when re-chunking a modified file, find which old chunks still
        match new chunks by ``content_hash`` so their cached embeddings
        can be reused instead of re-embedded."""
        hashes = list(content_hashes)
        if not hashes:
            return ()
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM chunks WHERE file_id = ? "
                f"AND content_hash IN ({placeholders})",
                [file_id, *hashes],
            ).fetchall()
        return tuple(self._row_to_chunk(r) for r in rows)

    def delete_chunks_by_file(self, file_id: int) -> None:
        """Drop all chunks for a file_id without touching the file
        record itself. Used by the indexer's incremental-update path
        before re-upserting the new chunk set so orphaned chunks (from
        a shrinking file) don't linger.

        Embedding rows are NOT freed here — the indexer freed them
        already in the diff phase to keep the EmbedStore / ChunkStore
        ownership split clean.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM chunks WHERE file_id = ?", (file_id,),
            )
        self._bm25_dirty = True

    def list_chunk_ids_in_scope(
        self, scope: Optional[ScopeFilter] = None,
    ) -> tuple[int, ...]:
        """Public scope → chunk_id tuple lookup. The searcher uses this
        to bound dense + lexical searches to a project subset without
        loading every chunk row."""
        with self._lock:
            ids = self._scope_chunk_ids(scope)
        # _scope_chunk_ids returns frozenset | None; map None → "all"
        # to a tuple of every id (matches the explicit "no filter"
        # caller expectation).
        if ids is None:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id FROM chunks ORDER BY id"
                ).fetchall()
            return tuple(int(r["id"]) for r in rows)
        return tuple(sorted(ids))

    def chunk_count(self, scope: Optional[ScopeFilter] = None) -> int:
        with self._lock:
            if scope is None or (scope.project is None and scope.project_in is None):
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks"
                ).fetchone()
            elif scope.project is not None:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks c "
                    "JOIN files f ON f.id = c.file_id "
                    "WHERE f.project = ?",
                    (scope.project,),
                ).fetchone()
            else:
                projects = scope.project_in or ()
                if not projects:
                    return 0
                placeholders = ",".join("?" * len(projects))
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks c "
                    "JOIN files f ON f.id = c.file_id "
                    f"WHERE f.project IN ({placeholders})",
                    list(projects),
                ).fetchone()
        return int(row["n"])

    # ------------------------------------------------- lexical (BM25) search

    def search_lexical(
        self,
        query_terms: list[str],
        k: int,
        scope: Optional[ScopeFilter] = None,
    ) -> tuple[Hit, ...]:
        """Top-K BM25 hits within ``scope``. Builds the in-memory
        BM25 corpus on first call after invalidation; subsequent calls
        reuse it. Empty query / empty corpus → empty result."""
        if k <= 0:
            return ()
        if not query_terms:
            return ()

        with self._lock:
            self._ensure_bm25()
            if self._bm25 is None or not self._bm25_chunk_ids:
                return ()

            scope_ids = self._scope_chunk_ids(scope)
            scores = self._bm25.get_scores(query_terms)
            n = len(self._bm25_chunk_ids)

            # Mask scores to scope. None = no scope filter, all in.
            if scope_ids is not None:
                scope_set = scope_ids
                mask = np.array(
                    [cid in scope_set for cid in self._bm25_chunk_ids],
                    dtype=bool,
                )
                masked = np.where(mask, scores, -np.inf)
            else:
                masked = scores

            # argsort ascending → take the tail for top-K.
            k_eff = min(k, n)
            top_idx = np.argpartition(-masked, kth=min(k_eff - 1, n - 1))[:k_eff]
            # Re-sort the top slice for deterministic ranked output.
            top_idx = top_idx[np.argsort(-masked[top_idx])]
            hits: list[Hit] = []
            for i in top_idx:
                score = float(masked[i])
                if not np.isfinite(score):
                    continue
                hits.append(Hit(chunk_id=int(self._bm25_chunk_ids[i]), score=score))
            return tuple(hits)

    def invalidate_lexical(self, file_id: Optional[int] = None) -> None:
        """Marks BM25 dirty so the next ``search_lexical`` rebuilds.
        ``file_id`` is currently unused — full rebuild is fast enough
        (~2-10s on 13M tokens) and the spec allows it. Granular per-file
        invalidation can land in Phase 3."""
        del file_id
        with self._lock:
            self._bm25_dirty = True

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Flush BM25 to disk + close the SQLite connection. Idempotent
        — calling twice is a no-op so cleanup paths can be defensive."""
        with self._lock:
            if self._conn is None:
                return
            try:
                if self._bm25 is not None and not self._bm25_dirty:
                    self._save_bm25()
            finally:
                try:
                    self._conn.execute("PRAGMA optimize")
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None    # type: ignore[assignment]

    # --------------------------------------------------- BM25 internal plumbing

    def _ensure_bm25(self) -> None:
        """Idempotent: only rebuilds when the dirty flag is set or the
        in-memory state is empty."""
        if not self._bm25_dirty and self._bm25 is not None:
            return
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Walks all chunks and builds a fresh ``BM25Okapi``.
        O(N) — fine at ~50K chunks. Persists the rebuilt index so the
        next process avoids the cost."""
        rows = self._conn.execute(
            "SELECT id, text FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            self._bm25 = None
            self._bm25_chunk_ids = []
            self._bm25_dirty = False
            return

        chunk_ids = [int(r["id"]) for r in rows]
        corpus = [_word_tokenize(r["text"]) for r in rows]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_chunk_ids = chunk_ids
        self._bm25_dirty = False
        try:
            self._save_bm25()
        except OSError as exc:
            _LOG.warning("BM25 persist failed (%s); will retry on close", exc)

    def _save_bm25(self) -> None:
        """Pickle the live BM25 + a validation header. Atomic via
        write-then-rename."""
        if self._bm25 is None:
            return
        ids = list(self._bm25_chunk_ids)
        rows = self._conn.execute(
            "SELECT content_hash FROM chunks WHERE id IN ("
            + ",".join("?" * len(ids))
            + ") ORDER BY id"
            if ids else
            "SELECT content_hash FROM chunks WHERE 0",
            ids,
        ).fetchall()
        digest = _digest_chunks(ids, [r["content_hash"] for r in rows])
        snap = _BM25Snapshot(
            chunk_count=len(ids), corpus_digest=digest,
            chunk_ids=tuple(ids),
        )
        tmp = self._bm25_path.with_suffix(self._bm25_path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"snapshot": snap, "bm25": self._bm25}, fh)
        os.replace(tmp, self._bm25_path)
        _try_chmod(self._bm25_path, INDEX_FILE_MODE)

    def _load_bm25_if_valid(self) -> None:
        """Try to load the persisted BM25; silently fall through to a
        rebuild on any mismatch. Defensive against pickle-version
        skew, partial writes, or chunks-changed-while-daemon-was-down."""
        if not self._bm25_path.exists():
            self._bm25_dirty = True
            return
        try:
            with open(self._bm25_path, "rb") as fh:
                blob = pickle.load(fh)
            snap: _BM25Snapshot = blob["snapshot"]
            bm25 = blob["bm25"]
        except (pickle.PickleError, OSError, KeyError, EOFError) as exc:
            _LOG.info("BM25 pickle unreadable (%s); will rebuild", exc)
            self._bm25_dirty = True
            return

        # Validate against current chunks table.
        rows = self._conn.execute(
            "SELECT id, content_hash FROM chunks ORDER BY id"
        ).fetchall()
        live_ids = [int(r["id"]) for r in rows]
        live_hashes = [r["content_hash"] for r in rows]
        live_digest = _digest_chunks(live_ids, live_hashes)

        if (snap.chunk_count != len(live_ids)
                or snap.corpus_digest != live_digest
                or list(snap.chunk_ids) != live_ids):
            _LOG.info(
                "BM25 pickle stale (count %d vs %d); will rebuild",
                snap.chunk_count, len(live_ids),
            )
            self._bm25_dirty = True
            return

        self._bm25 = bm25
        self._bm25_chunk_ids = list(snap.chunk_ids)
        self._bm25_dirty = False

    # ------------------------------------------------ row + scope plumbing

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            file_id=row["file_id"],
            chunk_index=row["chunk_index"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            token_count=row["token_count"],
            content_hash=row["content_hash"],
            text=row["text"],
            embedder_model=row["embedder_model"],
            embedder_sha256=row["embedder_sha256"],
            embedding_row=row["embedding_row"],
        )

    def _scope_chunk_ids(
        self, scope: Optional[ScopeFilter]
    ) -> Optional[set[int]]:
        """Resolve a scope to the set of chunk ids it includes.
        Returns ``None`` for unrestricted scope (so callers can skip
        the masking step entirely)."""
        if scope is None:
            return None
        if scope.project is None and not scope.project_in:
            return None
        projects: list[str]
        if scope.project_in is not None:
            projects = list(scope.project_in)
            if scope.project is not None and scope.project not in projects:
                projects.append(scope.project)
        else:
            projects = [scope.project]   # type: ignore[list-item]
        if not projects:
            return set()
        placeholders = ",".join("?" * len(projects))
        rows = self._conn.execute(
            "SELECT c.id AS id FROM chunks c "
            "JOIN files f ON f.id = c.file_id "
            f"WHERE f.project IN ({placeholders})",
            projects,
        ).fetchall()
        return {int(r["id"]) for r in rows}
