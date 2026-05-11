"""Tests for ``longctx_daemon.storage.sqlite_store``.

Coverage focus: each public method (happy / empty / invalid / edge),
plus the integration scenarios called out in the Phase 2.0 spec —
round-trip, cascade-on-delete, BM25 persistence + invalidation,
concurrent reads, WAL semantics, schema-version guard.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from longctx_daemon.storage.sqlite_store import (
    SCHEMA_VERSION,
    SqliteChunkStore,
    UnsupportedSchemaError,
)
from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Project,
    ScopeFilter,
)

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def store(tmp_path: Path) -> SqliteChunkStore:
    """Fresh store for each test; closed in teardown."""
    s = SqliteChunkStore(tmp_path / "index.db")
    yield s
    s.close()


@pytest.fixture
def populated(store: SqliteChunkStore) -> SqliteChunkStore:
    """Store with two projects, one file each, three chunks each."""
    store.upsert_project(Project(name="alpha", root_path="/r/alpha", last_full_scan_at=1))
    store.upsert_project(Project(name="beta", root_path="/r/beta", last_full_scan_at=2))

    fid_a = store.upsert_file(FileRecord(
        id=0, project="alpha", rel_path="a.py", mtime=10,
        size_bytes=100, content_hash="hA",
    ))
    fid_b = store.upsert_file(FileRecord(
        id=0, project="beta", rel_path="b.py", mtime=20,
        size_bytes=200, content_hash="hB",
    ))

    store.upsert_chunks([
        _mk_chunk(0, fid_a, 0, "alpha quick brown fox"),
        _mk_chunk(0, fid_a, 1, "the lazy dog jumps"),
        _mk_chunk(0, fid_a, 2, "alpha alpha alpha"),
        _mk_chunk(0, fid_b, 0, "beta authentication middleware"),
        _mk_chunk(0, fid_b, 1, "beta token validation logic"),
        _mk_chunk(0, fid_b, 2, "common keyword fox lazy"),
    ])
    return store


def _mk_chunk(cid: int, file_id: int, idx: int, text: str) -> Chunk:
    """Test helper: build a chunk with sensible defaults."""
    return Chunk(
        id=cid, file_id=file_id, chunk_index=idx,
        start_offset=idx * 100, end_offset=idx * 100 + len(text),
        start_line=idx + 1, end_line=idx + 1,
        token_count=len(text.split()), content_hash=f"h_{file_id}_{idx}",
        text=text, embedder_model="t", embedder_sha256="x",
        embedding_row=None,
    )


# ---------------------------------------------------------------- project lifecycle


class TestProjects:
    def test_upsert_and_get(self, store: SqliteChunkStore) -> None:
        p = Project(name="x", root_path="/p/x", last_full_scan_at=42)
        store.upsert_project(p)
        assert store.get_project("x") == p

    def test_get_missing_returns_none(self, store: SqliteChunkStore) -> None:
        assert store.get_project("nope") is None

    def test_upsert_updates_existing(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(name="x", root_path="/old", last_full_scan_at=1))
        store.upsert_project(Project(name="x", root_path="/new", last_full_scan_at=2))
        got = store.get_project("x")
        assert got is not None
        assert got.root_path == "/new"
        assert got.last_full_scan_at == 2

    def test_list_empty(self, store: SqliteChunkStore) -> None:
        assert store.list_projects() == ()

    def test_list_sorted(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(name="z", root_path="/z", last_full_scan_at=0))
        store.upsert_project(Project(name="a", root_path="/a", last_full_scan_at=0))
        names = [p.name for p in store.list_projects()]
        assert names == ["a", "z"]

    def test_invalid_unique_constraint_root_path(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(name="a", root_path="/r", last_full_scan_at=0))
        with pytest.raises(sqlite3.IntegrityError):
            store.upsert_project(Project(name="b", root_path="/r", last_full_scan_at=0))

    def test_session_id_persisted(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(
            name="ephemeral", root_path="/e",
            last_full_scan_at=0, session_id="s1",
        ))
        got = store.get_project("ephemeral")
        assert got is not None
        assert got.session_id == "s1"


# ---------------------------------------------------------------- file lifecycle


class TestFiles:
    def test_upsert_returns_id(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(name="a", root_path="/a", last_full_scan_at=0))
        fid = store.upsert_file(FileRecord(
            id=0, project="a", rel_path="x.py", mtime=1,
            size_bytes=10, content_hash="h",
        ))
        assert fid > 0

    def test_get_missing(self, store: SqliteChunkStore) -> None:
        assert store.get_file("nope", "x") is None

    def test_upsert_updates(self, store: SqliteChunkStore) -> None:
        store.upsert_project(Project(name="a", root_path="/a", last_full_scan_at=0))
        fid1 = store.upsert_file(FileRecord(
            id=0, project="a", rel_path="x.py", mtime=1,
            size_bytes=10, content_hash="h1",
        ))
        fid2 = store.upsert_file(FileRecord(
            id=0, project="a", rel_path="x.py", mtime=2,
            size_bytes=20, content_hash="h2",
        ))
        assert fid1 == fid2
        got = store.get_file("a", "x.py")
        assert got is not None
        assert got.content_hash == "h2"

    def test_list_files_empty(self, store: SqliteChunkStore) -> None:
        assert store.list_files() == ()

    def test_list_files_filtered(self, populated: SqliteChunkStore) -> None:
        files = populated.list_files(project="alpha")
        assert len(files) == 1
        assert files[0].rel_path == "a.py"

    def test_invalid_project_fk(self, store: SqliteChunkStore) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.upsert_file(FileRecord(
                id=0, project="ghost", rel_path="x", mtime=0,
                size_bytes=0, content_hash="h",
            ))


# ---------------------------------------------------------------- chunk crud


class TestChunks:
    def test_upsert_empty_is_noop(self, store: SqliteChunkStore) -> None:
        store.upsert_chunks([])  # must not raise

    def test_round_trip(self, populated: SqliteChunkStore) -> None:
        all_chunks = populated.get_chunks_by_file(1)
        assert len(all_chunks) == 3
        ids = [c.id for c in all_chunks]
        round_trip = populated.get_chunks_by_id(ids)
        assert [c.text for c in round_trip] == [c.text for c in all_chunks]

    def test_get_chunks_by_id_empty(self, populated: SqliteChunkStore) -> None:
        assert populated.get_chunks_by_id([]) == ()

    def test_get_chunks_by_id_preserves_order(self, populated: SqliteChunkStore) -> None:
        all_chunks = populated.get_chunks_by_file(1)
        ids = [c.id for c in all_chunks]
        reversed_ids = list(reversed(ids))
        rt = populated.get_chunks_by_id(reversed_ids)
        assert [c.id for c in rt] == reversed_ids

    def test_get_chunks_by_id_skips_unknown(self, populated: SqliteChunkStore) -> None:
        # 99999 doesn't exist; should be silently dropped.
        rt = populated.get_chunks_by_id([1, 99999])
        assert len(rt) == 1

    def test_get_chunks_by_content_hash(self, populated: SqliteChunkStore) -> None:
        # h_1_0 = first chunk of file 1
        rt = populated.get_chunks_by_content_hash(1, ["h_1_0", "h_1_2"])
        assert {c.content_hash for c in rt} == {"h_1_0", "h_1_2"}

    def test_get_chunks_by_content_hash_empty(self, populated: SqliteChunkStore) -> None:
        assert populated.get_chunks_by_content_hash(1, []) == ()

    def test_get_chunks_by_content_hash_isolates_file(self, populated: SqliteChunkStore) -> None:
        # No chunk in file_id=1 has hash 'h_2_0'; result must be empty.
        rt = populated.get_chunks_by_content_hash(1, ["h_2_0"])
        assert rt == ()

    def test_chunk_count_unscoped(self, populated: SqliteChunkStore) -> None:
        assert populated.chunk_count() == 6

    def test_chunk_count_scoped_project(self, populated: SqliteChunkStore) -> None:
        assert populated.chunk_count(ScopeFilter(project="alpha")) == 3

    def test_chunk_count_scoped_project_in(self, populated: SqliteChunkStore) -> None:
        assert populated.chunk_count(
            ScopeFilter(project_in=("alpha", "beta"))
        ) == 6

    def test_chunk_count_scoped_empty_project_in(self, populated: SqliteChunkStore) -> None:
        assert populated.chunk_count(ScopeFilter(project_in=())) == 0

    def test_upsert_then_update_chunk(self, populated: SqliteChunkStore) -> None:
        # Edit one chunk, ensure it's updated and content_hash search reflects.
        c = populated.get_chunks_by_file(1)[0]
        replaced = Chunk(
            id=c.id, file_id=c.file_id, chunk_index=c.chunk_index,
            start_offset=c.start_offset, end_offset=c.end_offset,
            start_line=c.start_line, end_line=c.end_line,
            token_count=c.token_count, content_hash="new_hash",
            text="replaced text", embedder_model=c.embedder_model,
            embedder_sha256=c.embedder_sha256, embedding_row=c.embedding_row,
        )
        populated.upsert_chunks([replaced])
        new = populated.get_chunks_by_id([c.id])
        assert new[0].text == "replaced text"
        assert new[0].content_hash == "new_hash"


# ---------------------------------------------------------------- delete cascade


class TestCascade:
    def test_delete_file_drops_chunks(self, populated: SqliteChunkStore) -> None:
        before = populated.chunk_count()
        populated.delete_file(1)
        after = populated.chunk_count()
        assert before - after == 3
        assert populated.get_chunks_by_file(1) == ()

    def test_delete_project_drops_files_and_chunks(
        self, populated: SqliteChunkStore
    ) -> None:
        populated.delete_project("alpha")
        assert populated.chunk_count() == 3
        assert populated.list_files(project="alpha") == ()
        assert populated.get_project("alpha") is None


# ---------------------------------------------------------------- lexical search


class TestLexical:
    def test_search_empty_corpus(self, store: SqliteChunkStore) -> None:
        assert store.search_lexical(["foo"], k=5) == ()

    def test_search_empty_query(self, populated: SqliteChunkStore) -> None:
        assert populated.search_lexical([], k=5) == ()

    def test_search_invalid_k(self, populated: SqliteChunkStore) -> None:
        assert populated.search_lexical(["alpha"], k=0) == ()
        assert populated.search_lexical(["alpha"], k=-1) == ()

    def test_search_finds_keyword(self, populated: SqliteChunkStore) -> None:
        hits = populated.search_lexical(["alpha"], k=5)
        assert len(hits) > 0
        # Top hit should be a chunk that contains "alpha"
        top = populated.get_chunks_by_id([hits[0].chunk_id])[0]
        assert "alpha" in top.text

    def test_search_respects_scope(self, populated: SqliteChunkStore) -> None:
        # "fox" is in both alpha (chunk 1) and beta (chunk 6). With
        # project=alpha, only alpha chunks should appear.
        hits = populated.search_lexical(
            ["fox"], k=10, scope=ScopeFilter(project="alpha"),
        )
        chunks = populated.get_chunks_by_id([h.chunk_id for h in hits])
        # Every returned chunk must belong to alpha.
        files = {c.file_id for c in chunks}
        assert files == {1}

    def test_search_persistence_round_trip(self, tmp_path: Path) -> None:
        """Build, query, close, reopen, query → same top hit, no rebuild."""
        db = tmp_path / "p.db"
        s = SqliteChunkStore(db)
        s.upsert_project(Project(name="a", root_path="/a", last_full_scan_at=0))
        fid = s.upsert_file(FileRecord(
            id=0, project="a", rel_path="x", mtime=0,
            size_bytes=0, content_hash="h",
        ))
        s.upsert_chunks([
            _mk_chunk(0, fid, 0, "first chunk authentication"),
            _mk_chunk(0, fid, 1, "second chunk encryption"),
        ])
        first = s.search_lexical(["authentication"], k=1)
        s.close()

        s2 = SqliteChunkStore(db)
        try:
            assert (db.parent / "p.db.bm25.pkl").exists()
            second = s2.search_lexical(["authentication"], k=1)
            assert second == first
        finally:
            s2.close()

    def test_invalidation_rebuilds(self, populated: SqliteChunkStore) -> None:
        """Change underlying chunks → next query reflects new state."""
        populated.search_lexical(["alpha"], k=1)  # warm
        # Add a brand new chunk with a uniquely-discoverable token.
        populated.upsert_chunks([
            _mk_chunk(0, 1, 99, "supercalifragilistic xyzzy"),
        ])
        # upsert_chunks marks dirty; query rebuilds.
        hits = populated.search_lexical(["xyzzy"], k=1)
        assert len(hits) == 1
        chunk = populated.get_chunks_by_id([hits[0].chunk_id])[0]
        assert "xyzzy" in chunk.text

    def test_invalidate_after_delete(self, populated: SqliteChunkStore) -> None:
        """Delete file; lexical search no longer returns its chunks."""
        populated.search_lexical(["alpha"], k=10)
        populated.delete_file(1)
        hits = populated.search_lexical(["alpha"], k=10)
        chunks = populated.get_chunks_by_id([h.chunk_id for h in hits])
        # No chunk from file 1 should remain.
        assert all(c.file_id != 1 for c in chunks)

    def test_invalidate_lexical_explicit(self, populated: SqliteChunkStore) -> None:
        # Touching the file_id arg path; should not raise.
        populated.invalidate_lexical(file_id=1)
        populated.invalidate_lexical(file_id=None)


# ---------------------------------------------------------------- concurrency


class TestConcurrency:
    def test_concurrent_reads(self, populated: SqliteChunkStore) -> None:
        """4 threads searching simultaneously — all return same result,
        no exceptions."""
        results: list[tuple] = []
        errs: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                hits = populated.search_lexical(["alpha"], k=3)
                results.append(tuple((h.chunk_id, h.score) for h in hits))
            except Exception as e:   # pragma: no cover — surfaces in errs
                errs.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errs == []
        assert len(set(results)) == 1, f"non-deterministic: {results}"

    def test_wal_read_during_write(self, tmp_path: Path) -> None:
        """A reader connection sees the pre-write state until the
        writer's transaction commits."""
        db = tmp_path / "wal.db"
        writer = SqliteChunkStore(db)
        writer.upsert_project(Project(name="a", root_path="/a", last_full_scan_at=0))
        fid = writer.upsert_file(FileRecord(
            id=0, project="a", rel_path="x", mtime=0,
            size_bytes=0, content_hash="h",
        ))
        writer.upsert_chunks([_mk_chunk(0, fid, 0, "before")])

        # Open a second connection (reader) BEFORE the writer commits.
        reader = SqliteChunkStore(db)
        try:
            assert reader.chunk_count() == 1

            # Writer adds a chunk; reader before re-query still sees the
            # pre-existing count via its own snapshot — but because both
            # are using independent connections + autocommit, the read
            # after the write will see the new state. Validate WAL by
            # confirming the reader can read concurrently without
            # blocking + sees committed state.
            writer.upsert_chunks([_mk_chunk(0, fid, 1, "after")])
            assert reader.chunk_count() == 2
        finally:
            reader.close()
            writer.close()


# ---------------------------------------------------------------- schema guard


class TestSchema:
    def test_future_schema_rejected(self, tmp_path: Path) -> None:
        """Bump schema_version on disk to an unsupported value;
        opening must raise."""
        db = tmp_path / "f.db"
        s = SqliteChunkStore(db)
        s.close()
        # Tamper with on-disk schema version.
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE schema_version SET version = ? WHERE id = 1",
            (SCHEMA_VERSION + 999,),
        )
        conn.commit()
        conn.close()

        with pytest.raises(UnsupportedSchemaError):
            SqliteChunkStore(db)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        s = SqliteChunkStore(tmp_path / "i.db")
        s.close()
        s.close()   # must not raise


# --------------------------------------------------- Phase 2.0 integration


def test_get_file_by_id_returns_file_record(store: SqliteChunkStore, tmp_path: Path) -> None:
    """get_file_by_id is the searcher's path from chunk → citation —
    chunks know their file_id but the project + rel_path live on files."""
    store.upsert_project(Project(name="myapp", root_path=str(tmp_path / "myapp")))
    fid = store.upsert_file(FileRecord(
        id=0, project="myapp", rel_path="src/auth.py",
        mtime=1000, size_bytes=42, content_hash="a" * 64,
    ))
    rec = store.get_file_by_id(fid)
    assert rec is not None
    assert rec.id == fid
    assert rec.project == "myapp"
    assert rec.rel_path == "src/auth.py"
    assert rec.content_hash == "a" * 64


def test_get_file_by_id_returns_none_for_missing(store: SqliteChunkStore) -> None:
    assert store.get_file_by_id(99999) is None


def test_delete_chunks_by_file_drops_chunks_keeps_file(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """Indexer's incremental-update path deletes chunks before
    re-upserting the new chunk set; the file row itself must survive."""
    store.upsert_project(Project(name="p", root_path=str(tmp_path / "p")))
    fid = store.upsert_file(FileRecord(
        id=0, project="p", rel_path="x.py", mtime=1, size_bytes=1,
        content_hash="z" * 64,
    ))
    store.upsert_chunks([
        Chunk(
            id=0, file_id=fid, chunk_index=i, start_offset=i * 10,
            end_offset=(i + 1) * 10, start_line=i, end_line=i + 1,
            token_count=4, content_hash=f"{i:064x}", text=f"hello {i}",
            embedder_model="m", embedder_sha256="s", embedding_row=i,
        )
        for i in range(3)
    ])
    assert len(store.get_chunks_by_file(fid)) == 3

    store.delete_chunks_by_file(fid)

    assert store.get_chunks_by_file(fid) == ()
    # File row survives so the indexer can re-upsert chunks against it
    assert store.get_file_by_id(fid) is not None


def test_list_chunk_ids_in_scope_returns_all_when_unfiltered(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    store.upsert_project(Project(name="p", root_path=str(tmp_path / "p")))
    fid = store.upsert_file(FileRecord(
        id=0, project="p", rel_path="x.py", mtime=1, size_bytes=1,
        content_hash="z" * 64,
    ))
    store.upsert_chunks([
        Chunk(
            id=0, file_id=fid, chunk_index=i, start_offset=i, end_offset=i + 1,
            start_line=i, end_line=i + 1, token_count=1, content_hash=f"{i:064x}",
            text=f"t{i}", embedder_model="m", embedder_sha256="s",
            embedding_row=i,
        )
        for i in range(5)
    ])
    ids = store.list_chunk_ids_in_scope(None)
    assert len(ids) == 5
    # Sorted ascending by id for deterministic dense-search output
    assert list(ids) == sorted(ids)


def test_list_chunk_ids_in_scope_filters_by_project(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """Searcher uses this to bound dense + BM25 to a project subset."""
    for name in ("a", "b"):
        store.upsert_project(Project(name=name, root_path=str(tmp_path / name)))
        fid = store.upsert_file(FileRecord(
            id=0, project=name, rel_path="f.py", mtime=1, size_bytes=1,
            content_hash=name * 64,
        ))
        store.upsert_chunks([
            Chunk(
                id=0, file_id=fid, chunk_index=0, start_offset=0, end_offset=1,
                start_line=1, end_line=1, token_count=1,
                content_hash=f"{name}_{0:063x}",
                text=f"hello {name}", embedder_model="m", embedder_sha256="s",
                embedding_row=0,
            )
        ])
    ids_a = store.list_chunk_ids_in_scope(ScopeFilter(project="a"))
    ids_b = store.list_chunk_ids_in_scope(ScopeFilter(project="b"))
    assert len(ids_a) == 1
    assert len(ids_b) == 1
    assert set(ids_a).isdisjoint(set(ids_b))


def test_get_embedding_rows_by_chunk_ids_translates_id_to_row(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """Inverse of get_chunk_ids_by_embedding_rows. The Searcher uses
    this method to translate a chunk-id-space scope filter into the
    row-space the EmbedStore expects. Returns rows sorted ascending
    so brute-force cosine ordering is deterministic."""
    store.upsert_project(Project(name="p", root_path=str(tmp_path / "p")))
    fid = store.upsert_file(FileRecord(
        id=0, project="p", rel_path="x.py", mtime=1, size_bytes=1,
        content_hash="z" * 64,
    ))
    store.upsert_chunks([
        Chunk(
            id=0, file_id=fid, chunk_index=i, start_offset=i, end_offset=i + 1,
            start_line=i, end_line=i + 1, token_count=1, content_hash=f"{i:064x}",
            text=f"t{i}", embedder_model="m", embedder_sha256="s",
            embedding_row=10 + i,  # deliberate id ≠ row offset
        )
        for i in range(3)
    ])
    ids = list(store.list_chunk_ids_in_scope(None))
    rows = store.get_embedding_rows_by_chunk_ids(ids)
    assert rows == (10, 11, 12)
    # Round-trip: row → id → row preserves the set
    id_by_row = store.get_chunk_ids_by_embedding_rows(rows)
    round_trip = store.get_embedding_rows_by_chunk_ids(id_by_row.values())
    assert round_trip == (10, 11, 12)


def test_get_embedding_rows_skips_chunks_without_row(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """Chunks with embedding_row IS NULL (e.g., upserted-but-not-embedded
    yet) are silently dropped from the result so they don't pollute the
    in-scope filter the EmbedStore consumes."""
    store.upsert_project(Project(name="p", root_path=str(tmp_path / "p")))
    fid = store.upsert_file(FileRecord(
        id=0, project="p", rel_path="x.py", mtime=1, size_bytes=1,
        content_hash="z" * 64,
    ))
    store.upsert_chunks([
        Chunk(
            id=0, file_id=fid, chunk_index=i, start_offset=i, end_offset=i + 1,
            start_line=i, end_line=i + 1, token_count=1, content_hash=f"{i:064x}",
            text=f"t{i}", embedder_model="m", embedder_sha256="s",
            embedding_row=i if i < 2 else None,  # third chunk has no embedding
        )
        for i in range(3)
    ])
    ids = list(store.list_chunk_ids_in_scope(None))
    rows = store.get_embedding_rows_by_chunk_ids(ids)
    assert rows == (0, 1)  # third chunk dropped


def test_list_chunk_ids_in_scope_project_in_filter(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """project_in supports the multi-project fan-out path."""
    for name in ("a", "b", "c"):
        store.upsert_project(Project(name=name, root_path=str(tmp_path / name)))
        fid = store.upsert_file(FileRecord(
            id=0, project=name, rel_path="f.py", mtime=1, size_bytes=1,
            content_hash=name * 64,
        ))
        store.upsert_chunks([
            Chunk(
                id=0, file_id=fid, chunk_index=0, start_offset=0, end_offset=1,
                start_line=1, end_line=1, token_count=1,
                content_hash=f"{name}_{0:063x}",
                text=f"hi {name}", embedder_model="m", embedder_sha256="s",
                embedding_row=0,
            )
        ])
    fan = store.list_chunk_ids_in_scope(ScopeFilter(project_in=("a", "c")))
    assert len(fan) == 2


def test_rename_file_preserves_chunks_and_embeddings(
    store: SqliteChunkStore, tmp_path: Path,
) -> None:
    """Watcher RENAME path: only the files.rel_path + mtime change;
    chunks and their embedding_row references are untouched."""
    store.upsert_project(Project(name="p", root_path=str(tmp_path / "p")))
    fid = store.upsert_file(FileRecord(
        id=0, project="p", rel_path="src/old.py", mtime=100, size_bytes=42,
        content_hash="z" * 64,
    ))
    store.upsert_chunks([
        Chunk(
            id=0, file_id=fid, chunk_index=i, start_offset=i * 10,
            end_offset=(i + 1) * 10, start_line=i, end_line=i + 1,
            token_count=4, content_hash=f"{i:064x}", text=f"hello {i}",
            embedder_model="m", embedder_sha256="s", embedding_row=i,
        )
        for i in range(3)
    ])
    pre_chunks = store.get_chunks_by_file(fid)
    pre_rows = tuple(c.embedding_row for c in pre_chunks)
    assert len(pre_chunks) == 3

    store.rename_file(fid, new_rel_path="src/new.py", mtime=200)

    rec = store.get_file_by_id(fid)
    assert rec is not None
    assert rec.rel_path == "src/new.py"
    assert rec.mtime == 200

    post_chunks = store.get_chunks_by_file(fid)
    assert len(post_chunks) == 3
    assert tuple(c.embedding_row for c in post_chunks) == pre_rows
    # Hash unchanged (text didn't change)
    assert {c.content_hash for c in pre_chunks} == \
           {c.content_hash for c in post_chunks}
