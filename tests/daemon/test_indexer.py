"""Indexer end-to-end tests.

Drive the indexer against fake ``ChunkStore`` + ``EmbedStore`` (see
``conftest.py``). No SQLite, no memmap, no real embedder. The fakes
are kept honest by mirroring the protocol surface and freelist
semantics.

Coverage targets (from agent spec §5):
  * add_project: rejects forbidden_dirs, registers, idempotent
  * full_scan: chunks every text file, skips by size / secret /
    gitignore / extension allowlist
  * update_file: no-op on unchanged content; reuses chunks on
    one-line edits; handles new-file and deleted-then-readded
  * delete_file / delete_project: free embedding rows + drop chunks
  * embedder mismatch: SHA change between two indexer instances
    raises EmbedderMismatchError
"""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx.rag.chunker import Chunker
from longctx_daemon.indexer import (
    EmbedderMismatchError,
    ForbiddenProjectError,
    Indexer,
    IndexerConfig,
    ScanResult,
    UpdateResult,
)
from longctx_daemon.types import Project


# --------------------------------------------------------------- helpers

def _make_indexer(
    fake_chunk_store,
    fake_embed_store,
    fake_embedder,
    config: IndexerConfig | None = None,
) -> Indexer:
    chunker = Chunker(tokens_per_chunk=64, token_overlap=8)
    return Indexer(
        chunk_store=fake_chunk_store,
        embed_store=fake_embed_store,
        embedder=fake_embedder,
        chunker=chunker,
        config=config or IndexerConfig(
            # Tighter chunks so 50-line files produce multiple chunks.
            embed_batch_size=8,
        ),
    )


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------- add_project

class TestAddProject:
    def test_registers_new_project(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        proj = idx.add_project("demo", tmp_path)
        assert isinstance(proj, Project)
        assert proj.name == "demo"
        assert Path(proj.root_path) == tmp_path.resolve()
        assert fake_chunk_store.get_project("demo") is not None

    def test_re_add_preserves_last_full_scan(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        # Simulate a prior scan timestamp.
        existing = fake_chunk_store.get_project("demo")
        fake_chunk_store.upsert_project(
            Project(
                name=existing.name,
                root_path=existing.root_path,
                last_full_scan_at=12345,
                session_id=existing.session_id,
            )
        )
        proj2 = idx.add_project("demo", tmp_path)
        assert proj2.last_full_scan_at == 12345

    def test_rejects_forbidden_root(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        bad = tmp_path / "secrets"
        bad.mkdir()
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        with pytest.raises(ForbiddenProjectError):
            idx.add_project("evil", bad)
        assert fake_chunk_store.get_project("evil") is None

    def test_rejects_missing_root(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        with pytest.raises(FileNotFoundError):
            idx.add_project("ghost", tmp_path / "does-not-exist")


# --------------------------------------------------------------- full_scan

class TestFullScan:
    def test_indexes_text_files(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "main.py", "print('hello world')\n" * 5)
        _write(tmp_path, "README.md", "# Title\n\nBody text.\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        assert isinstance(result, ScanResult)
        assert result.n_files == 2
        assert result.n_chunks_total >= 2
        assert result.n_chunks_new == result.n_chunks_total
        assert result.n_chunks_reused == 0
        # Embeddings actually went through the embedder.
        assert sum(len(c) for c in fake_embedder.encode_calls) >= 2

    def test_updates_last_full_scan_at(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "x = 1\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        proj = fake_chunk_store.get_project("demo")
        assert proj.last_full_scan_at > 0

    def test_unknown_project_raises(
        self, fake_chunk_store, fake_embed_store, fake_embedder,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        with pytest.raises(KeyError):
            idx.full_scan("ghost")

    def test_skips_files_over_size_cap(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "tiny.py", "ok\n")
        _write(tmp_path, "huge.txt", "x" * (2048 + 1))
        cfg = IndexerConfig(max_file_size_bytes=2048)
        idx = _make_indexer(
            fake_chunk_store, fake_embed_store, fake_embedder, cfg,
        )
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        assert result.n_files == 1
        assert result.n_files_skipped >= 1

    def test_skips_secret_files(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "ok.py", "ok\n")
        _write(tmp_path, ".env", "SECRET=abc\n")
        _write(tmp_path, "id_rsa", "private\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        assert result.n_files == 1  # only ok.py
        rel_paths = {f.rel_path for f in fake_chunk_store.list_files()}
        assert rel_paths == {"ok.py"}

    def test_skips_gitignored_files(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        # .gitignore itself is also indexable text (it doesn't ignore
        # itself by default) — assert the files that should be missing
        # rather than a global count.
        _write(tmp_path, ".gitignore", "ignored.py\n*.log\n")
        _write(tmp_path, "ok.py", "ok\n")
        _write(tmp_path, "ignored.py", "skip\n")
        _write(tmp_path, "trace.log", "trace\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        rel_paths = {f.rel_path for f in fake_chunk_store.list_files()}
        assert "ok.py" in rel_paths
        assert "ignored.py" not in rel_paths
        assert "trace.log" not in rel_paths
        # 2 files indexed (ok.py + .gitignore), 2 skipped.
        assert result.n_files_skipped >= 2

    def test_skips_excluded_dirs(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "node_modules/dep/index.js", "// dep\n")
        _write(tmp_path, ".git/HEAD", "ref: refs/heads/main\n")
        _write(tmp_path, "src/main.py", "ok\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        rel_paths = {f.rel_path for f in fake_chunk_store.list_files()}
        assert rel_paths == {"src/main.py"}
        assert result.n_files == 1

    def test_skips_binary_files(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        # Unknown extension + null bytes => binary.
        binary = tmp_path / "blob.bin"
        binary.write_bytes(b"\x00\x01\x02 not text \x00")
        _write(tmp_path, "code.py", "ok\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        assert result.n_files == 1

    def test_extension_allowlist_filters(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "code\n")
        _write(tmp_path, "b.md", "prose\n")
        cfg = IndexerConfig(include_extensions=(".py",))
        idx = _make_indexer(
            fake_chunk_store, fake_embed_store, fake_embedder, cfg,
        )
        idx.add_project("demo", tmp_path)
        result = idx.full_scan("demo")
        assert result.n_files == 1

    def test_chunks_carry_embedder_identity(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "x = 1\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        files = fake_chunk_store.list_files()
        chunks = fake_chunk_store.get_chunks_by_file(files[0].id)
        assert chunks[0].embedder_sha256 == fake_embedder.fake_sha
        assert chunks[0].embedder_model == fake_embedder.model_name

    def test_chunks_have_line_numbers(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "line1\nline2\nline3\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        files = fake_chunk_store.list_files()
        chunks = fake_chunk_store.get_chunks_by_file(files[0].id)
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= chunks[0].start_line


# --------------------------------------------------------------- update_file

class TestUpdateFile:
    def test_noop_on_unchanged_content(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        path = _write(tmp_path, "a.py", "x = 1\n" * 20)
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        encode_calls_before = len(fake_embedder.encode_calls)

        # Touch mtime via re-write of identical content.
        path.write_text("x = 1\n" * 20, encoding="utf-8")
        result = idx.update_file("demo", path)
        assert isinstance(result, UpdateResult)
        assert result.n_chunks_new == 0
        assert result.n_chunks_reused == result.n_chunks_total
        # No new embed call.
        assert len(fake_embedder.encode_calls) == encode_calls_before

    def test_one_line_edit_reuses_most_chunks(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        # Build a 50-line file that produces multiple chunks.
        lines = [f"line_{i:03d} body content here\n" for i in range(50)]
        path = _write(tmp_path, "a.py", "".join(lines))
        # Use a smaller chunker so 50 lines produces ≥3 chunks.
        chunker = Chunker(tokens_per_chunk=20, token_overlap=2)
        idx = Indexer(
            chunk_store=fake_chunk_store,
            embed_store=fake_embed_store,
            embedder=fake_embedder,
            chunker=chunker,
            config=IndexerConfig(),
        )
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        files = fake_chunk_store.list_files()
        first_pass_chunks = fake_chunk_store.get_chunks_by_file(files[0].id)
        assert len(first_pass_chunks) >= 3

        # Edit the very first line only — keep length so subsequent
        # chunk byte boundaries don't shift.
        old0 = lines[0]
        new0 = "line_AAA body content here\n"
        assert len(new0) == len(old0)
        lines[0] = new0
        path.write_text("".join(lines), encoding="utf-8")
        result = idx.update_file("demo", path)
        assert result.n_chunks_reused > 0
        assert result.n_chunks_new >= 1
        # Reuse > new for a single-line edit on a many-chunk file.
        assert result.n_chunks_reused > result.n_chunks_new

    def test_handles_new_file(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        # Project added but never scanned. New file shows up.
        path = _write(tmp_path, "fresh.py", "print('hi')\n" * 5)
        result = idx.update_file("demo", path)
        assert result.n_files == 1
        assert result.n_chunks_new > 0
        assert result.n_chunks_reused == 0

    def test_skips_filtered_files(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        secret = _write(tmp_path, ".env", "API=secret\n")
        result = idx.update_file("demo", secret)
        assert result.n_files == 0
        assert result.n_files_skipped == 1

    def test_shrinking_file_drops_orphans(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        # Big file -> many chunks
        big = "".join(f"line_{i:03d} content padded somewhat\n"
                      for i in range(80))
        path = _write(tmp_path, "a.py", big)
        chunker = Chunker(tokens_per_chunk=20, token_overlap=2)
        idx = Indexer(
            chunk_store=fake_chunk_store,
            embed_store=fake_embed_store,
            embedder=fake_embedder,
            chunker=chunker,
            config=IndexerConfig(),
        )
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        files = fake_chunk_store.list_files()
        n_before = len(fake_chunk_store.get_chunks_by_file(files[0].id))

        # Shrink.
        path.write_text("only one line\n", encoding="utf-8")
        idx.update_file("demo", path)
        files = fake_chunk_store.list_files()
        n_after = len(fake_chunk_store.get_chunks_by_file(files[0].id))
        assert n_after < n_before
        # Some embedding rows freed.
        assert len(fake_embed_store.frees) > 0

    def test_unknown_project_raises(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        path = _write(tmp_path, "a.py", "x\n")
        with pytest.raises(KeyError):
            idx.update_file("ghost", path)


# --------------------------------------------------------------- delete

class TestDelete:
    def test_delete_file_removes_chunks_and_frees_rows(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "ok\n" * 10)
        _write(tmp_path, "b.py", "ok\n" * 10)
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        rows_before = fake_embed_store.num_rows

        idx.delete_file("demo", "a.py")
        assert fake_chunk_store.get_file("demo", "a.py") is None
        # b.py untouched.
        assert fake_chunk_store.get_file("demo", "b.py") is not None
        # At least one row freed.
        assert len(fake_embed_store.frees) > 0
        # num_rows shouldn't grow on delete (rows freed, not removed).
        assert fake_embed_store.num_rows == rows_before

    def test_delete_file_missing_is_noop(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        # No file ever indexed; deleting should be silent.
        idx.delete_file("demo", "nope.py")

    def test_delete_project_cascades(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "ok\n" * 5)
        _write(tmp_path, "b.py", "ok\n" * 5)
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        idx.full_scan("demo")
        assert fake_chunk_store.get_project("demo") is not None
        idx.delete_project("demo")
        assert fake_chunk_store.get_project("demo") is None
        assert len(fake_chunk_store.list_files(project="demo")) == 0
        # Rows freed for every chunk.
        assert len(fake_embed_store.frees) > 0


# --------------------------------------------------------------- embedder ID

class TestEmbedderIdentity:
    def test_mismatch_raises_on_full_scan(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "ok\n")
        # First indexer with one SHA.
        idx1 = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx1.add_project("demo", tmp_path)
        idx1.full_scan("demo")

        # New embedder with a different SHA, same store. Use the same
        # _FakeEmbedder type instantiated again so its identity hooks
        # work without a stricter import path.
        fake2 = type(fake_embedder)(dim=4)
        fake2.fake_sha = "DIFFERENT-SHA-aaaaaa"
        idx2 = _make_indexer(fake_chunk_store, fake_embed_store, fake2)
        with pytest.raises(EmbedderMismatchError):
            idx2.full_scan("demo")

    def test_same_sha_does_not_raise(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        _write(tmp_path, "a.py", "ok\n")
        idx1 = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx1.add_project("demo", tmp_path)
        idx1.full_scan("demo")
        # New indexer, same embedder => same SHA.
        idx2 = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        # Should not raise.
        idx2.full_scan("demo")

    def test_first_run_no_chunks_no_check(
        self, fake_chunk_store, fake_embed_store, fake_embedder, tmp_path,
    ):
        # No existing chunks => embedder check is a no-op.
        _write(tmp_path, "a.py", "ok\n")
        idx = _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        idx.add_project("demo", tmp_path)
        # Should not raise even though SHA != stored (none stored yet).
        idx.full_scan("demo")


# --------------------------------------------------------------- misc

class TestEmbedderMaxSeqLength:
    def test_sets_max_seq_length_when_provided(
        self, fake_chunk_store, fake_embed_store, fake_embedder,
    ):
        cfg = IndexerConfig(embedder_max_seq_length=512)
        _make_indexer(
            fake_chunk_store, fake_embed_store, fake_embedder, cfg,
        )
        assert fake_embedder.max_seq_length == 512

    def test_skips_when_not_provided(
        self, fake_chunk_store, fake_embed_store, fake_embedder,
    ):
        _make_indexer(fake_chunk_store, fake_embed_store, fake_embedder)
        assert fake_embedder.max_seq_length is None
