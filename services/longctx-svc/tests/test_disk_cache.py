"""Disk cache tests. PRD §4 + smoke §7.10."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from longctx_svc.cache.disk import (
    cache_dir_for,
    cache_root_size_bytes,
    clean_older_than,
    list_cached,
    load_index,
    save_index,
)
from longctx_svc.config import ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect ~/.longctx to a tmp_path-rooted dir for the duration."""
    cache_root = tmp_path / "longctx-cache"
    cfg = ServiceConfig(cache_dir=cache_root)
    set_config(cfg)
    yield cache_root
    set_config(ServiceConfig())  # reset to env-driven default


def _make_index(scope_hash: str, scope_root: Path,
                n_chunks: int = 3) -> ScopeIndex:
    chunks = [Chunk(
        text=f"chunk {i} text\n",
        file_path=str(scope_root / f"file{i}.py"),
        start_line=1, end_line=10, file_type="code",
    ) for i in range(n_chunks)]
    embs = np.random.randn(n_chunks, 384).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return ScopeIndex(
        scope_root=scope_root, scope_hash=scope_hash,
        chunks=chunks, embeddings=embs,
        file_count=n_chunks, built_at=time.time(),
        embedder_name="test-model",
    )


def test_save_and_load_round_trip(cache_dir, tmp_path):
    root = tmp_path / "myproj"
    root.mkdir()
    idx = _make_index("abcd1234", root, n_chunks=5)
    save_index(idx, sentinel="package.json")

    # Files exist
    cdir = cache_dir / "abcd1234"
    assert (cdir / "embeddings.npy").is_file()
    assert (cdir / "chunks.jsonl").is_file()
    assert (cdir / "metadata.json").is_file()

    # Round trip
    loaded = load_index("abcd1234")
    assert loaded is not None
    idx2, meta = loaded
    assert idx2.chunk_count == 5
    np.testing.assert_array_almost_equal(idx2.embeddings, idx.embeddings)
    assert idx2.chunks[0].text == "chunk 0 text\n"
    assert meta["sentinel"] == "package.json"
    assert meta["scope_root"] == str(root)


def test_load_returns_none_when_absent(cache_dir):
    assert load_index("does_not_exist") is None


def test_load_returns_none_on_corrupt_metadata(cache_dir, tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    idx = _make_index("corrupt1", root)
    save_index(idx, sentinel="package.json")
    (cache_dir / "corrupt1" / "metadata.json").write_text("not json {{{")
    assert load_index("corrupt1") is None


def test_load_returns_none_on_version_mismatch(cache_dir, tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    idx = _make_index("vers1", root)
    save_index(idx, sentinel=".git")
    meta_path = cache_dir / "vers1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = 99999
    meta_path.write_text(json.dumps(meta))
    assert load_index("vers1") is None


def test_load_returns_none_on_chunk_count_mismatch(cache_dir, tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    idx = _make_index("mm1", root, n_chunks=4)
    save_index(idx, sentinel=".git")
    # Truncate chunks.jsonl
    chunks_path = cache_dir / "mm1" / "chunks.jsonl"
    lines = chunks_path.read_text().splitlines()
    chunks_path.write_text(lines[0] + "\n")
    assert load_index("mm1") is None


def test_save_noop_for_empty_index(cache_dir, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    idx = ScopeIndex(scope_root=root, scope_hash="empty1",
                     chunks=[], embeddings=None, file_count=0)
    save_index(idx, sentinel="package.json")
    cdir = cache_dir / "empty1"
    assert not (cdir / "embeddings.npy").exists()


def test_list_cached_returns_metadata(cache_dir, tmp_path):
    root = tmp_path / "p1"
    root.mkdir()
    idx = _make_index("h1", root, n_chunks=3)
    save_index(idx, sentinel="package.json")
    listing = list_cached()
    assert len(listing) == 1
    assert listing[0]["scope_hash"] == "h1"
    assert listing[0]["chunk_count"] == 3
    assert listing[0]["size_bytes"] > 0


def test_clean_older_than_drops_old_entries(cache_dir, tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    idx_old = _make_index("old1", root)
    idx_new = _make_index("new1", root)
    save_index(idx_old, sentinel="package.json")
    save_index(idx_new, sentinel="package.json")
    # Backdate the old one
    meta_path = cache_dir / "old1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["saved_at"] = time.time() - 60 * 86400
    meta_path.write_text(json.dumps(meta))

    removed = clean_older_than(days=30)
    assert removed == 1
    assert not (cache_dir / "old1").exists()
    assert (cache_dir / "new1").exists()


def test_cache_root_size_bytes(cache_dir, tmp_path):
    assert cache_root_size_bytes() == 0
    root = tmp_path / "p"
    root.mkdir()
    idx = _make_index("size1", root, n_chunks=10)
    save_index(idx, sentinel=".git")
    assert cache_root_size_bytes() > 0


# --- Smoke §7.10: cache reload across process restarts (simulated)

def test_register_scope_reloads_from_disk(cache_dir, tmp_path):
    """A previously-saved scope must be loadable on a fresh state."""
    root = tmp_path / "myapp"
    root.mkdir()
    (root / "package.json").write_text('{"name":"myapp"}')
    idx = _make_index("reload1", root, n_chunks=4)
    save_index(idx, sentinel="package.json")

    # Fresh state
    from longctx_svc.state import reset_state, get_state
    reset_state()
    t0 = time.time()
    entry = get_state().register_scope(root, "reload1", "package.json")
    elapsed_ms = (time.time() - t0) * 1000

    # Smoke §7.10: <500ms target. With 4 chunks this should be way under.
    assert elapsed_ms < 500
    assert entry.index is not None
    assert entry.index.chunk_count == 4
    assert entry.status == "ready"
    assert entry.from_disk is True
