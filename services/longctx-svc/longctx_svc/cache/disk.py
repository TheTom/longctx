"""Per-scope disk cache. PRD §4.

Layout:
    ~/.longctx/<scope-hash>/
        embeddings.npy        # (N, D) float32, L2-normalized
        chunks.jsonl          # one Chunk per line
        metadata.json         # scope_root, sentinel, embedder, built_at, etc.

Smoke §7.10 acceptance: reloading a previously-built scope on a fresh
process must take <500ms and yield identical retrieval output.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from longctx_svc.config import get_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk


CACHE_LAYOUT_VERSION = 1


def cache_dir_for(scope_hash: str) -> Path:
    """Resolve the per-scope cache directory under LONGCTX_CACHE_DIR."""
    return get_config().cache_dir / scope_hash


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_index(index: ScopeIndex, sentinel: str) -> Path:
    """Persist a built index. No-op if embeddings is None."""
    if index.embeddings is None or not index.chunks:
        return cache_dir_for(index.scope_hash)
    cdir = cache_dir_for(index.scope_hash)
    cdir.mkdir(parents=True, exist_ok=True)

    # 1. embeddings.npy (atomic via temp + replace).
    # np.save appends .npy if the path doesn't already end in it, so
    # write to a unique temp name then rename.
    import tempfile
    with tempfile.NamedTemporaryFile(
        dir=cdir, prefix=".embeddings.", suffix=".npy", delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
    np.save(tmp_path, index.embeddings.astype(np.float32),
            allow_pickle=False)
    os.replace(tmp_path, cdir / "embeddings.npy")

    # 2. chunks.jsonl
    lines = []
    for c in index.chunks:
        lines.append(json.dumps({
            "text": c.text,
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "file_type": c.file_type,
        }, ensure_ascii=False))
    _atomic_write_text(cdir / "chunks.jsonl", "\n".join(lines) + "\n")

    # 3. metadata.json
    meta = {
        "version": CACHE_LAYOUT_VERSION,
        "scope_root": str(index.scope_root),
        "scope_hash": index.scope_hash,
        "sentinel": sentinel,
        "file_count": index.file_count,
        "chunk_count": index.chunk_count,
        "embedder_name": index.embedder_name,
        "built_at": index.built_at,
        "saved_at": time.time(),
        "embedding_dim": int(index.embeddings.shape[1]),
        "embedding_dtype": str(index.embeddings.dtype),
    }
    _atomic_write_text(cdir / "metadata.json", json.dumps(meta, indent=2))
    return cdir


def load_index(scope_hash: str) -> tuple[ScopeIndex, dict] | None:
    """Reload a persisted index. Returns (index, metadata) or None if
    the cache dir is missing, partial, or version-mismatched.
    """
    cdir = cache_dir_for(scope_hash)
    meta_path = cdir / "metadata.json"
    npy_path = cdir / "embeddings.npy"
    chunks_path = cdir / "chunks.jsonl"
    if not (meta_path.is_file() and npy_path.is_file()
            and chunks_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("version") != CACHE_LAYOUT_VERSION:
        return None
    try:
        embs = np.load(npy_path)
    except (OSError, ValueError):
        return None
    chunks: list[Chunk] = []
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                chunks.append(Chunk(
                    text=d["text"],
                    file_path=d["file_path"],
                    start_line=int(d["start_line"]),
                    end_line=int(d["end_line"]),
                    file_type=d.get("file_type", "other"),
                ))
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    if embs.shape[0] != len(chunks):
        return None
    index = ScopeIndex(
        scope_root=Path(meta["scope_root"]),
        scope_hash=scope_hash,
        chunks=chunks,
        embeddings=embs.astype(np.float32),
        file_count=int(meta.get("file_count", 0)),
        built_at=float(meta.get("built_at", 0.0)),
        embedder_name=meta.get("embedder_name", ""),
    )
    return index, meta


def list_cached() -> list[dict]:
    """List every scope cache dir under ~/.longctx with a valid metadata."""
    cfg = get_config()
    if not cfg.cache_dir.exists():
        return []
    out = []
    for d in cfg.cache_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        size = sum(
            f.stat().st_size for f in d.iterdir() if f.is_file()
        )
        out.append({
            "scope_hash": d.name,
            "scope_root": meta.get("scope_root", ""),
            "sentinel": meta.get("sentinel", ""),
            "saved_at": float(meta.get("saved_at", 0.0)),
            "size_bytes": int(size),
            "chunk_count": int(meta.get("chunk_count", 0)),
        })
    return out


def clean_older_than(days: int) -> int:
    """Drop cache dirs whose `saved_at` is older than N days. Returns
    the count of removed scopes.
    """
    cutoff = time.time() - days * 86400
    removed = 0
    for entry in list_cached():
        if entry["saved_at"] < cutoff:
            try:
                shutil.rmtree(cache_dir_for(entry["scope_hash"]))
                removed += 1
            except OSError:
                pass
    return removed


def clean_all() -> int:
    """Remove the whole cache root. Returns count of removed scopes."""
    cfg = get_config()
    if not cfg.cache_dir.exists():
        return 0
    n = sum(1 for _ in cfg.cache_dir.iterdir())
    shutil.rmtree(cfg.cache_dir, ignore_errors=True)
    return n


def cache_root_size_bytes() -> int:
    cfg = get_config()
    if not cfg.cache_dir.exists():
        return 0
    total = 0
    for d in cfg.cache_dir.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
    return total
