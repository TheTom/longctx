"""Build a per-scope index: chunks + embeddings.

In-memory for v0.3.0 alpha skeleton. Disk persistence per PRD §4 lands
in a follow-up commit (write to ~/.longctx/<scope-hash>/).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from longctx_svc.config import get_config
from longctx_svc.indexer.chunker import Chunk, chunk_file


@dataclass
class ScopeIndex:
    """One per scope. Holds chunks + their normalized embeddings."""
    scope_root: Path
    scope_hash: str
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: np.ndarray | None = None       # shape (N, D), L2-normalized
    file_count: int = 0
    built_at: float = 0.0
    last_query_at: float = 0.0
    embedder_name: str = ""

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def memory_bytes(self) -> int:
        if self.embeddings is None:
            return 0
        return int(self.embeddings.nbytes)

    def touch(self) -> None:
        """Mark as recently queried (for eviction logic)."""
        self.last_query_at = time.time()


def _resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def update_files(index: ScopeIndex, files: list[Path],
                 embedder=None, device: str = "auto") -> ScopeIndex:
    """Incremental update: drop existing chunks for `files`, re-chunk and
    re-embed each, append back. Mutates `index` in-place and returns it.

    Files that no longer exist are simply removed.
    """
    cfg = get_config()
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(
            cfg.embedder_model, device=_resolve_device(device),
        )

    targets = {str(p) for p in files}
    if not targets:
        return index

    # 1. Drop existing chunks for these files (and their embedding rows).
    keep_idx: list[int] = []
    for i, c in enumerate(index.chunks):
        if c.file_path not in targets:
            keep_idx.append(i)
    if index.embeddings is not None and keep_idx:
        kept_embs = index.embeddings[keep_idx]
    else:
        kept_embs = None
    kept_chunks = [index.chunks[i] for i in keep_idx]

    # 2. Re-chunk + re-embed surviving files.
    new_chunks: list[Chunk] = []
    for f in files:
        if not f.exists():
            continue
        new_chunks.extend(chunk_file(f))
    if new_chunks:
        texts = [c.text for c in new_chunks]
        new_embs = embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        ).astype(np.float32)
        if kept_embs is not None and kept_embs.size:
            index.embeddings = np.vstack([kept_embs, new_embs])
        else:
            index.embeddings = new_embs
        index.chunks = kept_chunks + new_chunks
    else:
        index.embeddings = kept_embs if (kept_embs is not None
                                         and kept_embs.size) else None
        index.chunks = kept_chunks

    index.built_at = time.time()
    return index


def build_index(scope_root: Path, files: list[Path], scope_hash: str,
                embedder=None, device: str = "auto") -> ScopeIndex:
    """Chunk every file, embed every chunk, return a ScopeIndex.

    Caller is responsible for passing a sentence-transformers SentenceTransformer
    instance (or letting us instantiate the default).
    """
    cfg = get_config()
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(
            cfg.embedder_model, device=_resolve_device(device),
        )

    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(f))

    if not chunks:
        return ScopeIndex(
            scope_root=scope_root, scope_hash=scope_hash,
            chunks=[], embeddings=None, file_count=len(files),
            built_at=time.time(),
            embedder_name=cfg.embedder_model,
        )

    texts = [c.text for c in chunks]
    embs = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=64, show_progress_bar=False,
    )

    return ScopeIndex(
        scope_root=scope_root,
        scope_hash=scope_hash,
        chunks=chunks,
        embeddings=embs.astype(np.float32),
        file_count=len(files),
        built_at=time.time(),
        embedder_name=cfg.embedder_model,
    )
