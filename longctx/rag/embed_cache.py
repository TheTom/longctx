"""Disk-backed embedding cache.

Hash each chunk's text with sha256, store the embedding at
~/.cache/longctx/<embedder>/<2-char-shard>/<full-hash>.npy. On the next call,
only embed chunks whose hash isn't in cache.

Cache failures (read errors, write errors) are NEVER fatal — they degrade
to a fresh embed without raising. The cache is an optimization, not a
correctness boundary.

For "fresh haystack every query" workloads (one-shot PDF Q&A) the cache is
a no-op. For "ask many questions of one doc" or "persistent knowledge
base" patterns it drops embed time to ~0s after the first call.

See docs/embedding-roadmap.md for the broader strategy.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def _default_cache_dir() -> Path:
    """Resolve the platform-default cache root."""
    env = os.environ.get("LONGCTX_CACHE_DIR")
    if env:
        return Path(env)
    # XDG_CACHE_HOME or ~/.cache
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "longctx"


def _safe_dir_name(s: str) -> str:
    """Convert an embedder name to a filesystem-safe directory name."""
    return s.replace("/", "_").replace(":", "_")


class EmbedCache:
    """Disk-backed sha256-keyed embedding cache.

    Pass `cache_dir=None` (or omit) to disable caching. Pass `cache_dir="default"`
    or a path to enable. Pass an env var `LONGCTX_CACHE_DIR=/path` to override
    the platform default.

    Args:
        cache_dir: None (disable) | "default" (use ~/.cache/longctx) | path
        embedder_name: sub-directory name to keep different embedders' caches
            separate (a MiniLM-L6 vector is not interchangeable with a BGE one)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        embedder_name: str = "default",
    ) -> None:
        self.dir: Path | None = None
        if cache_dir is None:
            return
        if cache_dir == "default":
            cache_dir = _default_cache_dir()
        try:
            self.dir = Path(cache_dir) / _safe_dir_name(embedder_name)
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only home, missing perms, etc. Disable cache.
            self.dir = None

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _path_for(self, text: str) -> Path:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.dir / h[:2] / f"{h}.npy"  # type: ignore[union-attr]

    def get(self, text: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        path = self._path_for(text)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def set(self, text: str, emb: np.ndarray) -> None:
        if not self.enabled:
            return
        path = self._path_for(text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        except Exception:
            return  # silent: cache failures are non-fatal

    def encode_with_cache(
        self,
        embedder,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed `texts`, using cache where possible. Returns N × D array
        in original input order.
        """
        if not self.enabled:
            return embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )

        out: list[np.ndarray | None] = [None] * len(texts)
        misses_idx: list[int] = []
        misses_txt: list[str] = []
        for i, t in enumerate(texts):
            cached = self.get(t)
            if cached is not None:
                out[i] = cached
            else:
                misses_idx.append(i)
                misses_txt.append(t)

        if misses_txt:
            new_embs = embedder.encode(
                misses_txt,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            for j, idx in enumerate(misses_idx):
                out[idx] = new_embs[j]
                self.set(misses_txt[j], new_embs[j])

        return np.stack(out)  # type: ignore[arg-type]
