"""Memmap-backed ``EmbedStore`` implementation for Phase 2.0.

Persists L2-normalized dense embeddings as a numpy ``memmap`` plus a
small pickled state file. One physical row per chunk; the chunk's
``embedding_row`` field in :class:`SqliteChunkStore` indexes into this
store. Brute-force matmul is the search path — fine at the 50K-chunk
scale Phase 2 targets.

Embedder identity is recorded as ``model_name`` (HF id) AND
``model_sha256`` (actual file digest) so an upstream model swap can't
silently invalidate cached embeddings (PRD §4.4). Callers compute the
sha256 once at daemon startup and pass it in; the store never loads the
model itself.

What's NOT here yet:
  * No compaction pass for an ever-growing freelist — Phase 3+. The
    intended escape hatch is "shrink memmap on idle" in a separate
    background task.
  * No ANN index — pure brute-force matmul. Plenty fast for 50K rows
    on Apple Silicon; swap to FAISS / hnswlib in Phase 3 when needed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from longctx_daemon.types import Hit

if TYPE_CHECKING:   # pragma: no cover — only needed for type checkers
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- constants

#: Default initial allocation. Memmap grows by ``GROWTH_FACTOR`` from
#: this baseline, so the very first append doesn't double-pay for a
#: (rare) tiny corpus.
DEFAULT_INITIAL_ROWS = 1024

#: Multiplicative factor when the memmap is full and an append needs
#: more rows. 1.5 keeps wasted space bounded vs 2x while avoiding
#: pathological growth-spam at low row counts.
DEFAULT_GROWTH_FACTOR = 1.5

#: File mode for both .npy and .idx files. 0600 = owner-only RW.
EMBED_FILE_MODE = 0o600

#: Tolerance for the L2-norm assertion in ``append`` / ``append_batch``.
#: Tighter than typical fp32 round-trip noise; if you trip this, your
#: embeddings aren't actually normalized.
NORM_ATOL = 1e-5


_LOG = logging.getLogger(__name__)


class EmbedderMismatchError(RuntimeError):
    """Raised when an :class:`MemmapEmbedStore` is opened with a model
    sha256 that doesn't match the one persisted on disk. Refusing is
    the safer default — silently mixing two embedders' output rows
    breaks cosine geometry."""


# ------------------------------------------------------------------ helpers


def _safe_filename(name: str) -> str:
    """Make an HF model id safe to use as a filename — replace any non
    alnum/dash/dot/underscore character with ``_``. ``BAAI/bge-small-en-v1.5``
    becomes ``BAAI_bge-small-en-v1.5``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _try_chmod(path: Path, mode: int) -> None:
    """Best-effort POSIX chmod; tolerate platforms without it."""
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        _LOG.debug("chmod(%s, %o) unsupported on this platform", path, mode)


def _compute_embedder_sha256(model: SentenceTransformer) -> str:
    """Compute a stable hash of the loaded embedder's underlying weights.

    Walks the model's HF cache directory and hashes the (path, mtime,
    size) tuple of every file in deterministic order. Cheaper than
    rehashing N hundred MB of weights on every daemon start, and
    sufficient to detect "the upstream model id resolved to a different
    revision since last boot" (PRD §4.4).

    Callers that want a strict content-addressed hash can subclass +
    override; for the default daemon path file-stat is what we ship.
    """
    # ``model._model_card_vars["__model_path__"]`` is set by recent
    # sentence-transformers versions; fall back to the model's
    # ``cache_dir`` / first-parameter-tensor location otherwise. We do
    # NOT import sentence_transformers eagerly here (see TYPE_CHECKING
    # above) — the helper is opportunistic and tolerant.
    candidate_paths: list[Path] = []
    for attr in ("model_card_data", "_model_card_vars"):
        bag = getattr(model, attr, None)
        if isinstance(bag, dict) and "__model_path__" in bag:
            candidate_paths.append(Path(bag["__model_path__"]))
            break
    cache_dir = getattr(model, "cache_folder", None) or getattr(model, "_cache_folder", None)
    if cache_dir:
        candidate_paths.append(Path(cache_dir))
    # As a final fallback, hash the tensor data directly.
    if not candidate_paths:
        return _sha256_of_state_dict(model)

    hasher = hashlib.sha256()
    for root in candidate_paths:
        if not root.exists():
            continue
        if root.is_file():
            _hash_file(hasher, root)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                _hash_file(hasher, p)
    digest = hasher.hexdigest()
    if digest == hashlib.sha256().hexdigest():
        # Nothing on disk hashed — fall through to tensor digest.
        return _sha256_of_state_dict(model)
    return digest


def _hash_file(hasher: "hashlib._Hash", path: Path) -> None:
    """Stat-only contribution to a digest. We don't rehash file
    contents — that would scan ~250 MB on every boot. Stat is fine for
    "is this the same revision as last time"."""
    try:
        st = path.stat()
    except OSError:
        return
    hasher.update(str(path).encode())
    hasher.update(b"\0")
    hasher.update(str(st.st_size).encode())
    hasher.update(b"\0")
    hasher.update(str(int(st.st_mtime)).encode())
    hasher.update(b"\0")


def _sha256_of_state_dict(model: SentenceTransformer) -> str:
    """Last-resort fallback: hash the parameter tensors. Slower but
    correct."""
    import torch    # noqa: PLC0415 — only imported on the slow path

    hasher = hashlib.sha256()
    sd = model.state_dict() if hasattr(model, "state_dict") else {}
    for key in sorted(sd.keys()):
        t = sd[key]
        if isinstance(t, torch.Tensor):
            hasher.update(key.encode())
            hasher.update(b"\0")
            hasher.update(t.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ----------------------------------------------------------- state on disk


@dataclass
class _EmbedIndex:
    """Tiny pickled sidecar that captures everything we can't infer
    from the .npy itself."""
    num_rows: int = 0
    dim: int = 0
    model_name: str = ""
    model_sha256: str = ""
    freelist: list[int] = field(default_factory=list)
    capacity: int = 0   # informational; capacity also equals .npy shape[0]


# --------------------------------------------------------------- main class


class MemmapEmbedStore:
    """Concrete ``EmbedStore`` backed by a numpy memmap + freelist.

    Thread-safe for concurrent reads (memmap reads release the GIL);
    writes serialize through an internal RLock. Callers must pre-
    normalize their embeddings — the store asserts the L2 norm in
    ``validate=True`` mode (default) so a missing normalize call
    fails loudly the first time.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        model_name: str,
        model_sha256: str,
        dim: int,
        initial_rows: int = DEFAULT_INITIAL_ROWS,
        growth_factor: float = DEFAULT_GROWTH_FACTOR,
        validate: bool = True,
        on_mismatch: str = "raise",
    ) -> None:
        """
        Args:
            directory: directory holding ``<safe_name>.npy`` and
                ``<safe_name>.idx``. Created if missing.
            model_name: HF id of the embedder. Used for filename +
                identity recording.
            model_sha256: actual model-file SHA256, computed by the
                caller (see :func:`_compute_embedder_sha256`). Stored
                alongside ``model_name``; mismatch on reopen raises
                :class:`EmbedderMismatchError` by default.
            dim: embedding dimensionality. Must match any prior state.
            initial_rows: starting capacity of the memmap.
            growth_factor: multiplicative factor when grown.
            validate: assert L2-normalization on every append in debug
                mode. Pay the ~microsecond cost — bugs that ship are
                much more expensive than the asserts.
            on_mismatch: behaviour when reopened with a different
                ``model_sha256`` than recorded. ``"raise"`` (default)
                surfaces the change; ``"warn"`` logs and updates the
                stored sha (useful for ``longctx reembed --confirm``
                callers who already accepted the change).
        """
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if initial_rows <= 0:
            raise ValueError("initial_rows must be positive")
        if on_mismatch not in ("raise", "warn"):
            raise ValueError("on_mismatch must be 'raise' or 'warn'")

        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except (OSError, NotImplementedError):
            pass

        safe = _safe_filename(model_name)
        self._npy_path = self._dir / f"{safe}.npy"
        self._idx_path = self._dir / f"{safe}.idx"

        self._model_name = model_name
        self._model_sha256 = model_sha256
        self._dim = dim
        self._growth_factor = float(growth_factor)
        self._validate = validate
        self._lock = threading.RLock()

        if self._idx_path.exists() and self._npy_path.exists():
            self._open_existing(on_mismatch)
        else:
            self._create_new(initial_rows)

    # ------------------------------------------------------------ properties

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def num_rows(self) -> int:
        """Total appended rows. Includes freed rows (still occupy
        space). Capacity is `num_rows + len(freelist) + tail_unused`."""
        return self._state.num_rows

    @property
    def capacity(self) -> int:
        """Total physical rows in the memmap. Test/debug aid."""
        return self._state.capacity

    @property
    def freelist(self) -> tuple[int, ...]:
        return tuple(self._state.freelist)

    # ----------------------------------------------------------- write path

    def append(self, embedding: np.ndarray) -> int:
        """Append one row. Returns its physical index. ``embedding``
        must be ``(dim,)`` and L2-normalized. Reuses freelist before
        growing the memmap."""
        if embedding.shape != (self._dim,):
            raise ValueError(
                f"expected shape ({self._dim},), got {embedding.shape}"
            )
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        self._assert_normalized(embedding)

        with self._lock:
            if self._state.freelist:
                row = int(self._state.freelist.pop())
                self._mm[row] = embedding
            else:
                if self._state.num_rows >= self._state.capacity:
                    self._grow()
                row = int(self._state.num_rows)
                self._mm[row] = embedding
                self._state.num_rows += 1
            self._mm.flush()
            self._save_index()
            return row

    def append_batch(self, embeddings: np.ndarray) -> tuple[int, ...]:
        """Append N rows. ``embeddings`` shape ``(N, dim)``."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(
                f"expected shape (N, {self._dim}), got {embeddings.shape}"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.shape[0] == 0:
            return ()
        for i in range(embeddings.shape[0]):
            self._assert_normalized(embeddings[i])

        rows: list[int] = []
        with self._lock:
            # Use freelist first, then append at tail.
            n = embeddings.shape[0]
            consumed = 0
            while self._state.freelist and consumed < n:
                row = int(self._state.freelist.pop())
                self._mm[row] = embeddings[consumed]
                rows.append(row)
                consumed += 1
            remaining = n - consumed
            if remaining:
                while self._state.num_rows + remaining > self._state.capacity:
                    self._grow(min_extra=remaining)
                start = int(self._state.num_rows)
                self._mm[start:start + remaining] = embeddings[consumed:]
                rows.extend(range(start, start + remaining))
                self._state.num_rows += remaining
            self._mm.flush()
            self._save_index()
        return tuple(rows)

    def free_row(self, row: int) -> None:
        """Mark ``row`` free: zero out its data + push onto the freelist
        for reuse on next append. Idempotent — freeing an already-free
        row is a no-op."""
        with self._lock:
            if not 0 <= row < self._state.num_rows:
                raise IndexError(
                    f"row {row} out of range [0, {self._state.num_rows})"
                )
            if row in self._state.freelist:
                return
            self._mm[row] = 0.0
            self._state.freelist.append(row)
            self._mm.flush()
            self._save_index()

    # ------------------------------------------------------------ read path

    def get(self, row: int) -> np.ndarray:
        """Return one row. Caller gets a numpy view, no copy."""
        with self._lock:
            if not 0 <= row < self._state.num_rows:
                raise IndexError(
                    f"row {row} out of range [0, {self._state.num_rows})"
                )
            return self._mm[row]

    def get_batch(self, rows: list[int]) -> np.ndarray:
        """Gather multiple rows into one ``(len(rows), dim)`` array.
        Empty input returns an empty ``(0, dim)`` array."""
        with self._lock:
            if not rows:
                return np.empty((0, self._dim), dtype=np.float32)
            arr = np.empty((len(rows), self._dim), dtype=np.float32)
            for i, r in enumerate(rows):
                if not 0 <= r < self._state.num_rows:
                    raise IndexError(
                        f"row {r} out of range [0, {self._state.num_rows})"
                    )
                arr[i] = self._mm[r]
            return arr

    def search_dense(
        self,
        query_embedding: np.ndarray,
        k: int,
        chunk_ids_in_scope: Optional[tuple[int, ...]] = None,
    ) -> tuple[Hit, ...]:
        """Top-K cosine hits via brute-force matmul.

        ``chunk_ids_in_scope`` here is misleadingly named in the
        Protocol (it predates the chunk/embedding-row decoupling) — at
        the embed-store layer this is a tuple of *embedding row* indices
        to restrict to, not chunk IDs. The searcher resolves chunk IDs
        → row indices upstream. Treating it as row IDs preserves the
        Phase 2.0 contract while leaving the Protocol name stable.
        """
        if k <= 0:
            return ()
        if query_embedding.shape != (self._dim,):
            raise ValueError(
                f"expected query shape ({self._dim},), got {query_embedding.shape}"
            )

        with self._lock:
            n = self._state.num_rows
            if n == 0:
                return ()

            q = query_embedding.astype(np.float32, copy=False)
            # Read once into a numpy array; matmul against the live
            # memmap region. Slicing in numpy is a view, not a copy.
            view = np.asarray(self._mm[:n])
            scores = view @ q   # (n,)

            # Mask freelist (zeroed rows score 0; we don't want them).
            if self._state.freelist:
                mask = np.ones(n, dtype=bool)
                for fr in self._state.freelist:
                    if fr < n:
                        mask[fr] = False
                scores = np.where(mask, scores, -np.inf)

            if chunk_ids_in_scope is not None:
                allowed = np.zeros(n, dtype=bool)
                for r in chunk_ids_in_scope:
                    if 0 <= r < n:
                        allowed[r] = True
                scores = np.where(allowed, scores, -np.inf)

            k_eff = min(k, n)
            if k_eff <= 0:
                return ()
            # argpartition for top-K; full sort just on the K slice.
            top_idx = np.argpartition(-scores, kth=min(k_eff - 1, n - 1))[:k_eff]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            hits = []
            for i in top_idx:
                s = float(scores[i])
                if not np.isfinite(s):
                    continue
                hits.append(Hit(chunk_id=int(i), score=s))
            return tuple(hits)

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Flush memmap + save index. Idempotent."""
        with self._lock:
            if self._mm is None:
                return
            try:
                self._mm.flush()
            except (BufferError, ValueError):
                pass
            self._save_index()
            # Releasing the memmap is best-effort — numpy doesn't expose
            # an explicit close; we drop our reference and let GC + the
            # OS reclaim. Tests that need to reopen the file should
            # ``del store`` before reopening on Windows.
            self._mm = None    # type: ignore[assignment]

    # ----------------------------------------------------- create / open

    def _create_new(self, initial_rows: int) -> None:
        """Fresh-store path: allocate memmap + write index sidecar."""
        self._mm = np.memmap(
            self._npy_path, dtype=np.float32, mode="w+",
            shape=(initial_rows, self._dim),
        )
        self._mm[:] = 0.0
        self._mm.flush()
        _try_chmod(self._npy_path, EMBED_FILE_MODE)

        self._state = _EmbedIndex(
            num_rows=0,
            dim=self._dim,
            model_name=self._model_name,
            model_sha256=self._model_sha256,
            freelist=[],
            capacity=initial_rows,
        )
        self._save_index()

    def _open_existing(self, on_mismatch: str) -> None:
        """Reopen path: validate identity + remap onto the existing
        capacity."""
        with open(self._idx_path, "rb") as fh:
            state: _EmbedIndex = pickle.load(fh)

        if state.dim != self._dim:
            raise ValueError(
                f"existing memmap dim {state.dim} != requested {self._dim}"
            )
        if state.model_name != self._model_name:
            raise EmbedderMismatchError(
                f"existing model_name '{state.model_name}' != "
                f"'{self._model_name}'"
            )
        if state.model_sha256 != self._model_sha256:
            msg = (
                f"embedder sha256 changed: stored "
                f"'{state.model_sha256[:12]}…' vs supplied "
                f"'{self._model_sha256[:12]}…'"
            )
            if on_mismatch == "raise":
                raise EmbedderMismatchError(msg)
            _LOG.warning("%s; continuing per on_mismatch='warn'", msg)
            state.model_sha256 = self._model_sha256

        # ``mode='r+'`` opens for read/write on the existing file
        # without truncating. Capacity is the file's current row count.
        self._mm = np.memmap(
            self._npy_path, dtype=np.float32, mode="r+",
            shape=(state.capacity, self._dim),
        )
        self._state = state
        self._save_index()   # persist any sha256 normalization above

    # --------------------------------------------------------- internal

    def _grow(self, min_extra: int = 0) -> None:
        """Grow the memmap in place by ``growth_factor`` (or enough to
        accommodate ``min_extra`` new rows, whichever is larger)."""
        old_cap = self._state.capacity
        new_cap = max(int(old_cap * self._growth_factor) + 1,
                      old_cap + max(min_extra, 1))

        # numpy doesn't let us resize a memmap in place reliably across
        # platforms — the safe pattern is: flush, drop the mapping,
        # extend the file, remap. Saved data survives.
        self._mm.flush()
        self._mm = None    # type: ignore[assignment]
        with open(self._npy_path, "r+b") as fh:
            fh.seek(new_cap * self._dim * 4 - 1)   # float32 = 4 bytes
            fh.write(b"\0")
        self._mm = np.memmap(
            self._npy_path, dtype=np.float32, mode="r+",
            shape=(new_cap, self._dim),
        )
        # Zero the new tail (the truncate-extend on POSIX gives zeros
        # but we be explicit for the rare filesystems that don't).
        self._mm[old_cap:new_cap] = 0.0
        self._mm.flush()
        self._state.capacity = new_cap

    def _save_index(self) -> None:
        """Atomic pickle write of the sidecar."""
        tmp = self._idx_path.with_suffix(self._idx_path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self._state, fh)
        os.replace(tmp, self._idx_path)
        _try_chmod(self._idx_path, EMBED_FILE_MODE)

    def _assert_normalized(self, emb: np.ndarray) -> None:
        """Catch the "forgot to normalize" bug at write time. Off in
        production-style ``validate=False`` builds."""
        if not self._validate:
            return
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            raise AssertionError("zero-norm embedding rejected")
        # ``np.testing.assert_allclose`` semantics, inlined to avoid
        # the full message machinery on the hot path.
        diff = np.abs(emb - (emb / norm))
        if np.max(diff) > NORM_ATOL:
            raise AssertionError(
                f"embedding not L2-normalized (||x||={norm:.6f}, "
                f"max delta={float(np.max(diff)):.2e})"
            )
