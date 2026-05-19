"""Pack registry — declare pre-built (SqliteChunkStore, MemmapEmbedStore)
artifacts the daemon can mount as a project.

A "pack" is a directory produced by an offline ingest (e.g.
``corpora/swezero-12m/ingest.py``) containing:

  <pack_dir>/
    chunks.sqlite         (+ bm25 sidecar)
    embeds/
      <safe_name>.npy
      <safe_name>.idx

The pack is a self-contained artifact: it can be tarball'd, downloaded
from a CDN, dropped on disk, and registered with ``longctx pack add``
without re-running ingest.

v1 is intentionally a registry + on-disk pointer file (no daemon
mounting yet). The Searcher consumes packs via direct
``Pack.open_stores()`` in v1; the daemon learns to mount the registry
on startup in v2. The Mini-LLM controller (PRD #1) will add auto-pick
later. v1 keeps it manual.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_REGISTRY = Path("~/.longctx/packs.json").expanduser()
DEFAULT_PACK_ROOT = Path("~/.cache/longctx/corpora").expanduser()

# Heuristic: pack names starting with ``swezero-`` default to
# dedup_by_doc_root=True because their ingest pattern writes
# rel_path = ``{repo}/{instance}.HHHHHHHHHHHH`` with N near-duplicate
# chunks per (repo, instance) pair.
_AUTO_DEDUP_PREFIXES: tuple[str, ...] = ("swezero-",)


@dataclass(frozen=True)
class Pack:
    """One registered pack. Stored verbatim in the registry JSON file."""
    name: str
    path: str                  # absolute path to the pack dir
    embedder_model: str        # HF id, e.g. ``BAAI/bge-small-en-v1.5``
    embedder_sha256: str       # SHA256 of model state dict (validates compat)
    embed_dim: int             # dim of the memmap embeddings
    chunk_count: int           # COUNT(*) from chunks.sqlite at registration
    size_bytes: int            # du -sb of the pack dir at registration
    dedup_by_doc_root: bool    # applied at search time
    registered_at: float       # unix epoch
    last_used_at: float = 0.0  # updated by open_stores; 0 = never opened

    def open_stores(self):
        """Return (SqliteChunkStore, MemmapEmbedStore) for this pack.

        Imported lazily so this module stays cheap to import (no
        sentence_transformers etc. pulled in at import time).
        """
        from longctx_daemon.storage.memmap_store import MemmapEmbedStore
        from longctx_daemon.storage.sqlite_store import SqliteChunkStore

        pack_dir = Path(self.path)
        chunk_store = SqliteChunkStore(pack_dir / "chunks.sqlite")
        embed_store = MemmapEmbedStore(
            pack_dir / "embeds",
            model_name=self.embedder_model,
            model_sha256=self.embedder_sha256,
            dim=self.embed_dim,
        )
        return chunk_store, embed_store


# --------------------------------------------------------------- registry

class PackRegistry:
    """File-backed registry at ``~/.longctx/packs.json`` (override via
    LONGCTX_PACKS_FILE).

    Atomic writes via tmp+rename so a crash mid-write doesn't corrupt
    the registry. Per-pack stats are snapshotted at registration; call
    ``refresh(name)`` to re-sample after the pack changes on disk.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        env = os.environ.get("LONGCTX_PACKS_FILE")
        self._path = Path(env) if env else (path or DEFAULT_REGISTRY)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._packs: dict[str, Pack] = self._load()

    def _load(self) -> dict[str, Pack]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        out: dict[str, Pack] = {}
        for entry in raw.get("packs", []):
            try:
                p = Pack(**entry)
            except TypeError:
                # Skip entries with unknown fields rather than crashing
                # on a forward-compat registry from a newer version.
                continue
            out[p.name] = p
        return out

    def _save(self) -> None:
        payload = {
            "version": 1,
            "packs": [asdict(p) for p in sorted(
                self._packs.values(), key=lambda x: x.name
            )],
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self._path)

    # ------------------------------------------------------------- ops

    def list(self) -> tuple[Pack, ...]:
        return tuple(sorted(self._packs.values(), key=lambda p: p.name))

    def get(self, name: str) -> Optional[Pack]:
        return self._packs.get(name)

    def register(
        self,
        path: Path,
        *,
        name: Optional[str] = None,
        dedup_by_doc_root: Optional[bool] = None,
    ) -> Pack:
        """Register a pack at ``path``. Idempotent on (name, path) — a
        second call updates the snapshot in place.

        ``name`` defaults to the directory's basename when omitted.
        ``dedup_by_doc_root`` defaults to True for known patterns (see
        ``_AUTO_DEDUP_PREFIXES``), else False.
        """
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"pack path not a directory: {path}")

        chunks_sqlite = path / "chunks.sqlite"
        embeds_dir = path / "embeds"
        if not chunks_sqlite.exists():
            raise ValueError(
                f"pack {path} missing chunks.sqlite; not a valid pack"
            )
        if not embeds_dir.is_dir():
            raise ValueError(
                f"pack {path} missing embeds/ dir; not a valid pack"
            )

        # Pick name. Priority: explicit kwarg > <corpus>-<lang> when the
        # path matches ``.../corpora/<corpus>/<lang>`` shape (e.g.
        # ``.../corpora/swezero-12m/python`` → ``swezero-12m-python``,
        # which then matches the swezero- auto-dedup heuristic) >
        # directory basename.
        if name:
            resolved_name = name
        elif (
            len(path.parts) >= 3
            and path.parts[-3] == "corpora"
        ):
            corpus = path.parts[-2]
            lang = path.parts[-1]
            resolved_name = f"{corpus}-{lang}"
        else:
            resolved_name = path.name
        if not re.match(r"^[a-z0-9][a-z0-9_\-\.]*$", resolved_name):
            raise ValueError(
                f"pack name {resolved_name!r} must match "
                f"[a-z0-9][a-z0-9_\\-\\.]* (got from --name or auto-derive)"
            )

        # Snapshot: chunk count, embedder identity, size, embed dim
        try:
            with sqlite3.connect(str(chunks_sqlite)) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
                # Pull embedder identity from any chunk (all chunks share
                # one embedder per pack by design)
                row = conn.execute(
                    "SELECT embedder_model, embedder_sha256 "
                    "FROM chunks WHERE embedder_model IS NOT NULL LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:  # noqa: BLE001
            raise ValueError(f"could not read pack sqlite: {exc}") from exc
        if not row:
            raise ValueError(
                f"pack {path} has 0 chunks with embedder_model set; "
                "not a valid pack"
            )
        embedder_model, embedder_sha = row

        # Embed dim: read it from the .idx sidecar without loading the
        # memmap into RAM. The idx file's first 64 bytes are header.
        idx_files = list(embeds_dir.glob("*.idx"))
        if not idx_files:
            raise ValueError(f"pack {path} has no embeds/*.idx file")
        dim = _read_embed_dim(idx_files[0])

        size = _du_bytes(path)

        if dedup_by_doc_root is None:
            dedup_by_doc_root = any(
                resolved_name.startswith(prefix)
                for prefix in _AUTO_DEDUP_PREFIXES
            )

        now = time.time()
        prev = self._packs.get(resolved_name)
        registered_at = prev.registered_at if prev else now
        last_used_at = prev.last_used_at if prev else 0.0

        pack = Pack(
            name=resolved_name,
            path=str(path),
            embedder_model=embedder_model,
            embedder_sha256=embedder_sha,
            embed_dim=dim,
            chunk_count=count,
            size_bytes=size,
            dedup_by_doc_root=dedup_by_doc_root,
            registered_at=registered_at,
            last_used_at=last_used_at,
        )
        self._packs[resolved_name] = pack
        self._save()
        return pack

    def unregister(self, name: str) -> bool:
        """Remove from the registry. Does NOT delete on-disk artifacts.
        Returns True if the pack existed, False if it didn't."""
        if name not in self._packs:
            return False
        del self._packs[name]
        self._save()
        return True

    def refresh(self, name: str) -> Optional[Pack]:
        """Re-snapshot stats (chunk_count, size_bytes) for an
        already-registered pack. Returns None if not registered."""
        p = self._packs.get(name)
        if p is None:
            return None
        return self.register(
            Path(p.path), name=name, dedup_by_doc_root=p.dedup_by_doc_root,
        )

    def mark_used(self, name: str) -> None:
        """Update last_used_at to now. Called by daemon/search code
        when the pack's stores are opened."""
        p = self._packs.get(name)
        if p is None:
            return
        self._packs[name] = Pack(**{**asdict(p), "last_used_at": time.time()})
        self._save()


# --------------------------------------------------------------- helpers

def _du_bytes(path: Path) -> int:
    """Total byte size under ``path`` (recursive). Tolerant of broken
    symlinks etc. — they're skipped, never raise."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _read_embed_dim(idx_path: Path) -> int:
    """Read embed_dim from the MemmapEmbedStore .idx header (pickled
    ``_EmbedIndex`` dataclass — see
    ``longctx_daemon.storage.memmap_store``)."""
    import pickle
    try:
        with idx_path.open("rb") as f:
            state = pickle.load(f)
        return int(state.dim)
    except (pickle.UnpicklingError, OSError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"could not read embed dim from {idx_path}: {exc}"
        ) from exc
