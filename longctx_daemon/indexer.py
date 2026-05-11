"""Indexer for the longctx daemon (Phase 2.0).

Owns the project lifecycle: registering projects, walking their file
trees, chunking + embedding eligible files, and applying incremental
updates as the watcher observes changes.

Talks to storage via the ``ChunkStore`` + ``EmbedStore`` Protocols only
— no concrete backend coupling here. Reuses
``longctx.rag.chunker.Chunker`` for tokenization-aware splits so the
daemon and the Phase 1 RAG pipeline produce compatible chunks.

Covers the filter chain, the storage schema, and the incremental
update protocol (content-addressed chunk reuse).
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from longctx.rag.chunker import Chunker
from longctx.rag.coarse_filter import Chunk as RawChunk
from longctx_daemon.storage.protocol import ChunkStore, EmbedStore
from longctx_daemon.types import Chunk, FileRecord, Project

logger = logging.getLogger(__name__)


# ------------------------------------------------------------ exceptions

class ForbiddenProjectError(ValueError):
    """Raised when ``add_project`` is called with a root whose basename
    matches ``IndexerConfig.forbidden_dirs``. The whole directory is off
    limits — distinct from per-file ``secret_patterns``."""


class EmbedderMismatchError(RuntimeError):
    """Raised on operations that would mix embeddings from a different
    embedder file. Caller decides whether to honor (e.g., refuse to
    serve) or run ``longctx reembed --confirm`` to rebuild."""


# ------------------------------------------------------------ config + result

# Default exclude/include patterns. Kept as module-level constants
# so callers can extend rather than redefine.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "**/node_modules", "**/.git/objects", "**/__pycache__",
    "**/.venv", "**/venv", "**/.pytest_cache",
    "**/target", "**/build", "**/dist", "**/.next", "**/.svelte-kit",
    "**/*.lock", "**/*.min.js", "**/*.map",
)

DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env", ".env.*", "*.key", "*.pem", "id_rsa*", "id_ed25519*",
    "*.p12", "*.pfx", "**/secrets/**", "**/credentials/**",
    "**/.aws/**", "**/.ssh/**", "**/.gnupg/**",
)

DEFAULT_FORBIDDEN_DIRS: frozenset[str] = frozenset({
    "secrets", "credentials", ".aws", ".ssh", ".gnupg",
})

# Extension allowlist that boosts text-detection. Extensions outside the
# list still pass via the null-byte sniff; the list is just an early
# accept that avoids a stat for the obvious cases.
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".swift", ".py", ".pyi", ".cpp", ".cc", ".c", ".h", ".hpp", ".hh",
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala", ".rb", ".php", ".lua", ".sh", ".bash",
    ".zsh", ".fish", ".html", ".htm", ".xml", ".css", ".scss", ".sass",
    ".less", ".sql", ".graphql", ".proto", ".tex", ".bib", ".m", ".mm",
    ".swift", ".dart", ".vue", ".svelte", ".astro", ".elm", ".clj",
    ".cljs", ".ex", ".exs", ".erl", ".hs", ".ml", ".mli", ".fs",
    ".cs", ".vb", ".r", ".jl", ".nim", ".zig", ".v", ".sv", ".vhd",
    ".csv", ".tsv", ".log",
})


@dataclass(frozen=True)
class IndexerConfig:
    """Config for the indexer's filter chain + batching.

    Mirrors the user-facing config keys but in already-parsed form
    (tuples/frozensets, no Path resolution).
    """
    max_file_size_bytes: int = 5 * 1024 * 1024
    include_extensions: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    forbidden_dirs: frozenset[str] = DEFAULT_FORBIDDEN_DIRS
    respect_gitignore: bool = True
    embed_batch_size: int = 64
    embedder_max_seq_length: Optional[int] = None


@dataclass(frozen=True)
class ScanResult:
    """Counts emitted by ``Indexer.full_scan``. Used by the CLI status
    output and the per-call MCP trace (PRD §14.4)."""
    n_files: int
    n_chunks_total: int
    n_chunks_new: int
    n_chunks_reused: int
    n_files_skipped: int
    wall_secs: float


@dataclass(frozen=True)
class UpdateResult:
    """Counts emitted by ``Indexer.update_file``. Same shape as
    ``ScanResult`` so per-call traces don't have to branch."""
    n_files: int
    n_chunks_total: int
    n_chunks_new: int
    n_chunks_reused: int
    n_files_skipped: int
    wall_secs: float


# ------------------------------------------------------------ filter chain

class _GitignoreMatcher:
    """Wraps a ``pathspec`` PathSpec keyed at the project root. Loads
    ``.gitignore`` lazily on first use; misses are cheap (returns False).

    Kept private — callers go through ``should_index``."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._spec = None
        self._loaded = False

    def is_ignored(self, rel_path: str) -> bool:
        if not self._loaded:
            self._load()
        if self._spec is None:
            return False
        # pathspec expects forward-slash paths
        return self._spec.match_file(rel_path.replace(os.sep, "/"))

    def _load(self) -> None:
        self._loaded = True
        gi = self.project_root / ".gitignore"
        if not gi.is_file():
            return
        try:
            import pathspec
        except ImportError:
            logger.warning(
                "pathspec not installed; gitignore matching disabled"
            )
            return
        try:
            with gi.open("r", encoding="utf-8", errors="replace") as f:
                self._spec = pathspec.PathSpec.from_lines(
                    "gitignore", f.readlines()
                )
        except OSError as exc:
            logger.warning("failed to load .gitignore at %s: %s", gi, exc)


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    """Return True if ``rel_path`` matches any glob in ``patterns``.

    Uses ``pathspec`` (gitignore-style matching) so ``**`` works for
    deep wildcards. Patterns are interpreted in gitwildmatch mode —
    the same dialect users edit in ``.gitignore``, so the secret /
    exclude lists in config behave exactly like users expect.

    Patterns without a slash also match by basename (so ``.env``
    catches both ``.env`` and ``app/.env``).
    """
    pats = list(patterns)
    if not pats:
        return False
    p = rel_path.replace(os.sep, "/")
    name = p.rsplit("/", 1)[-1]
    try:
        import pathspec
    except ImportError:
        # Fallback to fnmatch-only if pathspec isn't installed. This
        # path loses ``**`` support but keeps the indexer running on
        # systems without the dep.
        from fnmatch import fnmatch
        for pat in pats:
            if fnmatch(p, pat) or fnmatch(name, pat):
                return True
        return False
    spec = pathspec.PathSpec.from_lines("gitignore", pats)
    if spec.match_file(p):
        return True
    # Basename pass: gitwildmatch's "no slash" rule already matches
    # basenames, but we make it explicit for the case where the path
    # is just a filename ("foo.lock") and the pattern is "**/*.lock".
    return spec.match_file(name)


def _looks_binary(path: Path) -> bool:
    """Sniff the first 8KB for null bytes / decode failures.

    Cheap, correct enough — full UTF-8 decode would be wasteful on
    real-world repos where the long tail is binary blobs (PNG / PDF /
    .lockb)."""
    try:
        with path.open("rb") as f:
            head = f.read(8192)
    except OSError:
        return True
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # Real-world note: many code files are utf-8; latin-1 prose is
        # rare in repos. We err on "skip" for the rare case rather than
        # try multiple codecs.
        return True
    return False


def should_index(
    path: Path,
    project_root: Path,
    config: IndexerConfig,
    gitignore_matcher: Optional[_GitignoreMatcher] = None,
) -> bool:
    """Apply the cheap-to-expensive filter chain from PRD §5.2.

    Filters in order:
      1. secret_patterns (always-excluded)
      2. exclude_patterns (user deny-list)
      3. .gitignore (if respect_gitignore)
      4. include_extensions allowlist (if non-empty)
      5. size cap
      6. binary detection (content sniff)

    Caller is responsible for ``forbidden_dirs`` at project-add time —
    we don't double-check here.
    """
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        return False
    rel_posix = rel.as_posix()

    if _matches_any(rel_posix, config.secret_patterns):
        return False
    if _matches_any(rel_posix, config.exclude_patterns):
        return False

    if config.respect_gitignore and gitignore_matcher is not None:
        if gitignore_matcher.is_ignored(rel_posix):
            return False

    if config.include_extensions:
        if path.suffix.lower() not in {
            e.lower() for e in config.include_extensions
        }:
            return False

    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > config.max_file_size_bytes:
        return False
    if size == 0:
        # Empty files have no chunks; skip cheaply.
        return False

    # Extension allowlist boost: skip the binary sniff for known-text
    # extensions. Saves a 8KB read on the common case.
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    if _looks_binary(path):
        return False

    return True


# ------------------------------------------------------------ embedder ID

def _embedder_sha256(embedder) -> str:  # noqa: ANN001 — duck-typed
    """Hash a stable file inside the embedder's local model dir.

    Walks SentenceTransformer's internals to find the underlying HF
    model path, then SHA-256s the first weights file we recognize
    (``model.safetensors`` preferred, ``pytorch_model.bin`` fallback).
    Returns ``f"unhashable:{model_name}"`` on failure so the indexer
    still has a deterministic identity to compare against.

    Test escape hatch: if the embedder exposes a ``fake_sha`` attribute
    (e.g., the test double in ``tests/daemon/conftest.py``), use that
    directly. This keeps the production path honest while letting
    tests flip the SHA to exercise the mismatch path.
    """
    fake = getattr(embedder, "fake_sha", None)
    if fake is not None:
        return str(fake)
    model_name = "unknown"
    model_path: Optional[Path] = None
    try:
        # SentenceTransformer wraps an HF Transformer at index 0.
        first = embedder._modules["0"]  # noqa: SLF001
        cfg = first.auto_model.config
        model_name = getattr(cfg, "_name_or_path", "unknown")
        if model_name and Path(model_name).is_dir():
            model_path = Path(model_name)
    except (AttributeError, KeyError, TypeError):
        pass

    # Fallback: query the HF cache by name. This handles the common
    # case where _name_or_path is the HF id (e.g.
    # ``BAAI/bge-small-en-v1.5``) and the actual files live in the
    # local huggingface cache.
    if model_path is None and model_name and "/" in model_name:
        try:
            from huggingface_hub import try_to_load_from_cache  # type: ignore
            for fname in ("model.safetensors", "pytorch_model.bin"):
                hit = try_to_load_from_cache(
                    repo_id=model_name, filename=fname,
                )
                # try_to_load_from_cache returns one of:
                #   - str path (cache hit)
                #   - None (never downloaded)
                #   - _CACHED_NO_EXIST sentinel (known-missing remote file)
                # The sentinel is a truthy non-str object; guard with isinstance.
                if isinstance(hit, str) and Path(hit).is_file():
                    return _sha256_file(Path(hit))
        except ImportError:
            pass

    if model_path is not None:
        for fname in ("model.safetensors", "pytorch_model.bin"):
            cand = model_path / fname
            if cand.is_file():
                return _sha256_file(cand)

    logger.warning(
        "embedder model file not found; using deterministic fallback "
        "identity for %s", model_name,
    )
    return f"unhashable:{model_name}"


def _sha256_file(path: Path) -> str:
    """Stream a file through SHA-256. 1 MiB chunks — kind to memory on
    multi-GB weights."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _embedder_model_name(embedder) -> str:  # noqa: ANN001
    """Best-effort model id for the ``Chunk.embedder_model`` field."""
    direct = getattr(embedder, "model_name", None)
    if isinstance(direct, str) and direct:
        return direct
    try:
        cfg = embedder._modules["0"].auto_model.config  # noqa: SLF001
        name = getattr(cfg, "_name_or_path", None)
        if name:
            return str(name)
    except (AttributeError, KeyError, TypeError):
        pass
    return "unknown"


# ------------------------------------------------------------ Indexer

class Indexer:
    """Drives full + incremental indexing for the daemon.

    Composition:
      * ``chunk_store`` / ``embed_store`` are storage Protocols. The
        indexer never touches a concrete backend — the daemon picks
        SQLite + memmap, tests pass fakes.
      * ``embedder`` is a sentence-transformers ``SentenceTransformer``
        (or any duck-typed object with an ``encode`` method that takes
        ``convert_to_numpy=True`` and ``normalize_embeddings=True``).
      * ``chunker`` is ``longctx.rag.chunker.Chunker`` — same chunker the
        Phase 1 pipeline uses.

    The indexer caches the embedder's content hash on construction so
    subsequent operations can fail fast on a model swap (PRD §4.4).
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        embed_store: EmbedStore,
        embedder,  # noqa: ANN001 — duck-typed SentenceTransformer
        chunker: Chunker,
        config: IndexerConfig,
    ) -> None:
        self._store = chunk_store
        self._embed = embed_store
        self._embedder = embedder
        self._chunker = chunker
        self._config = config
        self._embedder_sha = _embedder_sha256(embedder)
        self._embedder_model = _embedder_model_name(embedder)
        if config.embedder_max_seq_length is not None:
            try:
                embedder.max_seq_length = config.embedder_max_seq_length
            except (AttributeError, TypeError):
                # Embedders that don't expose a seq-length knob
                # (custom impls) — silently skip.
                pass

    # --------------------------------------------------------- public API

    def add_project(self, name: str, root_path: Path) -> Project:
        """Register or update a project; does not walk it.

        Raises ``ForbiddenProjectError`` if the root's basename matches
        a forbidden_dirs entry — guards against indexing
        ``~/dev/secrets/`` even if the user passes it explicitly.
        """
        root = Path(root_path).expanduser().resolve()
        if root.name in self._config.forbidden_dirs:
            raise ForbiddenProjectError(
                f"refusing to index forbidden_dirs root: {root}"
            )
        if not root.is_dir():
            raise FileNotFoundError(f"project root not a directory: {root}")
        existing = self._store.get_project(name)
        # Preserve last_full_scan_at on rename / re-add so we don't
        # lose freshness metadata.
        last_scan = existing.last_full_scan_at if existing is not None else 0
        project = Project(
            name=name,
            root_path=str(root),
            last_full_scan_at=last_scan,
            session_id=existing.session_id if existing is not None else None,
        )
        self._store.upsert_project(project)
        return project

    def full_scan(self, project: str) -> ScanResult:
        """Walk the project, chunk + embed every file passing the
        filter chain, persist to the stores."""
        self._assert_embedder_consistent()
        proj = self._store.get_project(project)
        if proj is None:
            raise KeyError(f"unknown project: {project}")
        root = Path(proj.root_path)
        gitignore = _GitignoreMatcher(root)

        t0 = time.perf_counter()
        n_files = 0
        n_chunks_total = 0
        n_chunks_new = 0
        n_chunks_reused = 0
        n_files_skipped = 0
        for path in self._walk(root):
            if not should_index(path, root, self._config, gitignore):
                n_files_skipped += 1
                continue
            res = self._index_file(proj.name, root, path)
            if res is None:
                n_files_skipped += 1
                continue
            n_files += 1
            n_chunks_total += res[0]
            n_chunks_new += res[1]
            n_chunks_reused += res[2]
            # Drain MPS / CUDA allocator caches every 500 files so RSS
            # stays bounded across million-LOC corpora. The allocator
            # holds released blocks in a process-local pool by default;
            # this returns them to the OS. Sub-second cost per drain.
            if n_files % 500 == 0:
                self._drain_allocator_cache()

        # Update last_full_scan_at; preserve other fields.
        self._store.upsert_project(Project(
            name=proj.name,
            root_path=proj.root_path,
            last_full_scan_at=int(time.time()),
            session_id=proj.session_id,
        ))
        return ScanResult(
            n_files=n_files,
            n_chunks_total=n_chunks_total,
            n_chunks_new=n_chunks_new,
            n_chunks_reused=n_chunks_reused,
            n_files_skipped=n_files_skipped,
            wall_secs=time.perf_counter() - t0,
        )

    def update_file(self, project: str, abs_path: Path) -> UpdateResult:
        """Incrementally update one file. No-op if content_hash matches
        the stored record. Otherwise re-chunks, diffs by chunk
        content_hash, embeds only new chunks."""
        self._assert_embedder_consistent()
        proj = self._store.get_project(project)
        if proj is None:
            raise KeyError(f"unknown project: {project}")
        root = Path(proj.root_path)
        path = Path(abs_path).resolve()

        t0 = time.perf_counter()
        gitignore = _GitignoreMatcher(root)
        if not should_index(path, root, self._config, gitignore):
            return UpdateResult(
                n_files=0, n_chunks_total=0,
                n_chunks_new=0, n_chunks_reused=0,
                n_files_skipped=1,
                wall_secs=time.perf_counter() - t0,
            )
        res = self._index_file(proj.name, root, path)
        if res is None:
            return UpdateResult(
                n_files=0, n_chunks_total=0,
                n_chunks_new=0, n_chunks_reused=0,
                n_files_skipped=1,
                wall_secs=time.perf_counter() - t0,
            )
        return UpdateResult(
            n_files=1,
            n_chunks_total=res[0],
            n_chunks_new=res[1],
            n_chunks_reused=res[2],
            n_files_skipped=0,
            wall_secs=time.perf_counter() - t0,
        )

    def delete_file(self, project: str, rel_path: str) -> None:
        """Drop a file's chunks and free their embedding rows.

        Cascade is handled by the store (chunks ON DELETE CASCADE) +
        ``free_row`` on each freed embedding. We just look up the file
        record."""
        rec = self._store.get_file(project, rel_path)
        if rec is None:
            return
        for chunk in self._store.get_chunks_by_file(rec.id):
            if chunk.embedding_row is not None:
                self._embed.free_row(chunk.embedding_row)
        self._store.delete_file(rec.id)

    def delete_project(self, name: str) -> None:
        """Drop a whole project. Frees every embedding row first to
        keep the memmap freelist tidy."""
        for rec in self._store.list_files(project=name):
            for chunk in self._store.get_chunks_by_file(rec.id):
                if chunk.embedding_row is not None:
                    self._embed.free_row(chunk.embedding_row)
        self._store.delete_project(name)

    def rename_file(
        self,
        project: str,
        old_rel_path: str,
        new_rel_path: str,
    ) -> bool:
        """Cheap rename: update the file's ``rel_path`` without re-embedding.

        Phase 2.2 path used by the watcher when the debounce coalesces a
        DELETE+CREATE on different basenames into a single RENAME (PRD
        §5.4). The chunk text is unchanged, so we want to avoid a full
        re-embed pass.

        Returns ``True`` when the cheap path applied (file row updated
        in place, embeddings preserved). Returns ``False`` when:

          * The old record doesn't exist (already deleted / never indexed).
          * The new path doesn't pass the filter chain (e.g. moved into
            ``node_modules``) — we drop the old record cleanly.
          * The backend doesn't support an in-place ``rename_file`` and
            the fallback "delete + reindex" applied — embeddings are
            lost in that path; logged at INFO so the operator sees it.

        Note for storage backend authors: implementing
        ``ChunkStore.rename_file(file_id, new_rel_path)`` makes this
        path zero-cost (one SQL UPDATE). Without it we degrade to
        delete + reindex which loses cached embeddings but still
        preserves correctness.
        """
        proj = self._store.get_project(project)
        if proj is None:
            return False
        existing = self._store.get_file(project, old_rel_path)
        if existing is None:
            return False
        root = Path(proj.root_path)
        new_abs = (root / new_rel_path).resolve()

        # Filter the destination first — if the new path violates
        # secret_patterns or exclude_patterns, we just drop the old row.
        gitignore = _GitignoreMatcher(root)
        if not should_index(new_abs, root, self._config, gitignore):
            self.delete_file(project, old_rel_path)
            return False

        # Cheap path: backend exposes a rename hook.
        backend_rename = getattr(self._store, "rename_file", None)
        if callable(backend_rename):
            try:
                # Stat the new path so the file row's mtime stays honest.
                try:
                    mtime = int(new_abs.stat().st_mtime)
                except OSError:
                    mtime = existing.mtime
                backend_rename(existing.id, new_rel_path, mtime=mtime)
                return True
            except Exception:  # noqa: BLE001
                logger.exception(
                    "backend rename_file failed for %s/%s -> %s; "
                    "falling back to delete+reindex",
                    project, old_rel_path, new_rel_path,
                )

        # Fallback: drop the old record, reindex under the new path.
        # _index_file's content-hash diff still reuses chunk text but the
        # embedding rows are gone (delete_file freed them above). Logged
        # so operators can spot when they should add the backend hook.
        logger.info(
            "rename_file falling back to delete+reindex for %s/%s -> %s "
            "(backend has no rename_file hook)",
            project, old_rel_path, new_rel_path,
        )
        self.delete_file(project, old_rel_path)
        if new_abs.is_file():
            self._index_file(project, root, new_abs)
        return False

    def update_files(
        self, project: str, abs_paths: Iterable[Path],
    ) -> UpdateResult:
        """Bulk variant of ``update_file`` — used by the watcher worker.

        Today this just iterates ``update_file`` and aggregates the
        counts; the upgrade lever is to share one ``encode`` call across
        all files in the batch (PRD §5.3 promises ~20 chunks across 20
        files = one batched embed). That optimization is left for Phase
        2.3 once the indexer's chunk-diff has been refactored to expose
        a "queue these chunks" entry point separate from the per-file
        commit.

        Aggregate ``wall_secs`` is the sum of per-file wall times — not
        a wall-clock measurement of the bulk pass — so the existing
        per-call trace shape stays intact.
        """
        n_files = 0
        n_total = 0
        n_new = 0
        n_reused = 0
        n_skipped = 0
        wall = 0.0
        for path in abs_paths:
            res = self.update_file(project, path)
            n_files += res.n_files
            n_total += res.n_chunks_total
            n_new += res.n_chunks_new
            n_reused += res.n_chunks_reused
            n_skipped += res.n_files_skipped
            wall += res.wall_secs
        return UpdateResult(
            n_files=n_files,
            n_chunks_total=n_total,
            n_chunks_new=n_new,
            n_chunks_reused=n_reused,
            n_files_skipped=n_skipped,
            wall_secs=wall,
        )

    # ----------------------------------------------------- internals

    def _walk(self, root: Path) -> Iterable[Path]:
        """Walk the project tree and yield candidate file paths.

        Uses ``os.walk`` with in-place ``dirs[:]`` filtering so we never
        descend into known-bad directories — a single root walk over a
        node_modules root would otherwise dominate.
        """
        forbidden_subdirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".pytest_cache", "target", "build", "dist", ".next",
            ".svelte-kit", ".tox", ".mypy_cache", ".ruff_cache",
        }
        for dirpath, dirs, files in os.walk(root):
            # Skip nested forbidden_dirs (defense in depth — project
            # root is checked at add_project time).
            dirs[:] = [
                d for d in dirs
                if d not in forbidden_subdirs
                and d not in self._config.forbidden_dirs
            ]
            for f in files:
                yield Path(dirpath) / f

    def _index_file(
        self,
        project_name: str,
        root: Path,
        path: Path,
    ) -> Optional[tuple[int, int, int]]:
        """Read, chunk, embed-only-new, persist. Returns
        ``(n_chunks_total, n_chunks_new, n_chunks_reused)`` or None on
        unreadable file or no-op (content unchanged)."""
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("read failed for %s: %s", path, exc)
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Best-effort fallback: replace bad bytes so we still index
            # mixed-encoding files. Hash the raw bytes either way for
            # change detection.
            text = raw.decode("utf-8", errors="replace")

        file_hash = hashlib.sha256(raw).hexdigest()
        rel = path.relative_to(root).as_posix()
        existing = self._store.get_file(project_name, rel)
        if existing is not None and existing.content_hash == file_hash:
            # Content unchanged — refresh mtime only if it diverged;
            # we still report 'reused' so tests can assert no-op.
            chunks = self._store.get_chunks_by_file(existing.id)
            return (len(chunks), 0, len(chunks))

        try:
            stat = path.stat()
        except OSError:
            return None

        # Upsert file record (creates id on first write).
        file_id = self._store.upsert_file(FileRecord(
            id=existing.id if existing is not None else 0,
            project=project_name,
            rel_path=rel,
            mtime=int(stat.st_mtime),
            size_bytes=stat.st_size,
            content_hash=file_hash,
        ))

        # Chunk the new content.
        raw_chunks = self._chunker.chunk(text)
        if not raw_chunks:
            # File is text-but-empty after chunking — drop any old
            # chunks, leave the file record so future updates are
            # detected as no-ops.
            for old in self._store.get_chunks_by_file(file_id):
                if old.embedding_row is not None:
                    self._embed.free_row(old.embedding_row)
            self._store.delete_chunks_by_file(file_id)
            return (0, 0, 0)

        # Build chunk records with line numbers + content hashes.
        new_records, new_hashes = self._build_records(
            file_id=file_id,
            text=text,
            raw_chunks=raw_chunks,
        )

        # Diff against existing chunks by content_hash.
        old_chunks = self._store.get_chunks_by_file(file_id)
        old_by_hash: dict[str, Chunk] = {c.content_hash: c for c in old_chunks}

        kept_rows: dict[str, int] = {}
        for h in new_hashes:
            if h in old_by_hash and old_by_hash[h].embedding_row is not None:
                kept_rows[h] = old_by_hash[h].embedding_row  # type: ignore[assignment]

        # Free rows for chunks that no longer exist in the new set.
        new_hash_set = set(new_hashes)
        for old in old_chunks:
            if old.content_hash not in new_hash_set:
                if old.embedding_row is not None:
                    self._embed.free_row(old.embedding_row)

        # Embed only the chunks whose hash isn't already cached.
        to_embed_idx: list[int] = [
            i for i, h in enumerate(new_hashes) if h not in kept_rows
        ]
        new_rows: dict[int, int] = {}
        if to_embed_idx:
            texts = [new_records[i].text for i in to_embed_idx]
            embeddings = self._encode_batch(texts)
            rows = self._embed.append_batch(embeddings)
            for i, row in zip(to_embed_idx, rows):
                new_rows[i] = row

        # Materialize final Chunk records with embedding_row set.
        final: list[Chunk] = []
        for i, rec in enumerate(new_records):
            if rec.content_hash in kept_rows:
                row = kept_rows[rec.content_hash]
            else:
                row = new_rows[i]
            final.append(Chunk(
                id=rec.id,
                file_id=rec.file_id,
                chunk_index=rec.chunk_index,
                start_offset=rec.start_offset,
                end_offset=rec.end_offset,
                start_line=rec.start_line,
                end_line=rec.end_line,
                token_count=rec.token_count,
                content_hash=rec.content_hash,
                text=rec.text,
                embedder_model=rec.embedder_model,
                embedder_sha256=rec.embedder_sha256,
                embedding_row=row,
            ))

        # Atomic replace: drop the old chunk set first, then upsert
        # the new one. Embedding rows have already been freed (orphans)
        # or carried over (kept_rows) above.
        self._store.delete_chunks_by_file(file_id)
        self._store.upsert_chunks(final)
        n_total = len(final)
        n_new = len(to_embed_idx)
        n_reused = n_total - n_new
        return (n_total, n_new, n_reused)

    def _build_records(
        self,
        file_id: int,
        text: str,
        raw_chunks: list[RawChunk],
    ) -> tuple[list[Chunk], list[str]]:
        """Map ``RawChunk`` (id/text/offsets/token_count) into daemon
        ``Chunk``s with file_id + line numbers + content hashes.

        Embedding rows are filled in by the caller after embedding.
        """
        # Precompute newline offsets once: bisect into this for each
        # chunk's start/end. O(n_lines + n_chunks * log n_lines).
        line_offsets = _line_start_offsets(text)
        records: list[Chunk] = []
        hashes: list[str] = []
        for idx, c in enumerate(raw_chunks):
            chunk_hash = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
            start_line = _offset_to_line(line_offsets, c.start_offset)
            # end_offset is exclusive in the chunker; subtract 1 for
            # line lookup so the line *containing* the last char is
            # reported.
            end_line = _offset_to_line(
                line_offsets, max(c.end_offset - 1, c.start_offset),
            )
            records.append(Chunk(
                id=0,  # store assigns
                file_id=file_id,
                chunk_index=idx,
                start_offset=c.start_offset,
                end_offset=c.end_offset,
                start_line=start_line,
                end_line=end_line,
                token_count=c.token_count,
                content_hash=chunk_hash,
                text=c.text,
                embedder_model=self._embedder_model,
                embedder_sha256=self._embedder_sha,
                embedding_row=None,
            ))
            hashes.append(chunk_hash)
        return records, hashes

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Wrap embedder.encode with the daemon's standard kwargs.

        L2-normalized so dot product == cosine on the dense store.

        Wrapped in ``torch.inference_mode()`` (stronger than the
        internal ``no_grad`` SentenceTransformer applies) so no grad
        graph builds even if a future caller forgets ``model.eval()``.
        Cheap belt-and-suspenders against autograd-cache growth.
        """
        try:
            import torch
            ctx = torch.inference_mode()
        except ImportError:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            out = self._embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self._config.embed_batch_size,
                show_progress_bar=False,
            )
        # Some test doubles return list[list[float]]; coerce to ndarray.
        return np.asarray(out, dtype=np.float32)

    def _drain_allocator_cache(self) -> None:
        """Release cached device memory back to the OS. PyTorch's MPS +
        CUDA allocators keep freed blocks in a per-process pool by
        default; over a long ``full_scan`` this drives RSS up linearly
        even though there is no real Python-level leak. Call every N
        files to bound the working set.
        """
        try:
            import gc, torch
        except ImportError:
            return
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _assert_embedder_consistent(self) -> None:
        """Compare the embedder's current SHA against the most-recent
        chunk's stored SHA. Mismatch raises ``EmbedderMismatchError``.

        First-time daemon (no chunks yet) is always consistent.
        """
        # Cheapest possible probe: list one project's files, take its
        # chunks. We don't want to load every chunk just to check.
        for proj in self._store.list_projects():
            for rec in self._store.list_files(project=proj.name):
                chunks = self._store.get_chunks_by_file(rec.id)
                if not chunks:
                    continue
                stored_sha = chunks[0].embedder_sha256
                if stored_sha != self._embedder_sha:
                    raise EmbedderMismatchError(
                        "embedder model file changed; run `longctx "
                        "reembed --confirm` to rebuild embeddings "
                        f"(stored={stored_sha[:12]}, "
                        f"current={self._embedder_sha[:12]})"
                    )
                return  # one match is enough


# ------------------------------------------------------------ helpers

def _line_start_offsets(text: str) -> list[int]:
    """Return a list of byte offsets where each line begins. The first
    line always starts at 0; subsequent lines start one past each
    newline.

    Used by ``_offset_to_line`` to convert chunk byte offsets into
    1-indexed line numbers via bisect.
    """
    offsets = [0]
    i = text.find("\n")
    while i != -1:
        offsets.append(i + 1)
        i = text.find("\n", i + 1)
    return offsets


def _offset_to_line(line_offsets: list[int], offset: int) -> int:
    """Map a character offset to a 1-indexed line number via bisect."""
    import bisect
    # bisect_right gives count of starts <= offset, which == 1-indexed line
    pos = bisect.bisect_right(line_offsets, offset)
    return max(1, pos)


# TODO(phase 2.1): when the watcher batches events, expose a
# ``update_files`` bulk path that shares one embed call across N files
# (PRD §5.3 batched-embed promise).
# TODO(phase 2.2): hook into the Tier 3 disk-budget check —
# delete_project here doesn't flush memmap freelist; that's the
# eviction loop's job.
