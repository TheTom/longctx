"""Global service state: scope registry + session manager + retrieve pipeline.

In-memory for v0.3.0a1. PRD §4 calls for ~/.longctx/<scope-hash>/ disk
persistence; lands in the next slice.

PRD §5.6: RW-lock per scope. We use threading.RLock() on the registry plus
a per-scope `threading.RLock()` exposed via `lock_for_scope()`.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from longctx_svc.config import get_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.retrieve.pipeline import RetrievePipeline
from longctx_svc.scope.tracker import ScopeClusterTracker
from longctx_svc.session.manager import SessionManager


@dataclass
class ScopeEntry:
    """A scope plus metadata for state-management."""
    scope_root: Path
    scope_hash: str
    sentinel: str
    index: ScopeIndex | None = None
    status: str = "pending"          # pending | indexing | ready | error | evicted | promoting
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    last_query_at: float = field(default_factory=time.time)
    from_disk: bool = False
    watcher: object | None = None
    # PRD §5.2 / v0.3.1 — track which mode we built at + indexed paths so
    # we can detect references that fall outside the Hot scope and
    # auto-promote to Package.
    mode: str = "hot"                # "hot" | "package"
    indexed_files: frozenset[str] = field(default_factory=frozenset)
    lock: threading.RLock = field(default_factory=threading.RLock)


class _State:
    """Singleton service state."""

    def __init__(self) -> None:
        self._registry_lock = threading.RLock()
        # OrderedDict to support LRU eviction by reinsert-on-touch.
        self._scopes: OrderedDict[str, ScopeEntry] = OrderedDict()
        self.sessions = SessionManager()
        self._pipeline: RetrievePipeline | None = None
        # Path-cluster scope trackers, keyed by session_id. Each session
        # gets its own tracker so cumulative path observations don't bleed
        # across unrelated conversations.
        self._cluster_lock = threading.RLock()
        self._cluster_trackers: dict[str, ScopeClusterTracker] = {}

    def cluster_tracker_for(self, session_id: str) -> ScopeClusterTracker:
        """Return (creating if needed) the per-session path-cluster tracker."""
        with self._cluster_lock:
            tracker = self._cluster_trackers.get(session_id)
            if tracker is None:
                tracker = ScopeClusterTracker()
                self._cluster_trackers[session_id] = tracker
            return tracker

    def evict_session_tracker(self, session_id: str) -> None:
        with self._cluster_lock:
            self._cluster_trackers.pop(session_id, None)

    @property
    def pipeline(self) -> RetrievePipeline:
        if self._pipeline is None:
            self._pipeline = RetrievePipeline()
        return self._pipeline

    def set_pipeline(self, pipeline: RetrievePipeline) -> None:
        self._pipeline = pipeline

    def get_scope(self, scope_hash: str) -> ScopeEntry | None:
        with self._registry_lock:
            entry = self._scopes.get(scope_hash)
            if entry is not None:
                # LRU touch
                self._scopes.move_to_end(scope_hash)
            return entry

    def register_scope(self, scope_root: Path, scope_hash: str,
                       sentinel: str) -> ScopeEntry:
        with self._registry_lock:
            entry = self._scopes.get(scope_hash)
            if entry is not None:
                self._scopes.move_to_end(scope_hash)
                return entry
            entry = ScopeEntry(
                scope_root=scope_root,
                scope_hash=scope_hash,
                sentinel=sentinel,
            )
            # Smoke §7.10: try a disk reload before declaring "pending".
            from longctx_svc.cache.disk import load_index
            loaded = load_index(scope_hash)
            if loaded is not None:
                entry.index = loaded[0]
                entry.status = "ready"
                entry.from_disk = True
            self._scopes[scope_hash] = entry
            self._evict_to_cap()
            return entry

    def evict_idle_indexes(self, now: float | None = None) -> int:
        """PRD §5.7: drop indexes whose last_query_at is older than
        index_idle_timeout_seconds. Returns count evicted.
        """
        cfg = get_config()
        idle = cfg.limits.index_idle_timeout_seconds
        now = now if now is not None else time.time()
        n = 0
        with self._registry_lock:
            for h, entry in list(self._scopes.items()):
                if entry.index is None:
                    continue
                if entry.index.embeddings is None:
                    continue
                if (now - entry.last_query_at) <= idle:
                    continue
                if entry.watcher is not None:
                    try:
                        entry.watcher.stop()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    entry.watcher = None
                entry.index = None
                entry.status = "evicted"
                n += 1
        return n

    def _evict_to_cap(self) -> None:
        """LRU evict in-memory indexes (NOT scope entries) past the cap.
        Drops index payloads but keeps the entry shell so re-loads can
        rehydrate later (from disk, once disk persistence lands).

        2026-05-11: explicitly null out `index.embeddings` + `index.chunks`
        and force gc.collect() after each eviction round. Previously the
        `entry.index = None` step alone left a non-trivial RSS overhead per
        evicted scope (~400 MB/scope on the bge-m3 + Python corpus run),
        because numpy ndarrays + chunk lists hung on through Python's
        delayed cyclic-collection. Verified by repro on 2026-05-11: 5 new
        scopes drove longctx-svc RSS 68 MB → 2289 MB with no drop after
        the 5th scope evicted the 1st.
        """
        cfg = get_config()
        cap = cfg.limits.max_indexes_in_memory
        # Count loaded indexes
        loaded = [h for h, e in self._scopes.items()
                  if e.index is not None and e.index.embeddings is not None]
        if len(loaded) <= cap:
            return
        excess = len(loaded) - cap
        evicted_any = False
        # Evict from the LEFT (oldest) of the OrderedDict.
        for h in list(self._scopes.keys()):
            if excess <= 0:
                break
            entry = self._scopes[h]
            if entry.index is not None and entry.index.embeddings is not None:
                # Stop the watcher; index lives on disk.
                if entry.watcher is not None:
                    try:
                        entry.watcher.stop()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                    entry.watcher = None
                # Explicit drop of the big arrays before nulling the index
                # ref. Numpy frees the buffer the moment its refcount hits
                # zero, but we want to make absolutely sure no transitive
                # ref (e.g. cached slice via Chunk) keeps the ndarray live.
                entry.index.embeddings = None
                entry.index.chunks = []
                entry.index = None
                entry.status = "evicted"
                excess -= 1
                evicted_any = True
        if evicted_any:
            # Force a cyclic-GC pass. ndarrays themselves are refcounted, but
            # the ScopeIndex ↔ Chunk ↔ inner-dict graph commonly has cycles
            # from chunker bookkeeping that would otherwise wait for the
            # next generation-2 sweep to collect. On macOS, where the
            # malloc arenas already don't aggressively return pages, this
            # delay shows up as a multi-GB RSS plateau.
            import gc
            gc.collect()

    def all_scopes(self) -> list[ScopeEntry]:
        with self._registry_lock:
            return list(self._scopes.values())

    def get_scopes(self, scope_hashes) -> list[ScopeEntry]:
        """Look up multiple scopes at once (for workspace queries)."""
        with self._registry_lock:
            out = []
            for h in scope_hashes:
                e = self._scopes.get(h)
                if e is not None and e.index is not None \
                        and e.index.embeddings is not None:
                    out.append(e)
            return out

    def maybe_promote_on_confidence(self, scope_hash: str,
                                     session_id: str,
                                     top_score: float) -> bool:
        """PRD §6.2 / v0.3.2: record the top-K cosine for this session
        and, if N consecutive turns dip below threshold, promote.

        Returns True if the promotion was triggered this call.
        """
        from longctx_svc.config import get_config
        if not session_id:
            return False
        sess = self.sessions.get(session_id)
        if sess is None:
            return False
        cfg = get_config()
        sess.record_confidence(top_score)
        if top_score < cfg.limits.confidence_threshold:
            sess.consecutive_low += 1
        else:
            sess.consecutive_low = 0
        if sess.consecutive_low < cfg.limits.confidence_consecutive_low:
            return False
        # Threshold tripped — promote and reset the streak so we don't
        # re-fire on every turn.
        sess.consecutive_low = 0
        entry = self.get_scope(scope_hash)
        if entry is None or entry.mode == "package" or entry.index is None:
            return False
        if entry.status in ("indexing", "promoting", "error", "evicted"):
            return False
        # Reuse the same promotion machinery as path-based promotion.
        return self._begin_package_rebuild(scope_hash, reason="low-confidence")

    def _begin_package_rebuild(self, scope_hash: str,
                                reason: str) -> bool:
        """Internal: start a Package-scope rebuild for the given scope.
        Used by both maybe_promote (path-based) and
        maybe_promote_on_confidence (confidence-based).
        """
        entry = self.get_scope(scope_hash)
        if entry is None or entry.mode == "package":
            return False
        if entry.status in ("indexing", "promoting"):
            return False
        entry.status = "promoting"
        entry.error = f"promotion reason: {reason}"
        import threading

        def _worker() -> None:
            from longctx_svc.cache.disk import save_index
            from longctx_svc.config import get_config
            from longctx_svc.indexer.builder import build_index
            from longctx_svc.scope.walk import walk_package_scope
            cfg = get_config()
            with entry.lock:
                try:
                    files = walk_package_scope(entry.scope_root)[:cfg.limits.max_files_package]
                    new_index = build_index(
                        entry.scope_root, files, entry.scope_hash,
                        embedder=getattr(self.pipeline, "_embedder", None),
                    )
                    entry.index = new_index
                    entry.indexed_files = frozenset(str(f) for f in files)
                    entry.mode = "package"
                    entry.status = ("ready"
                                    if new_index.embeddings is not None
                                    else "empty")
                    if entry.status == "ready":
                        save_index(new_index, entry.sentinel)
                    entry.error = None
                except Exception as exc:  # noqa: BLE001
                    entry.status = "error"
                    entry.error = f"promotion failed: {exc!s}"

        t = threading.Thread(
            target=_worker, daemon=True,
            name=f"longctx-promote-{scope_hash[:8]}",
        )
        t.start()
        return True

    def maybe_promote(self, scope_hash: str,
                      mentioned: list[str]) -> bool:
        """PRD §6.1 / v0.3.1: if any mentioned path is outside the Hot
        scope's indexed set, kick off a background Package-scope rebuild.

        Returns True if a promotion was scheduled. Idempotent — concurrent
        calls during promotion are safe; only one rebuild runs at a time.
        """
        entry = self.get_scope(scope_hash)
        if entry is None or entry.mode == "package":
            return False
        if entry.status in ("indexing", "promoting", "error", "evicted"):
            return False
        if entry.index is None:
            return False
        try:
            indexed = entry.indexed_files
        except AttributeError:
            return False
        if not indexed:
            return False
        outside = [m for m in mentioned if m and m not in indexed]
        if not outside:
            return False
        return self._begin_package_rebuild(scope_hash,
                                           reason="out-of-hot-path")

    def attach_watcher(self, scope_hash: str) -> None:
        """Spin up a debounced FileWatcher for the scope. Idempotent."""
        entry = self.get_scope(scope_hash)
        if entry is None or entry.watcher is not None:
            return
        from longctx_svc.config import get_config
        from longctx_svc.indexer.builder import update_files
        from longctx_svc.cache.disk import save_index
        from longctx_svc.watcher import FileWatcher

        def _on_change(touched: set) -> None:
            ent = self.get_scope(scope_hash)
            if ent is None or ent.index is None:
                return
            with ent.lock:
                if ent.index is None:
                    return
                try:
                    # Reuse the pipeline's embedder so we don't double-load.
                    pipe_embedder = getattr(self.pipeline, "_embedder", None)
                    if pipe_embedder is None:
                        pipe_embedder = self.pipeline._ensure_embedder()
                    update_files(ent.index, list(touched),
                                 embedder=pipe_embedder)
                    save_index(ent.index, ent.sentinel)
                except Exception as exc:  # noqa: BLE001
                    ent.error = f"watcher update failed: {exc!s}"

        cfg = get_config()
        w = FileWatcher(
            entry.scope_root, _on_change,
            debounce_seconds=cfg.limits.watcher_debounce_seconds,
        )
        w.start()
        entry.watcher = w

    def detach_watcher(self, scope_hash: str) -> None:
        entry = self.get_scope(scope_hash)
        if entry is None or entry.watcher is None:
            return
        try:
            entry.watcher.stop()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        entry.watcher = None

    def loaded_scope_count(self) -> int:
        with self._registry_lock:
            return sum(1 for e in self._scopes.values()
                       if e.index is not None
                       and e.index.embeddings is not None)

    def total_memory_bytes(self) -> int:
        with self._registry_lock:
            return sum(
                e.index.memory_bytes for e in self._scopes.values()
                if e.index is not None
            )


_STATE: _State | None = None


def get_state() -> _State:
    global _STATE
    if _STATE is None:
        _STATE = _State()
    return _STATE


def reset_state() -> None:
    """For tests: drop the singleton."""
    global _STATE
    if _STATE is not None:
        # Stop watchers from prior runs to avoid leaking observer threads.
        for entry in _STATE.all_scopes():
            if entry.watcher is not None:
                try:
                    entry.watcher.stop()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                entry.watcher = None
    _STATE = None
