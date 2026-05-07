"""Debounced file watcher per scope. PRD §5.6 / §R5.

Backed by `watchdog`. Coalesces FS events for `debounce_seconds`, then
fires `on_change(touched_paths)` exactly once for each idle interval.
The callback is responsible for re-embedding only the touched files
and updating the ScopeIndex in-place under the scope's RW lock.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    _HAS_WATCHDOG = True
except Exception:  # noqa: BLE001
    _HAS_WATCHDOG = False
    FileSystemEvent = object  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]


from longctx_svc.config import ALWAYS_SKIP_DIRS, BINARY_EXTS, LOCKFILE_NAMES


class FileWatcher:
    """Watch `scope_root` recursively, fire `on_change` after debounce."""

    def __init__(
        self,
        scope_root: Path,
        on_change: Callable[[set[Path]], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        self.scope_root = scope_root
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self.running = False
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_evt = threading.Event()
        self._observer = None
        self._debounce_thread: threading.Thread | None = None

    def _should_track(self, path: Path) -> bool:
        if path.name in LOCKFILE_NAMES:
            return False
        if path.suffix.lower() in BINARY_EXTS:
            return False
        # Case-insensitive containment on darwin (canonicalize_scope
        # lowercases scope_root; watchdog reports raw FS-event paths).
        try:
            rp = Path(os.path.realpath(path))
        except OSError:
            rp = path
        try:
            sr = Path(os.path.realpath(self.scope_root))
        except OSError:
            sr = self.scope_root
        rp_s = str(rp)
        sr_s = str(sr)
        if sys.platform == "darwin":
            rp_s = rp_s.lower()
            sr_s = sr_s.lower()
        if not rp_s.startswith(sr_s.rstrip("/") + os.sep) \
                and rp_s != sr_s:
            return False
        rel_parts = rp_s[len(sr_s):].lstrip(os.sep).split(os.sep)
        for part in rel_parts:
            if part in ALWAYS_SKIP_DIRS:
                return False
            if part.startswith(".") and part not in (".github",):
                return False
        return True

    def _enqueue(self, path: Path) -> None:
        if not self._should_track(path):
            return
        with self._lock:
            self._pending.add(path)
        self._wakeup.set()

    def _debounce_loop(self) -> None:
        while not self._stop_evt.is_set():
            # Wait for first event
            self._wakeup.wait()
            if self._stop_evt.is_set():
                return
            self._wakeup.clear()
            # Then idle for debounce_seconds — extra events extend the wait
            deadline = time.monotonic() + self.debounce_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self._wakeup.wait(timeout=remaining):
                    self._wakeup.clear()
                    deadline = time.monotonic() + self.debounce_seconds
                    if self._stop_evt.is_set():
                        return
            # Drain
            with self._lock:
                batch = self._pending
                self._pending = set()
            if batch:
                try:
                    self.on_change(batch)
                except Exception:  # noqa: BLE001
                    pass

    def start(self) -> None:
        if self.running or not _HAS_WATCHDOG:
            self.running = _HAS_WATCHDOG
            return
        watcher = self

        class _Handler(FileSystemEventHandler):  # type: ignore[misc]
            def on_modified(self, event: FileSystemEvent) -> None:  # type: ignore[name-defined]
                if event.is_directory:
                    return
                watcher._enqueue(Path(event.src_path))

            def on_created(self, event: FileSystemEvent) -> None:  # type: ignore[name-defined]
                if event.is_directory:
                    return
                watcher._enqueue(Path(event.src_path))

            def on_deleted(self, event: FileSystemEvent) -> None:  # type: ignore[name-defined]
                if event.is_directory:
                    return
                watcher._enqueue(Path(event.src_path))

            def on_moved(self, event: FileSystemEvent) -> None:  # type: ignore[name-defined]
                if event.is_directory:
                    return
                watcher._enqueue(Path(event.src_path))
                dest = getattr(event, "dest_path", None)
                if dest:
                    watcher._enqueue(Path(dest))

        self._observer = Observer()
        self._observer.schedule(
            _Handler(), str(self.scope_root), recursive=True,
        )
        self._observer.start()

        self._debounce_thread = threading.Thread(
            target=self._debounce_loop, daemon=True,
            name=f"longctx-watch-{self.scope_root.name}",
        )
        self._debounce_thread.start()
        self.running = True

    def stop(self) -> None:
        self.running = False
        self._stop_evt.set()
        self._wakeup.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            self._observer = None
        if self._debounce_thread is not None:
            self._debounce_thread.join(timeout=2.0)
            self._debounce_thread = None
