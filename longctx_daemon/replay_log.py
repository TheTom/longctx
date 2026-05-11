"""Append-only JSONL replay log for MCP calls.

Distinct from the operational log (``logging.py``):
  * full request + full response payload, no redaction (§14.8)
  * one file: ``~/.cache/longctx/interactions.jsonl`` by default
  * size-based rotation: when the active file exceeds the configured
    threshold (default 1 GB per §14.8.1), it is rotated to
    ``interactions-<timestamp>.jsonl.gz`` and a fresh active file
    is opened.

Phase 2.0 implements size-based rotation only. Time-based rotation
(§14.8.1's "30 days" branch) is left as a TODO — size is the runaway
case; daily turnover can be handled by a separate sweep job in 2.1.

Concurrency: single-process (the daemon writes; nothing else does).
We open in line-buffered append mode and call ``flush()`` after each
``record`` so a daemon crash loses at most the in-flight line.

TODO 2.1: time-based rotation, configurable retention sweep, an
async writer queue if synchronous flush ever shows up in profiles.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

# 1 GB default per §14.8.1.
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024


class ReplayLog:
    """Append-only JSONL replay log with size-based gzip rotation.

    Thread-safe via a single internal lock — the daemon expects multiple
    concurrent MCP handlers to call ``record`` from different async tasks
    that all run on the same event loop, but a worker thread for the
    embedder may also want to log, so we lock.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        """Open the active file in append mode.

        Args:
            path: filesystem path to the JSONL active file. Parent
                directories are created on demand.
            max_bytes: rotation threshold. When ``record`` would push the
                file over this size, the existing file is rotated to a
                gzip'd shard and a fresh active file is opened.
                Pass 0 to disable rotation (capture-everything mode,
                §14.8.1).
        """
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._fp: Optional[TextIO] = None
        self._closed = False
        self._open()

    # ----------------------------------------------------------- public API
    def record(self, call_data: dict[str, Any]) -> None:
        """Write one JSON line to the active file.

        Atomic-write: we serialize first, then write+flush under the
        lock. A partially-written line on crash is impossible at the
        JSON layer because ``write()`` is a single syscall for sub-page
        writes on local filesystems; if you ever route this to NFS,
        revisit.
        """
        if self._closed:
            raise RuntimeError("ReplayLog is closed")
        line = json.dumps(call_data, separators=(",", ":"), default=_default_encoder)
        encoded = line + "\n"
        with self._lock:
            assert self._fp is not None
            # Rotation check happens BEFORE the write so the rotated
            # shard never contains a partial trailing line.
            if self.max_bytes > 0 and self._fp.tell() + len(encoded) > self.max_bytes:
                self._rotate_locked()
            self._fp.write(encoded)
            self._fp.flush()

    def close(self) -> None:
        """Flush + close. Idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._fp is not None:
                self._fp.flush()
                self._fp.close()
                self._fp = None
            self._closed = True

    # ---------------------------------------------------------- internals
    def _open(self) -> None:
        """(Re)open the active file in append mode with line buffering."""
        # buffering=1 enables line buffering when in text mode.
        self._fp = open(self.path, "a", buffering=1, encoding="utf-8")

    def _rotate_locked(self) -> None:
        """Close active, gzip it to a timestamped shard, open fresh.

        Caller must hold ``self._lock``. Shard naming sorts
        lexicographically by rotation time so ``ls`` lists oldest first.
        """
        assert self._fp is not None
        self._fp.flush()
        self._fp.close()
        self._fp = None

        # ISO-style timestamp suitable for filenames (no colons). We
        # also add a microsecond suffix and, if a collision somehow
        # still happens (clock granularity, tests with mocked time), a
        # numeric counter that walks forward until a free name is found.
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}"
        # Insert the timestamp before the suffix(es). For a path like
        # interactions.jsonl, this becomes interactions-<ts>.jsonl.gz.
        stem = self.path.name.split(".", 1)[0]
        suffix = self.path.name[len(stem):] or ".jsonl"
        base_name = f"{stem}-{ts}{suffix}.gz"
        rotated = self.path.with_name(base_name)
        counter = 1
        while rotated.exists():
            rotated = self.path.with_name(f"{stem}-{ts}-{counter}{suffix}.gz")
            counter += 1

        # gzip the active file into the shard, then unlink the active.
        with open(self.path, "rb") as src, gzip.open(rotated, "wb") as dst:
            shutil.copyfileobj(src, dst)
        try:
            self.path.unlink()
        except FileNotFoundError:  # pragma: no cover
            pass

        self._open()

    # ----------------------------------------------------------- ctx mgr
    def __enter__(self) -> "ReplayLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _default_encoder(obj: Any) -> Any:
    """Fallback for objects json.dumps doesn't know.

    Keeps the replay log forgiving: dataclass instances and tuples
    serialize cleanly, while truly unknown types are coerced to str
    rather than raising mid-record.
    """
    # dataclass instance.
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    # tuple/set → list.
    if isinstance(obj, (tuple, set)):
        return list(obj)
    return str(obj)


__all__ = ["ReplayLog", "DEFAULT_MAX_BYTES"]
