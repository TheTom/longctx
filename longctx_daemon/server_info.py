"""Port discovery, ``server.info`` runtime descriptor, and singleton lock.

Implements PRD-phase2-mcp-daemon §8.1–§8.4:
  * ``bind_port`` walks forward from a preferred port until a free TCP slot
    is found.
  * ``ServerInfo`` is the dataclass written to
    ``~/.cache/longctx/server.info`` so other CLI subcommands and external
    MCP clients can find the running daemon without hard-coded ports.
  * ``SingletonLock`` enforces "exactly one daemon per user" via
    ``portalocker`` on ``~/.cache/longctx/server.lock``. Port walking is
    "the port is taken by someone else"; the singleton lock is "another
    longctx daemon is already running" — the two checks are independent
    by design (§8.3).

Stale-pid handling:
  * ``read_server_info`` validates the recorded PID with ``os.kill(pid, 0)``;
    if the process is gone we treat the file as garbage and let the
    caller reclaim it.
  * Same logic on lock acquisition: if the lock is held but the recorded
    PID is dead, the lock and info file are reclaimed.

Atomic write:
  * We always write ``server.info`` via tempfile + ``os.replace`` so a
    concurrent reader never sees a half-written file.

Cross-platform:
  * ``portalocker`` (already a transitive dep) is used over raw ``fcntl``
    so the same code works on Windows. macOS/Linux use ``fcntl.flock``
    under the hood; Windows uses ``msvcrt``.

Notes:
  * The ``mcp_port`` and ``status_port`` walk independently. Adjacency
    (8765 → 8766 → 8767) is preserved when possible by trying the
    `preferred + 1` slot for the status port first; if it's taken we
    walk further. We don't *require* adjacency — chasing it across
    process restarts gets fragile.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import portalocker


# ---------------------------------------------------------------- defaults
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "longctx"
DEFAULT_SERVER_INFO = DEFAULT_CACHE_DIR / "server.info"
DEFAULT_LOCK_PATH = DEFAULT_CACHE_DIR / "server.lock"


# ============================================================ exceptions
class PortBindError(RuntimeError):
    """No free port found in the requested range."""


# ============================================================ ServerInfo
@dataclass(frozen=True)
class ServerInfo:
    """Runtime descriptor of a live daemon.

    Written by the daemon on bind, deleted on graceful shutdown. Other
    CLI subcommands (``status``, ``port``, ``mcp-stdio``, ``reload``,
    ``stop``) read this file to find the running daemon.
    """

    pid: int
    started_at: str           # ISO 8601 UTC
    mcp_port: int
    mcp_transports: tuple[str, ...]
    status_port: int
    version: str

    # ---- serialization helpers
    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "started_at": self.started_at,
            "mcp_port": self.mcp_port,
            "mcp_transports": list(self.mcp_transports),
            "status_port": self.status_port,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ServerInfo":
        return cls(
            pid=int(payload["pid"]),
            started_at=str(payload["started_at"]),
            mcp_port=int(payload["mcp_port"]),
            mcp_transports=tuple(payload.get("mcp_transports", ())),
            status_port=int(payload["status_port"]),
            version=str(payload["version"]),
        )


def now_iso() -> str:
    """Current UTC time in ISO 8601 with the trailing ``Z`` per §8.2."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================ port walker
def bind_port(
    preferred: int,
    max_tries: int = 20,
    host: str = "127.0.0.1",
) -> tuple[socket.socket, int]:
    """Walk forward from ``preferred`` until we bind a TCP socket.

    Returns ``(bound_socket, actual_port)``. The caller is responsible
    for closing the socket if it's not handed off to the ASGI server.

    Raises:
        PortBindError: every port in ``[preferred, preferred + max_tries)``
            was taken or rejected.

    Per §8.1: we ONLY treat ``EADDRINUSE`` as walkable. Any other OSError
    (permission denied, address family unsupported) is re-raised — those
    are real errors, not "the port we wanted is busy".
    """
    if max_tries < 1:
        raise ValueError(f"max_tries must be >= 1, got {max_tries}")

    last_err: Optional[OSError] = None
    for offset in range(max_tries):
        port = preferred + offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # SO_REUSEADDR avoids TIME_WAIT bugs on rapid daemon
            # restarts; doesn't help with truly contended ports.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(128)
            return sock, port
        except OSError as e:
            sock.close()
            # EADDRINUSE = walk; anything else = bail.
            if e.errno not in {48, 98, 10048}:  # macOS, linux, windows
                raise
            last_err = e
    raise PortBindError(
        f"no free port in {preferred}..{preferred + max_tries - 1} "
        f"(host={host}, last error: {last_err})"
    )


def bind_pair(
    preferred_mcp: int,
    preferred_status: int,
    max_tries: int = 20,
    host: str = "127.0.0.1",
) -> tuple[tuple[socket.socket, int], tuple[socket.socket, int]]:
    """Bind ``(mcp, status)`` ports independently.

    The two walks are independent — the status port doesn't care if the
    MCP port walked. We try ``preferred_status`` first; if taken we walk
    forward independently.
    """
    mcp_sock, mcp_port = bind_port(preferred_mcp, max_tries=max_tries, host=host)
    try:
        status_sock, status_port = bind_port(
            preferred_status, max_tries=max_tries, host=host,
        )
    except Exception:
        mcp_sock.close()
        raise
    return (mcp_sock, mcp_port), (status_sock, status_port)


# ============================================================ server.info I/O
def _info_path(path: Optional[Path]) -> Path:
    return Path(path).expanduser() if path else DEFAULT_SERVER_INFO


def write_server_info(info: ServerInfo, path: Path | None = None) -> None:
    """Atomic ``server.info`` write — tempfile + ``os.replace``.

    A concurrent reader either sees the previous version or the new one,
    never a partial write. Parent directory is created if missing.
    """
    target = _info_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(info.to_dict(), indent=2)
    # ``delete=False`` because we hand the path to ``os.replace``.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=".server.info.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_name = fh.name
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_name, target)


def _pid_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the process
    is gone, ``PermissionError`` if it exists but we can't signal it.

    We treat permission-denied as alive — a daemon under a different
    user shouldn't get reclaimed by us.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_server_info(path: Path | None = None) -> Optional[ServerInfo]:
    """Read + validate ``server.info``.

    Returns ``None`` when:
      * the file is missing,
      * the JSON is malformed,
      * the recorded PID is no longer running.

    The caller decides whether to reclaim or error out.
    """
    target = _info_path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        info = ServerInfo.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None
    if not _pid_alive(info.pid):
        return None
    return info


def delete_server_info(path: Path | None = None) -> None:
    """Best-effort delete; missing file is not an error."""
    target = _info_path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ============================================================ singleton
class SingletonLock:
    """flock-based singleton — exactly one daemon per lock path.

    Usage::

        with SingletonLock() as lock:
            if not lock.acquire():
                sys.exit("daemon already running")
            ... # bind ports, run

    On ``__exit__`` the lock is released. The lock file content is the
    PID of the holder; readers can use it to confirm liveness when
    ``server.info`` is missing.

    Stale lock handling:
      * ``portalocker`` itself doesn't reclaim stale locks (the kernel
        releases them when the holder dies, so the next ``acquire`` just
        works).
      * If the previous holder is alive but the file is intact, we
        return ``False`` and the caller bails.
      * On Windows ``portalocker`` falls back to ``msvcrt``; same
        contract.
    """

    def __init__(self, lock_path: Path | None = None) -> None:
        self.lock_path: Path = (
            Path(lock_path).expanduser() if lock_path else DEFAULT_LOCK_PATH
        )
        self._fh = None  # opened on acquire
        self._acquired = False

    # ---------------------------------------------------------- lifecycle
    def acquire(self) -> bool:
        """Try to acquire the lock non-blocking.

        Returns:
            True  — we own the lock; safe to write server.info.
            False — another live daemon holds it.

        Side effect on True: writes our PID into the lock file content
        (helpful for diagnostics + stale detection).
        """
        if self._acquired:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in r+ if it exists, w+ otherwise — we need to read the
        # holder's PID even when we can't take the lock.
        fh = open(self.lock_path, "a+", encoding="utf-8")
        try:
            portalocker.lock(
                fh,
                portalocker.LOCK_EX | portalocker.LOCK_NB,
            )
        except portalocker.LockException:
            # Lock held by a live process. The kernel would have released
            # it had the process died, so we don't need a stale-pid
            # check here — but we close the handle cleanly.
            fh.close()
            return False
        # Acquired. Truncate + record our PID so external tooling can
        # see who owns the lock.
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
        except OSError:
            # Best effort; PID-in-content is diagnostic.
            pass
        self._fh = fh
        self._acquired = True
        return True

    def release(self) -> None:
        """Release the lock + close the file handle. Idempotent."""
        if not self._acquired or self._fh is None:
            return
        try:
            portalocker.unlock(self._fh)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        # Try to remove the lock file so a fresh ``ls ~/.cache/longctx``
        # doesn't show a stale .lock when no daemon is running. If
        # another waiter slipped in between unlock + unlink we leave it.
        try:
            self.lock_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        self._fh = None
        self._acquired = False

    # ---------------------------------------------------------- context
    def __enter__(self) -> "SingletonLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ---------------------------------------------------------- introspection
    @property
    def acquired(self) -> bool:
        return self._acquired

    def holder_pid(self) -> Optional[int]:
        """Read the PID stored in the lock file.

        Returns ``None`` if the file is missing, empty, or non-numeric.
        Useful for diagnostics when ``acquire()`` returns ``False``.
        """
        try:
            content = self.lock_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return None
        try:
            return int(content)
        except ValueError:
            return None


__all__ = [
    "PortBindError",
    "ServerInfo",
    "SingletonLock",
    "bind_pair",
    "bind_port",
    "delete_server_info",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LOCK_PATH",
    "DEFAULT_SERVER_INFO",
    "now_iso",
    "read_server_info",
    "write_server_info",
]
