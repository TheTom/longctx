"""Spawn and manage a longctx-svc sidecar subprocess.

Used by engine adapters that ship `--enable-longctx` (vllm-swift,
TheTom/vllm-turboquant, the llama-server-longctx wrapper) so users get
one-command setup: the engine starts longctx-svc on a free port, sets
its own retrieval endpoint to that port, and tears the sidecar down on
shutdown.

Tool stays optional: callers only invoke this when the user passed
the flag.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Sidecar:
    """Handle to a running longctx-svc subprocess."""
    url: str
    port: int
    proc: subprocess.Popen
    log_path: str | None = None

    def stop(self, timeout: float = 5.0) -> None:
        """Best-effort shutdown. Idempotent."""
        if self.proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
        except (ProcessLookupError, PermissionError):
            pass


class PortInUseError(RuntimeError):
    """Raised when a user-specified port is already bound by another process."""


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if `port` on `host` is currently bound by something.

    Tries to bind a fresh socket — the cheapest reliable check on POSIX.
    Connect-probes can succeed against half-open sockets, so we use bind.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # SO_REUSEADDR=0 (default) so we get OSError if anything is bound.
            s.bind((host, port))
        return False
    except OSError:
        return True


def _free_port(start: int = 8765) -> int:
    for p in range(start, start + 200):
        if not is_port_in_use(p):
            return p
    # Last resort: ask the OS
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pick_port(preferred: int | None = None,
              start: int = 8765,
              host: str = "127.0.0.1") -> int:
    """Return a free port. If `preferred` is given:
      - free → return it
      - in-use → raise PortInUseError with diagnostic guidance.

    If `preferred` is None: scan upward from `start`.
    """
    if preferred is not None:
        if is_port_in_use(preferred, host=host):
            raise PortInUseError(
                f"Port {preferred} on {host} is already in use. "
                f"Likely causes: another longctx-svc / inference server / "
                f"port-forward is running. Free it (lsof -ti :{preferred} "
                f"| xargs kill) or pick a different port via --port."
            )
        return preferred
    return _free_port(start)


def _wait_healthy(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=2.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.3)
    return False


def spawn_sidecar(
    *,
    port: int | None = None,
    cache_dir: str | None = None,
    log_path: str | None = None,
    python: str | None = None,
    extra_env: dict[str, str] | None = None,
    boot_timeout: float = 30.0,
) -> Sidecar:
    """Start a longctx-svc subprocess on a free port. Returns once the
    /healthz endpoint responds, or raises RuntimeError on timeout.

    Caller owns the lifecycle — call `.stop()` (or use `managed_sidecar`).
    """
    # Validate the chosen port before forking — saves a 30s health-check
    # timeout when the user already has something listening there.
    chosen = pick_port(preferred=port)
    py = python or sys.executable
    env = os.environ.copy()
    if cache_dir is not None:
        env["LONGCTX_CACHE_DIR"] = cache_dir
    if extra_env:
        env.update(extra_env)
    cmd = [
        py, "-m", "longctx_svc.cli",
        "serve", "--host", "127.0.0.1", "--port", str(chosen),
    ]
    if log_path:
        log_f = open(log_path, "ab")  # noqa: SIM115
    else:
        log_f = subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    url = f"http://127.0.0.1:{chosen}"
    if not _wait_healthy(url, timeout=boot_timeout):
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"longctx-svc sidecar did not become healthy at {url} "
            f"within {boot_timeout:.0f}s"
            + (f"; see log at {log_path}" if log_path else "")
        )
    return Sidecar(url=url, port=chosen, proc=proc, log_path=log_path)


@contextmanager
def managed_sidecar(**kwargs) -> Iterator[Sidecar]:
    """Context-managed spawn. Auto-stops on exit even if caller raises.

    Engines that have a clean foreground loop should prefer this.
    Engines that fork to a long-lived loop (uvicorn, web.run_app) should
    call `spawn_sidecar` and register `atexit` / signal handlers.
    """
    sc = spawn_sidecar(**kwargs)
    try:
        yield sc
    finally:
        sc.stop()
