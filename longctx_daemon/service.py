"""longctx service install/start/stop/status/uninstall.

Per spec §9.3. Phase 2.3 ships macOS launchd; Linux systemd + Windows
Service ship in 2.4 — they ride the same ``ServiceInstaller`` Protocol
so the CLI surface stays uniform across platforms.

Design:
  * Each platform implements ``ServiceInstaller``
  * The CLI dispatcher picks the right installer at runtime via
    ``current_installer()``
  * Installers are pure-Python (no shell-out templates) so unit tests
    can drive them on any platform via ``LaunchdInstaller(plist_dir=tmp_path)``
"""
from __future__ import annotations

import os
import platform
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


# ---------------------------------------------------------- Protocol

@dataclass(frozen=True)
class ServiceState:
    """Snapshot from ``status``."""
    installed: bool
    running: bool
    pid: Optional[int]
    label: str
    plist_path: Optional[Path]


class ServiceInstaller(Protocol):
    """Platform-agnostic interface."""

    def install(self, *, executable: Path, corpus_dir: Optional[Path],
                env: Optional[dict[str, str]] = None) -> Path:
        """Generate platform service definition (plist / unit / Service).
        Returns the path written. Idempotent: re-install overwrites
        the existing definition."""
        ...

    def uninstall(self) -> None:
        """Stop + remove the service definition."""
        ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> ServiceState: ...


# ---------------------------------------------------------- macOS launchd

class LaunchdInstaller:
    """macOS LaunchAgent installer.

    Writes a plist to ``~/Library/LaunchAgents/com.tomturney.longctx.plist``
    with ``RunAtLoad=true``, ``KeepAlive=true``, log paths under
    ``~/Library/Logs/longctx/``. Loaded via ``launchctl load``.
    """

    LABEL = "com.tomturney.longctx"

    def __init__(
        self,
        plist_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        launchctl: str = "launchctl",
    ) -> None:
        # plist_dir / log_dir overridable for tests
        self.plist_dir = (
            plist_dir
            if plist_dir is not None
            else Path.home() / "Library" / "LaunchAgents"
        )
        self.log_dir = (
            log_dir
            if log_dir is not None
            else Path.home() / "Library" / "Logs" / "longctx"
        )
        self.launchctl = launchctl

    @property
    def plist_path(self) -> Path:
        return self.plist_dir / f"{self.LABEL}.plist"

    # ----------------------------------------------------- install

    def install(
        self, *, executable: Path,
        corpus_dir: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> Path:
        self.plist_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        program_args = [str(executable), "serve"]
        if corpus_dir is not None:
            program_args += ["--corpus-dir", str(corpus_dir)]

        plist = {
            "Label": self.LABEL,
            "ProgramArguments": program_args,
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(self.log_dir / "stdout.log"),
            "StandardErrorPath": str(self.log_dir / "stderr.log"),
            "EnvironmentVariables": env or {},
            "ProcessType": "Background",
        }
        with self.plist_path.open("wb") as f:
            plistlib.dump(plist, f)
        try:
            os.chmod(self.plist_path, 0o644)
        except OSError:
            pass
        return self.plist_path

    # --------------------------------------------------- uninstall

    def uninstall(self) -> None:
        if self.plist_path.exists():
            # Best-effort unload (succeeds even if not loaded)
            self._launchctl(["unload", str(self.plist_path)],
                            check=False)
            self.plist_path.unlink()

    # -------------------------------------------------- start/stop

    def start(self) -> None:
        if not self.plist_path.exists():
            raise RuntimeError(
                "service not installed; run `longctx service install` first"
            )
        # ``launchctl load`` both registers + starts; idempotent
        self._launchctl(["load", str(self.plist_path)], check=False)

    def stop(self) -> None:
        self._launchctl(["unload", str(self.plist_path)], check=False)

    # -------------------------------------------------------- status

    def status(self) -> ServiceState:
        installed = self.plist_path.exists()
        if not installed:
            return ServiceState(False, False, None, self.LABEL, None)
        # ``launchctl list <label>`` returns plist-formatted output;
        # the PID line tells us if it's running.
        result = self._launchctl(
            ["list", self.LABEL], check=False, capture=True,
        )
        pid: Optional[int] = None
        running = False
        if result is not None and result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('"PID" = '):
                    val = line.removeprefix('"PID" = ').rstrip(";").strip()
                    if val.isdigit():
                        pid = int(val)
                        running = True
                        break
        return ServiceState(
            installed=True, running=running, pid=pid,
            label=self.LABEL, plist_path=self.plist_path,
        )

    # ----------------------------------------------------- internals

    def _launchctl(
        self, args: list[str], *, check: bool, capture: bool = False,
    ) -> Optional[subprocess.CompletedProcess]:
        """Wrapper so tests can monkeypatch + so we don't blow up if
        the binary is missing (e.g. running on Linux while testing)."""
        if not shutil.which(self.launchctl):
            return None
        try:
            return subprocess.run(
                [self.launchctl, *args],
                check=check, capture_output=capture, text=True,
            )
        except subprocess.CalledProcessError:
            if check:
                raise
            return None


# ---------------------------------------------------------- Linux systemd

class SystemdUserInstaller:
    """Linux systemd user-unit installer.

    Writes a unit file at ``~/.config/systemd/user/longctx.service``,
    enables + starts via ``systemctl --user``. Logs go to journald
    (``journalctl --user -u longctx``).
    """

    LABEL = "longctx"
    UNIT_NAME = "longctx.service"

    def __init__(
        self,
        unit_dir: Optional[Path] = None,
        systemctl: str = "systemctl",
    ) -> None:
        self.unit_dir = (
            unit_dir
            if unit_dir is not None
            else Path.home() / ".config" / "systemd" / "user"
        )
        self.systemctl = systemctl

    @property
    def unit_path(self) -> Path:
        return self.unit_dir / self.UNIT_NAME

    # The Phase 2.3 LaunchdInstaller exposes ``plist_path`` because its
    # platform-natural artifact is a plist. The systemd installer
    # exposes the same conceptual field via ``unit_path`` (different
    # name to keep the type signature semantic) — and ``ServiceState``
    # carries the path under ``plist_path`` regardless of platform
    # for one uniform CLI render.

    def install(
        self, *, executable: Path,
        corpus_dir: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> Path:
        self.unit_dir.mkdir(parents=True, exist_ok=True)

        exec_args = [str(executable), "serve"]
        if corpus_dir is not None:
            exec_args.extend(["--corpus-dir", str(corpus_dir)])

        env_lines = "\n".join(
            f"Environment={k}={v}" for k, v in (env or {}).items()
        )
        unit = (
            "[Unit]\n"
            "Description=longctx local-codebase Q&A daemon\n"
            "Documentation=https://github.com/TheTom/longctx\n"
            "After=network.target\n"
            "\n"
            "[Service]\n"
            f"ExecStart={' '.join(exec_args)}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            "Type=simple\n"
        )
        if env_lines:
            unit += env_lines + "\n"
        unit += (
            "\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        self.unit_path.write_text(unit)
        try:
            os.chmod(self.unit_path, 0o644)
        except OSError:
            pass
        # Tell systemd to pick up the new unit
        self._systemctl(["daemon-reload"], check=False)
        return self.unit_path

    def uninstall(self) -> None:
        if self.unit_path.exists():
            self._systemctl(["disable", "--now", self.UNIT_NAME], check=False)
            self.unit_path.unlink()
            self._systemctl(["daemon-reload"], check=False)

    def start(self) -> None:
        if not self.unit_path.exists():
            raise RuntimeError(
                "service not installed; run `longctx service install` first"
            )
        self._systemctl(["enable", "--now", self.UNIT_NAME], check=False)

    def stop(self) -> None:
        self._systemctl(["stop", self.UNIT_NAME], check=False)

    def status(self) -> ServiceState:
        installed = self.unit_path.exists()
        if not installed:
            return ServiceState(False, False, None, self.LABEL, None)
        # ``systemctl --user is-active <unit>`` returns active/inactive.
        # ``show -p MainPID <unit>`` returns the pid (0 when stopped).
        active = self._systemctl(
            ["is-active", self.UNIT_NAME],
            check=False, capture=True,
        )
        running = bool(
            active is not None and active.stdout.strip() == "active"
        )
        pid: Optional[int] = None
        if running:
            show = self._systemctl(
                ["show", "-p", "MainPID", self.UNIT_NAME],
                check=False, capture=True,
            )
            if show is not None and show.returncode == 0:
                line = show.stdout.strip()
                # "MainPID=12345"
                if "=" in line:
                    val = line.split("=", 1)[1]
                    if val.isdigit() and int(val) > 0:
                        pid = int(val)
        return ServiceState(
            installed=True, running=running, pid=pid,
            label=self.LABEL, plist_path=self.unit_path,
        )

    def _systemctl(
        self, args: list[str], *, check: bool, capture: bool = False,
    ) -> Optional[subprocess.CompletedProcess]:
        if not shutil.which(self.systemctl):
            return None
        try:
            return subprocess.run(
                [self.systemctl, "--user", *args],
                check=check, capture_output=capture, text=True,
            )
        except subprocess.CalledProcessError:
            if check:
                raise
            return None


# ---------------------------------------------------------- Windows

class WindowsServiceInstaller:
    """Windows Service installer — NOT supported in v1.

    macOS launchd + Linux systemd are the supported platforms.
    Windows users have two unsupported paths: run ``longctx serve``
    inside Task Scheduler, or run ``longctx serve --daemon`` and
    handle restart-on-reboot manually. Community PRs that wire
    pywin32 into a real ``ServiceInstaller`` impl are welcome.
    """

    LABEL = "longctx"
    _MSG = (
        "Windows Service auto-install is not supported in v1. Use "
        "`longctx serve --daemon` directly, or wrap it in Windows "
        "Task Scheduler. macOS + Linux are the supported daemon "
        "platforms; community PRs welcome at "
        "https://github.com/TheTom/longctx."
    )

    def install(self, *, executable, corpus_dir=None, env=None):  # noqa: ANN001
        raise NotImplementedError(self._MSG)

    def uninstall(self) -> None:
        raise NotImplementedError(self._MSG)

    def start(self) -> None:
        raise NotImplementedError(self._MSG)

    def stop(self) -> None:
        raise NotImplementedError(self._MSG)

    def status(self) -> ServiceState:
        return ServiceState(
            installed=False, running=False, pid=None,
            label=self.LABEL, plist_path=None,
        )


# ---------------------------------------------------------- dispatcher

def current_installer() -> ServiceInstaller:
    """Pick the right installer for the current platform."""
    sysname = platform.system()
    if sysname == "Darwin":
        return LaunchdInstaller()
    if sysname == "Linux":
        return SystemdUserInstaller()
    if sysname == "Windows":
        return WindowsServiceInstaller()
    raise RuntimeError(f"unsupported platform: {sysname}")
