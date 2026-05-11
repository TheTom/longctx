"""Tests for the macOS launchd service installer + Linux/Windows stubs."""
from __future__ import annotations

import plistlib
import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from longctx_daemon.service import (
    LaunchdInstaller,
    ServiceState,
    SystemdUserInstaller,
    WindowsServiceInstaller,
    current_installer,
)


# --------------------------------------------------------------- launchd

@pytest.fixture
def installer(tmp_path):
    """Launchd installer redirected at tmp paths so tests don't touch
    the user's ~/Library."""
    return LaunchdInstaller(
        plist_dir=tmp_path / "LaunchAgents",
        log_dir=tmp_path / "Logs",
        launchctl="launchctl",   # may or may not exist on the runner
    )


def test_install_writes_plist_with_expected_keys(installer, tmp_path):
    p = installer.install(
        executable=Path("/usr/local/bin/longctx"),
        corpus_dir=tmp_path / "myapp",
    )
    assert p.exists()
    with p.open("rb") as f:
        plist = plistlib.load(f)
    assert plist["Label"] == "com.tomturney.longctx"
    assert plist["ProgramArguments"][0] == "/usr/local/bin/longctx"
    assert plist["ProgramArguments"][1] == "serve"
    assert "--corpus-dir" in plist["ProgramArguments"]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert "stdout.log" in plist["StandardOutPath"]


def test_install_without_corpus(installer):
    p = installer.install(executable=Path("/usr/bin/longctx"))
    with p.open("rb") as f:
        plist = plistlib.load(f)
    assert plist["ProgramArguments"] == ["/usr/bin/longctx", "serve"]


def test_install_environment_variables(installer):
    p = installer.install(
        executable=Path("/usr/bin/longctx"),
        env={"LONGCTX_LOG_LEVEL": "DEBUG"},
    )
    with p.open("rb") as f:
        plist = plistlib.load(f)
    assert plist["EnvironmentVariables"] == {"LONGCTX_LOG_LEVEL": "DEBUG"}


def test_install_creates_log_dir(installer, tmp_path):
    installer.install(executable=Path("/usr/bin/longctx"))
    assert (tmp_path / "Logs").is_dir()


def test_install_idempotent(installer):
    """Calling install twice replaces; doesn't crash."""
    installer.install(executable=Path("/usr/bin/longctx"))
    installer.install(executable=Path("/usr/bin/longctx2"))
    with installer.plist_path.open("rb") as f:
        plist = plistlib.load(f)
    assert plist["ProgramArguments"][0] == "/usr/bin/longctx2"


def test_uninstall_removes_plist(installer):
    installer.install(executable=Path("/usr/bin/longctx"))
    assert installer.plist_path.exists()
    with patch.object(installer, "_launchctl") as mock_lc:
        mock_lc.return_value = None
        installer.uninstall()
    assert not installer.plist_path.exists()


def test_start_without_install_raises(installer):
    with pytest.raises(RuntimeError, match="not installed"):
        installer.start()


def test_status_when_not_installed(installer):
    state = installer.status()
    assert isinstance(state, ServiceState)
    assert state.installed is False
    assert state.running is False
    assert state.pid is None
    assert state.plist_path is None


def test_status_when_installed_but_launchctl_missing(installer):
    """If launchctl isn't available (e.g. running tests on Linux),
    status should report installed=True but running=False, no pid."""
    installer.install(executable=Path("/usr/bin/longctx"))
    with patch.object(installer, "_launchctl", return_value=None):
        state = installer.status()
    assert state.installed is True
    assert state.running is False
    assert state.pid is None


def test_status_parses_running_pid_from_launchctl_output(installer):
    """When launchctl list returns plist-format output with PID set,
    status reports running=True + the pid."""
    installer.install(executable=Path("/usr/bin/longctx"))
    fake = subprocess.CompletedProcess(
        args=["launchctl", "list", "com.tomturney.longctx"],
        returncode=0,
        stdout=(
            '{\n'
            '  "Label" = "com.tomturney.longctx";\n'
            '  "PID" = 12345;\n'
            '};\n'
        ),
        stderr="",
    )
    with patch.object(installer, "_launchctl", return_value=fake):
        state = installer.status()
    assert state.running is True
    assert state.pid == 12345


# ---------------------------------------------------------- linux + win

def test_systemd_install_writes_unit_file(tmp_path):
    """Systemd installer writes a real unit file at the chosen path."""
    inst = SystemdUserInstaller(
        unit_dir=tmp_path / "systemd-user", systemctl="systemctl",
    )
    with patch.object(inst, "_systemctl", return_value=None):
        p = inst.install(
            executable=Path("/usr/local/bin/longctx"),
            corpus_dir=tmp_path / "myapp",
            env={"LONGCTX_LOG_LEVEL": "DEBUG"},
        )
    assert p.exists()
    body = p.read_text()
    assert "ExecStart=/usr/local/bin/longctx serve" in body
    assert "--corpus-dir" in body
    assert "Environment=LONGCTX_LOG_LEVEL=DEBUG" in body
    assert "WantedBy=default.target" in body
    assert "Restart=on-failure" in body


def test_systemd_uninstall_removes_unit(tmp_path):
    inst = SystemdUserInstaller(unit_dir=tmp_path / "systemd-user")
    with patch.object(inst, "_systemctl", return_value=None):
        inst.install(executable=Path("/usr/bin/longctx"))
        assert inst.unit_path.exists()
        inst.uninstall()
    assert not inst.unit_path.exists()


def test_systemd_status_when_not_installed(tmp_path):
    inst = SystemdUserInstaller(unit_dir=tmp_path / "systemd-user")
    state = inst.status()
    assert state.installed is False


def test_systemd_status_parses_active_pid(tmp_path):
    inst = SystemdUserInstaller(unit_dir=tmp_path / "systemd-user")
    inst.unit_dir.mkdir(parents=True, exist_ok=True)
    inst.unit_path.write_text("[Unit]\nDescription=test\n")

    def fake(args, *, check, capture=False):
        if args[0] == "is-active":
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="active\n", stderr="",
            )
        if args[0] == "show":
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="MainPID=4242\n", stderr="",
            )
        return None

    with patch.object(inst, "_systemctl", side_effect=fake):
        state = inst.status()
    assert state.running is True
    assert state.pid == 4242


def test_systemd_start_without_install_raises(tmp_path):
    inst = SystemdUserInstaller(unit_dir=tmp_path / "missing")
    with pytest.raises(RuntimeError, match="not installed"):
        inst.start()


def test_windows_installer_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        WindowsServiceInstaller().install(executable=Path("longctx.exe"))


def test_systemd_status_returns_uninstalled():
    state = SystemdUserInstaller().status()
    assert state.installed is False


# ---------------------------------------------------------- dispatcher

def test_current_installer_picks_platform(monkeypatch):
    """``current_installer`` returns the right class for the current OS."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert isinstance(current_installer(), LaunchdInstaller)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert isinstance(current_installer(), SystemdUserInstaller)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert isinstance(current_installer(), WindowsServiceInstaller)


def test_current_installer_unsupported(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "OS/2")
    with pytest.raises(RuntimeError, match="unsupported platform"):
        current_installer()
