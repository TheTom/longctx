"""Tests for ``longctx init`` (Phase 2.1).

Coverage:
  * --non-interactive writes a default config + reports discovery
  * --parent-dir / --include / --no-auto-discover combos
  * --force overwrites; absence of --force errors out
  * config can be re-loaded after write (round-trip safety)
  * env-var overrides apply
  * exit codes on validation failure
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from longctx_daemon.cli import build_parser, main
from longctx_daemon.config import (
    DEFAULT_PARENT_DIR,
    ServiceConfig,
    load_config,
)


# --------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _clear_longctx_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LONGCTX_"):
            monkeypatch.delenv(k, raising=False)
    yield


def _make_project(parent: Path, name: str, sentinel: str = ".git") -> Path:
    proj = parent / name
    proj.mkdir(parents=True, exist_ok=True)
    if sentinel.startswith("."):
        (proj / sentinel).mkdir(exist_ok=True)
    else:
        (proj / sentinel).write_text("")
    return proj


def _run(*argv: str) -> int:
    return main(list(argv))


# --------------------------------------------------------------- non-interactive


class TestNonInteractive:
    def test_writes_default_config(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        assert rc == 0
        assert cfg_path.is_file()
        out = capsys.readouterr().out
        assert "Configuration written to" in out
        assert str(cfg_path) in out

    def test_round_trip_after_init(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert cfg.corpus.parent_dir == tmp_path / "dev"

    def test_with_parent_dir_override(self, tmp_path):
        custom_parent = tmp_path / "alt"
        custom_parent.mkdir()
        _make_project(custom_parent, "foo")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(custom_parent),
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert cfg.corpus.parent_dir == custom_parent

    def test_no_auto_discover_requires_include(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--no-auto-discover",
        )
        assert rc != 0
        err = capsys.readouterr().err
        assert "--include" in err

    def test_with_no_auto_discover_and_include(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
            "--no-auto-discover",
            "--include", "foo,bar",
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert cfg.corpus.auto_discover is False
        assert cfg.corpus.include == ("foo", "bar")

    def test_include_alone_treats_as_deny_list(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
            "--include", "skip-me",
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert cfg.corpus.auto_discover is True
        assert cfg.corpus.include == ("skip-me",)

    def test_include_csv_with_whitespace(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
            "--include", "  foo , bar ,baz ",
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert cfg.corpus.include == ("foo", "bar", "baz")


# --------------------------------------------------------------- discovery report


class TestDiscoveryReporting:
    def test_reports_discovered_count(self, tmp_path, capsys):
        parent = tmp_path / "dev"
        parent.mkdir()
        _make_project(parent, "alpha")
        _make_project(parent, "beta")
        _make_project(parent, "gamma")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(parent),
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Discovered 3" in out
        assert "3 will be indexed" in out

    def test_verbose_lists_projects(self, tmp_path, capsys):
        parent = tmp_path / "dev"
        parent.mkdir()
        _make_project(parent, "alpha")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--verbose",
            "--config", str(cfg_path),
            "--parent-dir", str(parent),
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "alpha" in out

    def test_warns_when_parent_dir_missing(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "absent"),
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "does not exist" in out

    def test_excluded_via_deny_list_not_indexed(self, tmp_path, capsys):
        parent = tmp_path / "dev"
        parent.mkdir()
        _make_project(parent, "skip")
        _make_project(parent, "keep")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(parent),
            "--include", "skip",
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Discovered 2" in out
        # auto_discover True + include=("skip",) → skip is denied, only
        # keep is indexed.
        assert "1 will be indexed" in out

    def test_longctxignore_skipped(self, tmp_path, capsys):
        parent = tmp_path / "dev"
        parent.mkdir()
        proj = _make_project(parent, "opt-out")
        (proj / ".longctxignore").write_text("# please skip me")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(parent),
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Discovered 1" in out
        assert "0 will be indexed" in out


# --------------------------------------------------------------- overwrite


class TestOverwriteSemantics:
    def test_existing_config_refuses_without_force(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text("# existing\n")
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        assert rc != 0
        err = capsys.readouterr().err
        assert "already exists" in err

    def test_force_overwrites(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text("[corpus]\nparent_dir = \"/old/path\"\n")
        rc = _run(
            "init",
            "--non-interactive",
            "--force",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        assert rc == 0
        cfg = load_config(cfg_path)
        assert str(cfg.corpus.parent_dir) != "/old/path"


# --------------------------------------------------------------- env overrides


class TestEnvOverridesViaInit:
    def test_parent_dir_env_used_when_no_flag(self, tmp_path, monkeypatch):
        # PRD §3.9: env vars override file values. ``init`` uses defaults
        # so env overrides apply through the same path. We don't expose
        # an env override at *init time* for parent-dir — the flag is
        # the init-time mechanism — but env vars affect the resulting
        # ServiceConfig's defaults via load_config later. This test
        # ensures env vars don't leak into the WRITTEN file (which
        # should be a clean snapshot of the user's choices).
        monkeypatch.setenv("LONGCTX_PARENT_DIR", str(tmp_path / "envparent"))
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
        )
        assert rc == 0
        # File should reflect the env-derived parent_dir (since no
        # --parent-dir was passed and the env override applies to
        # default_config's resolution path).
        cfg = load_config(cfg_path)
        # The env var influences load_config; the file was written
        # with whatever default_config said, then load_config re-applies
        # env. Confirm round-trip.
        assert cfg.corpus.parent_dir == Path(tmp_path / "envparent")

    def test_unknown_env_fails_init(self, tmp_path, monkeypatch, capsys):
        # An unknown LONGCTX_* env var causes load_config to raise. The
        # init command's round-trip-validate step should catch it.
        monkeypatch.setenv("LONGCTX_NOPE", "1")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init",
            "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        # The write happens before re-load, but re-load fails → exit 2
        assert rc != 0


# --------------------------------------------------------------- argparse wiring


class TestArgparseWiring:
    def test_init_parser_exists(self):
        parser = build_parser()
        # Doesn't blow up; parses with the defaults.
        ns = parser.parse_args(
            ["init", "--non-interactive", "--parent-dir", "/tmp"]
        )
        assert ns.command == "init"
        assert ns.non_interactive is True
        assert ns.parent_dir == "/tmp"

    def test_init_help_exits_clean(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["init", "--help"])
        assert exc.value.code == 0

    def test_no_command_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# --------------------------------------------------------------- defaults


class TestDefaultsWritten:
    def test_default_corpus_section_in_file(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init", "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(tmp_path / "dev"),
        )
        assert rc == 0
        text = cfg_path.read_text()
        # Sections present
        assert "[corpus]" in text
        assert "[server]" in text
        assert "[index]" in text
        assert "[watcher]" in text
        assert "[cleanup]" in text
        assert "[search]" in text
        assert "[logging]" in text

    def test_default_server_ports_in_file(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        _run("init", "--non-interactive",
             "--config", str(cfg_path),
             "--parent-dir", str(tmp_path / "dev"))
        text = cfg_path.read_text()
        assert "8765" in text
        assert "8766" in text

    def test_no_initial_indexing_per_prd_11_1(self, tmp_path, capsys):
        # init must not start any indexing pass; output should not
        # mention chunk counts.
        parent = tmp_path / "dev"
        parent.mkdir()
        proj = _make_project(parent, "foo")
        (proj / "main.py").write_text("print('x')")
        cfg_path = tmp_path / "config.toml"
        rc = _run(
            "init", "--non-interactive",
            "--config", str(cfg_path),
            "--parent-dir", str(parent),
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "chunks" not in out.lower()
        assert "embedder" not in out.lower() or "Next steps" in out

    def test_no_service_install_per_prd_11_1(self, tmp_path, capsys):
        cfg_path = tmp_path / "config.toml"
        rc = _run("init", "--non-interactive",
                  "--config", str(cfg_path),
                  "--parent-dir", str(tmp_path / "dev"))
        assert rc == 0
        out = capsys.readouterr().out
        # The output should suggest the next step but not perform it
        assert "service install" in out
        assert "service start" in out
