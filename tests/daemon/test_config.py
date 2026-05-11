"""Config TOML round-trip + env-override + validation tests.

Coverage targets (from agent spec §6):
  * default_config produces every section with sensible values
  * write_config + load_config round-trip preserves all fields
  * env-var overrides apply on top of file values
  * validation errors fire on malformed TOML, conflicting flags,
    invalid transports, etc.
  * config_path resolves via platformdirs (smoke check; not asserting
    absolute path on each platform)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from longctx_daemon.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_FORBIDDEN_DIRS,
    DEFAULT_PARENT_DIR,
    DEFAULT_SECRET_PATTERNS,
    DEFAULT_SENTINELS,
    CleanupConfig,
    ConfigError,
    CorpusConfig,
    IndexConfigSection,
    LoggingConfig,
    SearchConfig,
    ServerConfig,
    ServiceConfig,
    WatcherConfigSection,
    config_path,
    default_config,
    load_config,
    reconcile,
    write_config,
)
from longctx_daemon.discovery import DiscoveredProject


# --------------------------------------------------------------- env helpers

@pytest.fixture(autouse=True)
def _clear_longctx_env(monkeypatch):
    """Clear all LONGCTX_* env vars so test ordering doesn't leak."""
    for k in list(os.environ):
        if k.startswith("LONGCTX_"):
            monkeypatch.delenv(k, raising=False)
    yield


# --------------------------------------------------------------- defaults


class TestDefaults:
    def test_default_config_returns_service_config(self):
        cfg = default_config()
        assert isinstance(cfg, ServiceConfig)

    def test_default_corpus_section(self):
        cfg = default_config()
        assert isinstance(cfg.corpus, CorpusConfig)
        assert cfg.corpus.parent_dir == DEFAULT_PARENT_DIR
        assert cfg.corpus.auto_discover is True
        assert cfg.corpus.sentinels == DEFAULT_SENTINELS
        assert cfg.corpus.include == ()
        assert cfg.corpus.exclude_patterns == DEFAULT_EXCLUDE_PATTERNS
        assert cfg.corpus.secret_patterns == DEFAULT_SECRET_PATTERNS
        assert cfg.corpus.forbidden_dirs == DEFAULT_FORBIDDEN_DIRS
        assert cfg.corpus.respect_gitignore is True
        assert cfg.corpus.max_file_size_kb == 1024
        assert cfg.corpus.allow_secret_patterns is False

    def test_default_server_section(self):
        cfg = default_config()
        assert isinstance(cfg.server, ServerConfig)
        assert cfg.server.mcp_port_preferred == 8765
        assert cfg.server.status_port_preferred == 8766
        assert cfg.server.mcp_host == "127.0.0.1"
        assert "sse" in cfg.server.mcp_transports
        assert "streamable-http" in cfg.server.mcp_transports

    def test_default_index_section(self):
        cfg = default_config()
        assert isinstance(cfg.index, IndexConfigSection)
        assert cfg.index.embedder == "BAAI/bge-small-en-v1.5"
        assert cfg.index.chunk_size == 2048
        assert cfg.index.chunk_overlap == 128
        assert cfg.index.disk_budget_gb == 0.0

    def test_default_watcher_section(self):
        cfg = default_config()
        assert isinstance(cfg.watcher, WatcherConfigSection)
        assert cfg.watcher.debounce_ms == 200
        assert cfg.watcher.queue_size == 10000
        assert cfg.watcher.periodic_mtime_sweep_seconds == 30

    def test_default_cleanup_section(self):
        cfg = default_config()
        assert isinstance(cfg.cleanup, CleanupConfig)
        assert cfg.cleanup.missing_root_grace_days == 7
        assert cfg.cleanup.periodic_check_seconds == 3600
        assert cfg.cleanup.watcher_idle_pause_days == 30

    def test_default_search_section(self):
        cfg = default_config()
        assert isinstance(cfg.search, SearchConfig)
        assert cfg.search.default_wait_for_quiescence_ms == 500

    def test_default_logging_section(self):
        cfg = default_config()
        assert isinstance(cfg.logging, LoggingConfig)
        assert cfg.logging.level == "INFO"
        assert cfg.logging.file_logging is True
        assert cfg.logging.redact_query_text is False
        assert cfg.logging.redact_cwd is False
        assert cfg.logging.replay_retention_days == 7
        assert cfg.logging.replay_active_max_gb == 1.0
        assert cfg.logging.replay_active_max_days == 30

    def test_dataclasses_are_frozen(self):
        cfg = default_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.corpus.parent_dir = Path("/somewhere")  # type: ignore


# --------------------------------------------------------------- round-trip


class TestRoundTrip:
    def test_write_then_load_default(self, tmp_path):
        path = tmp_path / "config.toml"
        write_config(path, default_config())
        assert path.is_file()
        cfg = load_config(path)
        defaults = default_config()
        assert cfg == defaults

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deeply" / "nested" / "config.toml"
        write_config(path, default_config())
        assert path.is_file()

    def test_round_trip_preserves_custom_corpus(self, tmp_path):
        cfg = default_config()
        custom = CorpusConfig(
            parent_dir=Path("/srv/code"),
            auto_discover=False,
            sentinels=("Cargo.toml", ".git"),
            include=("foo", "bar"),
            exclude=(),
            exclude_patterns=("**/build", "**/dist"),
            secret_patterns=("*.key",),
            allow_secret_patterns=False,
            forbidden_dirs=frozenset({"secrets"}),
            respect_gitignore=False,
            max_file_size_kb=2048,
            include_extensions=(".rs", ".py"),
        )
        modified = ServiceConfig(
            corpus=custom,
            server=cfg.server,
            index=cfg.index,
            watcher=cfg.watcher,
            cleanup=cfg.cleanup,
            search=cfg.search,
            logging=cfg.logging,
        )
        p = tmp_path / "x.toml"
        write_config(p, modified)
        loaded = load_config(p)
        assert loaded.corpus == custom

    def test_round_trip_preserves_custom_server(self, tmp_path):
        cfg = default_config()
        new_server = ServerConfig(
            mcp_port_preferred=9000,
            status_port_preferred=9001,
            mcp_host="127.0.0.1",
            mcp_transports=("sse",),
        )
        modified = ServiceConfig(
            corpus=cfg.corpus, server=new_server, index=cfg.index,
            watcher=cfg.watcher, cleanup=cfg.cleanup,
            search=cfg.search, logging=cfg.logging,
        )
        p = tmp_path / "y.toml"
        write_config(p, modified)
        assert load_config(p).server == new_server

    def test_round_trip_preserves_custom_logging(self, tmp_path):
        cfg = default_config()
        new_log = LoggingConfig(
            level="DEBUG", file_logging=False,
            redact_query_text=True, redact_cwd=True,
            replay_retention_days=14, replay_active_max_gb=2.5,
            replay_active_max_days=60,
        )
        modified = ServiceConfig(
            corpus=cfg.corpus, server=cfg.server, index=cfg.index,
            watcher=cfg.watcher, cleanup=cfg.cleanup,
            search=cfg.search, logging=new_log,
        )
        p = tmp_path / "z.toml"
        write_config(p, modified)
        assert load_config(p).logging == new_log

    def test_load_missing_file_returns_defaults(self, tmp_path):
        # Doesn't exist
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg == default_config()

    def test_load_explicit_path_doesnt_touch_default_path(
        self, tmp_path, monkeypatch,
    ):
        # Even if config_path() points elsewhere, we use the override.
        path = tmp_path / "x.toml"
        write_config(path, default_config())
        assert load_config(path) == default_config()


# --------------------------------------------------------------- env overrides


class TestEnvOverrides:
    def test_parent_dir_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_PARENT_DIR", "/custom/path")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.corpus.parent_dir == Path("/custom/path")

    def test_auto_discover_via_env_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_AUTO_DISCOVER", "false")
        # When auto_discover is False we need a non-empty include list
        # for validation; set via env too.
        monkeypatch.setenv("LONGCTX_INCLUDE", "foo,bar")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.corpus.auto_discover is False
        assert cfg.corpus.include == ("foo", "bar")

    def test_auto_discover_via_env_true_variants(self, tmp_path, monkeypatch):
        # Liberal in what we accept
        for val in ("true", "1", "yes", "on", "TRUE"):
            monkeypatch.setenv("LONGCTX_AUTO_DISCOVER", val)
            cfg = load_config(tmp_path / "absent.toml")
            assert cfg.corpus.auto_discover is True

    def test_mcp_port_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_MCP_PORT", "9999")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.server.mcp_port_preferred == 9999

    def test_mcp_host_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_MCP_HOST", "0.0.0.0")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.server.mcp_host == "0.0.0.0"

    def test_embedder_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_EMBEDDER", "BAAI/bge-base-en-v1.5")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.index.embedder == "BAAI/bge-base-en-v1.5"

    def test_log_level_via_env_uppercased(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_LOG_LEVEL", "debug")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.logging.level == "DEBUG"

    def test_chunk_size_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_CHUNK_SIZE", "4096")
        monkeypatch.setenv("LONGCTX_CHUNK_OVERLAP", "256")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.index.chunk_size == 4096
        assert cfg.index.chunk_overlap == 256

    def test_disk_budget_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_DISK_BUDGET_GB", "5.5")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.index.disk_budget_gb == 5.5

    def test_env_overrides_file_values(self, tmp_path, monkeypatch):
        # File says port 1234; env says 5678. Env wins.
        cfg = default_config()
        new_server = ServerConfig(
            mcp_port_preferred=1234,
            status_port_preferred=1235,
            mcp_host="127.0.0.1",
            mcp_transports=("sse",),
        )
        modified = ServiceConfig(
            corpus=cfg.corpus, server=new_server, index=cfg.index,
            watcher=cfg.watcher, cleanup=cfg.cleanup,
            search=cfg.search, logging=cfg.logging,
        )
        p = tmp_path / "x.toml"
        write_config(p, modified)
        monkeypatch.setenv("LONGCTX_MCP_PORT", "5678")
        loaded = load_config(p)
        assert loaded.server.mcp_port_preferred == 5678
        # Other fields preserved from file
        assert loaded.server.mcp_transports == ("sse",)

    def test_unknown_longctx_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_UNKNOWN_FIELD", "x")
        with pytest.raises(ConfigError, match="unknown LONGCTX"):
            load_config(tmp_path / "absent.toml")

    def test_longctx_ts_env_is_allowed(self, tmp_path, monkeypatch):
        # Reserved for tree-sitter; should not error.
        monkeypatch.setenv("LONGCTX_TS", "1")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg == default_config()

    def test_bad_int_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_MCP_PORT", "not-a-number")
        with pytest.raises(ConfigError, match="expected integer"):
            load_config(tmp_path / "absent.toml")

    def test_bad_bool_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_AUTO_DISCOVER", "maybe")
        with pytest.raises(ConfigError, match="expected boolean"):
            load_config(tmp_path / "absent.toml")

    def test_csv_env_strips_whitespace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LONGCTX_INCLUDE", " a , b ,c, ")
        monkeypatch.setenv("LONGCTX_AUTO_DISCOVER", "false")
        cfg = load_config(tmp_path / "absent.toml")
        assert cfg.corpus.include == ("a", "b", "c")


# --------------------------------------------------------------- validation


class TestValidation:
    def test_malformed_toml_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("this is = not [valid] toml ===")
        with pytest.raises(ConfigError, match="malformed TOML"):
            load_config(p)

    def test_unknown_section_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text(
            '[mystery]\nfoo = 1\n[corpus]\nparent_dir = "/tmp"\n'
        )
        with pytest.raises(ConfigError, match="unknown config section"):
            load_config(p)

    def test_section_not_table_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text("corpus = 5\n")
        with pytest.raises(ConfigError, match=r"\[corpus\] must be"):
            load_config(p)

    def test_empty_parent_dir_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[corpus]\nparent_dir = ""\n')
        with pytest.raises(ConfigError, match="parent_dir must not be empty"):
            load_config(p)

    def test_no_auto_discover_without_include_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[corpus]\nauto_discover = false\ninclude = []\n')
        with pytest.raises(ConfigError, match="auto_discover=false requires"):
            load_config(p)

    def test_auto_discover_with_both_include_and_exclude_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text(
            '[corpus]\nauto_discover = true\n'
            'include = ["foo"]\nexclude = ["bar"]\n'
        )
        with pytest.raises(
            ConfigError,
            match="cannot set both `include`",
        ):
            load_config(p)

    def test_empty_sentinels_when_auto_discover_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[corpus]\nauto_discover = true\nsentinels = []\n')
        with pytest.raises(ConfigError, match="sentinels"):
            load_config(p)

    def test_negative_max_file_size_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text("[corpus]\nmax_file_size_kb = -1\n")
        with pytest.raises(ConfigError, match="max_file_size_kb must be >= 0"):
            load_config(p)

    def test_invalid_port_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text("[server]\nmcp_port_preferred = 99999\n")
        with pytest.raises(ConfigError, match="must be 1..65535"):
            load_config(p)

    def test_invalid_transport_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text(
            '[server]\nmcp_transports = ["websocket", "smoke-signal"]\n'
        )
        with pytest.raises(ConfigError, match="invalid entries"):
            load_config(p)

    def test_empty_transports_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[server]\nmcp_transports = []\n')
        with pytest.raises(ConfigError, match="must be non-empty"):
            load_config(p)

    def test_empty_embedder_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[index]\nembedder = ""\n')
        with pytest.raises(ConfigError, match="embedder must be non-empty"):
            load_config(p)

    def test_chunk_size_zero_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[index]\nchunk_size = 0\n')
        with pytest.raises(ConfigError, match="chunk_size must be > 0"):
            load_config(p)

    def test_chunk_overlap_negative_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[index]\nchunk_overlap = -1\n')
        with pytest.raises(ConfigError, match="chunk_overlap must be >= 0"):
            load_config(p)

    def test_overlap_greater_than_size_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[index]\nchunk_size = 100\nchunk_overlap = 200\n')
        with pytest.raises(ConfigError, match="must be < chunk_size"):
            load_config(p)

    def test_negative_disk_budget_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[index]\ndisk_budget_gb = -1.0\n')
        with pytest.raises(ConfigError, match="disk_budget_gb"):
            load_config(p)

    def test_negative_debounce_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[watcher]\ndebounce_ms = -10\n')
        with pytest.raises(ConfigError, match="debounce_ms"):
            load_config(p)

    def test_zero_queue_size_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[watcher]\nqueue_size = 0\n')
        with pytest.raises(ConfigError, match="queue_size must be > 0"):
            load_config(p)

    def test_zero_mtime_sweep_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[watcher]\nperiodic_mtime_sweep_seconds = 0\n')
        with pytest.raises(ConfigError, match="periodic_mtime_sweep_seconds"):
            load_config(p)

    def test_invalid_log_level_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[logging]\nlevel = "EVERYTHING"\n')
        with pytest.raises(ConfigError, match=r"\[logging\].level"):
            load_config(p)

    def test_negative_replay_retention_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[logging]\nreplay_retention_days = -1\n')
        with pytest.raises(ConfigError, match="replay_retention_days"):
            load_config(p)

    def test_negative_search_wait_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[search]\ndefault_wait_for_quiescence_ms = -5\n')
        with pytest.raises(ConfigError, match="default_wait_for_quiescence_ms"):
            load_config(p)

    def test_negative_grace_days_raises(self, tmp_path):
        p = tmp_path / "x.toml"
        p.write_text('[cleanup]\nmissing_root_grace_days = -2\n')
        with pytest.raises(ConfigError, match="missing_root_grace_days"):
            load_config(p)


# --------------------------------------------------------------- config_path


class TestConfigPath:
    def test_config_path_returns_path(self):
        p = config_path()
        assert isinstance(p, Path)
        assert p.name == "config.toml"
        # The directory should be under the user's config area on every
        # platform — we don't assert the exact value because platformdirs
        # is platform-dependent, but there should be a "longctx" segment.
        assert "longctx" in p.parts


# --------------------------------------------------------------- reconcile


def _proj(name: str, root: Path = Path("/tmp/x"),
          sentinel: str = ".git", ignored: bool = False) -> DiscoveredProject:
    return DiscoveredProject(
        name=name, root_path=root / name,
        sentinel=sentinel, has_longctxignore=ignored,
    )


class TestReconcile:
    def test_no_changes_no_diff(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg, discovered_now=[], currently_indexed=[],
        )
        assert diff.projects_to_add == ()
        assert diff.projects_to_remove == ()
        assert diff.embedder_changed is False
        assert diff.server_ports_changed is False
        assert diff.watcher_settings_changed is False
        assert diff.search_settings_changed is False
        assert diff.parent_dir_changed is False

    def test_new_project_added(self):
        cfg = default_config()
        new_proj = _proj("foo")
        diff = reconcile(
            cfg, cfg,
            discovered_now=[new_proj],
            currently_indexed=[],
        )
        assert diff.projects_to_add == (new_proj,)
        assert diff.projects_to_remove == ()

    def test_already_indexed_project_not_added(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo")],
            currently_indexed=["foo"],
        )
        assert diff.projects_to_add == ()
        assert diff.projects_to_remove == ()

    def test_removed_project_dropped(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[],
            currently_indexed=["foo"],
        )
        assert diff.projects_to_remove == ("foo",)

    def test_longctxignore_skipped(self):
        cfg = default_config()
        ignored = _proj("foo", ignored=True)
        diff = reconcile(
            cfg, cfg,
            discovered_now=[ignored],
            currently_indexed=[],
        )
        assert diff.projects_to_add == ()

    def test_auto_discover_include_acts_as_deny(self):
        # auto_discover=True + include=["foo"] → foo is denied
        new_corpus = CorpusConfig(
            parent_dir=Path("~/dev").expanduser(),
            auto_discover=True,
            sentinels=DEFAULT_SENTINELS,
            include=("foo",),
            exclude=(),
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
            secret_patterns=DEFAULT_SECRET_PATTERNS,
            allow_secret_patterns=False,
            forbidden_dirs=DEFAULT_FORBIDDEN_DIRS,
            respect_gitignore=True,
            max_file_size_kb=1024,
            include_extensions=(),
        )
        cfg = ServiceConfig(corpus=new_corpus)
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo"), _proj("bar")],
            currently_indexed=[],
        )
        names = [p.name for p in diff.projects_to_add]
        assert "foo" not in names
        assert "bar" in names

    def test_no_auto_discover_include_acts_as_allow(self):
        new_corpus = CorpusConfig(
            parent_dir=Path("~/dev").expanduser(),
            auto_discover=False,
            include=("foo",),
            exclude=(),
        )
        cfg = ServiceConfig(corpus=new_corpus)
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo"), _proj("bar")],
            currently_indexed=[],
        )
        names = [p.name for p in diff.projects_to_add]
        assert names == ["foo"]

    def test_embedder_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=IndexConfigSection(embedder="BAAI/bge-large-en-v1.5"),
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.embedder_changed is True

    def test_chunking_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=IndexConfigSection(chunk_size=4096),
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.chunking_changed is True
        assert diff.embedder_changed is False

    def test_port_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=ServerConfig(mcp_port_preferred=9000),
            index=old.index,
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.server_ports_changed is True

    def test_transport_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=ServerConfig(mcp_transports=("sse",)),
            index=old.index,
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.server_transports_changed is True

    def test_host_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=ServerConfig(mcp_host="0.0.0.0"),
            index=old.index,
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.server_host_changed is True

    def test_watcher_settings_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=old.index,
            watcher=WatcherConfigSection(debounce_ms=500),
            cleanup=old.cleanup,
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.watcher_settings_changed is True

    def test_cleanup_settings_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=old.index,
            watcher=old.watcher,
            cleanup=CleanupConfig(missing_root_grace_days=14),
            search=old.search,
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.cleanup_settings_changed is True

    def test_search_settings_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=old.index,
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=SearchConfig(default_wait_for_quiescence_ms=2000),
            logging=old.logging,
        )
        diff = reconcile(old, new)
        assert diff.search_settings_changed is True

    def test_logging_settings_change_detected(self):
        old = default_config()
        new = ServiceConfig(
            corpus=old.corpus,
            server=old.server,
            index=old.index,
            watcher=old.watcher,
            cleanup=old.cleanup,
            search=old.search,
            logging=LoggingConfig(level="DEBUG"),
        )
        diff = reconcile(old, new)
        assert diff.logging_settings_changed is True

    def test_corpus_filters_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            include=old.corpus.include,
            exclude=old.corpus.exclude,
            exclude_patterns=("**/something_new",),
            secret_patterns=old.corpus.secret_patterns,
            allow_secret_patterns=old.corpus.allow_secret_patterns,
            forbidden_dirs=old.corpus.forbidden_dirs,
            respect_gitignore=old.corpus.respect_gitignore,
            max_file_size_kb=old.corpus.max_file_size_kb,
            include_extensions=old.corpus.include_extensions,
        )
        new = ServiceConfig(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True

    def test_parent_dir_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(parent_dir=Path("/srv/code"))
        new = ServiceConfig(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.parent_dir_changed is True

    def test_explicit_deny_via_exclude_drops_existing(self):
        # Auto-discover, but exclude = ("foo",); foo currently indexed
        # → reconcile says "remove foo".
        new_corpus = CorpusConfig(
            parent_dir=Path("~/dev").expanduser(),
            auto_discover=True,
            include=(),
            exclude=("foo",),
        )
        cfg = ServiceConfig(corpus=new_corpus)
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo"), _proj("bar")],
            currently_indexed=["foo", "bar"],
        )
        assert "foo" in diff.projects_to_remove
        assert "bar" not in diff.projects_to_remove

    def test_remove_when_not_in_discovered(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("bar")],
            currently_indexed=["foo", "bar"],
        )
        assert "foo" in diff.projects_to_remove
        assert "bar" not in diff.projects_to_remove
