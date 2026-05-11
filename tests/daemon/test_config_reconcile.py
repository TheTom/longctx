"""Dedicated reconcile() unit tests.

Some overlap with test_config.py is intentional — these focus on
edge-case combinations the SIGHUP handler will hit in practice
(simultaneous embedder + ports + project changes; allow-list flips;
.longctxignore added between scans). The simple-diff cases live in
test_config.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_daemon.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_FORBIDDEN_DIRS,
    DEFAULT_PARENT_DIR,
    DEFAULT_SECRET_PATTERNS,
    DEFAULT_SENTINELS,
    CleanupConfig,
    ConfigDiff,
    CorpusConfig,
    IndexConfigSection,
    LoggingConfig,
    SearchConfig,
    ServerConfig,
    ServiceConfig,
    WatcherConfigSection,
    default_config,
    reconcile,
)
from longctx_daemon.discovery import DiscoveredProject


# --------------------------------------------------------------- helpers


def _proj(name: str, *, ignored: bool = False, sentinel: str = ".git",
          root: Path = Path("/tmp/dev")) -> DiscoveredProject:
    return DiscoveredProject(
        name=name,
        root_path=root / name,
        sentinel=sentinel,
        has_longctxignore=ignored,
    )


def _build(corpus: CorpusConfig | None = None,
           server: ServerConfig | None = None,
           index: IndexConfigSection | None = None,
           watcher: WatcherConfigSection | None = None,
           cleanup: CleanupConfig | None = None,
           search: SearchConfig | None = None,
           logging: LoggingConfig | None = None) -> ServiceConfig:
    base = default_config()
    return ServiceConfig(
        corpus=corpus or base.corpus,
        server=server or base.server,
        index=index or base.index,
        watcher=watcher or base.watcher,
        cleanup=cleanup or base.cleanup,
        search=search or base.search,
        logging=logging or base.logging,
    )


# --------------------------------------------------------------- shape


class TestDiffShape:
    def test_returns_config_diff_dataclass(self):
        cfg = default_config()
        diff = reconcile(cfg, cfg)
        assert isinstance(diff, ConfigDiff)

    def test_diff_is_frozen(self):
        cfg = default_config()
        diff = reconcile(cfg, cfg)
        with pytest.raises((AttributeError, TypeError)):
            diff.embedder_changed = True  # type: ignore

    def test_no_args_no_explicit_iterables_works(self):
        # Caller may omit discovered_now + currently_indexed
        diff = reconcile(default_config(), default_config())
        assert diff.projects_to_add == ()
        assert diff.projects_to_remove == ()


# --------------------------------------------------------------- combined


class TestCombinedDiffs:
    def test_simultaneous_embedder_and_port_change(self):
        old = default_config()
        new = _build(
            index=IndexConfigSection(embedder="other-model"),
            server=ServerConfig(mcp_port_preferred=9001),
        )
        diff = reconcile(old, new)
        assert diff.embedder_changed is True
        assert diff.server_ports_changed is True

    def test_project_add_with_embedder_change(self):
        old = default_config()
        new = _build(index=IndexConfigSection(embedder="other-model"))
        new_proj = _proj("foo")
        diff = reconcile(
            old, new,
            discovered_now=[new_proj],
            currently_indexed=[],
        )
        assert diff.embedder_changed is True
        assert diff.projects_to_add == (new_proj,)

    def test_simultaneous_add_and_remove(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("new")],
            currently_indexed=["gone"],
        )
        assert diff.projects_to_add[0].name == "new"
        assert diff.projects_to_remove == ("gone",)

    def test_simultaneous_logging_and_search_change(self):
        old = default_config()
        new = _build(
            logging=LoggingConfig(level="DEBUG"),
            search=SearchConfig(default_wait_for_quiescence_ms=1000),
        )
        diff = reconcile(old, new)
        assert diff.logging_settings_changed is True
        assert diff.search_settings_changed is True
        assert diff.embedder_changed is False


# --------------------------------------------------------------- ignore semantics


class TestLongctxIgnoreSemantics:
    def test_ignore_marker_appears_skips_add(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo", ignored=True)],
            currently_indexed=[],
        )
        assert diff.projects_to_add == ()

    def test_ignore_marker_does_not_remove_already_indexed(self):
        # If a user adds .longctxignore to a project that's already
        # indexed, reconcile DOES NOT auto-remove it. Reasoning:
        # explicit user action via `longctx clean` should be required
        # to drop existing data; the marker is a "don't add" hint.
        # (This may change in 2.2; tests pin the current contract.)
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[_proj("foo", ignored=True)],
            currently_indexed=["foo"],
        )
        # Project is "discovered_now" (just with the ignore marker), so
        # it's NOT in projects_to_remove.
        assert "foo" not in diff.projects_to_remove

    def test_no_changes_no_diff_minor(self):
        cfg = default_config()
        diff = reconcile(cfg, cfg)
        # Audit: every bool field is False
        bool_fields = (
            diff.embedder_changed,
            diff.server_ports_changed,
            diff.server_transports_changed,
            diff.server_host_changed,
            diff.watcher_settings_changed,
            diff.cleanup_settings_changed,
            diff.search_settings_changed,
            diff.logging_settings_changed,
            diff.chunking_changed,
            diff.corpus_filters_changed,
            diff.parent_dir_changed,
        )
        assert not any(bool_fields)


# --------------------------------------------------------------- allow-list flip


class TestAllowListSemantics:
    def test_flip_to_allow_only_adds_listed(self):
        # User flips auto_discover off + sets include=("a",); we should
        # only add "a" and remove anything else currently indexed.
        old = default_config()
        new = _build(
            corpus=CorpusConfig(
                parent_dir=Path("~/dev").expanduser(),
                auto_discover=False,
                include=("a",),
                exclude=(),
            ),
        )
        diff = reconcile(
            old, new,
            discovered_now=[_proj("a"), _proj("b"), _proj("c")],
            currently_indexed=["a", "b"],
        )
        # b was indexed, but isn't in include; should be removed.
        assert "b" in diff.projects_to_remove
        # a is already indexed; should NOT be in projects_to_add.
        assert all(p.name != "a" for p in diff.projects_to_add)

    def test_auto_discover_with_include_as_deny(self):
        # auto_discover=True with include=("a",): a is denied.
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=Path("~/dev").expanduser(),
            auto_discover=True,
            include=("a",),
            exclude=(),
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(
            old, new,
            discovered_now=[_proj("a"), _proj("b")],
            currently_indexed=[],
        )
        names = [p.name for p in diff.projects_to_add]
        assert names == ["b"]

    def test_empty_discovered_with_existing_indexed(self):
        # parent_dir was unmounted; nothing discovered. All currently
        # indexed projects show up as removals.
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[],
            currently_indexed=["a", "b", "c"],
        )
        assert sorted(diff.projects_to_remove) == ["a", "b", "c"]

    def test_remove_list_sorted(self):
        cfg = default_config()
        diff = reconcile(
            cfg, cfg,
            discovered_now=[],
            currently_indexed=["zebra", "apple", "mango"],
        )
        # Stable sorted output
        assert diff.projects_to_remove == ("apple", "mango", "zebra")


# --------------------------------------------------------------- corpus filters


class TestCorpusFiltersDiff:
    def test_secret_patterns_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            secret_patterns=("*.key", "*.pem"),
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True

    def test_respect_gitignore_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            respect_gitignore=False,
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True

    def test_max_file_size_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            max_file_size_kb=4096,
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True

    def test_include_extensions_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            include_extensions=(".py",),
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True

    def test_forbidden_dirs_change_detected(self):
        old = default_config()
        new_corpus = CorpusConfig(
            parent_dir=old.corpus.parent_dir,
            auto_discover=old.corpus.auto_discover,
            sentinels=old.corpus.sentinels,
            forbidden_dirs=frozenset({"new-forbidden"}),
        )
        new = _build(corpus=new_corpus)
        diff = reconcile(old, new)
        assert diff.corpus_filters_changed is True


# --------------------------------------------------------------- watcher diff granularity


class TestWatcherDiff:
    def test_debounce_change_only(self):
        old = default_config()
        new = _build(watcher=WatcherConfigSection(debounce_ms=400))
        diff = reconcile(old, new)
        assert diff.watcher_settings_changed is True

    def test_queue_size_change_only(self):
        old = default_config()
        new = _build(watcher=WatcherConfigSection(queue_size=20000))
        diff = reconcile(old, new)
        assert diff.watcher_settings_changed is True

    def test_periodic_sweep_change_only(self):
        old = default_config()
        new = _build(
            watcher=WatcherConfigSection(periodic_mtime_sweep_seconds=60),
        )
        diff = reconcile(old, new)
        assert diff.watcher_settings_changed is True


# --------------------------------------------------------------- cleanup diff


class TestCleanupDiff:
    def test_grace_days_change(self):
        old = default_config()
        new = _build(cleanup=CleanupConfig(missing_root_grace_days=30))
        diff = reconcile(old, new)
        assert diff.cleanup_settings_changed is True

    def test_idle_pause_days_change(self):
        old = default_config()
        new = _build(cleanup=CleanupConfig(watcher_idle_pause_days=60))
        diff = reconcile(old, new)
        assert diff.cleanup_settings_changed is True


# --------------------------------------------------------------- chunking diff


class TestChunkingDiff:
    def test_chunk_size_change_alone(self):
        old = default_config()
        new = _build(index=IndexConfigSection(chunk_size=1024))
        diff = reconcile(old, new)
        assert diff.chunking_changed is True
        assert diff.embedder_changed is False

    def test_chunk_overlap_change_alone(self):
        old = default_config()
        new = _build(index=IndexConfigSection(chunk_overlap=64))
        diff = reconcile(old, new)
        assert diff.chunking_changed is True

    def test_disk_budget_change_does_not_trigger_chunking(self):
        old = default_config()
        new = _build(index=IndexConfigSection(disk_budget_gb=10.0))
        diff = reconcile(old, new)
        assert diff.chunking_changed is False
        assert diff.embedder_changed is False
