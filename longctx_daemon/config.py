"""TOML config + auto-discovery for the longctx daemon (Phase 2.1).

Owns three things:

* The on-disk ``config.toml`` schema — load + write + defaults
* Environment-variable overrides (``LONGCTX_*`` per PRD §3.9)
* Pure ``reconcile`` step that diffs an old config against a new one
  and tells the daemon what action to take (project add/remove,
  embedder change, port rebind, etc.). The daemon (Agent F) wires the
  diff to SIGHUP; we just produce it.

Covers config format, live add/remove, env precedence, SIGHUP reload,
non-interactive defaults, the Tier-1 missing-root grace path, and
replay-log retention.

Public surface: ``ServiceConfig`` + section dataclasses, ``load_config``,
``write_config``, ``default_config``, ``config_path``, ``reconcile``,
plus ``ConfigError`` for bad input.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import tomli_w
import platformdirs

from longctx_daemon.discovery import DiscoveredProject


# --------------------------------------------------------------- exceptions

class ConfigError(ValueError):
    """Raised on malformed TOML, schema violations, or env-override
    coercion failures. Always carries a one-line user-facing message
    pointing at the field that's wrong; tests assert on its substring."""


# --------------------------------------------------------------- defaults

# Mirrored from PRD §3.2 — kept here as module constants so callers can
# extend (e.g. tests want to add a custom sentinel) without redefining
# the whole config.

DEFAULT_PARENT_DIR: Path = Path("~/dev").expanduser()

DEFAULT_SENTINELS: tuple[str, ...] = (
    ".git", "pyproject.toml", "package.json", "pnpm-workspace.yaml",
    "Cargo.toml", "go.mod", "Package.swift", "WORKSPACE", "BUILD.bazel",
    "pom.xml", "Gemfile", ".longctxinclude",
)

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "**/node_modules", "**/.git/objects", "**/__pycache__",
    "**/.venv", "**/venv", "**/.pytest_cache",
    "**/target", "**/build", "**/dist", "**/.next", "**/.svelte-kit",
    "**/*.lock", "**/*.min.js", "**/*.map",
)

DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env*", "*.key", "*.pem", "id_rsa*", "id_ed25519*",
    "*.p12", "*.pfx", "**/secrets/**", "**/credentials/**",
    "**/.aws/**", "**/.ssh/**", "**/.gnupg/**",
)

DEFAULT_FORBIDDEN_DIRS: frozenset[str] = frozenset({
    "secrets", "credentials", ".aws", ".ssh", ".gnupg",
})

VALID_TRANSPORTS: frozenset[str] = frozenset({
    "sse", "streamable-http", "stdio",
})

VALID_LOG_LEVELS: frozenset[str] = frozenset({
    "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL",
})


# --------------------------------------------------------------- sections

@dataclass(frozen=True)
class CorpusConfig:
    """``[corpus]`` section. See PRD §3.1, §3.2, §13.

    ``parent_dir`` is auto-discovered under for sentinel-marked subdirs.
    ``include`` doubles as deny-list when ``auto_discover=True`` and as
    explicit allow-list when ``auto_discover=False``.
    """
    parent_dir: Path = field(default_factory=lambda: DEFAULT_PARENT_DIR)
    auto_discover: bool = True
    sentinels: tuple[str, ...] = DEFAULT_SENTINELS
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS
    allow_secret_patterns: bool = False
    forbidden_dirs: frozenset[str] = DEFAULT_FORBIDDEN_DIRS
    respect_gitignore: bool = True
    max_file_size_kb: int = 1024
    include_extensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServerConfig:
    """``[server]`` section — MCP transports + port hints. See PRD §6.1, §8."""
    mcp_port_preferred: int = 8765
    status_port_preferred: int = 8766
    mcp_host: str = "127.0.0.1"
    mcp_transports: tuple[str, ...] = ("sse", "streamable-http")


@dataclass(frozen=True)
class IndexConfigSection:
    """``[index]`` section — embedder + chunking + disk budget. See PRD §4, §12.4.3.

    ``relevance_floor`` is the corpus-wide default dense-cosine cutoff
    that drives Phase 2.0.1's honest-retrieval behavior
    (``no_relevant_results=True`` when the top-1 dense cosine falls
    below the threshold). Default 0.50 is the safe-middle for a
    typical software-codebase corpus; per-corpus tuning is what
    ``longctx calibrate`` is for. 0.0 disables the floor entirely
    (legacy behavior — return whatever rank-1 wins).

    ``project_floors`` is the per-project override map. Keys are
    project names (matching ``Project.name``), values are floors
    that apply when the search scope resolves to that project.
    Phase 3 dogfood-audit found that 0.50 doesn't generalize — code
    corpora typically work at 0.50, mixed-content (docs+tests+code)
    needs ~0.60, prose-heavy corpora (knowledge bases / wikis) need
    ~0.65. Use ``longctx calibrate --project NAME --write-config``
    to populate.
    """
    embedder: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 2048
    chunk_overlap: int = 128
    disk_budget_gb: float = 0.0
    relevance_floor: float = 0.50
    project_floors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WatcherConfigSection:
    """``[watcher]`` section. See PRD §5.3, §5.5."""
    debounce_ms: int = 200
    queue_size: int = 10000
    periodic_mtime_sweep_seconds: int = 30


@dataclass(frozen=True)
class CleanupConfig:
    """``[cleanup]`` section. See PRD §12.4.1, §12.4.2."""
    missing_root_grace_days: int = 7
    periodic_check_seconds: int = 3600
    watcher_idle_pause_days: int = 30


@dataclass(frozen=True)
class SearchConfig:
    """``[search]`` section. See PRD §6.2 default ``wait_for_quiescence_ms``."""
    default_wait_for_quiescence_ms: int = 500


@dataclass(frozen=True)
class LoggingConfig:
    """``[logging]`` section — structured-log knobs. See PRD §14.8.1, §14.9."""
    level: str = "INFO"
    file_logging: bool = True
    redact_query_text: bool = False
    redact_cwd: bool = False
    replay_retention_days: int = 7
    replay_active_max_gb: float = 1.0
    replay_active_max_days: int = 30


@dataclass(frozen=True)
class ServiceConfig:
    """Top-level config. Composition of all sections.

    ``load_config`` produces this; the daemon hands the relevant slice to
    each component (indexer gets ``corpus`` + ``index``, watcher gets
    ``watcher`` + ``corpus``, etc.). Every dataclass is ``frozen`` so
    callers can pass slices around without worrying about shared
    mutation.
    """
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    index: IndexConfigSection = field(default_factory=IndexConfigSection)
    watcher: WatcherConfigSection = field(default_factory=WatcherConfigSection)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# --------------------------------------------------------------- diff for SIGHUP


@dataclass(frozen=True)
class ConfigDiff:
    """Pure description of what changed between two configs.

    Produced by ``reconcile``; consumed by the daemon's SIGHUP handler.
    Boolean fields are signals — caller decides whether to apply now,
    log a warning, or require a separate confirmation command (e.g.
    ``embedder_changed`` requires ``longctx reembed --confirm`` per PRD
    §4.6).
    """
    projects_to_add: tuple[DiscoveredProject, ...]
    projects_to_remove: tuple[str, ...]
    embedder_changed: bool
    server_ports_changed: bool
    server_transports_changed: bool
    server_host_changed: bool
    watcher_settings_changed: bool
    cleanup_settings_changed: bool
    search_settings_changed: bool
    logging_settings_changed: bool
    chunking_changed: bool
    corpus_filters_changed: bool
    parent_dir_changed: bool


# --------------------------------------------------------------- env vars

# PRD §3.9: env vars override config file but are overridden by CLI.
_ENV_PREFIX = "LONGCTX_"

# Every env-overridable field is enumerated here so unknown env vars
# can't silently change behavior. Format: env_key -> (section, field, coercer).
# Coercer signature: str -> typed value.
def _coerce_bool(s: str) -> bool:
    """Parse a boolish env var. Liberal in what we accept; strict
    on what fails."""
    low = s.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off", ""):
        return False
    raise ConfigError(f"expected boolean, got {s!r}")


def _coerce_int(s: str) -> int:
    try:
        return int(s.strip())
    except ValueError as exc:
        raise ConfigError(f"expected integer, got {s!r}") from exc


def _coerce_float(s: str) -> float:
    try:
        return float(s.strip())
    except ValueError as exc:
        raise ConfigError(f"expected float, got {s!r}") from exc


def _coerce_path(s: str) -> Path:
    return Path(s).expanduser()


def _coerce_csv(s: str) -> tuple[str, ...]:
    """Comma-separated list. Trims whitespace; drops empty entries."""
    return tuple(p.strip() for p in s.split(",") if p.strip())


_ENV_OVERRIDES: dict[str, tuple[str, str, Any]] = {
    "LONGCTX_PARENT_DIR": ("corpus", "parent_dir", _coerce_path),
    "LONGCTX_AUTO_DISCOVER": ("corpus", "auto_discover", _coerce_bool),
    "LONGCTX_INCLUDE": ("corpus", "include", _coerce_csv),
    "LONGCTX_EXCLUDE": ("corpus", "exclude", _coerce_csv),
    "LONGCTX_RESPECT_GITIGNORE": ("corpus", "respect_gitignore", _coerce_bool),
    "LONGCTX_MAX_FILE_SIZE_KB": ("corpus", "max_file_size_kb", _coerce_int),
    "LONGCTX_ALLOW_SECRET_PATTERNS": ("corpus", "allow_secret_patterns", _coerce_bool),
    "LONGCTX_MCP_PORT": ("server", "mcp_port_preferred", _coerce_int),
    "LONGCTX_STATUS_PORT": ("server", "status_port_preferred", _coerce_int),
    "LONGCTX_MCP_HOST": ("server", "mcp_host", lambda s: s.strip()),
    "LONGCTX_MCP_TRANSPORTS": ("server", "mcp_transports", _coerce_csv),
    "LONGCTX_EMBEDDER": ("index", "embedder", lambda s: s.strip()),
    "LONGCTX_CHUNK_SIZE": ("index", "chunk_size", _coerce_int),
    "LONGCTX_CHUNK_OVERLAP": ("index", "chunk_overlap", _coerce_int),
    "LONGCTX_DISK_BUDGET_GB": ("index", "disk_budget_gb", _coerce_float),
    "LONGCTX_DEBOUNCE_MS": ("watcher", "debounce_ms", _coerce_int),
    "LONGCTX_LOG_LEVEL": ("logging", "level", lambda s: s.strip().upper()),
}


def _apply_env_overrides(cfg: ServiceConfig) -> ServiceConfig:
    """Walk ``_ENV_OVERRIDES``, applying any set vars onto ``cfg``.

    Unknown ``LONGCTX_*`` env vars trigger a ``ConfigError`` so a typo
    can't silently no-op (the most painful failure mode in shell-driven
    config). Use ``LONGCTX_*_FOO`` rather than ``LONGCTX_FOO`` for
    daemon-internal vars (we reserve a few prefixes — currently only
    ``LONGCTX_TS`` for tree-sitter chunking, see PRD §18).
    """
    sections: dict[str, Any] = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    seen_unknown: list[str] = []
    for key, val in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        if key in _ENV_OVERRIDES:
            section, field_name, coercer = _ENV_OVERRIDES[key]
            try:
                typed = coercer(val)
            except ConfigError:
                raise
            current = sections[section]
            sections[section] = replace(current, **{field_name: typed})
        else:
            # Allow a small allowlist of daemon-internal vars to pass
            # without erroring. Anything else is a typo.
            if key in ("LONGCTX_TS", "LONGCTX_TS_LANGUAGES"):
                continue
            seen_unknown.append(key)
    if seen_unknown:
        raise ConfigError(
            "unknown LONGCTX_* environment variable(s): "
            + ", ".join(sorted(seen_unknown))
        )
    return ServiceConfig(**sections)


# --------------------------------------------------------------- TOML I/O

def config_path() -> Path:
    """Default config location, resolved via ``platformdirs``.

    macOS: ``~/Library/Application Support/longctx/config.toml``
    Linux: ``~/.config/longctx/config.toml``
    Windows: ``%APPDATA%\\longctx\\config.toml``
    """
    base = Path(platformdirs.user_config_dir("longctx", appauthor=False))
    return base / "config.toml"


def default_config() -> ServiceConfig:
    """Construct a ``ServiceConfig`` with all documented defaults.

    Useful for ``longctx init --non-interactive`` and as a baseline for
    test fixtures. No I/O — does not touch ``config_path()``.
    """
    return ServiceConfig()


def write_config(path: Path, config: ServiceConfig) -> None:
    """Serialize ``config`` to TOML at ``path``.

    Creates parent directories if missing. Writes with permissive 0644
    mode; the daemon's ``server.lock`` + ``server.info`` are the
    privileged files (PRD §13.1).
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _config_to_dict(config)
    with path.open("wb") as f:
        tomli_w.dump(doc, f)


def load_config(path: Optional[Path] = None) -> ServiceConfig:
    """Load TOML config from ``path`` (default: ``config_path()``).

    Order of resolution (PRD §3.9 minus per-call MCP args + sticky
    session, which are runtime-only):

      1. CLI flags — caller's responsibility, not done here
      2. Env vars (``LONGCTX_*``)
      3. Config file
      4. Defaults

    Missing file → defaults + env. Malformed TOML → ``ConfigError``.
    """
    if path is None:
        path = config_path()
    path = Path(path).expanduser()
    if path.is_file():
        try:
            with path.open("rb") as f:
                raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"malformed TOML in {path}: {exc}") from exc
        cfg = _config_from_dict(raw, source_path=path)
    else:
        cfg = default_config()
    cfg = _apply_env_overrides(cfg)
    _validate(cfg)
    return cfg


# ---------------------------------------------------------------- coercion helpers

def _config_to_dict(cfg: ServiceConfig) -> dict[str, Any]:
    """Convert ``ServiceConfig`` to a plain TOML-friendly dict."""
    return {
        "corpus": {
            "parent_dir": str(cfg.corpus.parent_dir),
            "auto_discover": cfg.corpus.auto_discover,
            "sentinels": list(cfg.corpus.sentinels),
            "include": list(cfg.corpus.include),
            "exclude": list(cfg.corpus.exclude),
            "exclude_patterns": list(cfg.corpus.exclude_patterns),
            "secret_patterns": list(cfg.corpus.secret_patterns),
            "allow_secret_patterns": cfg.corpus.allow_secret_patterns,
            "forbidden_dirs": sorted(cfg.corpus.forbidden_dirs),
            "respect_gitignore": cfg.corpus.respect_gitignore,
            "max_file_size_kb": cfg.corpus.max_file_size_kb,
            "include_extensions": list(cfg.corpus.include_extensions),
        },
        "server": {
            "mcp_port_preferred": cfg.server.mcp_port_preferred,
            "status_port_preferred": cfg.server.status_port_preferred,
            "mcp_host": cfg.server.mcp_host,
            "mcp_transports": list(cfg.server.mcp_transports),
        },
        "index": {
            "embedder": cfg.index.embedder,
            "chunk_size": cfg.index.chunk_size,
            "chunk_overlap": cfg.index.chunk_overlap,
            "disk_budget_gb": cfg.index.disk_budget_gb,
            "relevance_floor": cfg.index.relevance_floor,
            "project_floors": dict(cfg.index.project_floors),
        },
        "watcher": {
            "debounce_ms": cfg.watcher.debounce_ms,
            "queue_size": cfg.watcher.queue_size,
            "periodic_mtime_sweep_seconds": cfg.watcher.periodic_mtime_sweep_seconds,
        },
        "cleanup": {
            "missing_root_grace_days": cfg.cleanup.missing_root_grace_days,
            "periodic_check_seconds": cfg.cleanup.periodic_check_seconds,
            "watcher_idle_pause_days": cfg.cleanup.watcher_idle_pause_days,
        },
        "search": {
            "default_wait_for_quiescence_ms": cfg.search.default_wait_for_quiescence_ms,
        },
        "logging": {
            "level": cfg.logging.level,
            "file_logging": cfg.logging.file_logging,
            "redact_query_text": cfg.logging.redact_query_text,
            "redact_cwd": cfg.logging.redact_cwd,
            "replay_retention_days": cfg.logging.replay_retention_days,
            "replay_active_max_gb": cfg.logging.replay_active_max_gb,
            "replay_active_max_days": cfg.logging.replay_active_max_days,
        },
    }


def _expect_table(raw: dict[str, Any], key: str, source_path: Path) -> dict[str, Any]:
    """Pull ``raw[key]`` as a dict, defaulting to {} if absent. Type
    error if present but not a table."""
    val = raw.get(key, {})
    if not isinstance(val, dict):
        raise ConfigError(
            f"[{key}] must be a TOML table in {source_path}, "
            f"got {type(val).__name__}"
        )
    return val


def _config_from_dict(raw: dict[str, Any], source_path: Path) -> ServiceConfig:
    """Inverse of ``_config_to_dict``. Missing sections / fields → defaults.

    Unknown keys are tolerated (forward compat); unknown TOP-LEVEL
    sections trigger a warning-level ``ConfigError`` so users catch
    typos.
    """
    known_sections = {
        "corpus", "server", "index", "watcher", "cleanup", "search", "logging",
    }
    unknown_sections = set(raw.keys()) - known_sections
    if unknown_sections:
        raise ConfigError(
            f"unknown config section(s) in {source_path}: "
            f"{', '.join(sorted(unknown_sections))}. Known: "
            f"{', '.join(sorted(known_sections))}."
        )

    corpus_raw = _expect_table(raw, "corpus", source_path)
    server_raw = _expect_table(raw, "server", source_path)
    index_raw = _expect_table(raw, "index", source_path)
    watcher_raw = _expect_table(raw, "watcher", source_path)
    cleanup_raw = _expect_table(raw, "cleanup", source_path)
    search_raw = _expect_table(raw, "search", source_path)
    logging_raw = _expect_table(raw, "logging", source_path)

    raw_parent_dir = corpus_raw.get("parent_dir", str(DEFAULT_PARENT_DIR))
    if not isinstance(raw_parent_dir, str):
        raise ConfigError(
            f"[corpus].parent_dir must be a string, got "
            f"{type(raw_parent_dir).__name__}"
        )
    if raw_parent_dir.strip() == "":
        raise ConfigError("[corpus].parent_dir must not be empty")
    corpus = CorpusConfig(
        parent_dir=_coerce_path(raw_parent_dir),
        auto_discover=bool(corpus_raw.get("auto_discover", True)),
        sentinels=tuple(corpus_raw.get("sentinels", DEFAULT_SENTINELS)),
        include=tuple(corpus_raw.get("include", ())),
        exclude=tuple(corpus_raw.get("exclude", ())),
        exclude_patterns=tuple(corpus_raw.get("exclude_patterns", DEFAULT_EXCLUDE_PATTERNS)),
        secret_patterns=tuple(corpus_raw.get("secret_patterns", DEFAULT_SECRET_PATTERNS)),
        allow_secret_patterns=bool(corpus_raw.get("allow_secret_patterns", False)),
        forbidden_dirs=frozenset(
            corpus_raw.get("forbidden_dirs", DEFAULT_FORBIDDEN_DIRS)
        ),
        respect_gitignore=bool(corpus_raw.get("respect_gitignore", True)),
        max_file_size_kb=int(corpus_raw.get("max_file_size_kb", 1024)),
        include_extensions=tuple(corpus_raw.get("include_extensions", ())),
    )
    server = ServerConfig(
        mcp_port_preferred=int(server_raw.get("mcp_port_preferred", 8765)),
        status_port_preferred=int(server_raw.get("status_port_preferred", 8766)),
        mcp_host=str(server_raw.get("mcp_host", "127.0.0.1")),
        mcp_transports=tuple(server_raw.get("mcp_transports", ("sse", "streamable-http"))),
    )
    project_floors_raw = index_raw.get("project_floors", {}) or {}
    if not isinstance(project_floors_raw, dict):
        raise ConfigError(
            f"[index].project_floors must be a table mapping project "
            f"names to floors, got {type(project_floors_raw).__name__}"
        )
    project_floors: dict[str, float] = {}
    for k, v in project_floors_raw.items():
        try:
            project_floors[str(k)] = float(v)
        except (TypeError, ValueError) as e:
            raise ConfigError(
                f"[index].project_floors[{k!r}] must be a number, got {v!r}"
            ) from e
    index = IndexConfigSection(
        embedder=str(index_raw.get("embedder", "BAAI/bge-small-en-v1.5")),
        chunk_size=int(index_raw.get("chunk_size", 2048)),
        chunk_overlap=int(index_raw.get("chunk_overlap", 128)),
        disk_budget_gb=float(index_raw.get("disk_budget_gb", 0.0)),
        relevance_floor=float(index_raw.get("relevance_floor", 0.50)),
        project_floors=project_floors,
    )
    watcher = WatcherConfigSection(
        debounce_ms=int(watcher_raw.get("debounce_ms", 200)),
        queue_size=int(watcher_raw.get("queue_size", 10000)),
        periodic_mtime_sweep_seconds=int(
            watcher_raw.get("periodic_mtime_sweep_seconds", 30)
        ),
    )
    cleanup = CleanupConfig(
        missing_root_grace_days=int(cleanup_raw.get("missing_root_grace_days", 7)),
        periodic_check_seconds=int(cleanup_raw.get("periodic_check_seconds", 3600)),
        watcher_idle_pause_days=int(cleanup_raw.get("watcher_idle_pause_days", 30)),
    )
    search = SearchConfig(
        default_wait_for_quiescence_ms=int(
            search_raw.get("default_wait_for_quiescence_ms", 500)
        ),
    )
    logging_section = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")).upper(),
        file_logging=bool(logging_raw.get("file_logging", True)),
        redact_query_text=bool(logging_raw.get("redact_query_text", False)),
        redact_cwd=bool(logging_raw.get("redact_cwd", False)),
        replay_retention_days=int(logging_raw.get("replay_retention_days", 7)),
        replay_active_max_gb=float(logging_raw.get("replay_active_max_gb", 1.0)),
        replay_active_max_days=int(logging_raw.get("replay_active_max_days", 30)),
    )
    return ServiceConfig(
        corpus=corpus, server=server, index=index, watcher=watcher,
        cleanup=cleanup, search=search, logging=logging_section,
    )


# --------------------------------------------------------------- validation


def _validate(cfg: ServiceConfig) -> None:
    """Raise ``ConfigError`` on any nonsensical combination.

    Validation runs after env-overrides so a malicious env var can't
    sneak past. The daemon calls this once at startup; the wizard +
    ``longctx reload`` call it before write/apply.
    """
    # corpus.parent_dir must look like a path (we don't require it to
    # exist — handled at daemon-start refuse-to-do-stupid checks per
    # PRD §12.3 which is Agent F's deliverable). But it must not be
    # empty.
    if str(cfg.corpus.parent_dir).strip() == "":
        raise ConfigError("[corpus].parent_dir must not be empty")

    # auto_discover + include conflict: if auto_discover=False and
    # include is empty, we'd index nothing — usually a typo.
    if not cfg.corpus.auto_discover and not cfg.corpus.include:
        raise ConfigError(
            "[corpus].auto_discover=false requires a non-empty `include` "
            "list (otherwise nothing would be indexed)"
        )

    # When auto_discover=True, `include` is a deny-list semantically
    # equivalent to `exclude`. Forbid setting both to avoid ambiguity.
    if cfg.corpus.auto_discover and cfg.corpus.include and cfg.corpus.exclude:
        raise ConfigError(
            "[corpus] cannot set both `include` (deny-list when "
            "auto_discover=true) and `exclude` simultaneously; pick one"
        )

    # Sentinels list must be non-empty when auto_discover is on
    if cfg.corpus.auto_discover and not cfg.corpus.sentinels:
        raise ConfigError(
            "[corpus].sentinels must be non-empty when auto_discover=true"
        )

    if cfg.corpus.max_file_size_kb < 0:
        raise ConfigError(
            f"[corpus].max_file_size_kb must be >= 0, got "
            f"{cfg.corpus.max_file_size_kb}"
        )

    # Server: ports in 1024..65535
    for name, val in (
        ("mcp_port_preferred", cfg.server.mcp_port_preferred),
        ("status_port_preferred", cfg.server.status_port_preferred),
    ):
        if not (1 <= val <= 65535):
            raise ConfigError(
                f"[server].{name} must be 1..65535, got {val}"
            )

    bad_transports = set(cfg.server.mcp_transports) - VALID_TRANSPORTS
    if bad_transports:
        raise ConfigError(
            f"[server].mcp_transports has invalid entries: "
            f"{sorted(bad_transports)}. Valid: {sorted(VALID_TRANSPORTS)}."
        )
    if not cfg.server.mcp_transports:
        raise ConfigError("[server].mcp_transports must be non-empty")

    # Index: embedder must be a non-empty string; chunk sizes positive;
    # overlap < chunk_size.
    if not cfg.index.embedder.strip():
        raise ConfigError("[index].embedder must be non-empty")
    if cfg.index.chunk_size <= 0:
        raise ConfigError(
            f"[index].chunk_size must be > 0, got {cfg.index.chunk_size}"
        )
    if cfg.index.chunk_overlap < 0:
        raise ConfigError(
            f"[index].chunk_overlap must be >= 0, got {cfg.index.chunk_overlap}"
        )
    if cfg.index.chunk_overlap >= cfg.index.chunk_size:
        raise ConfigError(
            f"[index].chunk_overlap ({cfg.index.chunk_overlap}) must be < "
            f"chunk_size ({cfg.index.chunk_size})"
        )
    if cfg.index.disk_budget_gb < 0:
        raise ConfigError(
            f"[index].disk_budget_gb must be >= 0, got {cfg.index.disk_budget_gb}"
        )
    # 0.0 disables the floor; 1.0 would suppress everything (cosines are
    # bounded in [-1, 1] but in practice in [0, 1]). >1 is nonsensical.
    for proj, floor in cfg.index.project_floors.items():
        if not (0.0 <= floor <= 1.0):
            raise ConfigError(
                f"[index].project_floors[{proj!r}] must be in "
                f"[0.0, 1.0], got {floor}"
            )
    if not (0.0 <= cfg.index.relevance_floor <= 1.0):
        raise ConfigError(
            f"[index].relevance_floor must be in [0.0, 1.0], got "
            f"{cfg.index.relevance_floor}"
        )

    # Watcher
    if cfg.watcher.debounce_ms < 0:
        raise ConfigError(
            f"[watcher].debounce_ms must be >= 0, got {cfg.watcher.debounce_ms}"
        )
    if cfg.watcher.queue_size <= 0:
        raise ConfigError(
            f"[watcher].queue_size must be > 0, got {cfg.watcher.queue_size}"
        )
    if cfg.watcher.periodic_mtime_sweep_seconds <= 0:
        raise ConfigError(
            f"[watcher].periodic_mtime_sweep_seconds must be > 0, got "
            f"{cfg.watcher.periodic_mtime_sweep_seconds}"
        )

    # Cleanup
    if cfg.cleanup.missing_root_grace_days < 0:
        raise ConfigError(
            "[cleanup].missing_root_grace_days must be >= 0"
        )
    if cfg.cleanup.periodic_check_seconds <= 0:
        raise ConfigError(
            "[cleanup].periodic_check_seconds must be > 0"
        )

    # Search
    if cfg.search.default_wait_for_quiescence_ms < 0:
        raise ConfigError(
            "[search].default_wait_for_quiescence_ms must be >= 0"
        )

    # Logging
    if cfg.logging.level not in VALID_LOG_LEVELS:
        raise ConfigError(
            f"[logging].level={cfg.logging.level!r} not in "
            f"{sorted(VALID_LOG_LEVELS)}"
        )
    if cfg.logging.replay_retention_days < 0:
        raise ConfigError(
            "[logging].replay_retention_days must be >= 0"
        )
    if cfg.logging.replay_active_max_gb < 0:
        raise ConfigError(
            "[logging].replay_active_max_gb must be >= 0"
        )
    if cfg.logging.replay_active_max_days < 0:
        raise ConfigError(
            "[logging].replay_active_max_days must be >= 0"
        )

    # Privacy: forbidden_dirs ∩ secret_patterns must NOT be removable
    # by setting allow_secret_patterns alone — that flag covers per-file
    # globs only. Don't gate forbidden_dirs at config-load time; the
    # indexer enforces it (see ``ForbiddenProjectError``).


# --------------------------------------------------------------- reconcile


def reconcile(
    old: ServiceConfig,
    new: ServiceConfig,
    *,
    discovered_now: Iterable[DiscoveredProject] = (),
    currently_indexed: Iterable[str] = (),
) -> ConfigDiff:
    """Pure diff: what changed between ``old`` and ``new``?

    The daemon's SIGHUP handler calls this with the freshly-loaded
    config and the current set of indexed project names, then acts on
    each non-zero field. Pure: no I/O, no mutation. Caller is responsible
    for project discovery (we just take the result so this stays pure).

    Args:
      old: previously loaded ServiceConfig
      new: newly loaded ServiceConfig
      discovered_now: projects ``discover_projects(new.corpus.parent_dir, ...)``
        would return RIGHT NOW. Caller pre-computes; we don't walk the
        filesystem here.
      currently_indexed: names of projects in the daemon's index.

    Returns:
      ``ConfigDiff`` carrying actionable signals. Caller picks which to
      apply (some are immediate — port rebinds happen on next restart,
      not mid-flight; embedder change requires explicit confirmation).
    """
    discovered_names = {p.name for p in discovered_now}
    currently_indexed_set = set(currently_indexed)

    # Project add: discovered now AND not currently indexed AND not
    # in the deny list. The deny list semantics depend on auto_discover.
    deny: set[str] = set()
    if new.corpus.auto_discover:
        # When auto-discovering, both `include` (deny-list) and
        # `exclude` are dropped from the candidate set.
        deny = set(new.corpus.include) | set(new.corpus.exclude)
    else:
        # When NOT auto-discovering, only names in `include` survive.
        # Anything else is implicitly denied.
        allowed = set(new.corpus.include)
        deny = discovered_names - allowed

    to_add: tuple[DiscoveredProject, ...] = tuple(
        p for p in discovered_now
        if p.name not in currently_indexed_set
        and p.name not in deny
        and not p.has_longctxignore
    )

    # Project remove: in the index now BUT not in discovered_now (or
    # explicitly denied). The daemon also drops projects whose
    # root_path is gone via Tier 1 cleanup; this only handles the
    # config-level diff.
    to_remove: tuple[str, ...] = tuple(sorted(
        n for n in currently_indexed_set
        if n not in discovered_names or n in deny
    ))

    return ConfigDiff(
        projects_to_add=to_add,
        projects_to_remove=to_remove,
        embedder_changed=old.index.embedder != new.index.embedder,
        server_ports_changed=(
            old.server.mcp_port_preferred != new.server.mcp_port_preferred
            or old.server.status_port_preferred != new.server.status_port_preferred
        ),
        server_transports_changed=(
            tuple(old.server.mcp_transports) != tuple(new.server.mcp_transports)
        ),
        server_host_changed=old.server.mcp_host != new.server.mcp_host,
        watcher_settings_changed=old.watcher != new.watcher,
        cleanup_settings_changed=old.cleanup != new.cleanup,
        search_settings_changed=old.search != new.search,
        logging_settings_changed=old.logging != new.logging,
        chunking_changed=(
            old.index.chunk_size != new.index.chunk_size
            or old.index.chunk_overlap != new.index.chunk_overlap
        ),
        corpus_filters_changed=(
            old.corpus.exclude_patterns != new.corpus.exclude_patterns
            or old.corpus.secret_patterns != new.corpus.secret_patterns
            or old.corpus.respect_gitignore != new.corpus.respect_gitignore
            or old.corpus.max_file_size_kb != new.corpus.max_file_size_kb
            or old.corpus.include_extensions != new.corpus.include_extensions
            or old.corpus.forbidden_dirs != new.corpus.forbidden_dirs
        ),
        parent_dir_changed=old.corpus.parent_dir != new.corpus.parent_dir,
    )


# TODO(phase 2.3): config-migration helper (`from_v1_dict`) so we can
# rev the schema without breaking users' existing configs.
# TODO(phase 2.4): per-project overrides — allow nested `[corpus.projects.<name>]`
# tables for things like project-specific exclude_patterns.
