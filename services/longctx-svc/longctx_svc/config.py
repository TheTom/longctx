"""longctx-svc configuration: caps, sentinels, ignore patterns.

Maps to PRD §5.3 Caps and ignores. Defaults are conservative; users can
override via env vars (LONGCTX_*) or programmatic config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Project sentinels in priority order (PRD §5.1)
SENTINELS = [
    ".git",
    "pyproject.toml",
    "package.json",
    "pnpm-workspace.yaml",
    "Cargo.toml",
    "go.mod",
    "WORKSPACE",
    "BUILD.bazel",
    "pom.xml",
    "Gemfile",
]

# Always-skipped directories (PRD §5.3)
ALWAYS_SKIP_DIRS = frozenset({
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".tox",
})

# Lockfiles (always skipped)
LOCKFILE_NAMES = frozenset({
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "composer.lock",
    "Gemfile.lock",
})

# Code file extensions for chunking strategy selection
CODE_EXTS = frozenset({
    ".py", ".pyx", ".pyi",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".swift",
    ".java", ".kt", ".scala",
    ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hh",
    ".cs", ".fs", ".vb",
    ".rb", ".php", ".pl", ".lua", ".sh", ".bash", ".zsh", ".fish",
    ".sql",
})
PROSE_EXTS = frozenset({".md", ".markdown", ".txt", ".rst", ".adoc", ".org"})
CONFIG_EXTS = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".properties",
})

# Binary file extensions (skipped without magic-byte check for speed)
BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dylib", ".dll", ".o", ".a", ".lib", ".exe",
    ".pyc", ".pyo", ".class", ".jar",
    ".mp3", ".mp4", ".wav", ".flac", ".mov", ".avi", ".mkv", ".webm",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".sqlite", ".db", ".dat",
    ".pkl", ".npy", ".npz", ".bin", ".safetensors", ".gguf",
})


@dataclass
class Limits:
    """Caps and budgets per PRD §5.3."""
    max_file_size_bytes: int = 5 * 1024 * 1024            # 5 MB
    max_files_hot: int = 1000                              # PRD §5.2
    max_files_package: int = 50_000                        # PRD §5.2
    soft_index_seconds: int = 60
    hard_index_seconds: int = 5 * 60
    max_indexes_in_memory: int = 4                         # LRU eviction
    session_idle_timeout_seconds: int = 2 * 60 * 60        # 2h, PRD §5.7
    index_idle_timeout_seconds: int = 30 * 60              # 30 min, PRD §5.7
    chunk_lines_default: int = 50                          # PRD §5.4 line-window
    chunk_lines_overlap: int = 5
    watcher_debounce_seconds: float = 1.0                  # PRD §5.7
    # PRD §6.2 / v0.3.2 — confidence-driven promotion
    confidence_threshold: float = 0.30
    confidence_consecutive_low: int = 2
    # Scale-aware retrieval: bge-reranker-v2-m3 is 568M params on CPU
    # (~50ms/pair). With prefilter=100 that's ~5s per /retrieve. The
    # rerank is only worth it when the scope has enough chunks for the
    # ranking to matter. On small repos (<200 chunks) cosine R@8 is
    # already ~95%+, so skip rerank and save the 5s.
    rerank_min_chunks: int = 1000   # tiny/medium repos use cosine alone
    multiquery_min_chunks: int = 1000
    rerank_prefilter_small: int = 20   # cross-encoder pairs for <10k chunks
    rerank_prefilter_large: int = 100  # cross-encoder pairs at scale
    # Max tokens to ever splice into a request. 50-line code chunks at
    # ~30 tok/line × 8 = 12K tokens — blows out small-context windows.
    # Approximated as chars/4 (English ~4 chars/tok). Truncate longest
    # chunks first.
    splice_max_chars: int = 4096 * 4   # ~4K tokens

    @classmethod
    def from_env(cls) -> "Limits":
        return cls(
            max_file_size_bytes=int(os.environ.get(
                "LONGCTX_MAX_FILE_BYTES", cls.max_file_size_bytes)),
            max_files_hot=int(os.environ.get(
                "LONGCTX_MAX_HOT", cls.max_files_hot)),
            max_files_package=int(os.environ.get(
                "LONGCTX_MAX_PACKAGE", cls.max_files_package)),
        )


@dataclass
class ServiceConfig:
    """Top-level configuration."""
    cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("LONGCTX_CACHE_DIR")
        or (Path.home() / ".longctx")
    ))
    embedder_model: str = field(default_factory=lambda: os.environ.get(
        "LONGCTX_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"))
    reranker_model: str | None = field(default_factory=lambda: os.environ.get(
        "LONGCTX_RERANKER", "BAAI/bge-reranker-v2-m3"))
    use_multi_query: bool = field(default_factory=lambda: os.environ.get(
        "LONGCTX_MULTIQUERY", "1") == "1")
    use_treesitter: bool = field(default_factory=lambda: os.environ.get(
        "LONGCTX_TS", "0") == "1")
    limits: Limits = field(default_factory=Limits.from_env)


_config: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    """Return the singleton service config (env-driven)."""
    global _config
    if _config is None:
        _config = ServiceConfig()
    return _config


def set_config(cfg: ServiceConfig) -> None:
    """Override (for tests)."""
    global _config
    _config = cfg
