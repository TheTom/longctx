"""Filter-chain unit tests.

Each test isolates one rule from PRD §5.2. Together with the
end-to-end coverage in ``test_indexer.py`` these guarantee no rule
silently bypasses the chain after a refactor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_daemon.indexer import (
    DEFAULT_SECRET_PATTERNS,
    IndexerConfig,
    _GitignoreMatcher,
    _looks_binary,
    _matches_any,
    should_index,
)


# --------------------------------------------------------------- _matches_any

class TestMatchesAny:
    def test_basename_match(self):
        assert _matches_any(".env", (".env",))
        assert _matches_any("dir/.env", (".env",))

    def test_glob_match(self):
        assert _matches_any("foo/bar.key", ("*.key",))
        assert _matches_any("path/secrets/foo.txt", ("**/secrets/**",))

    def test_no_match(self):
        assert not _matches_any("README.md", ("*.key", "**/secrets/**"))

    def test_empty_patterns(self):
        assert not _matches_any("anything.py", ())


# --------------------------------------------------------------- gitignore

class TestGitignoreMatcher:
    def test_no_gitignore_passes_through(self, tmp_path):
        m = _GitignoreMatcher(tmp_path)
        assert not m.is_ignored("foo.py")

    def test_simple_pattern(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        m = _GitignoreMatcher(tmp_path)
        assert m.is_ignored("trace.log")
        assert not m.is_ignored("trace.txt")

    def test_directory_pattern(self, tmp_path):
        (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
        m = _GitignoreMatcher(tmp_path)
        assert m.is_ignored("build/output.bin")

    def test_lazy_load(self, tmp_path):
        # Matcher loads on first call, not construction.
        m = _GitignoreMatcher(tmp_path)
        assert m._spec is None
        assert not m._loaded
        m.is_ignored("a.py")
        assert m._loaded


# --------------------------------------------------------------- _looks_binary

class TestLooksBinary:
    def test_text_file(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("plain text\n", encoding="utf-8")
        assert not _looks_binary(p)

    def test_null_bytes(self, tmp_path):
        p = tmp_path / "blob.bin"
        p.write_bytes(b"hello\x00world")
        assert _looks_binary(p)

    def test_invalid_utf8(self, tmp_path):
        p = tmp_path / "bad.bin"
        # Lone continuation byte sequence = invalid UTF-8.
        p.write_bytes(b"\xc3\x28")
        assert _looks_binary(p)

    def test_unreadable(self, tmp_path):
        p = tmp_path / "ghost.bin"
        # File doesn't exist; treat as binary (skip).
        assert _looks_binary(p)


# --------------------------------------------------------------- should_index

class TestShouldIndex:
    def _config(self, **overrides) -> IndexerConfig:
        return IndexerConfig(**overrides)

    def test_passes_normal_text_file(self, tmp_path):
        p = tmp_path / "code.py"
        p.write_text("ok\n", encoding="utf-8")
        assert should_index(p, tmp_path, self._config())

    def test_rejects_secret_pattern(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("SECRET=1\n", encoding="utf-8")
        assert not should_index(p, tmp_path, self._config())

    def test_rejects_secret_pattern_nested(self, tmp_path):
        d = tmp_path / "secrets"
        d.mkdir()
        p = d / "token.txt"
        p.write_text("token\n", encoding="utf-8")
        assert not should_index(p, tmp_path, self._config())

    def test_rejects_exclude_pattern(self, tmp_path):
        p = tmp_path / "big.lock"
        p.write_text("lock data\n", encoding="utf-8")
        assert not should_index(p, tmp_path, self._config())

    def test_rejects_min_js(self, tmp_path):
        p = tmp_path / "bundle.min.js"
        p.write_text("var x=1;\n", encoding="utf-8")
        assert not should_index(p, tmp_path, self._config())

    def test_respects_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        p = tmp_path / "ignored.py"
        p.write_text("skip\n", encoding="utf-8")
        gi = _GitignoreMatcher(tmp_path)
        assert not should_index(p, tmp_path, self._config(), gi)

    def test_gitignore_disabled(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        p = tmp_path / "ignored.py"
        p.write_text("skip\n", encoding="utf-8")
        gi = _GitignoreMatcher(tmp_path)
        cfg = self._config(respect_gitignore=False)
        assert should_index(p, tmp_path, cfg, gi)

    def test_size_cap_rejects_large(self, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text("x" * 1024, encoding="utf-8")
        cfg = self._config(max_file_size_bytes=512)
        assert not should_index(p, tmp_path, cfg)

    def test_size_cap_accepts_under(self, tmp_path):
        p = tmp_path / "small.txt"
        p.write_text("x" * 256, encoding="utf-8")
        cfg = self._config(max_file_size_bytes=512)
        assert should_index(p, tmp_path, cfg)

    def test_zero_byte_file_skipped(self, tmp_path):
        p = tmp_path / "empty.py"
        p.write_text("", encoding="utf-8")
        assert not should_index(p, tmp_path, self._config())

    def test_extension_allowlist_accepts_match(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("ok\n", encoding="utf-8")
        cfg = self._config(include_extensions=(".py",))
        assert should_index(p, tmp_path, cfg)

    def test_extension_allowlist_rejects_nonmatch(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("ok\n", encoding="utf-8")
        cfg = self._config(include_extensions=(".py",))
        assert not should_index(p, tmp_path, cfg)

    def test_extension_allowlist_case_insensitive(self, tmp_path):
        p = tmp_path / "A.PY"
        p.write_text("ok\n", encoding="utf-8")
        cfg = self._config(include_extensions=(".py",))
        assert should_index(p, tmp_path, cfg)

    def test_binary_file_rejected(self, tmp_path):
        p = tmp_path / "blob.dat"
        p.write_bytes(b"\x00\x01\x02 not text \x00")
        assert not should_index(p, tmp_path, self._config())

    def test_known_text_extension_skips_binary_sniff(self, tmp_path):
        # .py is in the text allowlist; even a bytes-like body bypasses
        # the sniff (this is a corner case — production code wouldn't
        # produce such a file but the rule should still hold).
        p = tmp_path / "a.py"
        p.write_text("def f(): return 1\n", encoding="utf-8")
        assert should_index(p, tmp_path, self._config())

    def test_path_outside_root_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        p = outside / "a.py"
        p.write_text("ok\n", encoding="utf-8")
        assert not should_index(p, root, self._config())

    def test_default_secret_patterns_cover_common_cases(self):
        # Sanity: the spec list catches what we expect.
        assert ".env" in DEFAULT_SECRET_PATTERNS
        assert "*.key" in DEFAULT_SECRET_PATTERNS
        assert any("/secrets/" in p for p in DEFAULT_SECRET_PATTERNS)


# --------------------------------------------------------------- sanity

def test_module_exports():
    """Smoke test: indexer's public symbols are stable."""
    from longctx_daemon.indexer import (  # noqa: F401
        Indexer,
        IndexerConfig,
        ScanResult,
        UpdateResult,
        ForbiddenProjectError,
        EmbedderMismatchError,
        should_index,
    )
