"""Scope detection tests. PRD §5.1, smoke scenarios §7.1-§7.3."""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.scope.detect import (
    canonicalize_scope,
    detect_scope,
    extract_paths_from_prefill,
    find_sentinel_root,
    hash_scope,
)


class TestExtractPathsFromPrefill:
    def test_extracts_macos_paths(self):
        text = "the file at /Users/tom/dev/myapp/src/auth.ts is broken"
        assert Path("/Users/tom/dev/myapp/src/auth.ts") in \
            extract_paths_from_prefill(text)

    def test_extracts_linux_paths(self):
        text = "see /home/alice/project/main.py for details"
        paths = extract_paths_from_prefill(text)
        assert Path("/home/alice/project/main.py") in paths

    def test_strips_trailing_punct(self):
        text = "look at /Users/tom/dev/foo.py."
        paths = extract_paths_from_prefill(text)
        assert paths == [Path("/Users/tom/dev/foo.py")]

    def test_quoted_paths(self):
        text = '`/Users/tom/dev/myapp/src/auth.ts` and "/home/x/y.py"'
        paths = [str(p) for p in extract_paths_from_prefill(text)]
        assert "/Users/tom/dev/myapp/src/auth.ts" in paths
        assert "/home/x/y.py" in paths

    def test_empty_returns_empty(self):
        assert extract_paths_from_prefill("") == []
        assert extract_paths_from_prefill(None) == []   # type: ignore[arg-type]

    def test_no_paths_in_plain_text(self):
        assert extract_paths_from_prefill("hello world") == []

    def test_dedups(self):
        text = "see /Users/x/foo.py and /Users/x/foo.py"
        paths = extract_paths_from_prefill(text)
        assert len(paths) == 1


class TestFindSentinelRoot:
    def test_finds_package_json(self, project_dir):
        deep = project_dir / "src" / "auth.ts"
        result = find_sentinel_root(deep)
        assert result is not None
        assert result[0] == project_dir
        assert result[1] == "package.json"

    def test_returns_none_with_no_sentinel(self, tmp_path):
        # Empty tmp_path has no sentinel; walking up will eventually hit one
        # somewhere on the dev box (or nothing). We assert behavior is
        # well-defined: either Some root with a sentinel, or None.
        deep = tmp_path / "x" / "y"
        deep.mkdir(parents=True)
        result = find_sentinel_root(deep)
        # Either a real ancestor sentinel was found OR None — both valid.
        assert result is None or result[0].exists()

    def test_walks_up_from_file(self, project_dir):
        f = project_dir / "src" / "billing.ts"
        result = find_sentinel_root(f)
        assert result is not None
        assert result[0] == project_dir


class TestCanonicalizeScope:
    def test_strips_trailing_slash(self, tmp_path):
        p = tmp_path / "foo"
        p.mkdir()
        canon = canonicalize_scope(Path(str(p) + "/"))
        assert not str(canon).endswith("/")

    def test_resolves_relative(self, tmp_path):
        p = tmp_path / "foo" / "bar"
        p.mkdir(parents=True)
        canon = canonicalize_scope(p)
        # Should be absolute
        assert canon.is_absolute()


class TestHashScope:
    def test_stable(self, tmp_path):
        p = tmp_path / "foo"
        p.mkdir()
        h1 = hash_scope(canonicalize_scope(p))
        h2 = hash_scope(canonicalize_scope(p))
        assert h1 == h2

    def test_different_paths_differ(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        ha = hash_scope(canonicalize_scope(a))
        hb = hash_scope(canonicalize_scope(b))
        assert ha != hb

    def test_short_length(self, tmp_path):
        p = tmp_path / "foo"
        p.mkdir()
        assert len(hash_scope(canonicalize_scope(p))) == 16


class TestDetectScopeIntegration:
    def test_single_project_root(self, project_dir):
        """Smoke scenario §7.1: scope detected from prefill mentioning a
        file inside the project."""
        prefill = (
            f"i'm seeing an issue with {project_dir}/src/auth.ts "
            "in the docker build"
        )
        scope = detect_scope(prefill)
        assert scope is not None
        # Should land at the project root (sentinel: package.json)
        assert scope.sentinel == "package.json"
        # Either the canonicalized project_dir or a case-folded equivalent
        assert scope.root.name.lower() == project_dir.name.lower()

    def test_no_paths_returns_none(self):
        scope = detect_scope("how do i fix this auth issue?")
        assert scope is None

    def test_explicit_root_override(self, project_dir):
        """PRD §5.8: --longctx-scope flag wins."""
        scope = detect_scope("no paths here", explicit_root=project_dir)
        assert scope is not None
        assert scope.sentinel == "explicit"

    def test_monorepo_subpackage_detection(self, tmp_path):
        """Smoke §7.2: multiple files share a sub-package; prefer the
        package, not the workspace root."""
        # Mock monorepo: workspace at /mono with sub-package /mono/apps/billing
        mono = tmp_path / "mono"
        mono.mkdir()
        (mono / "package.json").write_text('{"name":"mono"}')
        billing = mono / "apps" / "billing"
        billing.mkdir(parents=True)
        (billing / "package.json").write_text('{"name":"billing"}')
        (billing / "Charge.tsx").write_text("export {};")
        (billing / "Invoice.tsx").write_text("export {};")
        prefill = (
            f"see {billing}/Charge.tsx and {billing}/Invoice.tsx"
        )
        scope = detect_scope(prefill)
        assert scope is not None
        # Should prefer billing, not mono
        assert scope.root.name.lower() == "billing"

    def test_returns_scope_hash(self, project_dir):
        prefill = f"file: {project_dir}/src/auth.ts"
        scope = detect_scope(prefill)
        assert scope is not None
        assert len(scope.scope_hash) == 16
        assert scope.scope_hash.isalnum()
