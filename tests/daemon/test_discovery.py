"""Auto-discovery tests for ``longctx_daemon.discovery``.

Coverage targets (agent spec §2):
  * sentinel detection — file vs directory, multiple sentinels, priority
  * .longctxignore opt-out flag (recorded, not filtered)
  * forbidden_dirs refusal
  * exclude patterns (basename, glob, full-path)
  * edge cases: broken symlinks, permissions, missing parent_dir,
    nested projects (we DON'T recurse — should be no-op),
    hidden directories
  * is_project_root predicate
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from longctx_daemon.discovery import (
    DiscoveredProject,
    discover_projects,
    is_project_root,
)


# --------------------------------------------------------------- helpers


def _make_project(parent: Path, name: str, sentinel: str = ".git") -> Path:
    """Create a fake project root with the given sentinel marker."""
    proj = parent / name
    proj.mkdir(parents=True, exist_ok=True)
    if sentinel == ".git" or sentinel.startswith("."):
        # ``.git`` is conventionally a directory; we make it a directory
        (proj / sentinel).mkdir(exist_ok=True)
    else:
        (proj / sentinel).write_text("")
    return proj


def _default_sentinels() -> tuple[str, ...]:
    return (".git", "pyproject.toml", "package.json", "Cargo.toml")


# --------------------------------------------------------------- is_project_root


class TestIsProjectRoot:
    def test_returns_sentinel_name_when_match(self, tmp_path):
        _make_project(tmp_path, "foo", sentinel=".git")
        assert is_project_root(tmp_path / "foo", _default_sentinels()) == ".git"

    def test_returns_first_match_in_priority_order(self, tmp_path):
        proj = tmp_path / "foo"
        proj.mkdir()
        # Both pyproject.toml AND Cargo.toml present
        (proj / "pyproject.toml").write_text("")
        (proj / "Cargo.toml").write_text("")
        # Order in sentinels list dictates which wins
        assert is_project_root(proj, ("pyproject.toml", "Cargo.toml")) == "pyproject.toml"
        assert is_project_root(proj, ("Cargo.toml", "pyproject.toml")) == "Cargo.toml"

    def test_returns_none_for_no_match(self, tmp_path):
        (tmp_path / "foo").mkdir()
        assert is_project_root(tmp_path / "foo", _default_sentinels()) is None

    def test_returns_none_for_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("not a dir")
        assert is_project_root(f, _default_sentinels()) is None

    def test_returns_none_for_missing_path(self, tmp_path):
        assert is_project_root(tmp_path / "absent", _default_sentinels()) is None

    def test_directory_sentinel_works(self, tmp_path):
        proj = tmp_path / "foo"
        proj.mkdir()
        (proj / ".git").mkdir()
        assert is_project_root(proj, (".git",)) == ".git"

    def test_file_sentinel_works(self, tmp_path):
        proj = tmp_path / "foo"
        proj.mkdir()
        (proj / "Cargo.toml").write_text("")
        assert is_project_root(proj, ("Cargo.toml",)) == "Cargo.toml"

    def test_empty_sentinels_returns_none(self, tmp_path):
        proj = _make_project(tmp_path, "foo")
        assert is_project_root(proj, ()) is None


# --------------------------------------------------------------- discover_projects


class TestDiscoverProjects:
    def test_empty_parent_dir_yields_no_projects(self, tmp_path):
        out = discover_projects(
            tmp_path,
            sentinels=_default_sentinels(),
            exclude=(),
            forbidden_dirs=frozenset(),
        )
        assert out == []

    def test_finds_one_project(self, tmp_path):
        _make_project(tmp_path, "foo", sentinel=".git")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert len(out) == 1
        assert out[0].name == "foo"
        assert out[0].sentinel == ".git"
        assert out[0].has_longctxignore is False

    def test_finds_multiple_projects_sorted_by_name(self, tmp_path):
        _make_project(tmp_path, "zebra")
        _make_project(tmp_path, "apple")
        _make_project(tmp_path, "mango")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["apple", "mango", "zebra"]

    def test_skips_directories_without_sentinel(self, tmp_path):
        # Directory exists, but no sentinel
        (tmp_path / "scratch").mkdir()
        _make_project(tmp_path, "real")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["real"]

    def test_skips_files_at_top_level(self, tmp_path):
        (tmp_path / "loose-file.txt").write_text("hi")
        _make_project(tmp_path, "real")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert len(out) == 1

    def test_does_not_recurse(self, tmp_path):
        # Nested: ~/dev/a/b is NOT discovered, only ~/dev/a
        _make_project(tmp_path, "a", sentinel=".git")
        # Make a sub-project of a
        sub = tmp_path / "a" / "b"
        sub.mkdir()
        (sub / "Cargo.toml").write_text("")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["a"]

    def test_longctxignore_marker_recorded(self, tmp_path):
        proj = _make_project(tmp_path, "foo")
        (proj / ".longctxignore").write_text("# opt out")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert len(out) == 1
        assert out[0].has_longctxignore is True

    def test_longctxignore_directory_not_treated_as_marker(self, tmp_path):
        # If someone creates a dir named .longctxignore, it shouldn't
        # mark the project (we test with .is_file()).
        proj = _make_project(tmp_path, "foo")
        (proj / ".longctxignore").mkdir()
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert out[0].has_longctxignore is False

    def test_forbidden_dirs_refused(self, tmp_path):
        _make_project(tmp_path, "secrets", sentinel=".git")
        _make_project(tmp_path, "myapp")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(),
            forbidden_dirs=frozenset({"secrets"}),
        )
        names = [p.name for p in out]
        assert "secrets" not in names
        assert "myapp" in names

    def test_forbidden_dirs_full_set(self, tmp_path):
        for name in ("secrets", "credentials", ".aws", ".ssh", ".gnupg"):
            _make_project(tmp_path, name)
        _make_project(tmp_path, "myapp")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(),
            forbidden_dirs=frozenset(
                {"secrets", "credentials", ".aws", ".ssh", ".gnupg"}
            ),
        )
        assert len(out) == 1
        assert out[0].name == "myapp"

    def test_exclude_basename(self, tmp_path):
        _make_project(tmp_path, "foo")
        _make_project(tmp_path, "bar")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=("foo",), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["bar"]

    def test_exclude_glob(self, tmp_path):
        _make_project(tmp_path, "playground-1")
        _make_project(tmp_path, "playground-2")
        _make_project(tmp_path, "real")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=("playground-*",), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["real"]

    def test_exclude_multiple_patterns(self, tmp_path):
        _make_project(tmp_path, "a")
        _make_project(tmp_path, "b")
        _make_project(tmp_path, "c")
        _make_project(tmp_path, "keep")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=("a", "b", "c"),
            forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["keep"]

    def test_dot_git_root_skipped(self, tmp_path):
        # A bare ``.git`` dir under parent_dir is NOT a project. (It
        # could be a bare repo; we don't want to index that.)
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("")
        _make_project(tmp_path, "real")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["real"]

    def test_hidden_directories_eligible(self, tmp_path):
        # Hidden dirs that aren't .git ARE eligible
        _make_project(tmp_path, ".hidden-project")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert ".hidden-project" in names

    def test_missing_parent_dir_returns_empty(self, tmp_path):
        out = discover_projects(
            tmp_path / "absent",
            sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert out == []

    def test_parent_dir_is_file_returns_empty(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("")
        out = discover_projects(
            f,
            sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert out == []

    def test_broken_symlink_skipped(self, tmp_path):
        # parent/foo -> /nonexistent
        link = tmp_path / "foo"
        try:
            os.symlink("/nonexistent/path", link)
        except OSError:
            pytest.skip("symlink not supported here")
        _make_project(tmp_path, "real")
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=(), forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert names == ["real"]

    def test_symlink_to_real_project_works(self, tmp_path):
        # Real project elsewhere; symlink under parent_dir points at it
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / ".git").mkdir()
        link = tmp_path / "linked"
        try:
            os.symlink(target, link)
        except OSError:
            pytest.skip("symlink not supported")
        # We need parent_dir to NOT contain the elsewhere/ dir directly
        # but DO contain the symlink. Move elsewhere out of parent_dir:
        # simpler: just use tmp_path as parent and have both. The
        # discovery should only pick up the symlink (which points at
        # a real .git dir).
        out = discover_projects(
            tmp_path, sentinels=_default_sentinels(),
            exclude=("elsewhere",),  # exclude the target so only the link
                                       # is a candidate
            forbidden_dirs=frozenset(),
        )
        names = [p.name for p in out]
        assert "linked" in names

    def test_priority_within_one_project(self, tmp_path):
        """If a project has both .git and Cargo.toml, the higher-priority
        sentinel from the list is reported."""
        proj = tmp_path / "x"
        proj.mkdir()
        (proj / ".git").mkdir()
        (proj / "Cargo.toml").write_text("")
        out = discover_projects(
            tmp_path,
            sentinels=(".git", "Cargo.toml"),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert out[0].sentinel == ".git"

    def test_priority_swap_within_one_project(self, tmp_path):
        proj = tmp_path / "x"
        proj.mkdir()
        (proj / ".git").mkdir()
        (proj / "Cargo.toml").write_text("")
        out = discover_projects(
            tmp_path,
            sentinels=("Cargo.toml", ".git"),
            exclude=(), forbidden_dirs=frozenset(),
        )
        assert out[0].sentinel == "Cargo.toml"

    def test_root_path_is_resolved_absolute(self, tmp_path):
        _make_project(tmp_path, "foo")
        out = discover_projects(
            tmp_path,
            sentinels=_default_sentinels(),
            exclude=(),
            forbidden_dirs=frozenset(),
        )
        assert out[0].root_path.is_absolute()
        assert out[0].root_path == (tmp_path / "foo").resolve()

    def test_returns_discovered_project_dataclass(self, tmp_path):
        _make_project(tmp_path, "foo")
        out = discover_projects(
            tmp_path,
            sentinels=_default_sentinels(),
            exclude=(),
            forbidden_dirs=frozenset(),
        )
        assert isinstance(out[0], DiscoveredProject)
        # Frozen dataclass
        with pytest.raises((AttributeError, TypeError)):
            out[0].name = "renamed"  # type: ignore

    def test_multiple_sentinels_one_dir(self, tmp_path):
        # Multiple projects, each with a different sentinel type
        _make_project(tmp_path, "py-proj", sentinel="pyproject.toml")
        _make_project(tmp_path, "rust-proj", sentinel="Cargo.toml")
        _make_project(tmp_path, "node-proj", sentinel="package.json")
        out = discover_projects(
            tmp_path,
            sentinels=("pyproject.toml", "Cargo.toml", "package.json"),
            exclude=(),
            forbidden_dirs=frozenset(),
        )
        sent_by_name = {p.name: p.sentinel for p in out}
        assert sent_by_name == {
            "py-proj": "pyproject.toml",
            "rust-proj": "Cargo.toml",
            "node-proj": "package.json",
        }
