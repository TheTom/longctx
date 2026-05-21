"""Unit tests for ``longctx_daemon.auto_learn`` pure-logic layer.

Covers the resolve_repo_root + forbidden-parent + naming helpers
without spinning up a daemon. Integration with MCPServer's
search_codebase hot path is tested separately in test_mcp_server.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from longctx_daemon.auto_learn import (
    FORBIDDEN_PARENTS,
    ambient_learning_enabled,
    is_forbidden_parent,
    project_name_from_root,
    resolve_repo_root,
)


# ----------------------------------------------------- env toggle

def test_ambient_learning_default_on(monkeypatch):
    monkeypatch.delenv("LONGCTX_AUTO_LEARN", raising=False)
    assert ambient_learning_enabled() is True


def test_ambient_learning_off_via_zero(monkeypatch):
    monkeypatch.setenv("LONGCTX_AUTO_LEARN", "0")
    assert ambient_learning_enabled() is False


def test_ambient_learning_off_via_false(monkeypatch):
    monkeypatch.setenv("LONGCTX_AUTO_LEARN", "false")
    assert ambient_learning_enabled() is False


def test_ambient_learning_on_via_1(monkeypatch):
    monkeypatch.setenv("LONGCTX_AUTO_LEARN", "1")
    assert ambient_learning_enabled() is True


# ----------------------------------------------------- forbidden parents

def test_forbidden_includes_home_dot_ssh(tmp_path):
    # ~/.ssh is in the list; resolve manually to confirm.
    home_ssh = Path.home() / ".ssh"
    if home_ssh.exists():
        assert is_forbidden_parent(home_ssh) is True


def test_forbidden_blocks_tmp(tmp_path):
    # /tmp is forbidden — random scratch under it must reject.
    p = Path("/tmp/random-scratch-dir-xyz")
    # We don't need it to exist; is_forbidden_parent checks ancestry.
    assert is_forbidden_parent(p) is True


def test_forbidden_blocks_downloads():
    # ~/Downloads in forbidden list (Tom Q3=G).
    p = Path.home() / "Downloads" / "some-random-repo"
    assert is_forbidden_parent(p) is True


def test_forbidden_blocks_library():
    p = Path.home() / "Library" / "Application Support" / "foo"
    assert is_forbidden_parent(p) is True


def test_forbidden_blocks_desktop():
    p = Path.home() / "Desktop"
    assert is_forbidden_parent(p) is True


def test_forbidden_blocks_bare_home():
    assert is_forbidden_parent(Path.home()) is True


def test_forbidden_blocks_filesystem_root():
    assert is_forbidden_parent(Path("/")) is True


def test_forbidden_allows_typical_dev_dir(tmp_path):
    # tmp_path is under /tmp/<pytest>, which IS in forbidden list,
    # so use a path explicitly outside the forbidden set.
    home_dev = Path.home() / "dev" / "fakerepo"
    assert is_forbidden_parent(home_dev) is False


def test_forbidden_allows_user_chosen_paths():
    # ~/code, ~/work, ~/projects — all should be allowed for
    # ambient learning (only explicit forbidden parents block).
    for name in ("code", "work", "projects", "research"):
        p = Path.home() / name / "myproject"
        assert is_forbidden_parent(p) is False, f"{name} unexpectedly forbidden"


# ----------------------------------------------------- resolve_repo_root

def test_resolve_returns_none_for_empty(tmp_path):
    assert resolve_repo_root(None) is None
    assert resolve_repo_root("") is None


def test_resolve_returns_none_for_nonexistent():
    # Path that does not exist.
    assert resolve_repo_root("/this/path/does/not/exist/anywhere") is None


def test_resolve_returns_none_for_forbidden(tmp_path):
    # ~/.ssh would be forbidden if it exists; safer test is /tmp.
    assert resolve_repo_root("/tmp") is None


def test_resolve_walks_to_git_root(tmp_path, monkeypatch):
    # Build a fake repo: <tmp>/myrepo/.git + <tmp>/myrepo/src/lib.rs
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / ".git").mkdir()
    src = repo / "src"
    src.mkdir()
    (src / "lib.rs").write_text("fn main() {}")

    # tmp_path falls under /tmp which is forbidden. Patch
    # FORBIDDEN_PARENTS to skip /tmp for this test only.
    monkeypatch.setattr(
        "longctx_daemon.auto_learn.FORBIDDEN_PARENTS",
        tuple(p for p in FORBIDDEN_PARENTS if not p.startswith("/tmp")
              and not p.startswith("/private/tmp")
              and not p == "/private/var"),
    )

    # cwd inside src/ → should walk up to myrepo/.
    out = resolve_repo_root(str(src))
    assert out == str(repo.resolve())


def test_resolve_treats_non_git_cwd_as_root(tmp_path, monkeypatch):
    # Q4=C: a directory with no .git ancestor is still eligible —
    # use cwd itself as the root.
    monkeypatch.setattr(
        "longctx_daemon.auto_learn.FORBIDDEN_PARENTS",
        tuple(p for p in FORBIDDEN_PARENTS if not p.startswith("/tmp")
              and not p.startswith("/private/tmp")
              and not p == "/private/var"),
    )

    plain_dir = tmp_path / "scratch-dir"
    plain_dir.mkdir()
    (plain_dir / "note.txt").write_text("hello")

    out = resolve_repo_root(str(plain_dir))
    assert out == str(plain_dir.resolve())


def test_resolve_file_path_uses_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "longctx_daemon.auto_learn.FORBIDDEN_PARENTS",
        tuple(p for p in FORBIDDEN_PARENTS if not p.startswith("/tmp")
              and not p.startswith("/private/tmp")
              and not p == "/private/var"),
    )
    d = tmp_path / "afile"
    d.mkdir()
    f = d / "x.rs"
    f.write_text("//")
    # Passing the file path should use its parent dir.
    out = resolve_repo_root(str(f))
    assert out == str(d.resolve())


def test_resolve_worktree_returns_worktree_path(tmp_path, monkeypatch):
    """Git worktrees use a .git FILE (not dir) pointing to the
    main repo's .git/worktrees/<name>. The walk should still
    detect that as a repo root and return the worktree path,
    not the main repo path."""
    monkeypatch.setattr(
        "longctx_daemon.auto_learn.FORBIDDEN_PARENTS",
        tuple(p for p in FORBIDDEN_PARENTS if not p.startswith("/tmp")
              and not p.startswith("/private/tmp")
              and not p == "/private/var"),
    )
    worktree = tmp_path / "wt"
    worktree.mkdir()
    # Worktree .git is a file containing "gitdir: <main>/.git/worktrees/wt"
    (worktree / ".git").write_text("gitdir: /tmp/main/.git/worktrees/wt\n")
    out = resolve_repo_root(str(worktree))
    assert out == str(worktree.resolve())


# ----------------------------------------------------- project_name_from_root

def test_name_from_simple_path():
    assert project_name_from_root("/Users/tom/dev/metaltile") == "metaltile"
    assert project_name_from_root("/tmp/mt-a") == "mt-a"


def test_name_disambiguates_generic_basenames():
    # ``src``, ``code`` etc would collide too often → join with parent.
    assert project_name_from_root("/Users/tom/work/foo/src") == "foo-src"
    assert project_name_from_root("/x/main") == "x-main"
    assert project_name_from_root("/Users/tom/projects/play") == "play"


def test_name_handles_trailing_slash():
    assert project_name_from_root("/Users/tom/dev/metaltile/") == "metaltile"
