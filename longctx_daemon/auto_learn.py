"""Ambient learning — cwd-driven repo discovery.

Phase 1 of the Ambient Indexing PRD (2026-05-21 v0.2). The daemon
learns which repos to index by observing the agent's ``cwd`` arg on
every search_codebase call. No filesystem walk on startup, no
allowlist roots — the daemon converges to the user's real working
set by signal alone.

This module is the pure-logic layer:
  * ``resolve_repo_root`` — walks up from cwd to a git root (or
    treats cwd itself as the root if no git ancestor exists),
    while honoring the forbidden-parents block-list.
  * ``is_forbidden_parent`` — checks whether a resolved path falls
    under one of the always-skip parents.
  * ``project_name_from_root`` — turns an absolute path into a
    short, registry-safe project name.

The MCP server (mcp_server.py) wires these helpers into the
search_codebase hot path. The actual background indexing + watcher
subscription lives in the server, not here, so this module stays
import-cheap + unit-testable without a daemon.

User-facing toggle: ``LONGCTX_AUTO_LEARN`` env var. Defaults to ``"1"``
(on); set to ``"0"`` to disable ambient learning entirely (explicit
``--corpus-dir`` and ``add_project`` keep working).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------- constants

# Paths under which ambient learning REFUSES to register a project,
# even if the agent's cwd points there. Picked per Tom's PRD review
# (2026-05-21, Q3=G): these are common parents we never want the
# daemon to stumble into. The list is intentionally biased toward
# false negatives — better to make the user explicitly add an unusual
# location than to ambient-index something sensitive or junky.
#
# Exact-match or ancestor-match: if the resolved cwd IS one of these,
# OR has one of these as a parent, the path is rejected. Order doesn't
# matter; the check is membership-based.
#
# Documented in README "Ambient Learning" section so users know what
# the daemon will never auto-touch.
FORBIDDEN_PARENTS: Tuple[str, ...] = (
    # Security-sensitive
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    # Junk / app data / non-source
    "~/Library",
    "~/Downloads",
    "~/.Trash",
    "~/Desktop",
    "/private/var",
    # Scratch space — explicit --corpus-dir still works, ambient
    # learning specifically should not auto-grab here
    "/tmp",
    "/private/tmp",
)

# Paths that block ONLY on exact match — these are "too broad to
# ancestor-block" because they're parents of everything. Bare ``/``
# would otherwise reject every cwd in the universe; bare ``~`` would
# reject every dir under the user's home. We still want to refuse if
# the agent literally points cwd AT one of these.
FORBIDDEN_EXACT: Tuple[str, ...] = (
    "/",
    "~",
)


# Env var that disables ambient learning. Default behavior is ON
# (PRD v0.2 ships default-on per Tom's Q1=A choice).
ENV_DISABLE = "LONGCTX_AUTO_LEARN"


def ambient_learning_enabled() -> bool:
    """True iff ambient learning should run for this daemon process.

    Reads ``LONGCTX_AUTO_LEARN`` env var. Any value other than ``"0"``,
    ``"false"``, or empty-string keeps the feature on. This is the
    user's escape hatch.
    """
    raw = os.environ.get(ENV_DISABLE, "1")
    return raw not in ("0", "false", "False", "")


# ---------------------------------------------------------- forbidden check

def _expand(path: str) -> Path:
    """Expand ``~`` and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def is_forbidden_parent(path: Path) -> bool:
    """True iff ``path`` is one of, or sits under, a forbidden parent.

    Used to keep ambient learning from registering projects in
    sensitive or non-source locations (see ``FORBIDDEN_PARENTS``).
    Worth noting: this only gates the ambient-learning auto-add
    path. Explicit ``--corpus-dir`` and the ``add_project`` MCP tool
    continue to work for any directory the user explicitly chooses.

    The comparison is on resolved absolute paths, so symlinks and
    ``..`` segments don't sneak through.
    """
    # Always resolve, even for absolute paths — on macOS ``/tmp`` is
    # a symlink to ``/private/tmp`` and we need both sides to use the
    # same canonical form to compare correctly.
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return True  # can't resolve → safer to refuse
    # Exact-match block: bare ``/`` and ``~`` would reject every
    # path under them if treated as ancestors, so they need a
    # tighter equality check.
    for raw in FORBIDDEN_EXACT:
        if resolved == _expand(raw):
            return True
    # Ancestor-match block: typical forbidden parents — sensitive
    # dirs and scratch areas. Membership covers both "path IS one
    # of these" and "path is under one of these".
    for raw in FORBIDDEN_PARENTS:
        parent = _expand(raw)
        if resolved == parent:
            return True
        try:
            resolved.relative_to(parent)
        except ValueError:
            continue
        return True
    return False


# --------------------------------------------------------- repo root walk

def _walk_for_git_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``.git`` entry.

    Returns the deepest ancestor whose ``.git`` entry exists
    (file OR directory — git worktrees use a ``.git`` file that
    points back to the main repo's ``.git/worktrees/<name>``).
    Returns ``None`` if no git ancestor is found before the
    filesystem root.

    Walking the deepest match first means worktrees correctly
    resolve to their own path, not the main repo's path. Two
    worktrees of the same upstream repo end up as separate
    project roots, which is what the ambient-learning tier
    model expects (each worktree's working state is distinct).
    """
    current = start if start.is_dir() else start.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(cwd: Optional[str]) -> Optional[str]:
    """Resolve an agent ``cwd`` to a project root path, or ``None``
    if ambient learning should not register the location.

    Decision flow:
      1. ``cwd`` is None or empty → return None (no signal).
      2. Resolve to absolute path. If the resolved path doesn't
         exist on disk, return None.
      3. If the resolved path is forbidden (see ``is_forbidden_parent``),
         return None — the daemon never ambient-indexes those areas.
      4. Walk up looking for a ``.git`` entry. If found, return
         the git root (which may be ``cwd`` itself or any ancestor
         up to a non-forbidden point).
      5. Otherwise (no git ancestor) treat ``cwd`` itself as the
         project root. Per Tom's Q4=C choice in the PRD review
         (2026-05-21), non-git directories are eligible for
         ambient learning so that scratch dirs the agent touches
         still get indexed.

    Returns:
        Absolute path string of the resolved root, or ``None`` if
        no eligible root was found.
    """
    if not cwd:
        return None
    try:
        resolved = _expand(cwd)
    except (OSError, RuntimeError):
        return None
    if not resolved.exists() or not resolved.is_dir():
        # Could be a file path — try its parent.
        if resolved.is_file():
            resolved = resolved.parent
        else:
            return None
    if is_forbidden_parent(resolved):
        return None
    git_root = _walk_for_git_root(resolved)
    if git_root is not None:
        # Re-check forbidden: a forbidden ancestor of the cwd
        # would also forbid the discovered git root.
        if is_forbidden_parent(git_root):
            return None
        return str(git_root)
    # Q4=C: non-git cwd → treat cwd itself as the root.
    return str(resolved)


# -------------------------------------------------------- naming

def project_name_from_root(root: str) -> str:
    """Derive a registry-safe project name from a root path.

    Uses the basename of the resolved root. If the basename is
    empty or matches a known generic dir name that would collide
    with other repos (e.g. ``src``, ``code``), falls back to the
    parent-basename joined with the basename.

    Examples:
        /Users/tom/dev/metaltile        → "metaltile"
        /tmp/mt-a                       → "mt-a"
        /Users/tom/work/foo/src         → "foo-src"
        /Users/tom/code/playground      → "playground"

    The MCP ``add_project`` handler rejects collisions by name
    against the existing registry, so distinct-but-same-named
    repos require user disambiguation. That's a Phase-2 polish
    target; for Phase 1 we accept the limitation.
    """
    p = Path(root)
    name = p.name or p.parts[-1]
    # Generic basenames that collide too often if used alone.
    _COLLISION_PRONE = frozenset({"src", "code", "main", "lib", "app", "project"})
    if name.lower() in _COLLISION_PRONE and p.parent != p:
        parent_name = p.parent.name or "root"
        name = f"{parent_name}-{name}"
    return name
