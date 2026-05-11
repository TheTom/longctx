"""Sentinel-based project auto-discovery (Phase 2.1).

Walks a parent directory (typically ``~/dev``) and returns every
immediate subdirectory that contains a sentinel file/directory marking
it as a project root. Plus a tiny helper, ``is_project_root``, for
"does THIS path look like a project?" callers (the cwd-walk path in
the searcher uses it too — see PRD §3.3).

Design constraints from PRD §3.1, §3.7:

* Cheap. Discovery runs at daemon start AND on every SIGHUP AND on
  every parent-watcher event. Aim for a single shallow ``listdir`` per
  parent dir, plus one ``Path.exists`` per sentinel per child.
* Tolerant. Broken symlinks, permission errors on a single subdir,
  unreadable ``.longctxignore`` — none of these block the rest of the
  walk.
* Defensive. Refuse to discover ``forbidden_dirs`` (PRD §13.3) even if
  they contain a sentinel.
* Pure. No mutation, no filesystem writes; we just read.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- types


@dataclass(frozen=True)
class DiscoveredProject:
    """One project picked up by ``discover_projects``.

    Attributes:
        name: directory basename (used as the project name in the
            index unless the user overrides it later).
        root_path: absolute path to the project root.
        sentinel: which sentinel matched (e.g. ``".git"``,
            ``"pyproject.toml"``). For logging + debugging — the
            indexer doesn't care which one fired.
        has_longctxignore: True if the project has an opt-out marker.
            Caller decides whether to skip; we surface the flag rather
            than filter so the wizard can show "X is opted out" in its
            project picker.
    """
    name: str
    root_path: Path
    sentinel: str
    has_longctxignore: bool


# --------------------------------------------------------------- predicates


def is_project_root(path: Path, sentinels: Sequence[str]) -> Optional[str]:
    """Return the matching sentinel name (in priority order) or None.

    Sentinels can be either a file (``pyproject.toml``) or a directory
    (``.git``); we test both. ``Path.exists`` follows symlinks; broken
    symlinks return False. Permission errors return False (logged at
    DEBUG so we don't spam in the common case).
    """
    if not path.is_dir():
        return None
    for s in sentinels:
        try:
            if (path / s).exists():
                return s
        except OSError as exc:
            # Broken symlink loop, EACCES, etc. Don't fail the whole
            # discovery — just skip this sentinel.
            logger.debug(
                "is_project_root sentinel-check failed for %s/%s: %s",
                path, s, exc,
            )
            continue
    return None


def _matches_exclude(name: str, full_path: Path, patterns: Sequence[str]) -> bool:
    """Match against a glob list. Patterns can be:

    * Bare basenames (``"node_modules"``) — match ``name`` exactly
    * Globs without slashes (``"foo-*"``) — match against ``name``
    * Globs with slashes (``"~/dev/legacy/*"``) — match against the
      absolute path's POSIX form

    Used by ``discover_projects`` to honor ``corpus.exclude``. Note:
    distinct from ``IndexerConfig.exclude_patterns``, which gates
    individual files. Discovery's exclude is a coarser per-project
    skip list.
    """
    if not patterns:
        return False
    p = full_path.as_posix()
    for pat in patterns:
        if pat == name:
            return True
        if "/" in pat:
            if fnmatch(p, pat) or fnmatch(p, str(Path(pat).expanduser())):
                return True
        else:
            if fnmatch(name, pat):
                return True
    return False


# --------------------------------------------------------------- main entry


def discover_projects(
    parent_dir: Path,
    sentinels: Sequence[str],
    exclude: Sequence[str],
    forbidden_dirs: frozenset[str],
) -> list[DiscoveredProject]:
    """Walk ``parent_dir``'s immediate children; return every project.

    Behavior:
        * Only IMMEDIATE subdirectories of ``parent_dir`` are checked
          (no recursion). If users want sub-projects of a project to
          be indexed independently, they explicitly point ``parent_dir``
          deeper or add the inner project via ``add_project``.
        * Subdirectories whose basename is in ``forbidden_dirs`` are
          refused even if they contain a sentinel. This is the
          ``~/dev/secrets/`` defense from PRD §13.3.
        * Subdirectories matching ``exclude`` are silently skipped.
        * ``.longctxignore`` opt-out is recorded but NOT filtered;
          callers decide. (The wizard wants to render "opted out" in
          its picker; the daemon's reconcile path filters them out.)
        * Broken symlinks, EACCES, etc. are logged at DEBUG and
          skipped — never raise.
        * Hidden directories (starting with ``.``) are eligible, but
          ``.git`` itself is forbidden (a bare repo is not a project).

    Returns list ordered by ``name`` for deterministic test output.
    """
    parent = Path(parent_dir).expanduser().resolve()
    if not parent.is_dir():
        logger.warning("discover_projects: parent_dir not a directory: %s", parent)
        return []

    out: list[DiscoveredProject] = []
    try:
        children = sorted(parent.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        logger.warning("discover_projects: failed to list %s: %s", parent, exc)
        return []

    for child in children:
        try:
            if not child.is_dir():
                continue
        except OSError:
            # Broken symlinks: skip silently
            continue

        name = child.name

        # Skip leading-dot dirs that aren't intentional projects
        # (the user almost never wants ``~/dev/.cache`` indexed). We
        # special-case ``.git`` -> never a project (a bare repo
        # checked into ~/dev would otherwise look like one).
        if name == ".git":
            continue

        # Forbidden by name (PRD §13.3) — refuse before sentinel check
        # so we don't even log discovery for sensitive dirs.
        if name in forbidden_dirs:
            logger.warning(
                "discover_projects: refusing forbidden dir name: %s", child
            )
            continue

        # Excluded by user config
        if _matches_exclude(name, child, exclude):
            logger.debug("discover_projects: excluded by config: %s", child)
            continue

        sentinel = is_project_root(child, sentinels)
        if sentinel is None:
            continue

        # Honor .longctxignore as a flag, not a filter. Caller decides.
        try:
            ignore_marker = (child / ".longctxignore").is_file()
        except OSError:
            ignore_marker = False

        out.append(DiscoveredProject(
            name=name,
            root_path=child.resolve(),
            sentinel=sentinel,
            has_longctxignore=ignore_marker,
        ))

    return out


# TODO(phase 2.2): cwd_to_project_root walks UP from cwd to find the
# nearest sentinel — used for per-session scope (PRD §3.3). Belongs
# here logically; lives elsewhere today since it predates this module.
# TODO(phase 2.4): Windows-symlink quirks: ``Path.is_dir`` follows
# junction points correctly on 3.10+ but not on 3.10.0 itself. Add a
# guard if we ever ship Windows pre-3.10.4.

# Convenience export so callers can iterate without repeating the
# os.walk pattern.
__all__ = [
    "DiscoveredProject",
    "discover_projects",
    "is_project_root",
]
