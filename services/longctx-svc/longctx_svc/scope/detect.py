"""Scope detection from prefill text. PRD §5.1.

The pipeline:
  1. Extract absolute paths from the prefill (system + user messages).
  2. For each mentioned path, walk up to the nearest project sentinel.
  3. If multiple mentioned files share a sub-package inside a monorepo,
     prefer the package, not the workspace root.
  4. If no path detected: no scope, no retrieval.

Designed to be synchronous and fast (<100ms target).
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from longctx_svc.config import SENTINELS


# Match absolute Unix paths (macOS /Users/..., Linux /home/...) plus a few
# common roots. Stop at whitespace or quoting/punct.
_PATH_RE = re.compile(
    r"(?:^|[\s'\"`(])"                       # boundary
    r"(/(?:Users|home|opt|var|usr|tmp|mnt|workspace|repo|repos|private|root|srv|data)/[^\s'\"`)\]]+)"
)

# Match plausibly-relative paths with a file extension and at least one
# directory separator, e.g. "django/core/handlers/exception.py". Must NOT
# start with '/' (those are absolute, handled above). Used when callers
# pass a `default_scope` to extract_paths_from_prefill so we can resolve
# relative mentions against the caller's repo root. Required for agents
# like Aider that cd into a repo and refer to files relatively.
_REL_PATH_RE = re.compile(
    r"(?:^|[\s'\"`(])"
    r"((?!/)[A-Za-z_][\w.-]*(?:/[\w.-]+)+\.[A-Za-z]\w{0,7})"
)


@dataclass(frozen=True)
class DetectedScope:
    """Result of scope detection."""
    root: Path
    sentinel: str
    scope_hash: str
    mentioned_paths: tuple[Path, ...] = ()

    @property
    def name(self) -> str:
        return self.root.name


def extract_paths_from_prefill(prefill_text: str,
                               default_scope: Path | str | None = None,
                               ) -> list[Path]:
    """Return distinct paths mentioned in prefill text.

    Absolute paths (``/Users/...``, ``/home/...``, etc.) are extracted as-is
    by ``_PATH_RE``.

    When ``default_scope`` is provided, relative paths that look like real
    file references (``django/core/foo.py``) are ALSO extracted, resolved
    against ``default_scope``, and included only if the resolved file
    actually exists. This lets agents that cd into a repo and refer to
    files relatively (Aider, Cursor agentic, OpenCode in workspace mode,
    direct MCP callers) still trigger scope routing without forcing every
    consumer to splice absolute paths into the prompt.

    Existence-check is the safety net against false positives from
    casual prose containing dotted tokens.
    """
    seen: set[str] = set()
    paths: list[Path] = []
    # 1. Absolute paths — same behavior as before.
    for m in _PATH_RE.finditer(prefill_text or ""):
        s = m.group(1).rstrip(".,;:!?")
        if s in seen:
            continue
        seen.add(s)
        # Drop fragments like /usr/bin/python which aren't user-project roots.
        # We keep them now and filter at sentinel-walk time.
        paths.append(Path(s))
    # 2. Relative paths resolved against default_scope.
    if default_scope is not None:
        root = Path(default_scope)
        try:
            root = root.expanduser().resolve()
        except OSError:
            root = Path(default_scope)
        for m in _REL_PATH_RE.finditer(prefill_text or ""):
            s = m.group(1).rstrip(".,;:!?")
            if s in seen:
                continue
            seen.add(s)
            full = root / s
            try:
                if full.exists():
                    paths.append(full)
            except OSError:
                pass
    return paths


def canonicalize_scope(p: Path) -> Path:
    """`realpath()` + macOS lower-case normalization + strip trailing slash.

    Per PRD §5.1: 'realpath() , lowercase on case-insensitive volumes,
    strip trailing slashes'. We treat darwin as case-insensitive by default.
    """
    try:
        rp = p.resolve()
    except OSError:
        return p
    s = str(rp).rstrip("/")
    if sys.platform == "darwin":
        # macOS default is case-insensitive APFS; normalize for hash stability.
        s = s.lower()
    return Path(s)


def hash_scope(p: Path) -> str:
    """Stable short hash of a canonicalized scope path for cache keys."""
    return hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:16]


def find_sentinel_root(start: Path, sentinels: list[str] | None = None
                       ) -> tuple[Path, str] | None:
    """Walk up from `start` until a sentinel is found. Returns (root, sentinel)
    or None if nothing found before /.

    `start` may be a file or directory. Walks up parents.
    """
    sentinels = sentinels or SENTINELS
    p = start if start.is_dir() else start.parent
    # Ensure we have an absolute path to walk
    try:
        p = p.resolve()
    except OSError:
        pass
    while True:
        for sent in sentinels:
            cand = p / sent
            if cand.exists():
                return p, sent
        if p.parent == p:
            return None
        p = p.parent


def _shared_subpath(paths: list[Path]) -> Path | None:
    """If all paths share a common ancestor strictly inside one of the
    candidates' sentinel roots, return that ancestor. Else None.
    """
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0] if paths[0].is_dir() else paths[0].parent
    parts_list = [p.parts for p in paths]
    common: list[str] = []
    for stack in zip(*parts_list):
        if all(x == stack[0] for x in stack):
            common.append(stack[0])
        else:
            break
    if not common:
        return None
    return Path(*common)


def detect_scopes(prefill_text: str,
                  sentinels: list[str] | None = None,
                  default_scope: Path | str | None = None,
                  ) -> list[DetectedScope]:
    """PRD §6.3 / v0.3.3: detect ALL distinct sentinel roots referenced
    in the prefill, not just the most common one. Used for multi-scope
    routing in a single conversation.

    ``default_scope`` enables relative-path resolution — paths that don't
    start with ``/`` are joined to it and kept iff the resolved file
    exists. Required for agents that cd into a repo and use relative
    paths in their prompts.
    """
    paths = extract_paths_from_prefill(prefill_text or "",
                                       default_scope=default_scope)
    if not paths:
        return []
    seen: dict[Path, tuple[Path, str, list[Path]]] = {}
    for p in paths:
        hit = find_sentinel_root(p, sentinels)
        if hit is None:
            continue
        root, sent = hit
        canon = canonicalize_scope(root)
        if canon not in seen:
            seen[canon] = (canon, sent, [])
        seen[canon][2].append(p)
    out: list[DetectedScope] = []
    for canon, sent, originals in seen.values():
        out.append(DetectedScope(
            root=canon,
            sentinel=sent,
            scope_hash=hash_scope(canon),
            mentioned_paths=tuple(originals),
        ))
    return out


def detect_scope(prefill_text: str,
                 explicit_root: Path | None = None,
                 sentinels: list[str] | None = None,
                 default_scope: Path | str | None = None,
                 ) -> DetectedScope | None:
    """Top-level scope detection.

    If `explicit_root` is given (e.g. via --longctx-scope), use it directly,
    skipping prefill-parsing.

    Otherwise:
      - Parse paths (absolute always; relative paths resolved against
        ``default_scope`` when one is provided, and kept iff they exist)
      - For each, find sentinel root
      - If multiple paths map to the same root and they have a deeper shared
        sub-directory that ALSO has a sentinel, prefer that (monorepo
        sub-package).
      - Else return the most-frequently-hit root.
      - Return None if nothing detected.
    """
    if explicit_root is not None:
        canon = canonicalize_scope(explicit_root)
        return DetectedScope(
            root=canon,
            sentinel="explicit",
            scope_hash=hash_scope(canon),
            mentioned_paths=(canon,),
        )

    paths = extract_paths_from_prefill(prefill_text or "",
                                       default_scope=default_scope)
    if not paths:
        return None

    # For each path, find nearest sentinel root.
    roots: list[tuple[Path, str, Path]] = []  # (root, sentinel, original)
    for p in paths:
        hit = find_sentinel_root(p, sentinels)
        if hit is None:
            continue
        roots.append((hit[0], hit[1], p))

    if not roots:
        return None

    # Check for shared sub-package inside the most common root (monorepo case).
    counter = Counter((r, s) for (r, s, _) in roots)
    (top_root, top_sent), _ = counter.most_common(1)[0]

    # Look for a deeper shared sub-path inside top_root that ALSO has a
    # sentinel — prefer that (monorepo sub-package detection).
    inside = [orig for (r, _, orig) in roots if r == top_root]
    shared = _shared_subpath(inside)
    if shared is not None and shared != top_root and top_root in shared.parents:
        sub_hit = find_sentinel_root(shared, sentinels)
        if sub_hit is not None and sub_hit[0] != top_root \
                and top_root in sub_hit[0].parents:
            top_root, top_sent = sub_hit

    canon = canonicalize_scope(top_root)
    return DetectedScope(
        root=canon,
        sentinel=top_sent,
        scope_hash=hash_scope(canon),
        mentioned_paths=tuple(p for (_, _, p) in roots if p),
    )
