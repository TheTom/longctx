"""Path-cluster scope detection.

The sentinel-walk approach (find nearest `.git` / `package.json` / etc.) is
brittle in three places:
  1. Fresh project dirs that haven't been `git init`'d yet.
  2. Monorepo sub-dirs where the sentinel walk goes too high.
  3. Scratch / experiment dirs that legitimately have no sentinel.

This module replaces "find the sentinel" with "find the working ancestor":
the deepest filesystem dir containing ≥ `min_files` distinct file mentions.
Sentinel becomes an optional refinement (handled by the caller), not a gate.

Pure, deterministic, no I/O. Threading concerns live in `tracker.py`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterResult:
    """A path cluster — a dir with a confidence floor of distinct mentions
    under it. ``unique_files`` is the cardinality of distinct paths that
    live (transitively) under ``ancestor``."""
    ancestor: Path
    unique_files: int
    total_paths: int           # input count after dedupe — informational


def cluster_paths(paths: list[Path], min_files: int = 3,
                  min_depth: int = 2) -> ClusterResult | None:
    """Return the best ancestor dir containing ≥ ``min_files`` distinct
    paths from ``paths``.

    "Best" rule, in order of priority:
      1. **More files captured wins.** Among qualifying ancestors, pick
         the one with the highest unique-file count. This keeps the
         broadest containing scope — e.g. picks the project root over a
         subdir that only holds a strict subset of the files mentioned.
         A scope that misses one file out of N would force longctx to
         fragment its index between the cluster pick and the missed
         file's eventual sentinel root.
      2. **Deeper wins on a tie.** Equal-cardinality ancestors get
         resolved to the more specific (deeper) one. Avoids picking ``/``
         when ``/work/proj`` and ``/`` both contain the same set.
      3. **Lex order wins on a second tie.** Stable across re-orderings.

    Returns None if no ancestor meets the threshold (including when the
    de-duped input is itself shorter than ``min_files``).

    The algorithm walks each path's parent chain, accumulating a
    ``ancestor -> {unique files under it}`` map. We deliberately do NOT
    cap the walk at a sentinel boundary — that lets the caller (a) layer
    sentinel preference on top of the cluster output, and (b) keep this
    function pure (no filesystem reads).

    Single-file inputs intentionally return None: one mention is too thin
    a signal to bind a scope to.
    """
    if min_files < 1:
        raise ValueError("min_files must be ≥ 1")
    if min_depth < 1:
        raise ValueError("min_depth must be ≥ 1")

    # Dedupe input. A path mentioned 10 times is one file's worth of evidence.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)

    if len(unique) < min_files:
        return None

    # Build ancestor -> set-of-paths-under-it map.
    by_ancestor: dict[Path, set[Path]] = defaultdict(set)
    for p in unique:
        # ``p.parents`` excludes ``p`` itself, which is what we want
        # (the ancestor is the dir, not the file).
        for parent in p.parents:
            by_ancestor[parent].add(p)

    # Filter shallow ancestors (``/`` by default). Otherwise the common-
    # ancestor of two unrelated subprojects would always win the
    # count-first comparator and produce a useless top-level scope.
    qualifying = [
        (anc, files) for anc, files in by_ancestor.items()
        if len(files) >= min_files and len(anc.parts) >= min_depth
    ]
    if not qualifying:
        return None

    # Sort: most files captured → deeper → lex. Reverse for "best first".
    qualifying.sort(
        key=lambda x: (len(x[1]), len(x[0].parts), str(x[0])),
        reverse=True,
    )
    best_anc, best_files = qualifying[0]
    return ClusterResult(
        ancestor=best_anc,
        unique_files=len(best_files),
        total_paths=len(unique),
    )


def all_clusters(paths: list[Path], min_files: int = 3,
                 ) -> list[ClusterResult]:
    """Return every qualifying ancestor (no parent/child filtering), sorted
    by the same key as :func:`cluster_paths`. Useful when a session
    legitimately spans multiple working dirs and the caller wants to
    speculatively index all of them.

    NOTE: this returns nested ancestors. If ``/a/b/c`` and ``/a/b`` both
    qualify, both appear. Caller dedupes if it wants only top-level
    distinct scopes.
    """
    if min_files < 1:
        raise ValueError("min_files must be ≥ 1")

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)

    if len(unique) < min_files:
        return []

    by_ancestor: dict[Path, set[Path]] = defaultdict(set)
    for p in unique:
        for parent in p.parents:
            by_ancestor[parent].add(p)

    qualifying = [
        ClusterResult(anc, len(files), len(unique))
        for anc, files in by_ancestor.items()
        if len(files) >= min_files
    ]
    qualifying.sort(
        key=lambda c: (c.unique_files, len(c.ancestor.parts), str(c.ancestor)),
        reverse=True,
    )
    return qualifying
