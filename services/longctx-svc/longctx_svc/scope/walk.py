"""Walk a detected scope to produce the file list for indexing.

Implements PRD §5.2 (Hot vs Package scope) and §5.3 (Caps + always-skip
+ .gitignore respect at every directory level).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pathspec

from longctx_svc.config import (
    ALWAYS_SKIP_DIRS,
    BINARY_EXTS,
    LOCKFILE_NAMES,
    Limits,
    get_config,
)


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Compile a PathSpec from a single .gitignore file, if present."""
    gi = root / ".gitignore"
    if not gi.is_file():
        return None
    try:
        with gi.open("r", errors="ignore") as f:
            return pathspec.PathSpec.from_lines(
                "gitwildmatch", f.readlines()
            )
    except OSError:
        return None


def _file_excluded(path: Path, root: Path,
                   gitignore_specs: list[tuple[Path, pathspec.PathSpec]],
                   max_size: int) -> bool:
    """Check skip rules for a single file."""
    if path.name in LOCKFILE_NAMES:
        return True
    if path.suffix.lower() in BINARY_EXTS:
        return True
    try:
        st = path.stat()
    except OSError:
        return True
    if st.st_size > max_size:
        return True
    if st.st_size == 0:
        return True
    # .gitignore at every level: walk up from the file's directory checking
    # which (gitignore-root, spec) pairs the path falls under.
    for gi_root, spec in gitignore_specs:
        try:
            rel = path.relative_to(gi_root)
        except ValueError:
            continue
        if spec.match_file(str(rel)):
            return True
    return False


def _walk_dir(root: Path, max_files: int, limits: Limits,
              recursive: bool = True) -> Iterator[Path]:
    """Yield non-skipped files under root.

    Walks with pruning: skip ALWAYS_SKIP_DIRS at any depth. Respects
    .gitignore files at each directory level (collected as we walk).
    """
    gitignore_specs: list[tuple[Path, pathspec.PathSpec]] = []
    root_spec = _load_gitignore(root)
    if root_spec is not None:
        gitignore_specs.append((root, root_spec))

    yielded = 0
    stack: list[tuple[Path, list[tuple[Path, pathspec.PathSpec]]]] = [
        (root, gitignore_specs)
    ]
    while stack and yielded < max_files:
        cur, specs = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        # Per-directory .gitignore augments the inherited list.
        local_gi = _load_gitignore(cur)
        local_specs = specs
        if local_gi is not None and cur != root:
            local_specs = specs + [(cur, local_gi)]

        for entry in entries:
            if yielded >= max_files:
                break
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in ALWAYS_SKIP_DIRS:
                        continue
                    if entry.name.startswith(".") and \
                            entry.name not in (".github", ".vscode-config"):
                        # Skip dotted dirs except whitelisted ones.
                        continue
                    if recursive:
                        stack.append((entry, local_specs))
                    continue
                if entry.is_file():
                    if _file_excluded(entry, root, local_specs,
                                      limits.max_file_size_bytes):
                        continue
                    yield entry
                    yielded += 1
            except OSError:
                continue


def walk_hot_scope(root: Path,
                   mentioned_paths: list[Path] | None = None
                   ) -> list[Path]:
    """Hot scope (PRD §5.2): mentioned files + their directories
    (non-recursive siblings) + nearest src/ recursive (capped).

    For the MVP we keep it simple: take mentioned files, their parent dir
    siblings, and (root / 'src') recursively if it exists. Cap at
    Limits.max_files_hot.
    """
    cfg = get_config()
    out: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        out.append(p)

    # 1. Mentioned files (only if inside root)
    for mp in mentioned_paths or []:
        try:
            real = mp.resolve()
        except OSError:
            continue
        try:
            real.relative_to(root)
        except ValueError:
            continue
        if real.is_file():
            add(real)
        elif real.is_dir():
            for sib in _walk_dir(real, cfg.limits.max_files_hot - len(out),
                                 cfg.limits, recursive=False):
                add(sib)

    # 2. Sibling files in the parent of each mentioned file
    for mp in mentioned_paths or []:
        try:
            real = mp.resolve()
        except OSError:
            continue
        if real.is_file():
            parent = real.parent
            for sib in _walk_dir(parent,
                                 cfg.limits.max_files_hot - len(out),
                                 cfg.limits, recursive=False):
                add(sib)

    # 3. nearest src/ recursive
    src = root / "src"
    if src.is_dir():
        for f in _walk_dir(src, cfg.limits.max_files_hot - len(out),
                           cfg.limits, recursive=True):
            add(f)

    # 4. If still small, scan immediate root files
    if len(out) < 50:
        for f in _walk_dir(root, cfg.limits.max_files_hot - len(out),
                           cfg.limits, recursive=False):
            add(f)

    return out


def walk_package_scope(root: Path) -> list[Path]:
    """Package scope (PRD §5.2): the entire detected project root,
    recursive, capped at Limits.max_files_package.
    """
    cfg = get_config()
    return list(_walk_dir(root, cfg.limits.max_files_package,
                          cfg.limits, recursive=True))
