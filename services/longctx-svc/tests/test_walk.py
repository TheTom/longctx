"""Walk + .gitignore + caps tests. PRD §5.2, §5.3."""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.scope.walk import walk_hot_scope, walk_package_scope


def test_walk_package_finds_all_non_ignored_files(project_dir: Path):
    files = walk_package_scope(project_dir)
    paths = [str(f) for f in files]
    assert any(f.endswith("auth.ts") for f in paths)
    assert any(f.endswith("billing.ts") for f in paths)
    assert any(f.endswith("README.md") for f in paths)
    assert any(f.endswith("package.json") for f in paths)


def test_walk_respects_gitignore(project_dir: Path):
    """Scenario §7.7: .gitignore'd dist/ is not indexed."""
    files = walk_package_scope(project_dir)
    paths = [str(f) for f in files]
    assert not any("dist/ignored.js" in f for f in paths)


def test_walk_skips_node_modules(tmp_path: Path):
    """Scenario §7.7: node_modules always skipped even if not in gitignore."""
    root = tmp_path / "mod"
    root.mkdir()
    (root / "package.json").write_text('{}')
    nm = root / "node_modules" / "lodash"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("// skipped")
    (root / "main.js").write_text("// kept")
    files = walk_package_scope(root)
    rels = [str(f.relative_to(root)) for f in files]
    assert any("main.js" in p for p in rels)
    assert not any("node_modules" in p for p in rels)


def test_walk_skips_large_files(tmp_path: Path):
    """Scenario §7.8: >5MB files skipped without crashing."""
    root = tmp_path / "big"
    root.mkdir()
    (root / "package.json").write_text('{}')
    huge = root / "huge.bin"
    huge.write_bytes(b"x" * (6 * 1024 * 1024))
    (root / "small.py").write_text("x = 1\n")
    files = walk_package_scope(root)
    paths = [str(f) for f in files]
    # Note: .bin is also in BINARY_EXTS so doubly skipped, but the size cap
    # alone should also catch it for non-binary extensions.
    assert not any("huge.bin" in p for p in paths)


def test_walk_skips_lockfiles(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "package.json").write_text('{}')
    (root / "package-lock.json").write_text('{"lockfile": true}')
    (root / "yarn.lock").write_text("# yarn lock")
    (root / "main.js").write_text("// kept")
    files = walk_package_scope(root)
    paths = [str(f) for f in files]
    assert any("main.js" in p for p in paths)
    assert not any("package-lock.json" in p for p in paths)
    assert not any("yarn.lock" in p for p in paths)


def test_walk_hot_scope_picks_mentioned(project_dir: Path):
    auth = project_dir / "src" / "auth.ts"
    files = walk_hot_scope(project_dir, mentioned_paths=[auth])
    paths = [str(f) for f in files]
    # Mentioned file must be present
    assert any(f.endswith("auth.ts") for f in paths)
    # Sibling billing.ts should also be in (siblings of mentioned files)
    assert any(f.endswith("billing.ts") for f in paths)


def test_walk_hot_scope_includes_src_recursive(project_dir: Path):
    files = walk_hot_scope(project_dir, mentioned_paths=[])
    paths = [str(f) for f in files]
    assert any(f.endswith("auth.ts") for f in paths)


def test_walk_skips_dotted_dirs(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "package.json").write_text('{}')
    (root / ".idea").mkdir()
    (root / ".idea" / "workspace.xml").write_text("<x/>")
    (root / "main.py").write_text("x = 1\n")
    files = walk_package_scope(root)
    paths = [str(f) for f in files]
    assert any("main.py" in p for p in paths)
    assert not any(".idea" in p for p in paths)


def test_walk_no_symlink_traversal(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "package.json").write_text('{}')
    (root / "real.py").write_text("x = 1")
    target = tmp_path / "outside"
    target.mkdir()
    (target / "secret.py").write_text("# should not be reached via symlink")
    try:
        (root / "link").symlink_to(target)
    except OSError:
        pytest.skip("no symlink support")
    files = walk_package_scope(root)
    paths = [str(f) for f in files]
    assert not any("secret.py" in p for p in paths)
