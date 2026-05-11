"""Tests for the live ``add_project`` MCP tool (PRD §3.8).

Persist=True writes a config entry; persist=False binds the project
to the current session via ``session_id`` (so the daemon GCs it on
disconnect). Forbidden_dirs are refused; already-indexed roots
return an informative error. After registration, a background
``full_scan`` is launched and the response returns immediately with
``status=indexing``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from longctx_daemon.mcp_server import (
    ConnectionContext,
    MCPServer,
    set_active_connection,
    reset_active_connection,
)
from longctx_daemon.types import Project


# -------------------------------------------------------------- fakes
@dataclass
class _IndexerCfg:
    forbidden_dirs: frozenset[str] = frozenset(
        {"secrets", "credentials", ".aws", ".ssh", ".gnupg"}
    )


class _FakeIndexer:
    """Captures add_project + full_scan calls. Doesn't actually walk
    a filesystem — that's the indexer's own test surface."""

    def __init__(self) -> None:
        self.config = _IndexerCfg()
        self.add_calls: list[tuple[str, Path]] = []
        self.scan_calls: list[str] = []

    def add_project(self, *, name: str, root_path: Path) -> Project:
        self.add_calls.append((name, root_path))
        return Project(
            name=name, root_path=str(root_path),
            last_full_scan_at=0, session_id=None,
        )

    def full_scan(self, project: str):
        self.scan_calls.append(project)
        return None

    def status(self):
        raise NotImplementedError


class _FakeChunkStore:
    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}

    def get_project(self, name: str) -> Optional[Project]:
        return self._projects.get(name)

    def list_projects(self) -> tuple[Project, ...]:
        return tuple(self._projects.values())

    def list_files(self, project: Optional[str] = None):
        return ()

    def get_chunks_by_file(self, file_id: int):
        return ()

    def upsert_project(self, project: Project) -> None:
        self._projects[project.name] = project


def _make_server(
    *,
    indexer: Optional[_FakeIndexer] = None,
    chunk_store: Optional[_FakeChunkStore] = None,
    config_path: Optional[Path] = None,
    background_runner=None,
) -> MCPServer:
    indexer = indexer or _FakeIndexer()
    chunk_store = chunk_store or _FakeChunkStore()

    # Pass a sync background_runner that just calls the function in the
    # foreground for deterministic testing.
    if background_runner is None:
        background_runner = lambda fn: fn()

    return MCPServer(
        searcher=MagicMock(),
        indexer=indexer,
        chunk_store=chunk_store,
        config_path=config_path,
        background_runner=background_runner,
    )


# -------------------------------------------------------------- tests
class TestAddProjectPathHandling:
    def test_rejects_nonexistent_path(self, tmp_path):
        server = _make_server()
        out = asyncio.run(server._dispatch_tool(
            "add_project",
            {"path": str(tmp_path / "no-such-dir")},
        ))
        assert "error" in out
        assert "does not exist" in out["error"]

    def test_rejects_file_path(self, tmp_path):
        f = tmp_path / "not-a-dir.txt"
        f.write_text("nope")
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(f)})
        )
        assert "error" in out
        assert "not a directory" in out["error"]

    def test_resolves_relative_paths(self, tmp_path, monkeypatch):
        proj_dir = tmp_path / "myproj"
        proj_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": "myproj"})
        )
        assert out["status"] == "indexing"
        assert out["project"] == "myproj"
        assert Path(out["root_path"]).is_absolute()

    def test_expands_tilde(self, tmp_path, monkeypatch):
        # Pretend HOME is tmp_path so ~/my-thing resolves under tmp.
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "my-thing"
        target.mkdir()
        server = _make_server()
        out = asyncio.run(server._dispatch_tool(
            "add_project", {"path": "~/my-thing"},
        ))
        assert out["status"] == "indexing"
        assert out["project"] == "my-thing"


class TestForbiddenDirs:
    def test_rejects_forbidden_basename(self, tmp_path):
        secrets = tmp_path / "secrets"
        secrets.mkdir()
        server = _make_server()
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(secrets)})
        )
        assert "error" in out
        assert "forbidden_dirs" in out
        assert "secrets" in out["forbidden_dirs"]

    def test_uses_indexer_config_forbidden_dirs(self, tmp_path):
        indexer = _FakeIndexer()
        # Tests that custom forbidden set is honored.
        indexer.config = _IndexerCfg(forbidden_dirs=frozenset({"private"}))
        priv = tmp_path / "private"
        priv.mkdir()
        server = _make_server(indexer=indexer)
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(priv)})
        )
        assert "error" in out
        assert "private" in out["forbidden_dirs"]


class TestAlreadyIndexed:
    def test_rejects_duplicate(self, tmp_path):
        d = tmp_path / "myapp"
        d.mkdir()
        store = _FakeChunkStore()
        store._projects["myapp"] = Project(
            name="myapp", root_path=str(d), last_full_scan_at=0,
        )
        server = _make_server(chunk_store=store)
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(d)})
        )
        assert "error" in out
        assert out["error"] == "project already indexed"
        assert out["name"] == "myapp"
        assert out["root_path"] == str(d)


class TestPersistTrue:
    def test_writes_config_block(self, tmp_path):
        d = tmp_path / "myapp"
        d.mkdir()
        cfg = tmp_path / "longctx-config.toml"
        cfg.write_text('# existing config\n')
        server = _make_server(config_path=cfg)
        out = asyncio.run(server._dispatch_tool(
            "add_project", {"path": str(d), "persist": True},
        ))
        assert out["status"] == "indexing"
        assert out["persist"] is True
        # Config file gained a [[projects]] block.
        body = cfg.read_text()
        assert "[[projects]]" in body
        assert 'name = "myapp"' in body
        assert str(d) in body

    def test_persist_clears_session_id(self, tmp_path):
        d = tmp_path / "myapp"
        d.mkdir()
        store = _FakeChunkStore()

        class _IndexerWithSession(_FakeIndexer):
            def add_project(self, *, name, root_path):
                self.add_calls.append((name, root_path))
                # Pretend the indexer somehow set a session_id; the
                # MCP server should clear it when persist=True.
                return Project(
                    name=name, root_path=str(root_path),
                    last_full_scan_at=0, session_id="ses_old",
                )

        idx = _IndexerWithSession()
        server = _make_server(indexer=idx, chunk_store=store)
        asyncio.run(server._dispatch_tool(
            "add_project", {"path": str(d), "persist": True},
        ))
        # The chunk_store was updated with session_id=None.
        assert store._projects["myapp"].session_id is None

    def test_no_config_path_skips_write(self, tmp_path):
        # When config_path is None we still succeed — the persist=True
        # path just records via chunk_store, no TOML write.
        d = tmp_path / "myapp"
        d.mkdir()
        server = _make_server(config_path=None)
        out = asyncio.run(server._dispatch_tool(
            "add_project", {"path": str(d), "persist": True},
        ))
        assert out["status"] == "indexing"


class TestPersistFalse:
    def test_session_bound_stamps_session_id(self, tmp_path):
        d = tmp_path / "ad-hoc"
        d.mkdir()
        store = _FakeChunkStore()

        # Drive add_project under a known ConnectionContext so we can
        # assert the session_id flows through.
        ctx = ConnectionContext(
            session_id="ses-12345", connection_id="conn-abcdef",
        )

        async def _run():
            tok = set_active_connection(ctx)
            try:
                server = _make_server(chunk_store=store)
                return await server._dispatch_tool(
                    "add_project", {"path": str(d), "persist": False},
                )
            finally:
                reset_active_connection(tok)

        out = asyncio.run(_run())
        assert out["status"] == "indexing"
        assert out["session_id"] == "ses-12345"
        assert store._projects["ad-hoc"].session_id == "ses-12345"

    def test_default_persist_is_false(self, tmp_path):
        d = tmp_path / "ad-hoc"
        d.mkdir()
        store = _FakeChunkStore()
        server = _make_server(chunk_store=store)
        out = asyncio.run(server._dispatch_tool(
            "add_project", {"path": str(d)},
        ))
        # No "persist" arg → defaults to False → session-bound row.
        assert out["status"] == "indexing"
        assert out["persist"] is False
        assert "session_id" in out


class TestBackgroundScan:
    def test_full_scan_kicked_off(self, tmp_path):
        d = tmp_path / "myapp"
        d.mkdir()
        idx = _FakeIndexer()
        scans: list[str] = []

        def _runner(fn):
            # Synchronous runner — fire immediately.
            fn()

        server = _make_server(
            indexer=idx, background_runner=_runner,
        )
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(d)})
        )
        assert out["status"] == "indexing"
        # Scan was queued + ran via the runner.
        assert idx.scan_calls == ["myapp"]

    def test_scan_failure_does_not_break_response(self, tmp_path):
        """If the scan errors, the response is already on the wire —
        we just swallow + log inside the indexer."""
        d = tmp_path / "myapp"
        d.mkdir()

        class _BoomIndexer(_FakeIndexer):
            def full_scan(self, project: str):
                raise RuntimeError("disk full")

        idx = _BoomIndexer()

        def _runner(fn):
            fn()  # synchronous

        server = _make_server(indexer=idx, background_runner=_runner)
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(d)})
        )
        assert out["status"] == "indexing"


class TestIndexerAddProjectFailure:
    def test_propagates_as_error_response(self, tmp_path):
        d = tmp_path / "myapp"
        d.mkdir()

        class _BadIndexer(_FakeIndexer):
            def add_project(self, *, name, root_path):
                raise ValueError("indexer rejected")

        server = _make_server(indexer=_BadIndexer())
        out = asyncio.run(
            server._dispatch_tool("add_project", {"path": str(d)})
        )
        assert "error" in out
        assert "indexer rejected" in out["error"]
