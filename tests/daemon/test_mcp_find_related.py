"""Tests for the live ``find_related`` MCP tool (PRD §6.2 revised).

Semantic similarity only — NOT static analysis. The chunk at
``file_path``:``line`` is embedded; the top-K nearest other chunks
are returned. Source chunk excluded from the result. ``file_path``
accepts either ``project/rel/path.py`` or bare ``rel/path.py``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from longctx_daemon.mcp_server import MCPServer
from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
)


# ---------------------------------------------------------- fakes

class _FakeChunkStore:
    """Just enough store surface for find_related."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._files: dict[int, FileRecord] = {}
        self._chunks_by_file: dict[int, list[Chunk]] = {}
        self._next_file_id = 1
        self._next_chunk_id = 1

    # ---- project + file CRUD (test helper)
    def add_project(self, name: str, root: str = "/x") -> None:
        self._projects[name] = Project(
            name=name, root_path=root, last_full_scan_at=0,
        )

    def add_file_with_chunks(
        self, project: str, rel_path: str,
        chunk_specs: list[tuple[int, int, str]],  # (start_line, end_line, text)
        embedding_rows: Optional[list[Optional[int]]] = None,
    ) -> tuple[int, list[int]]:
        """Insert one file plus N chunks; returns (file_id, chunk_ids)."""
        file_id = self._next_file_id
        self._next_file_id += 1
        self._files[file_id] = FileRecord(
            id=file_id, project=project, rel_path=rel_path,
            mtime=0, size_bytes=0, content_hash="h",
        )
        chunks: list[Chunk] = []
        ids: list[int] = []
        for idx, (start, end, text) in enumerate(chunk_specs):
            cid = self._next_chunk_id
            self._next_chunk_id += 1
            row = (
                embedding_rows[idx]
                if embedding_rows is not None and idx < len(embedding_rows)
                else cid - 1
            )
            chunks.append(Chunk(
                id=cid, file_id=file_id, chunk_index=idx,
                start_offset=0, end_offset=len(text),
                start_line=start, end_line=end,
                token_count=len(text), content_hash="hch",
                text=text, embedder_model="m", embedder_sha256="s",
                embedding_row=row,
            ))
            ids.append(cid)
        self._chunks_by_file[file_id] = chunks
        return file_id, ids

    # ---- store API surface used by MCPServer
    def list_projects(self):
        return tuple(self._projects.values())

    def get_project(self, name: str):
        return self._projects.get(name)

    def list_files(self, project: Optional[str] = None):
        if project is None:
            return tuple(self._files.values())
        return tuple(r for r in self._files.values() if r.project == project)

    def get_file(self, project: str, rel_path: str):
        for r in self._files.values():
            if r.project == project and r.rel_path == rel_path:
                return r
        return None

    def get_file_by_id(self, file_id: int):
        return self._files.get(file_id)

    def get_chunks_by_file(self, file_id: int):
        return tuple(self._chunks_by_file.get(file_id, ()))

    def get_chunks_by_id(self, ids):
        ids_set = set(ids)
        out = []
        for chunks in self._chunks_by_file.values():
            for c in chunks:
                if c.id in ids_set:
                    out.append(c)
        return tuple(out)


class _FakeEmbedStore:
    """Minimum surface: ``get(row)`` returns the row's vector;
    ``search_dense`` returns Hits from a precomputed ranking we set per
    test.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._rows: dict[int, np.ndarray] = {}
        self._dense_ranking: list[Hit] = []
        self.last_search_query: Optional[np.ndarray] = None

    def set_row(self, row: int, vec) -> None:
        self._rows[row] = np.asarray(vec, dtype=np.float32)

    def set_dense_ranking(self, hits: list[Hit]) -> None:
        self._dense_ranking = list(hits)

    def get(self, row: int) -> np.ndarray:
        return self._rows[row]

    def search_dense(self, query_emb, k, chunk_ids_in_scope=None):
        self.last_search_query = np.asarray(query_emb)
        return tuple(self._dense_ranking[:k])


def _make_server(
    *,
    chunk_store: Optional[_FakeChunkStore] = None,
    embed_store: Optional[_FakeEmbedStore] = None,
) -> MCPServer:
    return MCPServer(
        searcher=MagicMock(),
        indexer=MagicMock(),
        chunk_store=chunk_store or _FakeChunkStore(),
        embed_store=embed_store,
    )


# --------------------------------------------------- happy-path tests
class TestFindRelatedHappyPath:
    def test_finds_similar_chunks_excluding_source(self):
        store = _FakeChunkStore()
        store.add_project("myapp")
        # Source file: 1 chunk at lines 1-30.
        f1, [c1] = store.add_file_with_chunks(
            "myapp", "src/a.py",
            [(1, 30, "def authenticate(user): ...")],
        )
        # Two other chunks in different files.
        f2, [c2] = store.add_file_with_chunks(
            "myapp", "src/b.py",
            [(10, 20, "def authorize(user): ...")],
        )
        f3, [c3] = store.add_file_with_chunks(
            "myapp", "src/c.py",
            [(5, 15, "def login(user): ...")],
        )

        emb = _FakeEmbedStore()
        emb.set_row(0, [1.0, 0.0, 0.0, 0.0])
        emb.set_row(1, [0.9, 0.1, 0.0, 0.0])
        emb.set_row(2, [0.7, 0.3, 0.0, 0.0])
        # Note search_dense returns ranking; source chunk first to test
        # exclusion logic.
        emb.set_dense_ranking([
            Hit(chunk_id=c1, score=1.0),
            Hit(chunk_id=c2, score=0.9),
            Hit(chunk_id=c3, score=0.7),
        ])

        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related",
            {"file_path": "myapp/src/a.py", "max_results": 5},
        ))
        # Source chunk excluded — only c2 + c3 returned.
        assert "chunks" in out
        ids_returned = [
            c["file_path"] for c in out["chunks"]
        ]
        assert "myapp/src/b.py" in ids_returned
        assert "myapp/src/c.py" in ids_returned
        assert "myapp/src/a.py" not in ids_returned

    def test_max_results_caps(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "a.py", [(1, 5, "x")])
        store.add_file_with_chunks("p", "b.py", [(1, 5, "y")])
        store.add_file_with_chunks("p", "c.py", [(1, 5, "z")])
        store.add_file_with_chunks("p", "d.py", [(1, 5, "w")])

        emb = _FakeEmbedStore()
        for i in range(4):
            emb.set_row(i, [1.0, 0.0, 0.0, 0.0])
        # Source = chunk_id 1; rank: 1, 2, 3, 4
        emb.set_dense_ranking([
            Hit(chunk_id=1, score=1.0),
            Hit(chunk_id=2, score=0.9),
            Hit(chunk_id=3, score=0.8),
            Hit(chunk_id=4, score=0.7),
        ])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related",
            {"file_path": "p/a.py", "max_results": 2},
        ))
        # 2 results (excluding the source chunk).
        assert len(out["chunks"]) == 2

    def test_default_max_results_is_5(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "a.py", [(1, 5, "x")])
        for i in range(10):
            store.add_file_with_chunks("p", f"b{i}.py", [(1, 5, f"y{i}")])

        emb = _FakeEmbedStore()
        for i in range(11):
            emb.set_row(i, [1.0, 0.0, 0.0, 0.0])
        emb.set_dense_ranking(
            [Hit(chunk_id=1, score=1.0)] +
            [Hit(chunk_id=i, score=1.0 - i * 0.05) for i in range(2, 12)]
        )
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related",
            {"file_path": "p/a.py"},
        ))
        assert len(out["chunks"]) == 5

    def test_response_chunks_have_required_fields(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "a.py", [(1, 5, "src")])
        store.add_file_with_chunks("p", "b.py", [(7, 12, "other")])
        emb = _FakeEmbedStore()
        emb.set_row(0, [1, 0, 0, 0])
        emb.set_row(1, [0.5, 0.5, 0, 0])
        emb.set_dense_ranking([
            Hit(chunk_id=1, score=1.0),
            Hit(chunk_id=2, score=0.5),
        ])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related",
            {"file_path": "p/a.py"},
        ))
        c = out["chunks"][0]
        for k in (
            "project", "file_path", "start_line", "end_line",
            "text", "relevance_score",
        ):
            assert k in c

    def test_includes_source_chunk_summary(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "a.py", [(1, 30, "src text")])
        store.add_file_with_chunks("p", "b.py", [(7, 12, "other")])
        emb = _FakeEmbedStore()
        emb.set_row(0, [1, 0, 0, 0])
        emb.set_row(1, [0.5, 0.5, 0, 0])
        emb.set_dense_ranking([
            Hit(chunk_id=2, score=0.7),
        ])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related",
            {"file_path": "p/a.py"},
        ))
        src = out["source"]
        assert src["project"] == "p"
        assert src["file_path"] == "p/a.py"
        assert src["start_line"] == 1
        assert src["end_line"] == 30


# --------------------------------------------------- line targeting
class TestFindRelatedLineTargeting:
    def test_picks_chunk_for_line(self):
        store = _FakeChunkStore()
        store.add_project("p")
        # Three chunks: 1-10, 11-20, 21-30
        store.add_file_with_chunks(
            "p", "a.py",
            [(1, 10, "a"), (11, 20, "b"), (21, 30, "c")],
        )
        store.add_file_with_chunks("p", "other.py", [(1, 5, "other")])
        emb = _FakeEmbedStore()
        for i in range(4):
            emb.set_row(i, [1.0, 0.0, 0.0, 0.0])
        # When line=15, source should be chunk_id=2 (chunk_index=1).
        emb.set_dense_ranking([Hit(chunk_id=4, score=0.7)])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/a.py", "line": 15},
        ))
        # The source range should match chunk 2's lines.
        assert out["source"]["start_line"] == 11
        assert out["source"]["end_line"] == 20

    def test_line_none_picks_first_chunk(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks(
            "p", "a.py",
            [(1, 10, "first"), (11, 20, "second")],
        )
        store.add_file_with_chunks("p", "o.py", [(1, 5, "o")])
        emb = _FakeEmbedStore()
        for i in range(3):
            emb.set_row(i, [1.0, 0, 0, 0])
        emb.set_dense_ranking([Hit(chunk_id=3, score=0.5)])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/a.py"},
        ))
        # First chunk wins (chunk_index=0 → line 1-10).
        assert out["source"]["start_line"] == 1
        assert out["source"]["end_line"] == 10

    def test_line_outside_chunks_falls_back_to_first(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks(
            "p", "a.py",
            [(1, 10, "first")],
        )
        store.add_file_with_chunks("p", "o.py", [(1, 5, "o")])
        emb = _FakeEmbedStore()
        for i in range(2):
            emb.set_row(i, [1, 0, 0, 0])
        emb.set_dense_ranking([Hit(chunk_id=2, score=0.5)])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/a.py", "line": 9999},
        ))
        # Falls back to first chunk.
        assert out["source"]["start_line"] == 1


# --------------------------------------------------- error paths
class TestFindRelatedErrors:
    def test_no_embed_store_returns_error(self):
        server = _make_server(embed_store=None)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "any/path.py"},
        ))
        assert "error" in out

    def test_unknown_file_returns_error(self):
        store = _FakeChunkStore()
        store.add_project("p")
        emb = _FakeEmbedStore()
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/missing.py"},
        ))
        assert out["error"] == "file not indexed"
        assert out["file_path"] == "p/missing.py"

    def test_file_with_no_chunks_returns_error(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "empty.py", [])
        emb = _FakeEmbedStore()
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/empty.py"},
        ))
        assert "error" in out
        assert "no indexed chunks" in out["error"]

    def test_chunk_without_embedding_returns_error(self):
        store = _FakeChunkStore()
        store.add_project("p")
        # embedding_rows=[None] makes the chunk unembedded.
        store.add_file_with_chunks(
            "p", "a.py", [(1, 10, "x")], embedding_rows=[None],
        )
        emb = _FakeEmbedStore()
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/a.py"},
        ))
        assert "error" in out
        assert "no embedding" in out["error"]


# --------------------------------------------------- file_path forms
class TestFindRelatedFilePathForms:
    def test_accepts_bare_rel_path(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "src/a.py", [(1, 10, "x")])
        store.add_file_with_chunks("p", "other.py", [(1, 5, "y")])
        emb = _FakeEmbedStore()
        for i in range(2):
            emb.set_row(i, [1, 0, 0, 0])
        emb.set_dense_ranking([Hit(chunk_id=2, score=0.5)])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "src/a.py"},
        ))
        assert out["source"]["file_path"] == "p/src/a.py"

    def test_accepts_canonical_form(self):
        store = _FakeChunkStore()
        store.add_project("p")
        store.add_file_with_chunks("p", "src/a.py", [(1, 10, "x")])
        store.add_file_with_chunks("p", "other.py", [(1, 5, "y")])
        emb = _FakeEmbedStore()
        for i in range(2):
            emb.set_row(i, [1, 0, 0, 0])
        emb.set_dense_ranking([Hit(chunk_id=2, score=0.5)])
        server = _make_server(chunk_store=store, embed_store=emb)
        out = asyncio.run(server._dispatch_tool(
            "find_related", {"file_path": "p/src/a.py"},
        ))
        assert out["source"]["file_path"] == "p/src/a.py"
