"""Workspace + multi-scope routing tests. PRD §6.3 / v0.3.3."""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.scope.detect import detect_scope, detect_scopes


# ---------------------------------------------------------------------------
# detect_scopes (multi-mention)
# ---------------------------------------------------------------------------

def test_detect_scopes_returns_one_for_single_mention(project_dir):
    auth = project_dir / "src" / "auth.ts"
    scopes = detect_scopes(f"see {auth}")
    assert len(scopes) == 1
    assert scopes[0].sentinel == "package.json"


def test_detect_scopes_returns_multiple_distinct_roots(tmp_path):
    """Two different projects mentioned in one prefill → both detected."""
    a = tmp_path / "proj_a"
    b = tmp_path / "proj_b"
    a.mkdir()
    b.mkdir()
    (a / "package.json").write_text('{"name":"a"}')
    (b / "package.json").write_text('{"name":"b"}')
    (a / "alpha.py").write_text("x = 1\n")
    (b / "beta.py").write_text("y = 2\n")
    prefill = f"see {a}/alpha.py and {b}/beta.py"
    scopes = detect_scopes(prefill)
    roots = {s.root.name.lower() for s in scopes}
    assert roots == {"proj_a", "proj_b"}


def test_detect_scopes_dedups_same_root(project_dir):
    """Two paths inside the same project → one DetectedScope."""
    auth = project_dir / "src" / "auth.ts"
    bill = project_dir / "src" / "billing.ts"
    scopes = detect_scopes(f"see {auth} and also {bill}")
    assert len(scopes) == 1


def test_detect_scopes_returns_empty_when_no_paths():
    assert detect_scopes("how do i fix this?") == []


# ---------------------------------------------------------------------------
# Session: scope_hashes accumulates across binds
# ---------------------------------------------------------------------------

def test_session_accumulates_scope_hashes():
    from longctx_svc.session.manager import SessionManager
    sm = SessionManager()
    sm.bind("user1", "scope-a")
    sm.bind("user1", "scope-b")
    e = sm.get("user1")
    assert e.scope_hash == "scope-b"          # primary = most recent
    assert e.scope_hashes == {"scope-a", "scope-b"}


# ---------------------------------------------------------------------------
# /retrieve workspace mode (ws:)
# ---------------------------------------------------------------------------

def test_workspace_returns_no_scope_when_session_empty(client):
    """ws: without any prior scopes bound to the session → no-scope."""
    r = client.post(
        "/retrieve",
        headers={"x-session-affinity": "ws-empty"},
        json={
            "prefill_text": "no path here",
            "query": "anything", "top_k": 4,
            "explicit_scope": "ws:",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["scope_status"] == "no-scope"
    assert data["scope_path"] == "ws:"
    assert data["chunks"] == []


def test_workspace_searches_across_bound_scopes(client, tmp_path):
    """Bind two scopes via separate retrieves, then ws: searches both."""
    a = tmp_path / "proj_a"
    b = tmp_path / "proj_b"
    a.mkdir()
    b.mkdir()
    (a / "package.json").write_text('{"name":"a"}')
    (b / "package.json").write_text('{"name":"b"}')
    (a / "alpha.py").write_text(
        "# alpha module — handles authentication\n"
        "def alpha_func(): return 'alpha'\n"
    )
    (b / "beta.py").write_text(
        "# beta module — handles billing\n"
        "def beta_func(): return 'beta'\n"
    )

    sid = "ws-multi"
    # Visit A
    r1 = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": f"see {a}/alpha.py",
              "query": "alpha", "top_k": 2},
    )
    assert r1.status_code == 200
    # Visit B
    r2 = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": f"see {b}/beta.py",
              "query": "beta", "top_k": 2},
    )
    assert r2.status_code == 200

    # Now workspace query
    r3 = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": "ignored when ws: is set",
              "query": "alpha beta", "top_k": 4,
              "explicit_scope": "ws:"},
    )
    assert r3.status_code == 200
    data = r3.json()
    assert data["scope_path"] == "ws:"
    assert data["scope_status"] == "ready"
    file_paths = {c["file_path"] for c in data["chunks"]}
    # We expect chunks from BOTH projects
    assert any("alpha.py" in p for p in file_paths)
    assert any("beta.py" in p for p in file_paths)


def test_workspace_emits_workspace_sentinel(client, tmp_path):
    """sentinel field reads 'workspace' for ws: responses."""
    a = tmp_path / "proj_a"
    a.mkdir()
    (a / "package.json").write_text('{}')
    (a / "x.py").write_text("x = 1\n")
    sid = "ws-sentinel"
    client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": f"see {a}/x.py", "query": "x", "top_k": 1},
    )
    r = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": "n/a", "query": "x", "top_k": 1,
              "explicit_scope": "ws:"},
    )
    assert r.json()["scope_sentinel"] == "workspace"


# ---------------------------------------------------------------------------
# Multi-scope routing within a single turn
# ---------------------------------------------------------------------------

def test_single_turn_mentioning_two_projects_binds_both(client, tmp_path):
    """A single prefill mentioning files from 2 projects binds both
    scopes to the session — so a follow-up ws: query can find them."""
    a = tmp_path / "proj_a"
    b = tmp_path / "proj_b"
    for d in (a, b):
        d.mkdir()
        (d / "package.json").write_text(f'{{"name":"{d.name}"}}')
    (a / "alpha.py").write_text("# alpha\n")
    (b / "beta.py").write_text("# beta\n")

    sid = "multi-mention"
    # Single turn, both files mentioned
    r = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={
            "prefill_text": f"compare {a}/alpha.py vs {b}/beta.py",
            "query": "compare", "top_k": 2,
        },
    )
    assert r.status_code == 200
    # Now workspace query — both scopes should be reachable
    rws = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": "n/a", "query": "alpha beta", "top_k": 4,
              "explicit_scope": "ws:"},
    )
    assert rws.status_code == 200
    file_paths = {c["file_path"] for c in rws.json()["chunks"]}
    assert any("alpha.py" in p for p in file_paths)
    assert any("beta.py" in p for p in file_paths)


# ---------------------------------------------------------------------------
# pipeline.retrieve_multi (unit)
# ---------------------------------------------------------------------------

def test_retrieve_multi_merges_by_score(fake_embedder, fake_reranker,
                                          tmp_path):
    """Cross-scope merge picks top-K by score regardless of which scope."""
    import time
    import numpy as np
    from longctx_svc.indexer.builder import ScopeIndex
    from longctx_svc.indexer.chunker import Chunk
    from longctx_svc.retrieve.pipeline import RetrievePipeline

    pipe = RetrievePipeline(embedder=fake_embedder, reranker=fake_reranker)
    # Two scopes, each with one chunk; query designed so chunk-A wins
    idx_a = ScopeIndex(
        scope_root=tmp_path, scope_hash="A",
        chunks=[Chunk(text="alpha module text",
                      file_path="/p/a.py",
                      start_line=1, end_line=1, file_type="code")],
        embeddings=np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        file_count=1, built_at=time.time(),
    )
    idx_b = ScopeIndex(
        scope_root=tmp_path, scope_hash="B",
        chunks=[Chunk(text="beta module text",
                      file_path="/p/b.py",
                      start_line=1, end_line=1, file_type="code")],
        embeddings=np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
        file_count=1, built_at=time.time(),
    )
    res = pipe.retrieve_multi("alpha", indexes=[idx_a, idx_b],
                              top_k=2, use_rerank=False)
    paths = [c.file_path for c in res.chunks]
    assert "/p/a.py" in paths
    assert "/p/b.py" in paths


def test_retrieve_multi_empty_indexes_returns_empty(fake_embedder,
                                                       fake_reranker):
    from longctx_svc.retrieve.pipeline import RetrievePipeline
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=fake_reranker)
    res = pipe.retrieve_multi("anything", indexes=[], top_k=4)
    assert res.chunks == []
    assert res.scores == []
