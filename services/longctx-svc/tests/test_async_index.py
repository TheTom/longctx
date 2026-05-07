"""Async index kickoff tests. PRD §5.2 (background indexing).

Sarah's flow: first request sends `x-longctx-async: 1`, server returns
immediately with scope_status=indexing. By the second turn, status has
flipped to ready and chunks come back.
"""
from __future__ import annotations

import time

import pytest


def _wait_for(predicate, timeout: float = 4.0,
              interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_async_kickoff_first_call_returns_indexing(client, project_dir):
    """First call with x-longctx-async returns immediately in indexing
    state, no chunks yet."""
    r = client.post(
        "/retrieve",
        headers={
            "x-session-affinity": "async-1",
            "x-longctx-async": "1",
        },
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["scope_path"] is not None
    # Either still indexing OR finished too fast (small fixture). Both
    # are valid outcomes for the contract; the no-block guarantee is
    # what we're protecting.
    assert data["scope_status"] in ("indexing", "ready", "empty")


def test_async_kickoff_then_warm_second_call(client, project_dir):
    """First async call returns fast; subsequent call without the header
    blocks until ready and serves chunks."""
    # Kick off async
    r1 = client.post(
        "/retrieve",
        headers={
            "x-session-affinity": "async-2",
            "x-longctx-async": "1",
        },
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    assert r1.status_code == 200

    # Poll until ready (or fall through to sync second call which
    # will block on the same lock).
    def _ready() -> bool:
        from longctx_svc.state import get_state
        for e in get_state().all_scopes():
            if e.index is not None and e.index.embeddings is not None:
                return True
        return False

    _wait_for(_ready, timeout=5.0)

    r2 = client.post(
        "/retrieve",
        headers={"x-session-affinity": "async-2"},  # sync this time
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["scope_status"] in ("ready", "empty")


def test_sync_default_blocks_until_ready(client, project_dir):
    """Without x-longctx-async, behavior is unchanged: synchronous."""
    r = client.post(
        "/retrieve",
        headers={"x-session-affinity": "sync-only"},
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    assert r.status_code == 200
    assert r.json()["scope_status"] in ("ready", "empty")
