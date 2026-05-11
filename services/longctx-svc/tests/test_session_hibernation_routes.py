"""HTTP integration tests for the session hibernation endpoints.

Covers POST /sessions/{id}/hibernate, /resume, and GET /hibernation-status
through the FastAPI handlers. Uses a fake embedder so no MiniLM weights
are pulled at test time.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from longctx_svc import eviction_store
from longctx_svc.eviction_store import EvictionStore


class _DeterministicEmbedder:
    """4-dim keyword-presence embedder. Stable across calls so the
    on-disk snapshot of a chunk's embedding round-trips exactly."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, **_):
        out = []
        for t in texts:
            low = t.lower()
            vec = np.array([
                1.0 if "alpha" in low else 0.0,
                1.0 if "bravo" in low else 0.0,
                1.0 if "secret" in low else 0.0,
                0.5,  # constant — keeps norms non-zero
            ], dtype=np.float32)
            n = float(np.linalg.norm(vec) + 1e-8)
            out.append(vec / n)
        return np.stack(out)


@pytest.fixture
def hibernation_client(monkeypatch, tmp_path):
    """FastAPI TestClient with a fresh store pinned to tmp_path so each
    test gets isolated cold-storage."""
    fake_store = EvictionStore(
        embedder=_DeterministicEmbedder(),
        hibernation_dir=tmp_path / "snapshots",
    )
    monkeypatch.setattr(eviction_store, "_GLOBAL", fake_store)
    from longctx_svc.app import app
    with TestClient(app) as c:
        yield c


def _write_chunks(client: TestClient, sid: str, texts: list[str]):
    """Helper: write evicted chunks via the /evict/write endpoint."""
    payload = {
        "session_id": sid,
        "chunks": [
            {
                "text": t, "token_range": [0, len(t)],
                "layer": 0, "score": 0.5,
            }
            for t in texts
        ],
    }
    r = client.post("/evict/write", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_hibernate_route_writes_snapshot(hibernation_client):
    """POST /sessions/{id}/hibernate persists + drops RAM."""
    _write_chunks(hibernation_client, "sess-A",
                  ["alpha line", "bravo line", "the secret token"])
    r = hibernation_client.post("/sessions/sess-A/hibernate")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-A"
    assert body["hibernated"] is True
    assert body["n_chunks"] == 3

    # Status reflects on-disk only, RAM empty
    s = hibernation_client.get("/sessions/sess-A/hibernation-status").json()
    assert s["in_ram"] is False
    assert s["on_disk"] is True
    assert s["n_chunks_in_ram"] == 0
    assert s["n_chunks_on_disk"] == 3
    assert s["embedder_fingerprint"] is not None


def test_hibernate_empty_session(hibernation_client):
    """Empty session → hibernated=False, status shows nothing anywhere."""
    r = hibernation_client.post("/sessions/never-touched/hibernate")
    assert r.status_code == 200
    body = r.json()
    assert body["hibernated"] is False
    assert body["n_chunks"] == 0


def test_resume_route_restores_chunks(hibernation_client):
    """POST /sessions/{id}/resume reloads chunks into RAM."""
    _write_chunks(hibernation_client, "sess-R", ["alpha thing", "secret thing"])
    hibernation_client.post("/sessions/sess-R/hibernate")
    # Confirm dropped
    s = hibernation_client.get("/sessions/sess-R/hibernation-status").json()
    assert s["in_ram"] is False and s["on_disk"] is True

    r = hibernation_client.post("/sessions/sess-R/resume")
    assert r.status_code == 200
    body = r.json()
    assert body["resumed"] is True
    assert body["n_chunks"] == 2
    assert body["error"] is None

    # Now in RAM
    s = hibernation_client.get("/sessions/sess-R/hibernation-status").json()
    assert s["in_ram"] is True
    assert s["n_chunks_in_ram"] == 2


def test_resume_no_snapshot_returns_zero(hibernation_client):
    """Resume on a session with no snapshot is a clean no-op."""
    r = hibernation_client.post("/sessions/no-such-session/resume")
    assert r.status_code == 200
    body = r.json()
    assert body["resumed"] is False
    assert body["n_chunks"] == 0


def test_evict_retrieve_autoresumes_from_disk(hibernation_client):
    """The big integration: hibernate a session, then call /evict/retrieve.
    The store must auto-resume from disk + return the same top hit."""
    _write_chunks(hibernation_client, "sess-AR",
                  ["alpha line", "bravo line", "secret token here"])
    hibernation_client.post("/sessions/sess-AR/hibernate")

    # Direct /evict/retrieve should pull the session back from cold storage
    r = hibernation_client.post("/evict/retrieve", json={
        "session_id": "sess-AR",
        "query": "secret",
        "top_k": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["chunks"]) == 1
    assert "secret" in body["chunks"][0]["text"]
    # session_total should reflect the resumed chunks
    assert body["session_total"] == 3


def test_hibernation_status_reports_both_tiers(hibernation_client):
    """When a session lives in RAM AND has an older disk snapshot, status
    should report both. (Edge case: snapshot was hibernated, then a
    new /evict/write or /resume re-populated RAM.)"""
    _write_chunks(hibernation_client, "sess-BOTH", ["alpha thing"])
    hibernation_client.post("/sessions/sess-BOTH/hibernate")
    hibernation_client.post("/sessions/sess-BOTH/resume")

    s = hibernation_client.get("/sessions/sess-BOTH/hibernation-status").json()
    # Both tiers populated
    assert s["in_ram"] is True
    assert s["on_disk"] is True
    assert s["n_chunks_in_ram"] == 1
    assert s["n_chunks_on_disk"] == 1
