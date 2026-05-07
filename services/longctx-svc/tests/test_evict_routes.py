"""HTTP smoke tests for /evict/write + /evict/retrieve.

The eviction_store unit tests exercise the store directly. These tests
exercise the FastAPI handlers — the wire shape is what V3 actually POSTs
to from inside vLLM, so we want to be sure schema + status codes + the
session_total / accepted accounting all hold over HTTP.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from longctx_svc import eviction_store
from longctx_svc.eviction_store import EvictionStore


class _DeterministicEmbedder:
    """5-dim keyword-presence embedder. Same family as
    test_eviction_store.py but lives here so this file is self-contained
    for the HTTP layer. Last dim flags "off-topic" content so the
    score-floor test can distinguish near-orthogonal chunks."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, **_):
        out = []
        for t in texts:
            low = t.lower()
            on_topic = (
                "password" in low or "meeting" in low or "deploy" in low
                or "what" in low
            )
            vec = np.array([
                1.0 if "password" in low else 0.0,
                1.0 if "meeting" in low else 0.0,
                1.0 if "deploy" in low else 0.0,
                1.0 if "?" in t else 0.0,
                0.0 if on_topic else 1.0,
            ], dtype=np.float32)
            n = float(np.linalg.norm(vec) + 1e-8)
            out.append(vec / n)
        return np.stack(out)


@pytest.fixture
def evict_client(monkeypatch):
    """FastAPI TestClient with the eviction store singleton replaced by a
    fresh deterministic-embedder instance. Avoids loading MiniLM."""
    fake_store = EvictionStore(embedder=_DeterministicEmbedder())
    monkeypatch.setattr(eviction_store, "_GLOBAL", fake_store)
    from longctx_svc.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /evict/write
# ---------------------------------------------------------------------------


class TestEvictWrite:

    def test_empty_chunks_returns_zero(self, evict_client):
        r = evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [],
        })
        assert r.status_code == 200
        body = r.json()
        assert body == {"accepted": 0, "session_total": 0}

    def test_accepts_chunks_and_reports_total(self, evict_client):
        r = evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [
                {"text": "the password is hunter2",
                 "token_range": [10, 20], "layer": 5, "score": 0.91},
                {"text": "the meeting is at 3pm",
                 "token_range": [40, 50], "layer": 7, "score": 0.83},
            ],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["accepted"] == 2
        assert body["session_total"] == 2

    def test_session_total_accumulates_across_calls(self, evict_client):
        evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [{"text": "first", "token_range": [0, 5],
                        "layer": 0, "score": 0.1}],
        })
        r2 = evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [{"text": "second", "token_range": [5, 10],
                        "layer": 1, "score": 0.2}],
        })
        assert r2.json()["session_total"] == 2

    def test_sessions_isolated(self, evict_client):
        evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [{"text": "a", "token_range": [0, 1],
                        "layer": 0, "score": 0.0}],
        })
        r = evict_client.post("/evict/write", json={
            "session_id": "s2",
            "chunks": [{"text": "b", "token_range": [0, 1],
                        "layer": 0, "score": 0.0}],
        })
        assert r.json()["session_total"] == 1  # s2 has only its own write


# ---------------------------------------------------------------------------
# /evict/retrieve
# ---------------------------------------------------------------------------


class TestEvictRetrieve:

    def test_empty_session_returns_no_chunks(self, evict_client):
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "ghost",
            "query": "anything",
            "top_k": 8,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["chunks"] == []
        assert body["session_total"] == 0

    def test_roundtrip_returns_relevant_chunk(self, evict_client):
        evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [
                {"text": "the password is hunter2",
                 "token_range": [10, 20], "layer": 5, "score": 0.9},
                {"text": "the deploy schedule slipped to friday",
                 "token_range": [40, 60], "layer": 7, "score": 0.4},
            ],
        })
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "what was the password?",
            "top_k": 2,
        })
        assert r.status_code == 200
        body = r.json()
        # password chunk ranks above deploy chunk for a password query.
        assert len(body["chunks"]) == 2
        assert "password" in body["chunks"][0]["text"]
        assert body["session_total"] == 2

    def test_score_floor_filters(self, evict_client):
        evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [
                {"text": "the password is hunter2",
                 "token_range": [10, 20], "layer": 5, "score": 0.9},
                {"text": "completely unrelated lorem ipsum content",
                 "token_range": [30, 50], "layer": 6, "score": 0.1},
            ],
        })
        # High floor should drop the irrelevant chunk.
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "what was the password?",
            "top_k": 5,
            "score_floor": 0.5,
        })
        body = r.json()
        for c in body["chunks"]:
            assert "password" in c["text"]

    def test_session_isolated_on_retrieve(self, evict_client):
        evict_client.post("/evict/write", json={
            "session_id": "alice",
            "chunks": [{"text": "the password is hunter2",
                        "token_range": [0, 5], "layer": 0, "score": 0.9}],
        })
        # Bob retrieves with the same query — should get nothing.
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "bob",
            "query": "what was the password?",
            "top_k": 5,
        })
        assert r.json()["chunks"] == []

    def test_metadata_round_trips(self, evict_client):
        """token_range, layer, score must survive write→retrieve."""
        evict_client.post("/evict/write", json={
            "session_id": "s1",
            "chunks": [{"text": "the password is hunter2",
                        "token_range": [42, 99], "layer": 13, "score": 0.77}],
        })
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "password?",
            "top_k": 1,
        })
        c = r.json()["chunks"][0]
        assert c["token_range"] == [42, 99]
        assert c["layer"] == 13
        # Score of the WRITTEN chunk (not the cosine sim from retrieval).
        assert c["score"] == pytest.approx(0.77, abs=1e-5)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestEvictValidation:

    def test_write_rejects_missing_session_id(self, evict_client):
        r = evict_client.post("/evict/write", json={
            "chunks": [],
        })
        assert r.status_code == 422  # FastAPI validation error

    def test_retrieve_rejects_top_k_zero(self, evict_client):
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "anything",
            "top_k": 0,
        })
        assert r.status_code == 422

    def test_retrieve_rejects_top_k_too_large(self, evict_client):
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "anything",
            "top_k": 1000,
        })
        assert r.status_code == 422

    def test_retrieve_rejects_score_floor_above_one(self, evict_client):
        r = evict_client.post("/evict/retrieve", json={
            "session_id": "s1",
            "query": "anything",
            "score_floor": 1.5,
        })
        assert r.status_code == 422
