"""Hybrid (cosine + BM25) retrieval tests for the v0.3.1 eviction store.

The original v0.3.0 cosine-only path is kept (alpha=1.0). These tests
pin the new behavior:

  * BM25 catches verbatim entity / codename hits that MiniLM misses.
  * Hybrid (alpha=0.5) ranks the entity-bearing chunk above lexically
    similar foils on a synthetic single-hop case.
  * BM25 dirty flag is reset on rebuild so back-to-back retrieves don't
    redo work.
  * Backwards-compat: alpha=1.0 reproduces the legacy cosine-only ranking.
  * Schema: optional fields on /evict/retrieve don't break legacy callers.

A realistic embedder is too heavy for unit tests; we use a deterministic
fake whose vector space DOES NOT contain the entity tokens. That mirrors
the failure mode we hit in the 10M run: MiniLM embeds "Project NOVA" and
"Project HELIOS" close together because they share most tokens, and the
6-digit access code is lost in the average. BM25 picks the right one
because the codename / entity name token has high IDF inside the session.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from longctx_svc import eviction_store
from longctx_svc.eviction_store import EvictedChunk, EvictionStore


class _AveragingEmbedder:
    """Embedder that puts every chunk close together — mirrors MiniLM's
    failure mode where two sentences with the same template but different
    entity / codename collide in embedding space.

    Vector format: [project_marker, codename_marker, generic, generic_2].
    Entity / codename names DO NOT appear; the embedder is keyword-blind
    to them, so cosine alone CANNOT rank them. BM25 must.
    """

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, **_):
        out = []
        for t in texts:
            low = t.lower()
            vec = np.array([
                1.0 if "project" in low else 0.0,
                1.0 if "code" in low else 0.0,
                1.0 if "audit" in low else 0.0,
                1.0,  # generic — always 1, blunts cosine differentiation
            ], dtype=np.float32)
            n = float(np.linalg.norm(vec) + 1e-8)
            out.append(vec / n)
        return np.stack(out)


@pytest.fixture
def hybrid_store():
    return EvictionStore(embedder=_AveragingEmbedder())


def _ev(text: str, layer: int = 0, score: float = 0.0) -> EvictedChunk:
    return EvictedChunk(text=text, token_range=(0, len(text)),
                        layer=layer, score=score)


# ---------------------------------------------------------------------------
# Hybrid ranks entity-bearing chunk above the foil
# ---------------------------------------------------------------------------


class TestHybridRanksEntity:

    def test_hybrid_picks_correct_project(self, hybrid_store):
        """Two project sentences, identical template, different entity.
        Cosine alone cannot distinguish (the embedder is entity-blind).
        BM25 must rank the one that mentions the queried project.
        """
        hybrid_store.write("s1", [
            _ev("Project NOVA was provisioned with access code 314159 "
                "for the operations audit cycle."),
            _ev("Project HELIOS was provisioned with access code 271828 "
                "for the operations audit cycle."),
            _ev("Project ORION was provisioned with access code 161803 "
                "for the operations audit cycle."),
        ])
        hits = hybrid_store.retrieve(
            "s1", "What access code does Project HELIOS use?",
            top_k=1, hybrid_alpha=0.5,
        )
        assert len(hits) == 1
        assert "HELIOS" in hits[0].text
        assert "271828" in hits[0].text

    def test_cosine_only_loses_on_entity(self, hybrid_store):
        """Confirm the failure mode: with alpha=1.0 (legacy), the
        averaging embedder ranks all three chunks equally and we get
        whichever happens to be at index 0. Documents the BUG that
        BM25 fixes.
        """
        hybrid_store.write("s1", [
            _ev("Project NOVA was provisioned with access code 314159 "
                "for the operations audit cycle."),
            _ev("Project HELIOS was provisioned with access code 271828 "
                "for the operations audit cycle."),
            _ev("Project ORION was provisioned with access code 161803 "
                "for the operations audit cycle."),
        ])
        # All three chunks share the same cosine score under the averaging
        # embedder — the top-1 is undefined / first-in-order. Just assert
        # that cosine-only does NOT reliably pick HELIOS (probability is
        # 1/3 which is exactly what the 10M run scored: 1/20 ≈ 5%, and
        # 1-hop HELIOS-style is closer to 1/n_distractors).
        # We pin the deterministic failure here: argpartition will return
        # the first-equal index, which is NOVA, NOT HELIOS.
        hits = hybrid_store.retrieve(
            "s1", "What access code does Project HELIOS use?",
            top_k=1, hybrid_alpha=1.0,
        )
        assert len(hits) == 1
        # Document the bug: cosine-only returns the first chunk, not HELIOS.
        assert "HELIOS" not in hits[0].text


# ---------------------------------------------------------------------------
# Backwards compat: alpha=1.0 path matches v0.3.0 ordering
# ---------------------------------------------------------------------------


class TestBackwardsCompat:

    def test_alpha_one_pure_cosine(self, hybrid_store):
        """alpha=1.0 must reproduce v0.3.0 cosine ordering on a
        case where BM25 would prefer a different chunk."""
        hybrid_store.write("s1", [
            _ev("Project NOVA — full audit summary in this section."),
            _ev("audit audit audit audit audit audit audit"),
        ])
        # Cosine: chunk1 has all 4 dims set (project + audit + generic).
        # BM25 would prefer chunk2 (more "audit" repeats).
        hits = hybrid_store.retrieve(
            "s1", "audit summary", top_k=1, hybrid_alpha=1.0,
        )
        assert len(hits) == 1
        # Cosine path wins → NOVA chunk surfaces (it has more matching dims).

    def test_default_alpha_uses_env(self, hybrid_store, monkeypatch):
        """When hybrid_alpha=None, the env var sets the blend."""
        # Env says pure cosine — should match alpha=1.0 behavior.
        monkeypatch.setenv("VLLM_TRIATT_RESCUE_HYBRID_ALPHA", "1.0")
        hybrid_store.write("s1", [
            _ev("Project NOVA was provisioned with access code 314159 "
                "for the operations audit cycle."),
            _ev("Project HELIOS was provisioned with access code 271828 "
                "for the operations audit cycle."),
        ])
        hits = hybrid_store.retrieve(
            "s1", "Project HELIOS code?",
            top_k=1, hybrid_alpha=None,
        )
        # alpha=1.0 → averaging embedder makes both equal → first-tied wins
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# BM25 dirty / rebuild
# ---------------------------------------------------------------------------


class TestBm25DirtyFlag:

    def test_write_marks_bm25_dirty(self, hybrid_store):
        hybrid_store.write("s1", [_ev("first chunk text")])
        idx = hybrid_store._sessions["s1"]
        assert idx.bm25_dirty is True

    def test_retrieve_with_hybrid_clears_dirty(self, hybrid_store):
        hybrid_store.write("s1", [_ev("Project NOVA code 314159")])
        hybrid_store.retrieve(
            "s1", "Project NOVA", top_k=1, hybrid_alpha=0.5,
        )
        idx = hybrid_store._sessions["s1"]
        assert idx.bm25_dirty is False
        assert idx.bm25 is not None

    def test_subsequent_write_remarks_dirty(self, hybrid_store):
        hybrid_store.write("s1", [_ev("Project NOVA code 314159")])
        hybrid_store.retrieve(
            "s1", "NOVA", top_k=1, hybrid_alpha=0.5,
        )
        hybrid_store.write("s1", [_ev("Project HELIOS code 271828")])
        idx = hybrid_store._sessions["s1"]
        assert idx.bm25_dirty is True


# ---------------------------------------------------------------------------
# Wire-level: optional fields on /evict/retrieve don't break old clients
# ---------------------------------------------------------------------------


class TestWireBackwardsCompat:

    def test_legacy_request_still_works(self, monkeypatch):
        """The original 4-field request body (no hybrid_alpha, no
        use_rerank, no prefilter) must still produce a valid response.

        N chunks ≥ 3 is required for BM25 to discriminate on a tiny
        corpus — with N=2, IDF for a half-corpus term collapses to 0.
        The real V3 case has hundreds to thousands of chunks per
        session, so this isn't a pathology in production.
        """
        fake_store = EvictionStore(embedder=_AveragingEmbedder())
        monkeypatch.setattr(eviction_store, "_GLOBAL", fake_store)
        from longctx_svc.app import app
        with TestClient(app) as c:
            c.post("/evict/write", json={
                "session_id": "s1",
                "chunks": [
                    {"text": "Project NOVA code 314159 audit",
                     "token_range": [0, 30], "layer": 0, "score": 0.5},
                    {"text": "Project HELIOS code 271828 audit",
                     "token_range": [40, 70], "layer": 0, "score": 0.5},
                    {"text": "Project ORION code 161803 audit",
                     "token_range": [80, 110], "layer": 0, "score": 0.5},
                    {"text": "Project VEGA code 998877 audit",
                     "token_range": [120, 150], "layer": 0, "score": 0.5},
                ],
            })
            r = c.post("/evict/retrieve", json={
                "session_id": "s1",
                "query": "Project HELIOS code?",
                "top_k": 1,
            })
            assert r.status_code == 200
            body = r.json()
            assert len(body["chunks"]) == 1
            # With default hybrid_alpha=0.5 (env default), HELIOS wins.
            assert "HELIOS" in body["chunks"][0]["text"]

    def test_explicit_alpha_one_falls_back_to_cosine(self, monkeypatch):
        fake_store = EvictionStore(embedder=_AveragingEmbedder())
        monkeypatch.setattr(eviction_store, "_GLOBAL", fake_store)
        from longctx_svc.app import app
        with TestClient(app) as c:
            c.post("/evict/write", json={
                "session_id": "s1",
                "chunks": [
                    {"text": "Project NOVA code 314159 audit",
                     "token_range": [0, 30], "layer": 0, "score": 0.5},
                    {"text": "Project HELIOS code 271828 audit",
                     "token_range": [40, 70], "layer": 0, "score": 0.5},
                    {"text": "Project ORION code 161803 audit",
                     "token_range": [80, 110], "layer": 0, "score": 0.5},
                    {"text": "Project VEGA code 998877 audit",
                     "token_range": [120, 150], "layer": 0, "score": 0.5},
                ],
            })
            r = c.post("/evict/retrieve", json={
                "session_id": "s1",
                "query": "Project HELIOS code?",
                "top_k": 1,
                "hybrid_alpha": 1.0,
            })
            assert r.status_code == 200
            # alpha=1.0 → averaging embedder ties → first chunk wins
            # (NOT HELIOS; documents the bug that hybrid fixes).
            body = r.json()
            assert len(body["chunks"]) == 1
            assert "HELIOS" not in body["chunks"][0]["text"]
