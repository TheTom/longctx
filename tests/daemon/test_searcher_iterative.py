"""Iterative retrieval API tests for ``Searcher.search``.

Covers the AutoCodeRover-style retry pattern wired in 2026-05-19:
``prior_context`` / ``prior_context_weight`` / ``suppress_ids``.

The synthetic-store tests exercise the API surface deterministically.
The live-pack test runs against the swezero-12m python pack when
available — skipped when the pack isn't on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from longctx_daemon.searcher import Searcher, SearcherConfig
from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
    ScopeFilter,
)


# Use the java pack for live smoke — 1.4 GB embeddings vs python's 4.9 GB.
# The point is to exercise the real Searcher + real memmap path end to end,
# not to stress-test brute-force matmul throughput; smaller pack = faster
# test without losing what we are actually verifying (suppress_ids round trip
# against a live corpus).
SWEZERO_LIVE_PACK = Path("~/.cache/longctx/corpora/swezero-12m/java").expanduser()
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


# -------------------------------------------------------------- synthetic stack

class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.

    Returns one fixed unit vector per known string. Unknown strings
    map to a default zero-aligned vector so behavior stays predictable
    without bringing the real transformer into the test path.
    """

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = {k: v / max(float(np.linalg.norm(v)), 1e-12)
                         for k, v in vectors.items()}
        self._fallback = np.zeros(next(iter(vectors.values())).shape,
                                  dtype=np.float32)

    def encode(self, texts, *, convert_to_numpy=True, normalize_embeddings=True):
        out = np.stack([
            self._vectors.get(t, self._fallback)
            for t in texts
        ]).astype(np.float32)
        return out


class _FakeChunkStore:
    def __init__(self, chunks: list[Chunk], files: list[FileRecord]):
        self._chunks = {c.id: c for c in chunks}
        self._files = {f.id: f for f in files}

    def list_projects(self):
        return (Project(name="p", root_path="/p"),)

    def search_lexical(self, terms, top_n, scope_filter):
        return ()

    def get_chunks_by_id(self, ids):
        return [self._chunks[i] for i in ids if i in self._chunks]

    def get_file_by_id(self, fid):
        return self._files.get(fid)

    def get_chunk_ids_by_embedding_rows(self, rows):
        # 1:1 identity: row i ↔ chunk.id i+1 (matches the constructor below)
        return {r: r + 1 for r in rows}

    def list_chunk_ids_in_scope(self, scope_filter):
        return tuple(self._chunks.keys())

    def get_embedding_rows_by_chunk_ids(self, chunk_ids):
        return tuple(sorted(cid - 1 for cid in chunk_ids))

    def chunk_count(self):
        return len(self._chunks)


class _FakeEmbedStore:
    """Brute-force cosine over a small in-memory matrix.

    ``search_dense`` returns Hits whose ``chunk_id`` field carries the
    *row index* of the matched embedding — matches the production
    contract of ``MemmapEmbedStore.search_dense``.
    """

    def __init__(self, matrix: np.ndarray):
        self._m = matrix  # [N, D] normalized

    def search_dense(self, query_emb, top_n, scope_rows):
        scores = self._m @ query_emb
        order = np.argsort(-scores)[:top_n]
        return tuple(Hit(chunk_id=int(i), score=float(scores[i])) for i in order)


def _build_searcher(vectors_for_chunks: list[np.ndarray],
                    embed_lookup: dict[str, np.ndarray]) -> Searcher:
    n = len(vectors_for_chunks)
    files = [
        FileRecord(id=i + 1, project="p", rel_path=f"f{i}.py",
                   mtime=0, size_bytes=10, content_hash=f"file-h{i}")
        for i in range(n)
    ]
    chunks = [
        Chunk(id=i + 1, file_id=i + 1, chunk_index=0, start_offset=0,
              end_offset=10, start_line=1, end_line=1, token_count=10,
              content_hash=f"h{i}", text=f"chunk-{i} text",
              embedder_model=EMBED_MODEL, embedder_sha256="x",
              embedding_row=i)
        for i in range(n)
    ]
    matrix = np.stack(vectors_for_chunks).astype(np.float32)
    return Searcher(
        chunk_store=_FakeChunkStore(chunks, files),
        embed_store=_FakeEmbedStore(matrix),
        embedder=_FakeEmbedder(embed_lookup),
        config=SearcherConfig(bm25_weight=0.0, relevance_floor=0.0),
    )


# ------------------------------------------------------------------ unit tests

def test_zero_weight_prior_context_is_identical_to_baseline():
    """Regression guard: weight=0 → behavior identical to baseline."""
    q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunk_vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
    ]
    s = _build_searcher(
        chunk_vecs,
        {"q": q_vec, "prior": np.array([0, 1, 0], dtype=np.float32)},
    )
    base = s.search("q", relevance_floor=0.0, max_results=3)
    iterative = s.search(
        "q",
        prior_context="prior",
        prior_context_weight=0.0,
        relevance_floor=0.0,
        max_results=3,
    )
    base_ids = tuple(c.chunk_id for c in base.chunks)
    iter_ids = tuple(c.chunk_id for c in iterative.chunks)
    assert base_ids == iter_ids


def test_prior_context_shifts_ranking_toward_prior_direction():
    """Mixing prior with weight > 0 must change the top-1 chunk
    toward the prior's semantic direction."""
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prior = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    chunk_vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),   # row 0: matches q
        np.array([0.0, 1.0, 0.0], dtype=np.float32),   # row 1: matches prior
        np.array([0.7, 0.7, 0.0], dtype=np.float32),
    ]
    s = _build_searcher(chunk_vecs, {"q": q, "prior": prior})
    no_prior = s.search("q", relevance_floor=0.0, max_results=3)
    heavy_prior = s.search(
        "q",
        prior_context="prior",
        prior_context_weight=0.9,
        relevance_floor=0.0,
        max_results=3,
    )
    assert no_prior.chunks[0].chunk_id == 1   # row 0 → chunk.id 1
    assert heavy_prior.chunks[0].chunk_id == 2   # row 1 → chunk.id 2


def test_suppress_ids_drops_named_chunks_from_results():
    """``suppress_ids`` must remove matching chunks before the budget take."""
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunk_vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.4, 0.0], dtype=np.float32),
        np.array([0.7, 0.7, 0.0], dtype=np.float32),
    ]
    s = _build_searcher(chunk_vecs, {"q": q})
    base = s.search("q", relevance_floor=0.0, max_results=3)
    top_ids = tuple(c.chunk_id for c in base.chunks)
    assert top_ids == (1, 2, 3)

    suppressed = s.search(
        "q",
        suppress_ids={1, 2},
        relevance_floor=0.0,
        max_results=3,
    )
    new_ids = tuple(c.chunk_id for c in suppressed.chunks)
    assert 1 not in new_ids and 2 not in new_ids
    assert new_ids == (3,)


def test_search_chunk_carries_chunk_id_for_round_tripping():
    """The whole iterative-retrieval loop relies on SearchChunk.chunk_id
    being populated so callers can pass the prior result back as
    ``suppress_ids``. Spot-check the field is non-None on real results."""
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunk_vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    s = _build_searcher(chunk_vecs, {"q": q})
    out = s.search("q", relevance_floor=0.0, max_results=1)
    assert out.chunks[0].chunk_id is not None
    assert isinstance(out.chunks[0].chunk_id, int)


def test_search_multi_accumulates_suppress_ids_across_groups():
    """Across sub-queries in one search_multi, a chunk surfaced by
    group N must not reappear in group N+1."""
    qa = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    qb = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # same direction → same hits
    chunk_vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.4, 0.0], dtype=np.float32),
        np.array([0.7, 0.7, 0.0], dtype=np.float32),
    ]
    s = _build_searcher(chunk_vecs, {"qa": qa, "qb": qb})
    multi = s.search_multi(
        ["qa", "qb"],
        relevance_floor=0.0,
        max_results=2,
    )
    a_ids = {c.chunk_id for c in multi.groups[0].chunks}
    b_ids = {c.chunk_id for c in multi.groups[1].chunks}
    assert a_ids and b_ids
    assert a_ids.isdisjoint(b_ids), (
        f"search_multi groups overlap: {a_ids & b_ids}"
    )


def test_empty_prior_context_string_is_noop():
    """Whitespace-only / empty prior_context string must not trigger
    the encode + mix path."""
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunk_vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    s = _build_searcher(chunk_vecs, {"q": q})
    base = s.search("q", relevance_floor=0.0, max_results=1)
    spaces = s.search(
        "q", prior_context="   ", prior_context_weight=0.5,
        relevance_floor=0.0, max_results=1,
    )
    empty_list = s.search(
        "q", prior_context=[], prior_context_weight=0.5,
        relevance_floor=0.0, max_results=1,
    )
    assert (base.chunks[0].chunk_id
            == spaces.chunks[0].chunk_id
            == empty_list.chunks[0].chunk_id)


# ------------------------------------------------------------------- live test

@pytest.mark.skipif(
    not (SWEZERO_LIVE_PACK / "chunks.sqlite").exists()
    or os.environ.get("LONGCTX_SKIP_LIVE_PACK") == "1",
    reason="swezero-12m live pack not present (set LONGCTX_SKIP_LIVE_PACK=1 to force skip)",
)
def test_iterative_retrieval_against_swezero_live_pack():
    """End-to-end smoke against a real swezero pack.

    Q1 retrieves baseline. Q2 = same query with Q1's chunk ids in
    suppress_ids. Must return zero overlap. Validates the full
    SentenceTransformer + memmap + sqlite pipeline; the API math is
    covered deterministically by the synthetic-store tests above so
    this exists to catch wiring breaks against real data, not to
    re-prove the algorithm.
    """
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    from longctx_daemon.storage.memmap_store import (
        MemmapEmbedStore,
        _compute_embedder_sha256,
    )
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    sha = _compute_embedder_sha256(embedder)
    chunk_store = SqliteChunkStore(SWEZERO_LIVE_PACK / "chunks.sqlite")
    embed_store = MemmapEmbedStore(
        SWEZERO_LIVE_PACK / "embeds",
        model_name=EMBED_MODEL,
        model_sha256=sha,
        dim=EMBED_DIM,
    )
    searcher = Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(
            dedup_by_doc_root=True,
            relevance_floor=0.0,
        ),
    )

    query = "fix a flaky test that intermittently fails on CI"
    first = searcher.search(query, max_results=5, max_tokens=100_000)
    first_ids = {c.chunk_id for c in first.chunks}
    # dedup_by_doc_root collapses near-duplicate trajectories so the top-N
    # in a narrow query can be small; assert >= 1 to keep the test from
    # being brittle to corpus content while still proving chunks came back.
    assert len(first_ids) >= 1, f"baseline returned nothing: {first.chunks}"
    assert all(cid is not None for cid in first_ids)

    second = searcher.search(
        query,
        suppress_ids=first_ids,
        max_results=5,
        max_tokens=100_000,
    )
    second_ids = {c.chunk_id for c in second.chunks}
    assert first_ids.isdisjoint(second_ids), (
        f"suppress_ids leaked: overlap={first_ids & second_ids}"
    )
