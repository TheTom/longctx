"""Coarse filter tests — BM25 + dense + RRF on synthetic corpora.

Mocks ``SentenceTransformer`` so tests run on CPU without downloading a
model. BM25 stays real (rank_bm25 is pure-Python and fast). The point is
to exercise the fusion + ranking logic, not the embedder math.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class _KeywordEmbedder:
    """Deterministic embedder: vector dims keyed by keyword presence.

    Keeps dense-stage scoring boring and predictable so we can assert
    on rank order. Identical encode calls return identical vectors so
    the disk cache stays well-defined under repeat runs.
    """

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               batch_size=32, **_):
        out = []
        for t in texts:
            tl = t.lower()
            vec = np.array([
                1.0 if "needle" in tl else 0.0,
                1.0 if "alpha" in tl else 0.0,
                1.0 if "beta" in tl else 0.0,
                0.5,  # baseline so vectors aren't zero
            ], dtype=np.float32)
            n = np.linalg.norm(vec) + 1e-8
            out.append(vec / n)
        return np.stack(out)


def _make_filter(cache_dir=None):
    """Build a CoarseFilter with the embedder mocked. Cache disabled by
    default so tests don't touch the user's ~/.cache."""
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=_KeywordEmbedder(),
    ):
        from longctx.rag.coarse_filter import CoarseFilter
        return CoarseFilter(cache_dir=cache_dir)


def _chunk(cid: str, text: str):
    from longctx.rag.coarse_filter import Chunk
    return Chunk(id=cid, text=text)


# --------------------------------------------------------------- imports

def test_imports():
    from longctx.rag.coarse_filter import Chunk, CoarseFilter, _rrf_score
    assert Chunk is not None
    assert CoarseFilter is not None
    # RRF is canonical: rank 1 with k=60 → 1/61
    assert _rrf_score(1, k=60) == pytest.approx(1.0 / 61)


# --------------------------------------------------------------- basics

def test_empty_input():
    f = _make_filter()
    assert f.filter([], "anything", top_k=10) == []


def test_single_chunk():
    f = _make_filter()
    chunks = [_chunk("c0", "the needle is here")]
    out = f.filter(chunks, "needle", top_k=10)
    assert len(out) == 1
    assert out[0][0].id == "c0"


def test_top_k_ge_n_returns_all_sorted():
    """When top_k >= N, callers still expect a fully-ranked list."""
    f = _make_filter()
    chunks = [
        _chunk("noise", "totally unrelated content"),
        _chunk("hit", "the needle is right here in this chunk"),
        _chunk("noise2", "more unrelated content"),
    ]
    out = f.filter(chunks, "needle", top_k=99)
    assert len(out) == 3
    assert out[0][0].id == "hit"


# ---------------------------------------------------------------- BM25

def test_bm25_ranking_finds_keyword_match():
    """Rank-1 BM25 result should be the chunk containing the query terms."""
    f = _make_filter()
    chunks = [
        _chunk(f"noise{i}", f"unrelated filler text {i}") for i in range(20)
    ]
    chunks.append(_chunk("planted", "the secret access code is 481729"))
    ranks = f._bm25_rankings(chunks, "secret access code")
    assert ranks["planted"] == 1


def test_bm25_only_mode_via_zero_dense_weight():
    """``dense_weight=0`` skips the dense stage entirely."""
    f = _make_filter()
    chunks = [
        _chunk("a", "alpha here"),
        _chunk("b", "beta here"),
        _chunk("c", "needle planted in this chunk"),
    ]
    out = f.filter(chunks, "needle", top_k=1, dense_weight=0.0)
    assert out[0][0].id == "c"


def test_dense_only_mode_via_zero_bm25_weight():
    """``bm25_weight=0`` skips the BM25 stage entirely."""
    f = _make_filter()
    chunks = [
        _chunk("a", "alpha goes here"),
        _chunk("b", "beta goes here"),
        _chunk("c", "needle goes here"),
    ]
    out = f.filter(chunks, "needle", top_k=1, bm25_weight=0.0)
    assert out[0][0].id == "c"


def test_both_weights_zero_raises():
    f = _make_filter()
    chunks = [_chunk("a", "x")]
    with pytest.raises(ValueError):
        f.filter(chunks, "x", top_k=1, bm25_weight=0.0, dense_weight=0.0)


# ----------------------------------------------------------------- RRF

def test_rrf_fuses_bm25_and_dense():
    """RRF should put a chunk top-1 even when only one source ranks it #1.

    Builds a corpus where BM25 ranks chunk A first (lexical match) but
    dense ranks chunk B first (semantic match). Without RRF the top-1
    flips with weighting. With equal-weight RRF we expect both A and B
    in the top-2 — neither is dropped.
    """
    f = _make_filter()
    chunks = [
        # Alpha-heavy — dense embedder returns high similarity
        _chunk("dense_hit", "alpha alpha alpha"),
        # Lexically rich — BM25 returns high similarity
        _chunk("bm25_hit", "needle special-keyword distinctive-phrase"),
        # Pure noise
        *[_chunk(f"noise{i}", "filler") for i in range(8)],
    ]
    out = f.filter(chunks, "needle special-keyword distinctive-phrase alpha", top_k=2)
    ids = {c.id for c, _ in out}
    assert "bm25_hit" in ids
    assert "dense_hit" in ids


def test_top_k_limits_output():
    f = _make_filter()
    chunks = [
        _chunk("planted", "needle is right here"),
        *[_chunk(f"n{i}", f"noise text {i}") for i in range(50)],
    ]
    out = f.filter(chunks, "needle", top_k=5)
    assert len(out) == 5
    assert out[0][0].id == "planted"


def test_results_descending_by_score():
    f = _make_filter()
    chunks = [_chunk(f"c{i}", f"text {i} alpha" if i == 3 else f"text {i}")
              for i in range(10)]
    chunks.append(_chunk("hit", "needle alpha both keywords"))
    out = f.filter(chunks, "needle alpha", top_k=5)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------- weight tilting

def test_high_bm25_weight_tilts_to_lexical():
    """Heavy BM25 weight should pull a chunk that has the rare query
    term to #1, even when the dense embedder prefers a different chunk.

    Construction:
      - ``alpha_only`` is the dense embedder's favourite (contains
        ``alpha``, which the keyword embedder weights heavily).
      - ``lex_hit`` lacks ``alpha`` but is the only chunk containing the
        rare query term ``zebraduck`` so BM25 ranks it #1.
      - Plus enough background noise so BM25 IDF actually discriminates.
    """
    f = _make_filter()
    chunks = [
        _chunk("alpha_only", "alpha alpha alpha"),
        _chunk("lex_hit", "zebraduck zebraduck zebraduck zebraduck zebraduck"),
        *[_chunk(f"noise{i}", f"plain filler text number {i}") for i in range(20)],
    ]
    # Query mentions only the BM25-distinctive term; without weight the
    # dense embedder still likes alpha_only (baseline 0.5 dim) more than
    # lex_hit (also baseline 0.5). With BM25 weighted up, lex_hit wins.
    out = f.filter(chunks, "zebraduck", top_k=1,
                   bm25_weight=10.0, dense_weight=0.1)
    assert out[0][0].id == "lex_hit"


# ------------------------------------------------------------ scale smoke

def test_multi_query_fusion_returns_sorted_pairs():
    """Filter with N paraphrases — needle still surfaces in top-K.

    Each paraphrase shares at least one discriminative term with the
    needle so all three rankings actually contribute signal. (If the
    paraphrases were pure noise, RRF would correctly drown out the
    one-good-query signal — that's a separate behavior.)
    """
    f = _make_filter()
    chunks = [_chunk(f"n{i}", f"unrelated filler content {i}")
              for i in range(50)]
    chunks.insert(20, _chunk("planted", "the secret needle is here"))
    queries = [
        "secret needle planted",
        "find the secret needle item",
        "where is the planted needle",
    ]
    out = f.filter_multi_query(chunks, queries, top_k=5)
    ids = [c.id for c, _ in out]
    assert "planted" in ids
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_multi_query_with_one_query_matches_single_filter():
    """``filter_multi_query([q])`` must produce the same ordering as
    ``filter(q)`` — multi-query is a strict generalization."""
    f = _make_filter()
    chunks = [
        _chunk("a", "alpha is here"),
        _chunk("b", "beta is here"),
        _chunk("c", "needle here"),
    ]
    a = f.filter(chunks, "needle", top_k=3)
    b = f.filter_multi_query(chunks, ["needle"], top_k=3)
    assert [c.id for c, _ in a] == [c.id for c, _ in b]


def test_multi_query_empty_queries_raises():
    f = _make_filter()
    chunks = [_chunk("a", "x")]
    with pytest.raises(ValueError):
        f.filter_multi_query(chunks, [], top_k=1)


def test_batch_size_threads_through_to_encode(tmp_path):
    """Custom ``batch_size`` arg should be forwarded to the chunk
    encode call. (Query encode goes through a different code path that
    doesn't take batch_size — only the chunk encode does.)"""
    calls: list[dict] = []

    class _RecordingEmbedder:
        def encode(self, texts, convert_to_numpy=True,
                   normalize_embeddings=True, batch_size=32, **_):
            calls.append({"n_texts": len(texts), "batch_size": batch_size})
            return np.zeros((len(texts), 4), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer",
               return_value=_RecordingEmbedder()):
        from longctx.rag.coarse_filter import CoarseFilter
        cf = CoarseFilter(cache_dir=None, batch_size=4)
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(3)]
    cf.filter(chunks, "query", top_k=2)
    # The chunk-encode call (n_texts == n_chunks) must use batch_size=4.
    chunk_calls = [c for c in calls if c["n_texts"] == 3]
    assert chunk_calls, f"no chunk encode found in calls={calls}"
    assert chunk_calls[0]["batch_size"] == 4


def test_max_seq_length_isolates_cache_subdirs(tmp_path):
    """A 512-cap run must not share cache entries with an 8192-cap run.
    Verified by checking that the EmbedCache subdir paths differ."""
    with patch("sentence_transformers.SentenceTransformer",
               return_value=_KeywordEmbedder()):
        from longctx.rag.coarse_filter import CoarseFilter
        cf_default = CoarseFilter(cache_dir=str(tmp_path),
                                  embedder_model="x/y")
        cf_capped = CoarseFilter(cache_dir=str(tmp_path),
                                 embedder_model="x/y",
                                 max_seq_length=512)
    assert cf_default._cache.dir != cf_capped._cache.dir
    assert "seq512" in str(cf_capped._cache.dir)


def test_synthetic_haystack_finds_needle_in_top_10():
    """Plant one needle in 200 noise chunks; coarse filter must surface
    it in top-10 with default weights. This is the cheapest possible
    end-to-end assertion that fusion actually works."""
    f = _make_filter()
    chunks = [_chunk(f"n{i}", f"unrelated filler content number {i}")
              for i in range(200)]
    chunks.insert(137, _chunk("needle_chunk",
                              "the secret needle is planted at position 137"))
    out = f.filter(chunks, "secret needle position", top_k=10)
    ids = [c.id for c, _ in out]
    assert "needle_chunk" in ids
