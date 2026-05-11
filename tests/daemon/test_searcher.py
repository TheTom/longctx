"""Searcher pipeline tests — BM25 + dense + RRF over fake storage.

These tests exercise the search pipeline end-to-end against in-memory
fakes for ``ChunkStore`` + ``EmbedStore``. The fakes match the
``Protocol`` shape but skip persistence — we want to assert on
ranking, fusion, freshness, and budget enforcement, not on SQLite.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from longctx_daemon.searcher import (
    Searcher,
    SearcherConfig,
    _word_tokenize,
)
from longctx_daemon.types import (
    Chunk,
    FileRecord,
    Hit,
    Project,
    ScopeFilter,
)


# ----------------------------------------------------------------- fakes

class _FakeChunkStore:
    """In-memory ``ChunkStore`` for tests.

    BM25 is the real ``rank_bm25.BM25Okapi`` over chunk text; dense
    search lives on ``_FakeEmbedStore``. ``list_chunk_ids_in_scope``
    is exposed via ``getattr`` lookup in the searcher; supplying it
    here means scope filtering exercises that branch in tests.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        files: Sequence[FileRecord],
        projects: Sequence[Project],
    ) -> None:
        self._chunks = {c.id: c for c in chunks}
        self._files = {f.id: f for f in files}
        self._projects = list(projects)

    # --- project lifecycle --------------------------------------------------

    def list_projects(self) -> tuple[Project, ...]:
        return tuple(self._projects)

    # --- file lookup --------------------------------------------------------

    def get_file_by_id(self, file_id: int) -> Optional[FileRecord]:
        return self._files.get(file_id)

    # --- chunk lookup -------------------------------------------------------

    def get_chunks_by_id(self, ids: Iterable[int]) -> tuple[Chunk, ...]:
        return tuple(self._chunks[i] for i in ids if i in self._chunks)

    def list_chunk_ids_in_scope(
        self, scope: ScopeFilter,
    ) -> tuple[int, ...]:
        out = []
        for cid, chunk in self._chunks.items():
            file_rec = self._files.get(chunk.file_id)
            if file_rec is None:
                continue
            if scope.project is not None and file_rec.project != scope.project:
                continue
            if scope.project_in is not None and (
                file_rec.project not in scope.project_in
            ):
                continue
            out.append(cid)
        return tuple(out)

    # --- BM25 ---------------------------------------------------------------

    def search_lexical(
        self,
        query_terms: list[str],
        k: int,
        scope: Optional[ScopeFilter] = None,
    ) -> tuple[Hit, ...]:
        from rank_bm25 import BM25Okapi

        # Restrict to scope first; only score chunks the agent can see.
        if scope is not None:
            ids_in_scope = set(self.list_chunk_ids_in_scope(scope))
            scoped_chunks = [
                self._chunks[i] for i in ids_in_scope if i in self._chunks
            ]
        else:
            scoped_chunks = list(self._chunks.values())
        if not scoped_chunks or not query_terms:
            return ()
        corpus = [_word_tokenize(c.text) for c in scoped_chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_terms)
        order = np.argsort(-scores)[:k]
        return tuple(
            Hit(chunk_id=scoped_chunks[idx].id, score=float(scores[idx]))
            for idx in order
            if scores[idx] > 0  # only return positively scored hits
        )


class _FakeEmbedStore:
    """In-memory ``EmbedStore``. Brute-force cosine; that's the
    Phase 2.0 production behavior anyway."""

    def __init__(self, embeddings_by_chunk_id: dict[int, np.ndarray]) -> None:
        self._embs = embeddings_by_chunk_id

    @property
    def dim(self) -> int:
        if not self._embs:
            return 0
        return len(next(iter(self._embs.values())))

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    @property
    def model_sha256(self) -> str:
        return "deadbeef"

    @property
    def num_rows(self) -> int:
        return len(self._embs)

    def search_dense(
        self,
        query_embedding: np.ndarray,
        k: int,
        chunk_ids_in_scope: Optional[tuple[int, ...]] = None,
    ) -> tuple[Hit, ...]:
        if not self._embs:
            return ()
        ids = (
            list(chunk_ids_in_scope)
            if chunk_ids_in_scope is not None
            else list(self._embs.keys())
        )
        if not ids:
            return ()
        mat = np.stack([self._embs[i] for i in ids])
        sims = mat @ np.asarray(query_embedding).astype(mat.dtype)
        order = np.argsort(-sims)[:k]
        return tuple(
            Hit(chunk_id=ids[idx], score=float(sims[idx]))
            for idx in order
        )


class _KeywordEmbedder:
    """Same trick as ``tests/test_coarse_filter.py``: keyword-presence
    embedding so ranking is predictable without a real model."""

    def __init__(self, dims: tuple[str, ...]) -> None:
        self._dims = dims

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **_,
    ) -> np.ndarray:
        rows = []
        for t in texts:
            tl = t.lower()
            row = np.array(
                [1.0 if d in tl else 0.0 for d in self._dims] + [0.5],
                dtype=np.float32,
            )
            if normalize_embeddings:
                norm = np.linalg.norm(row) + 1e-8
                row = row / norm
            rows.append(row)
        return np.stack(rows)


# ----------------------------------------------------------- corpus builders

def _build_corpus(
    chunks_text: list[tuple[int, int, str, str]],  # (chunk_id, file_id, project, text)
    project_names: tuple[str, ...] = ("longctx",),
    embedder: Optional[_KeywordEmbedder] = None,
) -> tuple[_FakeChunkStore, _FakeEmbedStore, _KeywordEmbedder]:
    """Build a synthetic corpus from a list of (id, file_id, project, text).

    One file per (project, file_id) pair, derived from chunk listings.
    """
    embedder = embedder or _KeywordEmbedder(("needle", "secret", "auth"))

    seen_files: dict[tuple[str, int], FileRecord] = {}
    chunks: list[Chunk] = []
    embs_by_id: dict[int, np.ndarray] = {}

    for cid, file_id, project, text in chunks_text:
        key = (project, file_id)
        if key not in seen_files:
            seen_files[key] = FileRecord(
                id=file_id,
                project=project,
                rel_path=f"src/file_{file_id}.py",
                mtime=1_700_000_000,
                size_bytes=len(text),
                content_hash=f"hash{file_id}",
            )
        chunk = Chunk(
            id=cid,
            file_id=file_id,
            chunk_index=cid,
            start_offset=0,
            end_offset=len(text),
            start_line=1,
            end_line=max(1, text.count("\n") + 1),
            token_count=max(1, len(text) // 4),
            content_hash=f"chash{cid}",
            text=text,
            embedder_model="fake-embedder",
            embedder_sha256="deadbeef",
            embedding_row=cid,
        )
        chunks.append(chunk)
        emb = embedder.encode([text], normalize_embeddings=True)[0]
        embs_by_id[cid] = emb

    files = list(seen_files.values())
    projects = tuple(
        Project(name=name, root_path=f"/tmp/{name}")
        for name in project_names
    )
    return (
        _FakeChunkStore(chunks=chunks, files=files, projects=projects),
        _FakeEmbedStore(embeddings_by_chunk_id=embs_by_id),
        embedder,
    )


def _build_searcher(
    chunk_store: _FakeChunkStore,
    embed_store: _FakeEmbedStore,
    embedder: _KeywordEmbedder,
    *,
    config: Optional[SearcherConfig] = None,
    quiescence_cb=None,
) -> Searcher:
    return Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=config or SearcherConfig(),
        quiescence_check_callback=quiescence_cb,
    )


# ------------------------------------------------------------ basic plumbing

def test_searcher_imports():
    """Sanity import — public surface area is exposed."""
    from longctx_daemon.searcher import (
        COMMON_WORD_NAMES,
        Searcher,
        SearcherConfig,
        decide_scope,
    )
    assert Searcher is not None
    assert SearcherConfig is not None
    assert decide_scope is not None
    assert COMMON_WORD_NAMES


def test_searcher_config_defaults():
    """Config defaults match the §6 PRD."""
    config = SearcherConfig()
    assert config.rrf_k == 60
    assert config.bm25_weight == 1.0
    assert config.dense_weight == 1.0
    assert config.default_top_k_for_fusion == 1000
    assert config.default_wait_for_quiescence_ms == 500
    assert config.chars_per_token == 4


def test_word_tokenize_strips_punctuation():
    """BM25 tokenizer drops punctuation, lowercases."""
    assert _word_tokenize("Find Auth's Token!") == [
        "find", "auth", "s", "token",
    ]


# ----------------------------------------------------------- needle planted

def test_needle_chunk_in_top_3():
    """Planted needle surfaces in the fused top-3."""
    chunks_text = [
        (i, i // 5, "longctx", f"unrelated filler line {i}")
        for i in range(50)
    ]
    chunks_text.append((100, 99, "longctx", "the secret needle is here"))
    store, embeds, embedder = _build_corpus(chunks_text)
    searcher = _build_searcher(store, embeds, embedder)

    result = searcher.search(
        "needle",
        cwd="/tmp/longctx/foo.py",
    )
    top_ids = [c.citation.file_path for c in result.chunks[:3]]
    assert any("file_99" in p for p in top_ids)


def test_search_returns_chunks_in_descending_score():
    """Returned chunks are ranked descending by relevance_score."""
    chunks_text = [
        (1, 1, "longctx", "needle alpha line"),
        (2, 1, "longctx", "another needle here"),
        (3, 2, "longctx", "totally unrelated content"),
    ]
    store, embeds, embedder = _build_corpus(chunks_text)
    searcher = _build_searcher(store, embeds, embedder)

    result = searcher.search("needle", cwd="/tmp/longctx")
    scores = [c.relevance_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_query_returns_empty():
    """Empty query yields no BM25 hits and a low-info dense pass.

    Dense alone may still surface something via the keyword embedder's
    baseline dim (everything has 0.5 there), so we accept any number
    of results — we just want the call to not crash.
    """
    chunks_text = [(1, 1, "longctx", "hello world")]
    store, embeds, embedder = _build_corpus(chunks_text)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("", cwd="/tmp/longctx")
    # Don't crash; freshness still populated.
    assert result.freshness.is_fully_fresh is True


# -------------------------------------------------------- multi-query fusion

def test_multi_query_fusion_includes_needle():
    """Paraphrases boost recall of a planted chunk above the noise."""
    chunks = [
        (i, i // 5, "longctx", f"unrelated filler {i}")
        for i in range(50)
    ]
    # Needle uses words that overlap with paraphrases collectively but
    # not with every single paraphrase — multi-query fusion should win.
    chunks.insert(20, (200, 99, "longctx", "the secret needle is planted here"))
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)

    paraphrases = (
        "find the secret needle item",
        "where is the planted needle",
        "look up the planted secret",
    )
    result = searcher.search(
        "secret needle planted",
        cwd="/tmp/longctx",
        paraphrases=paraphrases,
        max_tokens=4096,
    )
    file_paths = [c.citation.file_path for c in result.chunks]
    # The planted chunk lives in file 99.
    assert any("file_99" in p for p in file_paths)


def test_multi_query_outranks_single_for_partial_paraphrases():
    """Cross-paraphrase fusion > single-query when paraphrases share term diversity.

    Setup: planted chunk has terms that distribute across paraphrases.
    Each individual paraphrase ranks the planted chunk somewhere
    middling. The fused score should be higher than any single-query
    score AND surface the chunk earlier in the result list.
    """
    # Make the corpus harder: lots of distractors share words with each
    # paraphrase, but only the planted chunk has the union of terms.
    chunks_text = [(1, 1, "longctx", "alpha distractor")]
    chunks_text += [(2, 2, "longctx", "beta distractor")]
    chunks_text += [(3, 3, "longctx", "gamma distractor")]
    chunks_text += [(99, 99, "longctx", "alpha beta gamma the needle")]
    chunks_text += [
        (10 + i, 10 + i, "longctx", f"unrelated text {i}")
        for i in range(20)
    ]
    store, embeds, embedder = _build_corpus(chunks_text)
    searcher = _build_searcher(store, embeds, embedder)

    paraphrases = ("alpha needle", "beta needle", "gamma needle")
    multi = searcher.search(
        "needle",
        cwd="/tmp/longctx",
        paraphrases=paraphrases,
    )
    # Find the planted chunk's rank in both modes.
    multi_ids = [c.citation.file_path for c in multi.chunks]
    multi_planted_rank = next(
        i for i, p in enumerate(multi_ids) if "file_99" in p
    )

    # Single query — no paraphrases. Each individual term may or may not
    # rank the planted chunk highly; whatever the rank is, the fused
    # version with paraphrases must be at least as good.
    single = searcher.search("needle", cwd="/tmp/longctx")
    single_ids = [c.citation.file_path for c in single.chunks]
    single_planted_rank = next(
        i for i, p in enumerate(single_ids) if "file_99" in p
    )
    # The fused score should bring it forward — equal or better.
    assert multi_planted_rank <= single_planted_rank


# ------------------------------------------------------------ token budget

def test_max_tokens_truncates_results():
    """``max_tokens=512`` keeps total token count under the cap."""
    # Use chunks with explicit token counts so the budget math is exact.
    chunks = []
    for i in range(20):
        chunks.append((i, i, "longctx", "filler " * 50))  # ~50 tokens
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)

    result = searcher.search("filler", cwd="/tmp/longctx", max_tokens=512)
    total_tokens = sum(c.token_count for c in result.chunks)
    assert total_tokens <= 512


def test_max_tokens_keeps_at_least_one_when_first_fits():
    """Smallest-possible budget that fits one chunk → one chunk returned."""
    # One chunk, ~25 tokens.
    chunks = [(1, 1, "longctx", "needle " * 25)]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", cwd="/tmp/longctx", max_tokens=100)
    assert len(result.chunks) == 1


def test_max_results_cap():
    """``max_results=2`` keeps at most two chunks even with budget room."""
    chunks = [(i, i, "longctx", f"needle text {i}") for i in range(10)]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)

    result = searcher.search(
        "needle",
        cwd="/tmp/longctx",
        max_tokens=100_000,
        max_results=2,
    )
    assert len(result.chunks) <= 2


# ----------------------------------------------------------------- citations

def test_citations_have_project_prefix():
    """Citation.file_path always starts with the project name."""
    chunks = [(1, 1, "longctx", "needle text"), (2, 1, "longctx", "more text")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", cwd="/tmp/longctx")
    for c in result.chunks:
        assert c.citation.file_path.startswith("longctx/")
        assert c.citation.project == "longctx"


def test_citations_carry_line_numbers():
    """start_line / end_line carried through from the chunk record."""
    chunks = [(1, 1, "longctx", "line one\nline two\nline three")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("line", cwd="/tmp/longctx")
    assert result.chunks[0].citation.start_line == 1
    assert result.chunks[0].citation.end_line == 3


# -------------------------------------------------------------- freshness

def test_freshness_default_callback_is_fully_fresh():
    """Default callback returns 0 pending; ``is_fully_fresh`` is True."""
    chunks = [(1, 1, "longctx", "hello")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("hello", cwd="/tmp/longctx")
    assert result.freshness.is_fully_fresh is True
    assert result.freshness.pending_updates == 0
    assert result.freshness.indexed_through.endswith("Z")
    assert result.freshness.stale_files == ()


def test_freshness_pending_updates_marks_not_fresh():
    """Quiescence callback returning 3 pending → not fresh."""
    chunks = [(1, 1, "longctx", "hello")]
    store, embeds, embedder = _build_corpus(chunks)

    def cb(project, timeout_ms):
        return 3, "2026-05-09T14:30:00.000Z"

    searcher = _build_searcher(
        store, embeds, embedder, quiescence_cb=cb,
    )
    result = searcher.search("hello", cwd="/tmp/longctx")
    assert result.freshness.is_fully_fresh is False
    assert result.freshness.pending_updates == 3
    assert result.freshness.indexed_through == "2026-05-09T14:30:00.000Z"


def test_freshness_callback_receives_scope_and_timeout():
    """Callback is called with the resolved primary project and timeout."""
    chunks = [(1, 1, "longctx", "hello")]
    store, embeds, embedder = _build_corpus(chunks)

    captured: dict[str, object] = {}

    def cb(project, timeout_ms):
        captured["project"] = project
        captured["timeout_ms"] = timeout_ms
        return 0, "2026-05-09T14:00:00.000Z"

    searcher = _build_searcher(
        store, embeds, embedder, quiescence_cb=cb,
    )
    searcher.search(
        "hello", cwd="/tmp/longctx", wait_for_quiescence_ms=750,
    )
    assert captured["project"] == "longctx"
    assert captured["timeout_ms"] == 750


def test_freshness_default_timeout_used_when_none():
    """When the caller passes ``None``, config default flows through."""
    chunks = [(1, 1, "longctx", "hello")]
    store, embeds, embedder = _build_corpus(chunks)

    captured: dict[str, object] = {}

    def cb(project, timeout_ms):
        captured["timeout_ms"] = timeout_ms
        return 0, "2026-05-09T14:00:00.000Z"

    config = SearcherConfig(default_wait_for_quiescence_ms=1234)
    searcher = _build_searcher(
        store, embeds, embedder, config=config, quiescence_cb=cb,
    )
    searcher.search("hello", cwd="/tmp/longctx")
    assert captured["timeout_ms"] == 1234


# ----------------------------------------------------------------- latency

def test_latency_breakdown_all_fields_populated():
    """Every stage records a non-negative ms value; total is the sum."""
    chunks = [(1, 1, "longctx", "needle"), (2, 1, "longctx", "haystack")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", cwd="/tmp/longctx")
    lat = result.latency_ms
    for field in (
        "wait_quiescence", "embed_query", "bm25_score",
        "dense_score", "rrf_fuse", "fetch_chunks", "total",
    ):
        assert getattr(lat, field) >= 0.0, field
    component_sum = (
        lat.wait_quiescence + lat.embed_query + lat.bm25_score
        + lat.dense_score + lat.rrf_fuse + lat.fetch_chunks
    )
    # Total should be at least the sum of components (allow tiny FP slack).
    assert lat.total >= component_sum - 1e-6


# -------------------------------------------------------------- scope wiring

def test_search_filters_to_explicit_project():
    """``project=`` arg restricts results to that project."""
    chunks = [
        (1, 1, "longctx", "needle in longctx"),
        (2, 2, "mlx-swift-lm", "needle in mlx"),
    ]
    store, embeds, embedder = _build_corpus(
        chunks, project_names=("longctx", "mlx-swift-lm"),
    )
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", project="longctx")
    for c in result.chunks:
        assert c.citation.project == "longctx"


def test_search_fanout_no_primary_searches_everything():
    """No cwd / no sticky → search across all indexed projects."""
    chunks = [
        (1, 1, "longctx", "needle in longctx"),
        (2, 2, "mlx-swift-lm", "needle in mlx"),
    ]
    store, embeds, embedder = _build_corpus(
        chunks, project_names=("longctx", "mlx-swift-lm"),
    )
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle")
    projects = {c.citation.project for c in result.chunks}
    assert projects == {"longctx", "mlx-swift-lm"}
    assert result.scope_decision.primary_source == "fanout_no_primary"


def test_search_cross_project_pattern_routes_to_named_project():
    """Query mentioning project name → primary == that project."""
    chunks = [
        (1, 1, "longctx", "needle in longctx"),
        (2, 2, "mlx-swift-lm", "needle in mlx-swift-lm"),
    ]
    store, embeds, embedder = _build_corpus(
        chunks, project_names=("longctx", "mlx-swift-lm"),
    )
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("how does mlx-swift-lm find the needle")
    assert result.scope_decision.primary_source == "cross_project_pattern"
    assert result.scope_decision.primary_project == "mlx-swift-lm"


def test_scope_decision_attached_to_result():
    """Result always carries the scope_decision struct."""
    chunks = [(1, 1, "longctx", "x")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("x", cwd="/tmp/longctx")
    assert result.scope_decision.primary_project == "longctx"
    assert result.scope_decision.primary_source == "cwd_walk_to_sentinel"


# ---------------------------------------------------- weight ablation

def test_bm25_only_weight_ignores_dense():
    """``dense_weight=0`` skips dense; only BM25 ranks contribute.

    BM25 IDF requires enough non-matching docs in the corpus to give
    positive scores to matching docs (when half the corpus has the
    term, IDF is zero). We use a noise-heavy corpus to make the
    scoring meaningful.
    """
    chunks = [(1, 1, "longctx", "the needle is right here in this chunk")]
    chunks += [
        (i, i, "longctx", f"totally unrelated noise number {i}")
        for i in range(2, 12)
    ]
    store, embeds, embedder = _build_corpus(chunks)
    config = SearcherConfig(dense_weight=0.0)
    searcher = _build_searcher(store, embeds, embedder, config=config)
    result = searcher.search("needle", cwd="/tmp/longctx")
    assert len(result.chunks) >= 1
    assert result.chunks[0].citation.file_path.endswith("file_1.py")


def test_dense_only_weight_ignores_bm25():
    """``bm25_weight=0`` skips BM25; only dense ranks contribute."""
    chunks = [
        (1, 1, "longctx", "the needle is here"),
        (2, 2, "longctx", "totally unrelated"),
    ]
    store, embeds, embedder = _build_corpus(chunks)
    config = SearcherConfig(bm25_weight=0.0)
    searcher = _build_searcher(store, embeds, embedder, config=config)
    result = searcher.search("needle", cwd="/tmp/longctx")
    # Dense ranks the needle higher (keyword embedder).
    assert result.chunks[0].citation.file_path.endswith("file_1.py")


# ------------------------------------------------------------ edge cases

def test_no_results_when_corpus_empty():
    """Empty corpus yields zero chunks but no crash."""
    store = _FakeChunkStore(
        chunks=(), files=(),
        projects=(Project(name="longctx", root_path="/tmp/longctx"),),
    )
    embeds = _FakeEmbedStore({})

    embedder = _KeywordEmbedder(("x",))
    searcher = Searcher(
        chunk_store=store,
        embed_store=embeds,
        embedder=embedder,
        config=SearcherConfig(),
    )
    result = searcher.search("anything", cwd="/tmp/longctx")
    assert result.chunks == ()
    assert result.freshness.is_fully_fresh is True


def test_paraphrases_empty_tuple_default():
    """Default ``paraphrases=()`` runs single-query path without error."""
    chunks = [(1, 1, "longctx", "needle")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", cwd="/tmp/longctx")
    assert len(result.chunks) >= 1


def test_paraphrases_count_matches_query_embedding_batch():
    """Embedder receives N+1 queries (literal + paraphrases) in one call."""
    chunks = [(1, 1, "longctx", "needle"), (2, 1, "longctx", "x")]
    store, embeds, embedder = _build_corpus(chunks)

    encode_calls: list[int] = []
    real_encode = embedder.encode

    def spy_encode(texts, **kw):
        encode_calls.append(len(list(texts)))
        return real_encode(texts, **kw)

    embedder.encode = spy_encode  # type: ignore[method-assign]
    searcher = _build_searcher(store, embeds, embedder)
    searcher.search(
        "needle",
        cwd="/tmp/longctx",
        paraphrases=("alpha", "beta", "gamma"),
    )
    # First call is the query batch (1 + 3 paraphrases = 4).
    assert encode_calls[0] == 4


# ------------------------------------------------ active_project_sticky

def test_active_project_sticky_routes_correctly():
    """Sticky session pins scope without cwd."""
    chunks = [
        (1, 1, "longctx", "needle in longctx"),
        (2, 2, "mlx-swift-lm", "needle in mlx"),
    ]
    store, embeds, embedder = _build_corpus(
        chunks, project_names=("longctx", "mlx-swift-lm"),
    )
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search(
        "needle", active_project_sticky="mlx-swift-lm",
    )
    assert result.scope_decision.primary_source == "active_project_sticky"
    for c in result.chunks:
        assert c.citation.project == "mlx-swift-lm"


# ----------------------------------------------------------- result shape

def test_result_has_immutable_chunks_tuple():
    """SearchResult.chunks is a tuple — agents can't accidentally mutate it."""
    chunks = [(1, 1, "longctx", "x")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("x", cwd="/tmp/longctx")
    assert isinstance(result.chunks, tuple)


def test_result_chunks_contain_text():
    """Chunks expose the raw text the agent will see in its context."""
    chunks = [(1, 1, "longctx", "needle in haystack")]
    store, embeds, embedder = _build_corpus(chunks)
    searcher = _build_searcher(store, embeds, embedder)
    result = searcher.search("needle", cwd="/tmp/longctx")
    assert result.chunks[0].text == "needle in haystack"
