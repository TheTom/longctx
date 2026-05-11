"""Searcher — BM25 + dense + RRF over the persistent storage layer.

Phase 2.0 component. Reads chunks from a ``ChunkStore`` (lexical) and
embeddings from an ``EmbedStore`` (dense), fuses rankings with
reciprocal-rank-fusion, and returns a ``SearchResult`` whose freshness
fields tell the agent whether what it just saw is current.

The algorithmic core is the same as ``longctx.rag.coarse_filter`` —
multi-query RRF over BM25 + dense — but the storage is persistent and
the result type carries citations + freshness signals defined in
``longctx_daemon.types`` rather than the in-memory pipeline shape.

By design we DO NOT import ``longctx.rag.coarse_filter`` here. Phase 2
keeps the daemon package free of cross-package dependencies on internal
helpers so the daemon can be split out later without dragging the
research-pipeline surface area along. The few helpers we share
(``_word_tokenize``, ``_rrf_score``) are short enough to copy.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np

from longctx_daemon.storage.protocol import ChunkStore, EmbedStore
from longctx_daemon.types import (
    Citation,
    Hit,
    LatencyBreakdown,
    ScopeDecision,
    ScopeFilter,
    SearchChunk,
    SearchFreshness,
    SearchResult,
)


# ----------------------------------------------------------------- constants

# Project names that collide with common English words used in code-search
# queries. These require a syntactic cue ("in <name>", "the <name> repo",
# etc.) to be treated as a project mention rather than a content word.
# Sourced from the §3.5 spec table.
COMMON_WORD_NAMES: frozenset[str] = frozenset({
    "auth", "core", "lib", "api", "client", "server", "tools", "utils",
    "platform",
})


# Syntactic cue patterns. ``{name}`` is interpolated with the regex-escaped
# project name. Cases are case-insensitive at match time. The patterns are
# chosen so a bare mention of a name (no surrounding cue) only fires for
# distinctive names, never common-word names.
_CUE_TEMPLATES: tuple[str, ...] = (
    r"\bin\s+{name}\b",
    r"\b{name}\s+repo\b",
    r"\b{name}\s+module\b",
    r"\b{name}\s+project\b",
    r"\b{name}'s\b",
    r"\bthe\s+{name}\b",
)


# ------------------------------------------------------------------- helpers

def _word_tokenize(text: str) -> list[str]:
    """Word-level tokenizer used for BM25 query expansion.

    The persistent BM25 index inside the ``ChunkStore`` does its own
    chunk tokenization at write time; this helper only tokenizes the
    incoming query so we hand a list-of-tokens (not a raw string) to
    ``search_lexical``. Same regex as ``coarse_filter._word_tokenize``
    so the term distribution lines up across the two layers.
    """
    return re.findall(r"\w+", text.lower())


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal-rank-fusion score for one ranking source.

    Standard formulation from Cormack, Clarke, Buettcher 2009:
        score(d) = sum over rankings r of  1 / (k + r(d))
    Rank is 1-indexed; rank 1 is the best document.
    """
    return 1.0 / (k + rank)


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing ``Z`` (per spec §6.3)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _default_quiescence_callback() -> tuple[int, str]:
    """Default quiescence stub. Returns (pending_updates=0, indexed_through=now).

    The watcher integration in Phase 2.2 swaps this for the real queue
    drain. For Phase 2.0 we return zero pending updates so tests that
    don't care about freshness can still assert ``is_fully_fresh=True``.
    """
    return 0, _utc_now_iso()


# ------------------------------------------------------------------- config

@dataclass(frozen=True)
class SearcherConfig:
    """Tunables for the searcher. All values match the §3 + §6 PRD defaults.

    Attributes:
        rrf_k: dampening constant in 1/(k+rank). 60 is canonical.
        bm25_weight: scalar multiplier on every BM25 RRF contribution.
            Set to 0 to skip the lexical stage entirely (debug only).
        dense_weight: same, for the dense stage.
        default_top_k_for_fusion: how many hits each ranker returns
            BEFORE fusion. Larger = more recall but more fusion cost.
            Trimmed to ``max_results`` (or ``max_tokens``) AFTER fusion.
        default_wait_for_quiescence_ms: timeout cap for the quiescence
            wait stage. Phase 2.0 doesn't actually wait; the value is
            forwarded to the quiescence callback for accounting.
        chars_per_token: rough heuristic used as a fallback when a chunk
            has no recorded ``token_count``. ~4 chars/token matches
            sentencepiece-style BPE on English code-and-prose.
    """
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    default_top_k_for_fusion: int = 1000
    default_wait_for_quiescence_ms: int = 500
    chars_per_token: int = 4


# ------------------------------------------------------------- scope routing

def _build_cue_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    """Compile the §3.5 syntactic cue patterns for one project name."""
    escaped = re.escape(name)
    return tuple(
        re.compile(t.format(name=escaped), re.IGNORECASE)
        for t in _CUE_TEMPLATES
    )


def _project_mentioned(query: str, name: str) -> bool:
    """True iff ``query`` contains ``name`` as a bare token (case-insensitive).

    Bare mention is sufficient for distinctive project names; common-word
    names need ``_project_mentioned_with_cue``.

    Boundary rules: word boundaries are stricter than the default
    ``\\b`` semantics — surrounding characters must be whitespace,
    start/end of string, or punctuation. Critically, ``-`` is NOT a
    boundary, so ``longctx-other`` does NOT match project ``longctx``
    (hyphen is treated as part of an identifier, not a word break)
    while ``the longctx repo`` does.
    """
    # Build a custom-boundary regex: outside-of-name chars must be whitespace
    # or one of the listed punctuation. Anything word-like (incl. hyphens
    # and underscores) is rejected so we don't substring-match into a
    # longer identifier.
    boundary = r"(?:^|(?<=[\s.,;:!?()\[\]{}\"'`]))"
    boundary_end = r"(?=$|[\s.,;:!?()\[\]{}\"'`])"
    pattern = re.compile(
        boundary + re.escape(name) + boundary_end, re.IGNORECASE,
    )
    return pattern.search(query) is not None


def _project_mentioned_with_cue(query: str, name: str) -> bool:
    """True iff ``query`` mentions ``name`` adjacent to a syntactic cue.

    Used for common-word project names where a bare match would mistake
    a content word for a project reference (the canonical case from the
    spec: "where do I handle auth" should NOT route to project ``auth``).
    """
    return any(p.search(query) is not None for p in _build_cue_patterns(name))


def decide_scope(
    query: str,
    cwd: Optional[str],
    project: Optional[str],
    active_project_sticky: Optional[str],
    projects_in_index: Sequence[str],
    *,
    project_roots: Optional[dict[str, str]] = None,
) -> ScopeDecision:
    """Pick the search scope using the §3.5 + §3.3 routing rules.

    Priority order (highest first):
        1. ``project=`` argument set         → explicit_project
        2. Cross-project name in the query   → cross_project_pattern
        3. ``active_project_sticky`` set     → active_project_sticky
        4. ``cwd`` falls under a project     → cwd_walk_to_sentinel
        5. Nothing                           → fanout_no_primary

    The cross-project step honors the §3.5 tiebreakers: distinctive
    names match on bare mention; common-word names need a syntactic
    cue. Multiple matches → fanout (e.g. "compare auth in longctx vs
    mlx-swift-lm" matches both projects).

    Args:
        query: user-supplied search string.
        cwd: working directory of the calling agent; used for the
            walk-up-to-sentinel step when nothing more specific applies.
        project: explicit project name supplied by the caller; wins
            unconditionally when set.
        active_project_sticky: project pinned via ``set_active_project``
            for the current session.
        projects_in_index: every project the daemon currently has
            indexed. Cross-project matches must intersect this set.
        project_roots: optional ``{name: root_path}`` map used by the
            cwd walk. When omitted, cwd routing falls back to the
            project-name-substring heuristic that's good enough for
            the Phase 2.0 single-process tests; production wiring
            passes the real map from ``ChunkStore.list_projects``.

    Returns:
        A ``ScopeDecision`` with ``primary_source`` reflecting the
        rule that fired.
    """
    indexed = tuple(projects_in_index)

    # Tier 1 — explicit project argument
    if project is not None:
        return ScopeDecision(
            primary_project=project,
            primary_source="explicit_project",
            fanout_projects=(project,),
            cross_project_pattern_matched=None,
            active_project_sticky=active_project_sticky,
        )

    # Tier 2 — cross-project name in the query (handle BEFORE sticky/cwd)
    matched: list[str] = []
    for name in indexed:
        if name.lower() in COMMON_WORD_NAMES:
            # Common word — require a syntactic cue.
            if _project_mentioned_with_cue(query, name):
                matched.append(name)
        else:
            # Distinctive name — bare mention is enough; cue still wins
            # too (covers "in mlx-swift-lm" alongside "mlx-swift-lm").
            if (
                _project_mentioned(query, name)
                or _project_mentioned_with_cue(query, name)
            ):
                matched.append(name)
    if matched:
        primary = matched[0]
        return ScopeDecision(
            primary_project=primary,
            primary_source="cross_project_pattern",
            fanout_projects=tuple(matched),
            cross_project_pattern_matched=primary,
            active_project_sticky=active_project_sticky,
        )

    # Tier 3 — sticky session
    if active_project_sticky is not None:
        return ScopeDecision(
            primary_project=active_project_sticky,
            primary_source="active_project_sticky",
            fanout_projects=(active_project_sticky,),
            cross_project_pattern_matched=None,
            active_project_sticky=active_project_sticky,
        )

    # Tier 4 — cwd walk-up
    if cwd is not None:
        walked = _walk_to_project(cwd, indexed, project_roots)
        if walked is not None:
            return ScopeDecision(
                primary_project=walked,
                primary_source="cwd_walk_to_sentinel",
                fanout_projects=(walked,),
                cross_project_pattern_matched=None,
                active_project_sticky=active_project_sticky,
            )

    # Tier 5 — nothing
    return ScopeDecision(
        primary_project=None,
        primary_source="fanout_no_primary",
        fanout_projects=indexed,
        cross_project_pattern_matched=None,
        active_project_sticky=active_project_sticky,
    )


def _walk_to_project(
    cwd: str,
    indexed: Sequence[str],
    project_roots: Optional[dict[str, str]],
) -> Optional[str]:
    """Find the indexed project that owns ``cwd``.

    Two strategies, tried in order:

    1. Real walk: if ``project_roots`` was supplied, find the project
       whose ``root_path`` is a prefix of ``cwd``. Among multiple
       matches the deepest root wins (handles nested workspaces).
    2. Substring fallback: find any indexed project whose name appears
       as a path segment of ``cwd``. Matches "/Users/tom/dev/longctx/x"
       to project ``longctx``. Good enough for the Phase 2.0 tests
       where we don't have real on-disk roots.

    Returns ``None`` when no indexed project covers ``cwd`` — the
    caller then falls through to ``fanout_no_primary``.
    """
    # Strategy 1 — real walk against project_roots.
    if project_roots:
        best_name: Optional[str] = None
        best_len = -1
        for name, root in project_roots.items():
            if name not in indexed:
                continue
            # Normalize trailing separators on root so "/dev/longctx" and
            # "/dev/longctx/" behave identically.
            root_norm = root.rstrip("/")
            if cwd == root_norm or cwd.startswith(root_norm + "/"):
                if len(root_norm) > best_len:
                    best_name = name
                    best_len = len(root_norm)
        if best_name is not None:
            return best_name

    # Strategy 2 — substring fallback. Splits cwd on "/" so
    # "/Users/tom/dev/longctx-other/foo" doesn't match project "longctx".
    parts = [p for p in cwd.split("/") if p]
    parts_set = set(parts)
    for name in indexed:
        if name in parts_set:
            return name
    return None


# ------------------------------------------------------------------ pipeline


class Searcher:
    """BM25 + dense + RRF searcher over the persistent storage layer.

    Builds zero in-memory indexes itself — both rankers are delegated
    to the storage backend. The searcher's job is:

      1. Decide which projects to search (§3.5 routing).
      2. Embed the query (and any caller-supplied paraphrases).
      3. Fan out BM25 + dense lookups across the union of queries.
      4. RRF-fuse the rankings (BM25 across queries + dense across
         queries, both weighted).
      5. Materialize the top-K chunks via ``get_chunks_by_id``.
      6. Trim to fit the caller's ``max_tokens`` budget.
      7. Build citations + freshness signaling per §6.3.

    The class is intentionally synchronous; the daemon runs each MCP
    call in a worker thread so concurrency comes from the runtime, not
    from this module.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        embed_store: EmbedStore,
        embedder,
        config: SearcherConfig,
        *,
        quiescence_check_callback: Optional[
            Callable[[Optional[str], int], tuple[int, str]]
        ] = None,
    ) -> None:
        """Wire the searcher to its dependencies.

        Args:
            chunk_store: persistent chunk + lexical index. Must outlive
                the searcher; closed elsewhere.
            embed_store: persistent dense-embedding store.
            embedder: the same SentenceTransformer the indexer used —
                callers must enforce model identity (§4.4) BEFORE
                handing the embedder to this class. We just call
                ``.encode``.
            config: tunables; see ``SearcherConfig``.
            quiescence_check_callback: ``(project, timeout_ms) ->
                (pending_updates, indexed_through_iso)``. Phase 2.0
                wiring is a stub; the watcher in 2.2 supplies the real
                queue-drain implementation. Left injectable so tests
                can simulate "3 pending writes" without a real watcher.
        """
        self._chunk_store = chunk_store
        self._embed_store = embed_store
        self._embedder = embedder
        self._config = config
        self._quiescence_cb = quiescence_check_callback

    # ------------------------------------------------------------ public API

    def search(
        self,
        query: str,
        *,
        cwd: Optional[str] = None,
        project: Optional[str] = None,
        max_tokens: int = 4096,
        max_results: Optional[int] = None,
        wait_for_quiescence_ms: Optional[int] = None,
        active_project_sticky: Optional[str] = None,
        paraphrases: tuple[str, ...] = (),
    ) -> SearchResult:
        """Run one search end-to-end and return a ``SearchResult``.

        Args mirror the MCP ``search_codebase`` signature (§6.2). The
        behavior in §6.3 — every response carries scope, freshness, and
        timings — is what makes this method's return type fat.

        ``paraphrases`` is accepted as a tuple even though it's only
        ever produced by the MCP layer (Phase 2.x will synthesize
        paraphrases via a side-channel LLM call). Keeping the parameter
        on the searcher means the multi-query fusion code path is
        unit-testable without standing up the MCP transport.
        """
        # ---- Stage: scope decision (cheap, but timed for completeness)
        projects = self._list_indexed_projects()
        scope = decide_scope(
            query=query,
            cwd=cwd,
            project=project,
            active_project_sticky=active_project_sticky,
            projects_in_index=projects,
        )
        scope_filter = self._scope_to_filter(scope)

        # ---- Stage: wait_quiescence
        timeout_ms = (
            wait_for_quiescence_ms
            if wait_for_quiescence_ms is not None
            else self._config.default_wait_for_quiescence_ms
        )
        t0 = time.perf_counter()
        pending_updates, indexed_through = self._wait_quiescence(
            scope.primary_project, timeout_ms,
        )
        wait_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage: embed_query
        queries = (query, *paraphrases)
        t0 = time.perf_counter()
        query_embs = self._embed_queries(queries)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        top_n = self._config.default_top_k_for_fusion

        # ---- Stage: bm25_score
        bm25_rankings: list[tuple[Hit, ...]] = []
        t0 = time.perf_counter()
        if self._config.bm25_weight > 0:
            for q in queries:
                terms = _word_tokenize(q)
                if not terms:
                    bm25_rankings.append(())
                    continue
                bm25_rankings.append(
                    self._chunk_store.search_lexical(terms, top_n, scope_filter)
                )
        bm25_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage: dense_score
        dense_rankings: list[tuple[Hit, ...]] = []
        t0 = time.perf_counter()
        if self._config.dense_weight > 0:
            scope_chunk_ids = self._scope_to_chunk_ids(scope)
            for emb in query_embs:
                dense_rankings.append(
                    self._embed_store.search_dense(emb, top_n, scope_chunk_ids)
                )
        dense_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage: rrf_fuse
        t0 = time.perf_counter()
        fused = self._rrf_fuse(bm25_rankings, dense_rankings)
        # Cap the post-fusion list at top_n so we don't pay for
        # materializing chunks that won't make it past the token budget.
        cap = max_results if max_results is not None else top_n
        fused = fused[:max(cap, 1)]
        rrf_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage: fetch_chunks
        t0 = time.perf_counter()
        chunks_by_id = {
            c.id: c
            for c in self._chunk_store.get_chunks_by_id([cid for cid, _ in fused])
        }
        ranked_chunks = [
            (chunks_by_id[cid], score)
            for cid, score in fused
            if cid in chunks_by_id
        ]
        # Token-budget enforcement: greedy take-until-overflow. Ordered by
        # fused score so the highest-relevance chunks are kept.
        kept: list[tuple[int, SearchChunk]] = []
        budget = max_tokens
        for chunk, score in ranked_chunks:
            tok = self._chunk_token_count(chunk)
            if tok > budget:
                # Single chunk overflows; stop here so we never exceed the cap.
                break
            citation = self._build_citation(chunk)
            kept.append((
                chunk.id,
                SearchChunk(
                    citation=citation,
                    text=chunk.text,
                    relevance_score=float(score),
                    token_count=tok,
                ),
            ))
            budget -= tok
            if max_results is not None and len(kept) >= max_results:
                break
        fetch_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Freshness
        # TODO(2.2): compare each chunk's source-file mtime against
        # ``last_indexed_at`` (need a ChunkStore.get_file_mtime helper)
        # to populate stale_files. For now we leave it empty and rely on
        # pending_updates as the binary signal.
        stale_files: tuple[str, ...] = ()
        is_fully_fresh = pending_updates == 0 and not stale_files
        freshness = SearchFreshness(
            is_fully_fresh=is_fully_fresh,
            pending_updates=pending_updates,
            indexed_through=indexed_through,
            stale_files=stale_files,
        )

        total_ms = wait_ms + embed_ms + bm25_ms + dense_ms + rrf_ms + fetch_ms
        latency = LatencyBreakdown(
            wait_quiescence=wait_ms,
            embed_query=embed_ms,
            bm25_score=bm25_ms,
            dense_score=dense_ms,
            rrf_fuse=rrf_ms,
            fetch_chunks=fetch_ms,
            total=total_ms,
        )

        return SearchResult(
            chunks=tuple(sc for _, sc in kept),
            freshness=freshness,
            scope_decision=scope,
            latency_ms=latency,
        )

    # --------------------------------------------------------- internals

    def _list_indexed_projects(self) -> tuple[str, ...]:
        """Project names from the chunk store, in stable order.

        Used only by ``decide_scope``; we don't bother caching since
        SQLite-backed ``list_projects`` returns in milliseconds.
        """
        return tuple(p.name for p in self._chunk_store.list_projects())

    def _scope_to_filter(self, scope: ScopeDecision) -> Optional[ScopeFilter]:
        """Translate a ``ScopeDecision`` into a ``ScopeFilter`` for the store.

        - Single-project fanout → ``ScopeFilter(project=...)``
        - Multi-project fanout  → ``ScopeFilter(project_in=...)``
        - Empty fanout          → None (search everything)

        Empty fanout shouldn't happen in practice (tier 5 returns all
        indexed projects), but is handled defensively so a caller with
        an empty corpus doesn't blow up here.
        """
        fanout = scope.fanout_projects
        if not fanout:
            return None
        if len(fanout) == 1:
            return ScopeFilter(project=fanout[0])
        return ScopeFilter(project_in=fanout)

    def _scope_to_chunk_ids(
        self, scope: ScopeDecision
    ) -> Optional[tuple[int, ...]]:
        """Resolve a scope to the chunk-id allow-list for ``search_dense``.

        ``EmbedStore.search_dense`` doesn't take a ``ScopeFilter`` — it
        takes ``chunk_ids_in_scope``. So when fanout is non-empty we
        ask the chunk store which chunk-ids belong to those projects.

        Returns ``None`` for global search (no filter) — equivalent to
        passing ``ScopeFilter(project=None, project_in=None)``.

        Performance note: building the allow-list is O(N_chunks_in_scope).
        At Phase 2.0 corpus sizes (~50K chunks/project) this is fine; if
        it ever shows up in profiling, we move the filter into the embed
        store itself (it has the chunk-id metadata anyway via the
        memmap-row mapping).
        """
        scope_filter = self._scope_to_filter(scope)
        if scope_filter is None:
            return None
        # Light path: query the chunk store for the ids in scope.
        # Sorting keeps brute-force cosine deterministic across runs.
        ids = sorted(
            getattr(self._chunk_store, "list_chunk_ids_in_scope", lambda f: ())(
                scope_filter,
            )
        )
        return tuple(ids)

    def _wait_quiescence(
        self, project: Optional[str], timeout_ms: int,
    ) -> tuple[int, str]:
        """Phase 2.0 stub. Defers to the injected callback or returns idle.

        The watcher work in Phase 2.2 will replace this with a real
        queue drain that blocks up to ``timeout_ms``. Keep the stub
        signature stable so the swap is internal.
        """
        if self._quiescence_cb is not None:
            return self._quiescence_cb(project, timeout_ms)
        return _default_quiescence_callback()

    def _embed_queries(self, queries: tuple[str, ...]) -> np.ndarray:
        """Single batched encode of all queries (literal + paraphrases).

        SentenceTransformer-style ``.encode`` is the assumed contract;
        we always request normalized embeddings so the dense backend's
        dot-product can be treated as cosine.
        """
        if not queries:
            return np.empty((0,), dtype=np.float32)
        embs = self._embedder.encode(
            list(queries),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embs)

    def _rrf_fuse(
        self,
        bm25_rankings: Sequence[Sequence[Hit]],
        dense_rankings: Sequence[Sequence[Hit]],
    ) -> list[tuple[int, float]]:
        """Combine all per-query rankings into one ``(chunk_id, score)`` list.

        Each ranker's per-query top-N gets the standard 1/(k+rank) score
        weighted by ``bm25_weight`` / ``dense_weight``. Scores accumulate
        across queries — a chunk that's strong in one paraphrase and
        weak in another wins over a chunk that's mediocre everywhere.

        Output is sorted descending by fused score; ties broken
        deterministically by chunk_id for replay stability.
        """
        k = self._config.rrf_k
        bw = self._config.bm25_weight
        dw = self._config.dense_weight

        fused: dict[int, float] = {}
        for ranking in bm25_rankings:
            for rank, hit in enumerate(ranking, start=1):
                fused[hit.chunk_id] = (
                    fused.get(hit.chunk_id, 0.0) + bw * _rrf_score(rank, k)
                )
        for ranking in dense_rankings:
            for rank, hit in enumerate(ranking, start=1):
                fused[hit.chunk_id] = (
                    fused.get(hit.chunk_id, 0.0) + dw * _rrf_score(rank, k)
                )
        # Stable sort: primary key is descending score, tiebreak ascending id.
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

    def _build_citation(self, chunk) -> Citation:
        """Turn a stored ``Chunk`` into a ``Citation`` with project prefix.

        Spec §6.2: file_path is repo-relative AND prefixed with the
        project name so cross-project hits are unambiguous in the
        agent's context window. We need the file's project + rel_path
        to assemble it.
        """
        # Look up the file record. Tolerant of stores that don't expose
        # ``get_file_by_id`` (e.g. unit-test fakes) — fall back to the
        # chunk's own attributes.
        get_file = getattr(self._chunk_store, "get_file_by_id", None)
        if callable(get_file):
            file_rec = get_file(chunk.file_id)
            if file_rec is not None:
                project = file_rec.project
                rel_path = file_rec.rel_path
                return Citation(
                    project=project,
                    file_path=f"{project}/{rel_path}",
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
        # Fallback: chunk-derived attributes (test fakes set these directly).
        project = getattr(chunk, "project", "")
        rel_path = getattr(chunk, "rel_path", f"chunk-{chunk.id}")
        prefix = f"{project}/" if project else ""
        return Citation(
            project=project,
            file_path=f"{prefix}{rel_path}",
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

    def _chunk_token_count(self, chunk) -> int:
        """Best-effort token count for budget enforcement.

        Prefer the chunker-recorded value; fall back to a chars-per-token
        estimate when the count is missing (legacy fixtures, in-memory
        fakes that don't bother to compute it).
        """
        recorded = getattr(chunk, "token_count", 0) or 0
        if recorded > 0:
            return int(recorded)
        return max(1, len(chunk.text) // self._config.chars_per_token)
