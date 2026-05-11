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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np

from longctx.rag.symbol_augment import (
    extract_symbols,
    file_type_weight,
    has_code_signal,
    query_features,
    symbol_grep_repo,
)
from longctx_daemon.storage.protocol import ChunkStore, EmbedStore
from longctx_daemon.types import (
    Citation,
    Hit,
    LatencyBreakdown,
    MultiSearchResult,
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


# Lightweight code-token regex used by ``classify_query_type``. Matches
# things that strongly suggest a code paste rather than a question:
# parenthesis-followed identifiers, dunder names, brace blocks, common
# language keywords with structural punctuation. Deliberately
# conservative — we only want to flag the obvious blob cases. Natural-
# language sentences that mention code names ("the foo() function") DO
# NOT trip these patterns.
_CODE_BLOB_RE = re.compile(
    r"(?:"
    r"\bdef\s+\w+\s*\("                       # python def
    r"|\bclass\s+\w+\s*[:(]"                  # class declaration
    r"|\bfunction\s+\w+\s*\("                 # JS/TS function decl
    r"|\bfunc\s+\w+\s*\("                     # Go / Swift func decl
    r"|\bfn\s+\w+\s*\("                       # Rust fn decl
    r"|\bimport\s+\w[\w.]*"                   # import statements
    r"|^\s*(?:from\s+\w[\w.]*\s+import|require\s*\()"   # py from / node require
    r"|\}\s*else\s*\{"                        # } else { block
    r"|\)\s*->\s*\w"                          # type-arrow return
    r"|^\s*//\s|^\s*#\s*[A-Z]"                # comment-led code
    r"|::\w+\s*\("                            # C++/Rust :: call
    r")",
    re.MULTILINE,
)
_INTERROGATIVE_RE = re.compile(
    r"\b(?:what|where|when|who|why|how|which|"
    r"is|are|does|do|did|can|could|should|would|will|won't|isn't|aren't|"
    r"show|tell|find|give|explain|describe|list|name)\b",
    re.IGNORECASE,
)
_TRACEBACK_RE = re.compile(
    r"(?:Traceback\s*\(|\bat\s+\w[\w./]*\.\w+:\d+|"
    r"^\s*File\s+['\"][^'\"]+['\"],\s*line\s+\d+)",
    re.MULTILINE,
)


def classify_query_type(query: str) -> str:
    """Classify ``query`` as ``"natural_language"`` or ``"find_similar"``.

    Heuristic-based; deliberately conservative. The goal is to flag
    the obvious code-blob / paste-with-question cases without false-
    flagging natural-language questions that happen to mention code
    identifiers.

    Decision tree:
      1. Multi-line input + (code-block pattern OR traceback marker)
         → ``find_similar``. Pasted stack traces and code snippets
         almost always span multiple lines.
      2. Single-line input → never ``find_similar``. Even if the line
         contains code (``where is auth_middleware``), the
         interrogative-only-on-one-line case is overwhelmingly a
         question, not a find-similar request.
      3. Multi-line + interrogative cue at start/end (e.g. blob pasted
         then "where is this?" appended) → still ``natural_language``
         because the agent likely DID intend to ask a question; the
         relevance-floor + score will determine if it works.
      4. Multi-line + code blob and NO interrogative anywhere
         → ``find_similar``.

    The Phase 2.0.1 contract is to TAG the result, not to change
    ranking. Future iterations can swap modes (e.g. dense-only on
    full blob for find_similar) once we observe how callers use
    the tag.
    """
    if not query or "\n" not in query:
        return "natural_language"
    has_code = bool(_CODE_BLOB_RE.search(query))
    has_traceback = bool(_TRACEBACK_RE.search(query))
    if not (has_code or has_traceback):
        return "natural_language"
    has_interrogative = bool(_INTERROGATIVE_RE.search(query))
    if has_interrogative:
        return "natural_language"
    return "find_similar"


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
        relevance_floor: minimum top-1 dense cosine for a result to be
            considered "real". Below this, ``Searcher.search`` returns
            empty chunks + ``no_relevant_results=True`` instead of a
            low-confidence false positive (e.g. off-corpus questions
            like "capital of france" sit at ~0.44 against a code corpus
            and used to confidently return wrong chunks). Anchored on
            DENSE COSINE, not the fused RRF score, because RRF is
            rank-driven (~0.032 for top-1 always). Calibration data:
            benchmark/messy_queries/RESULTS.md. Defensible default 0.50;
            clean matches sit at 0.6+, noise at 0.4-0.5. Set to 0.0 to
            disable the filter entirely.
    """
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    default_top_k_for_fusion: int = 1000
    default_wait_for_quiescence_ms: int = 500
    chars_per_token: int = 4
    relevance_floor: float = 0.50
    project_floors: dict[str, float] = field(default_factory=dict)
    """Per-project floor overrides — when the search's primary project
    matches a key, that floor takes precedence over ``relevance_floor``.
    Phase 3 dogfood-audit found 0.50 doesn't generalize across corpus
    types; populated via ``longctx calibrate --project NAME --write-config``
    (Phase 3 follow-up). Empty dict (default) keeps the global floor."""


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
        relevance_floor: Optional[float] = None,
        auto_policy: bool = False,
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

        ``relevance_floor`` overrides ``SearcherConfig.relevance_floor``
        per call (e.g. an agent that wants raw results passes 0.0;
        an agent that wants strict honesty passes 0.65). The floor
        gates on TOP-1 DENSE COSINE, not the fused RRF score. When the
        top-1 cosine falls below it, ``chunks`` returns empty and
        ``no_relevant_results=True`` so the agent knows the corpus
        had nothing for this question.

        ``auto_policy`` opts into the context-size + query-shape
        adaptive router (``longctx_daemon.policy``). When True, the
        searcher detects the query's shape (symbolic / prose / mixed),
        estimates corpus size from the chunk store, looks up the
        ``RetrievalPolicy`` for that cell, and overrides
        ``bm25_weight`` / ``dense_weight`` for THIS call. The policy's
        rationale + embedder hint are surfaced on the result so the
        caller can see why this stack was picked. Default False keeps
        production behavior unchanged.
        """
        # Per-call policy resolution. ``query_shape`` is always
        # populated (cheap regex); the *_weight overrides only apply
        # when ``auto_policy`` was opted into.
        (
            effective_bm25_weight,
            effective_dense_weight,
            policy_rationale,
            embedder_hint_text,
            query_shape_str,
        ) = self._resolve_policy(query=query, auto_policy=auto_policy)
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
        if effective_bm25_weight > 0:
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
        if effective_dense_weight > 0:
            scope_embedding_rows = self._scope_to_embedding_rows(scope)
            raw_dense_rankings: list[tuple[Hit, ...]] = []
            for emb in query_embs:
                raw_dense_rankings.append(
                    self._embed_store.search_dense(
                        emb, top_n, scope_embedding_rows,
                    )
                )
            # Normalize: ``EmbedStore.search_dense`` returns row indices
            # in ``Hit.chunk_id``, not chunk.id. Translate so RRF fuses
            # against the same key space as BM25 hits (which DO emit
            # real chunk.id values). Without this, the same chunk
            # appears under two different keys and fusion silently
            # double-counts / mis-orders.
            row_set: set[int] = set()
            for ranking in raw_dense_rankings:
                row_set.update(h.chunk_id for h in ranking)
            row_to_chunk_id: dict[int, int] = {}
            if row_set:
                getter = getattr(
                    self._chunk_store, "get_chunk_ids_by_embedding_rows",
                    None,
                )
                if callable(getter):
                    row_to_chunk_id = getter(row_set) or {}
            for ranking in raw_dense_rankings:
                normalized: list[Hit] = []
                for h in ranking:
                    cid = row_to_chunk_id.get(h.chunk_id, h.chunk_id)
                    normalized.append(Hit(chunk_id=cid, score=h.score))
                dense_rankings.append(tuple(normalized))
        dense_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage: rrf_fuse
        t0 = time.perf_counter()
        fused = self._rrf_fuse(
            bm25_rankings, dense_rankings,
            bm25_weight=effective_bm25_weight,
            dense_weight=effective_dense_weight,
        )
        # Cap the post-fusion list at top_n so we don't pay for
        # materializing chunks that won't make it past the token budget.
        cap = max_results if max_results is not None else top_n
        fused = fused[:max(cap, 1)]

        # Build chunk_id → max-cosine-across-queries map. Dense store
        # returns chunk-row indices keyed as ``Hit.chunk_id``. Same key
        # space the searcher uses across the rest of the pipeline.
        # Skip BM25-only hits (they have no dense entry).
        dense_cosine_by_id: dict[int, float] = {}
        for ranking in dense_rankings:
            for hit in ranking:
                prev = dense_cosine_by_id.get(hit.chunk_id, -1.0)
                if hit.score > prev:
                    dense_cosine_by_id[hit.chunk_id] = float(hit.score)
        rrf_ms = (time.perf_counter() - t0) * 1000.0

        # Top-1 dense cosine drives the relevance floor. Anchor on the
        # FUSED RANK-1 CHUNK specifically — not the global max-cosine
        # across all chunks. The user's question is "is the thing I'm
        # about to return actually relevant?", so we test the chunk
        # the agent will see, not whichever-chunk-had-a-good-match-
        # somewhere. This is the strict honest-retrieval reading and
        # what the messy-query data points at: run-on / fragment / off-
        # corpus all have fused-rank-1 cosines ≤ 0.49, which we want
        # to suppress.
        top1_cosine = 0.0
        top2_cosine = 0.0
        if fused:
            top1_cosine = dense_cosine_by_id.get(fused[0][0], 0.0)
            if len(fused) >= 2:
                top2_cosine = dense_cosine_by_id.get(fused[1][0], 0.0)
        confidence_gap = max(0.0, top1_cosine - top2_cosine)

        # Classify query shape (natural-language vs find-similar).
        # Tag-only for Phase 2.0.1 — does not change ranking. Lets the
        # agent render results differently for code-blob inputs.
        query_type = classify_query_type(query)

        # Apply the relevance floor BEFORE materializing chunks: if the
        # top-1 dense cosine is below threshold, we know the chunks
        # would be low-confidence noise, so skip the fetch + return
        # the no-results signal. This is the "honest" path — better to
        # tell the agent we have nothing than ship false positives.
        #
        # Two cases skip the floor entirely:
        #   * floor <= 0 → user explicitly disabled the filter
        #   * dense_weight <= 0 → no dense signal to anchor on; the
        #     floor is meaningless and applying it would zero out
        #     pure-BM25 callers. They get a permissive pass through.
        # Floor resolution priority:
        #   1. per-call ``relevance_floor`` arg (explicit override)
        #   2. per-project floor from ``SearcherConfig.project_floors``
        #      keyed on the resolved primary project
        #   3. global ``SearcherConfig.relevance_floor``
        # Phase 3 dogfood-audit (2026-05-09) found 0.50 doesn't
        # generalize across corpus types — code corpora typically
        # work at 0.50, mixed at ~0.60, prose-heavy at ~0.65.
        if relevance_floor is not None:
            floor = relevance_floor
        else:
            primary = scope.primary_project
            project_floors = getattr(self._config, "project_floors", {}) or {}
            if primary and primary in project_floors:
                floor = project_floors[primary]
            else:
                floor = self._config.relevance_floor
        no_relevant_results = (
            floor > 0.0
            and effective_dense_weight > 0
            and (not fused or top1_cosine < floor)
        )

        # ---- Stage: fetch_chunks
        t0 = time.perf_counter()
        kept: list[tuple[int, SearchChunk]] = []
        if not no_relevant_results:
            chunks_by_id = {
                c.id: c
                for c in self._chunk_store.get_chunks_by_id(
                    [cid for cid, _ in fused],
                )
            }
            ranked_chunks = [
                (chunks_by_id[cid], score)
                for cid, score in fused
                if cid in chunks_by_id
            ]
            # Symbol-aware re-rank (2026-05-11). For code-signal queries
            # (traceback / error type / class/def mention), grep the
            # primary project's root for `class X` / `def X` definitions
            # of every identifier in the query. Boost chunks whose source
            # file is a definition site, then apply a file-type prior
            # (.py boost, .rst/.md demote). Bridges the BM25+dense bias
            # toward docs/changelogs over source on SWE-bench-style
            # code-fix queries. No-op when the primary project has no
            # known root or the query has no extractable identifiers.
            qf = query_features(query)
            if has_code_signal(qf) and scope.primary_project:
                project_root: Optional[str] = None
                try:
                    for p in self._chunk_store.list_projects():
                        if p.name == scope.primary_project:
                            project_root = p.root_path
                            break
                except Exception:  # noqa: BLE001
                    project_root = None
                sym_paths_set: set[str] = set()
                if project_root:
                    try:
                        from pathlib import Path as _Path
                        syms = extract_symbols(query)
                        if syms:
                            sym_paths_set = {
                                str(_Path(p).resolve())
                                for p in symbol_grep_repo(
                                    syms, _Path(project_root),
                                )
                            }
                    except Exception:  # noqa: BLE001
                        sym_paths_set = set()

                def _augment_key(item: tuple) -> tuple[float, float]:
                    chunk_obj, score_val = item
                    abs_path = self._chunk_path_abs(chunk_obj)
                    sym_boost = 1.5 if abs_path in sym_paths_set else 1.0
                    type_w = file_type_weight(abs_path, qf)
                    # Sort descending by (sym_boost, type_w, score).
                    return (-(sym_boost * type_w), -float(score_val))

                ranked_chunks.sort(key=_augment_key)

            # Token-budget enforcement: greedy take-until-overflow.
            # Ordered by fused score so the highest-relevance chunks
            # are kept.
            budget = max_tokens
            for chunk, score in ranked_chunks:
                tok = self._chunk_token_count(chunk)
                if tok > budget:
                    # Single chunk overflows; stop here so we never
                    # exceed the cap.
                    break
                citation = self._build_citation(chunk)
                cos = dense_cosine_by_id.get(chunk.id)
                kept.append((
                    chunk.id,
                    SearchChunk(
                        citation=citation,
                        text=chunk.text,
                        relevance_score=float(score),
                        token_count=tok,
                        dense_cosine=cos,
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

        retrieval_quality = self._compute_retrieval_quality(
            top1_cosine=top1_cosine, confidence_gap=confidence_gap,
            no_relevant_results=no_relevant_results,
            n_chunks=len(kept),
        )
        return SearchResult(
            chunks=tuple(sc for _, sc in kept),
            freshness=freshness,
            scope_decision=scope,
            latency_ms=latency,
            no_relevant_results=no_relevant_results,
            top1_dense_cosine=top1_cosine,
            query_type=query_type,
            confidence_gap=confidence_gap,
            query_shape=query_shape_str,
            applied_policy_rationale=policy_rationale,
            embedder_hint=embedder_hint_text,
            retrieval_quality=retrieval_quality,
        )

    # ---------------------------------------------------- multi-question API

    def search_multi(
        self,
        queries: Sequence[str],
        *,
        cwd: Optional[str] = None,
        project: Optional[str] = None,
        max_tokens: int = 4096,
        max_results: Optional[int] = None,
        wait_for_quiescence_ms: Optional[int] = None,
        active_project_sticky: Optional[str] = None,
        relevance_floor: Optional[float] = None,
    ) -> MultiSearchResult:
        """Run N independent searches for N sub-queries.

        Each sub-query goes through the full BM25 + dense + RRF pipeline
        on its own. Results are NOT merged or de-duplicated across
        groups — the caller needs to know which chunks came from which
        sub-question. ``max_tokens`` is per-group (the caller's budget
        is shared across groups; we don't try to enforce a global cap
        because the caller has more context for that decision).

        The shared response fields (``scope_decision``, ``freshness``,
        total ``latency_ms``) collapse the per-group values:
          - ``scope_decision`` is taken from the FIRST query (typically
            the agent's primary intent; subsequent queries usually
            inherit the same scope).
          - ``freshness`` is the worst-case across groups: if any group
            saw stale data, the whole response reports stale.
          - ``latency_ms.total`` is the SUM of all groups' totals; the
            per-stage breakdown sums each stage.

        Args mirror ``search`` plus accept a Sequence of queries instead
        of a string + paraphrases. Empty list → empty MultiSearchResult
        with a synthetic scope_decision (no_primary, no fanout) and
        zeroed freshness; doesn't raise.
        """
        queries_t = tuple(queries)
        if not queries_t:
            now_iso = _utc_now_iso()
            empty_scope = ScopeDecision(
                primary_project=None,
                primary_source="fanout_no_primary",
                fanout_projects=(),
                cross_project_pattern_matched=None,
                active_project_sticky=active_project_sticky,
            )
            return MultiSearchResult(
                queries=(),
                groups=(),
                scope_decision=empty_scope,
                freshness=SearchFreshness(
                    is_fully_fresh=True, pending_updates=0,
                    indexed_through=now_iso, stale_files=(),
                ),
                latency_ms=LatencyBreakdown(),
            )

        groups: list[SearchResult] = []
        for q in queries_t:
            groups.append(self.search(
                q,
                cwd=cwd,
                project=project,
                max_tokens=max_tokens,
                max_results=max_results,
                wait_for_quiescence_ms=wait_for_quiescence_ms,
                active_project_sticky=active_project_sticky,
                relevance_floor=relevance_floor,
            ))

        # ---- Collapse shared fields
        first_scope = groups[0].scope_decision
        any_stale = any(not g.freshness.is_fully_fresh for g in groups)
        max_pending = max(g.freshness.pending_updates for g in groups)
        # Earliest indexed_through wins (oldest snapshot dominates).
        # ISO-8601 strings sort lexicographically when zero-padded, so
        # min() on the strings is correct.
        oldest_indexed_through = min(
            g.freshness.indexed_through for g in groups
        )
        # Union of stale files across groups
        stale_union: set[str] = set()
        for g in groups:
            stale_union.update(g.freshness.stale_files)
        merged_freshness = SearchFreshness(
            is_fully_fresh=not any_stale and max_pending == 0,
            pending_updates=max_pending,
            indexed_through=oldest_indexed_through,
            stale_files=tuple(sorted(stale_union)),
        )
        # Sum stage timings across groups.
        total_lat = LatencyBreakdown(
            wait_quiescence=sum(g.latency_ms.wait_quiescence for g in groups),
            embed_query=sum(g.latency_ms.embed_query for g in groups),
            bm25_score=sum(g.latency_ms.bm25_score for g in groups),
            dense_score=sum(g.latency_ms.dense_score for g in groups),
            rrf_fuse=sum(g.latency_ms.rrf_fuse for g in groups),
            fetch_chunks=sum(g.latency_ms.fetch_chunks for g in groups),
            total=sum(g.latency_ms.total for g in groups),
        )
        return MultiSearchResult(
            queries=queries_t,
            groups=tuple(groups),
            scope_decision=first_scope,
            freshness=merged_freshness,
            latency_ms=total_lat,
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

    def _resolve_policy(
        self, *, query: str, auto_policy: bool,
    ) -> tuple[float, float, str, str, str]:
        """Compute the per-call (bm25_weight, dense_weight, rationale,
        embedder_hint, query_shape_str) for one search invocation.

        ``query_shape`` is always detected (cheap regex) and surfaced
        on the result so callers can render the heuristic
        classification regardless of whether they opted into auto-
        policy. The weight overrides only apply when ``auto_policy``
        is True.

        Corpus-size estimate uses ``chunk_count() × ~8000 chars``
        (rough average for a 2K-token chunk × 4 chars/token). Good
        enough for size bucketing.
        """
        from longctx_daemon.policy import (
            QueryShape, detect_query_shape, select_policy,
        )
        shape = detect_query_shape(query)
        shape_str = shape.value
        if not auto_policy:
            return (
                self._config.bm25_weight,
                self._config.dense_weight,
                "",   # no policy applied
                "",   # no embedder hint
                shape_str,
            )
        # Cheap corpus-size estimate. Skips a SQLite count when
        # the chunk store doesn't expose ``chunk_count()`` (defensive
        # against custom Protocol impls in tests).
        corpus_chars = 0
        try:
            cc = self._chunk_store.chunk_count()
            # Average chunk size ≈ 2K tokens × 4 chars/token = 8000.
            # Conservative; over-estimates rather than under so we
            # don't accidentally pick the SHORT bucket for a real
            # codebase that's just under the 64K threshold.
            corpus_chars = cc * 8000
        except Exception:  # noqa: BLE001 — best-effort
            corpus_chars = 0
        policy = select_policy(
            corpus_size_chars=corpus_chars, query_shape=shape,
        )
        return (
            policy.bm25_weight,
            policy.dense_weight,
            policy.rationale,
            policy.embedder_hint or "",
            shape_str,
        )

    def _compute_retrieval_quality(
        self, *, top1_cosine: float, confidence_gap: float,
        no_relevant_results: bool, n_chunks: int,
    ) -> str:
        """Coarse confidence summary derived from the dense-cosine
        signals. See ``SearchResult.retrieval_quality`` docstring for
        the labels' semantics. Thresholds eyeballed from dogfood data
        — per-corpus calibration may shift them later."""
        if no_relevant_results:
            return "abstain"
        if n_chunks == 0:
            return "unknown"
        if top1_cosine >= 0.75 and confidence_gap >= 0.10:
            return "high"
        if top1_cosine >= 0.60:
            return "medium"
        if top1_cosine >= 0.50 and confidence_gap >= 0.15:
            return "medium"
        return "low"

    def _scope_to_embedding_rows(
        self, scope: ScopeDecision
    ) -> Optional[tuple[int, ...]]:
        """Resolve a scope to the embedding-row allow-list that
        ``EmbedStore.search_dense`` actually consumes.

        Two key spaces collide here:
          * ``ChunkStore.list_chunk_ids_in_scope`` returns SQLite
            chunk.id values (1-indexed PKs).
          * ``EmbedStore.search_dense``'s scope-filter parameter is
            keyed on **embedding row indices** (0-indexed memmap rows).

        Earlier versions of this method handed chunk.ids straight to
        ``search_dense``, which silently masked the wrong rows whenever
        chunk.id ≠ embedding_row (any time chunks were deleted or
        re-embedded). The bug surfaces as the row-0 chunk being
        invisible in scoped dense search. The fix is to translate via
        ``ChunkStore.get_embedding_rows_by_chunk_ids`` before returning.

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
        chunk_ids = getattr(
            self._chunk_store, "list_chunk_ids_in_scope", lambda f: ()
        )(scope_filter)
        if not chunk_ids:
            return ()
        # Translate chunk.id → embedding_row. The store sorts the result
        # ascending so brute-force cosine ordering stays deterministic
        # across runs.
        translator = getattr(
            self._chunk_store, "get_embedding_rows_by_chunk_ids", None,
        )
        if callable(translator):
            return tuple(translator(chunk_ids) or ())
        # Fallback for chunk-store implementations that don't (yet) have
        # the translator: pass chunk.ids and accept the latent bug. Real
        # impls (SqliteChunkStore) always have the translator.
        return tuple(sorted(chunk_ids))

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
        *,
        bm25_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
    ) -> list[tuple[int, float]]:
        """Combine all per-query rankings into one ``(chunk_id, score)`` list.

        Each ranker's per-query top-N gets the standard 1/(k+rank) score
        weighted by ``bm25_weight`` / ``dense_weight``. Scores accumulate
        across queries — a chunk that's strong in one paraphrase and
        weak in another wins over a chunk that's mediocre everywhere.

        Output is sorted descending by fused score; ties broken
        deterministically by chunk_id for replay stability.

        ``bm25_weight`` / ``dense_weight`` override the
        ``SearcherConfig`` defaults for this fusion only — used by the
        auto-policy router to flip retrieval shape per call without
        mutating the global config. ``None`` falls back to the config.
        """
        k = self._config.rrf_k
        bw = bm25_weight if bm25_weight is not None else self._config.bm25_weight
        dw = dense_weight if dense_weight is not None else self._config.dense_weight

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

    def _chunk_path_abs(self, chunk) -> str:
        """Best-effort absolute path for a chunk's source file.

        Used by the symbol-aware re-rank to match against rg's grep
        output. Tolerant of stores that don't expose ``get_file_by_id``
        / ``list_projects`` (test fakes, partial mocks) — returns the
        empty string when it can't resolve, which makes the augment a
        no-op for that chunk.
        """
        get_file = getattr(self._chunk_store, "get_file_by_id", None)
        if not callable(get_file):
            return ""
        try:
            file_rec = get_file(chunk.file_id)
        except Exception:  # noqa: BLE001
            return ""
        if file_rec is None:
            return ""
        try:
            for p in self._chunk_store.list_projects():
                if p.name == file_rec.project:
                    from pathlib import Path as _Path
                    return str((_Path(p.root_path) / file_rec.rel_path).resolve())
        except Exception:  # noqa: BLE001
            return ""
        return ""

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
