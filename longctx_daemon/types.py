"""Shared types for the longctx daemon.

These dataclasses are the contract every component speaks. Storage,
indexer, searcher, and the MCP layer all import from here so a change
in one shape ripples through the type checker.

Conservatively typed — no Any, no untyped dicts in public signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------- corpus types

@dataclass(frozen=True)
class Project:
    """One indexable corpus root.

    Attributes:
        name: stable display name (used in citations + tool args).
            Derived from the directory's basename unless overridden.
        root_path: absolute path to the project root.
        last_full_scan_at: unix timestamp of the most recent complete walk.
        session_id: when set, this project is bound to an MCP session
            (added via add_project(persist=False)) and gets GC'd when
            that session ends. None for permanent projects.
    """
    name: str
    root_path: str
    last_full_scan_at: int = 0
    session_id: Optional[str] = None


@dataclass(frozen=True)
class FileRecord:
    """One indexed file. Tracks identity + change-detection metadata."""
    id: int
    project: str
    rel_path: str
    mtime: int
    size_bytes: int
    content_hash: str   # sha256 of full file content


@dataclass(frozen=True)
class Chunk:
    """One retrievable chunk. Has stable identity via content_hash so
    incremental updates can reuse cached embeddings when the chunk text
    is unchanged across a file mutation."""
    id: int
    file_id: int
    chunk_index: int
    start_offset: int     # byte offset in source file
    end_offset: int
    start_line: int       # 1-indexed
    end_line: int
    token_count: int
    content_hash: str
    text: str
    embedder_model: str
    embedder_sha256: str
    embedding_row: Optional[int]   # row in the memmap; None if unembedded


# --------------------------------------------------------------- search types

@dataclass(frozen=True)
class ScopeFilter:
    """Constraint passed to ChunkStore.search_*. ``project`` selects
    one project; ``project_in`` selects multiple. Both None = search
    everything indexed."""
    project: Optional[str] = None
    project_in: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class Hit:
    """One ranking hit — chunk + score from a single ranker pass.
    The score's interpretation depends on the ranker:
      - dense: cosine similarity in [-1, 1] (normalized embeddings)
      - bm25:  raw Okapi BM25 score (unbounded)
      - fused: RRF fused score (~0.005-0.04 with k=60)
    """
    chunk_id: int
    score: float


@dataclass(frozen=True)
class Citation:
    """Citation surfaced to the agent. Identifies WHERE a chunk lives."""
    project: str
    file_path: str         # repo-relative, with project name prefix
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SearchChunk:
    """A search result chunk + its citation + relevance.
    Distinct from ``Chunk`` (the persisted form) so the search response
    type stays decoupled from the storage schema.

    ``relevance_score`` is the fused RRF score (rank-driven, useful for
    ordering but not absolute thresholding). ``dense_cosine`` is the
    pre-fusion semantic match anchor — that's the field to threshold
    on if you want to act per-chunk (e.g. drop borderline chunks). It
    is ``None`` for hits that came purely from BM25 (no embedding
    lookup happened for that chunk)."""
    citation: Citation
    text: str
    relevance_score: float
    token_count: int
    dense_cosine: Optional[float] = None


@dataclass(frozen=True)
class SearchFreshness:
    """Freshness signaling included in every search response so agents
    don't have to guess. ``is_fully_fresh`` is the cheap binary;
    ``stale_files`` is for the granular case."""
    is_fully_fresh: bool
    pending_updates: int
    indexed_through: str           # ISO 8601 UTC
    stale_files: tuple[str, ...]   # citations of files behind on indexing


@dataclass(frozen=True)
class SearchResult:
    """Top-level response from the searcher / MCP search_codebase tool.

    Attributes:
        chunks: top-K hits in ranked order (highest relevance first).
            Empty when ``no_relevant_results`` is True.
        freshness: per-response freshness signaling.
        scope_decision: which projects were searched + how the daemon
            picked them. Surfaced for observability — agents can ignore.
        latency_ms: per-stage timings, useful for the per-call MCP trace.
        no_relevant_results: True when the top-1 dense cosine fell below
            the configured ``relevance_floor``. The chunks are dropped
            (returned empty) so the agent doesn't cite low-confidence
            matches and gets a clean "I don't have anything for this"
            signal. Anchored on dense cosine — NOT on the fused RRF
            score, which is rank-driven (~0.032 for top-1 always) and
            useless for thresholding.
        top1_dense_cosine: the pre-fusion dense cosine for the highest-
            ranked chunk in ANY query (max across rankings). Surfaced
            so the agent / monitoring can see the actual semantic-match
            anchor that drives ``no_relevant_results``. Equals 0.0 when
            no chunks were considered.
        query_type: classification of the input. ``"natural_language"``
            (default) means an interrogative question or directive;
            ``"find_similar"`` means the input looks like a code or
            text blob the agent wants matched against the corpus
            rather than answered. Lets the caller render results
            differently (find-similar mode benefits from showing more
            context than top-1; question mode stops at the answer).
    """
    chunks: tuple[SearchChunk, ...]
    freshness: SearchFreshness
    scope_decision: ScopeDecision
    latency_ms: LatencyBreakdown
    no_relevant_results: bool = False
    top1_dense_cosine: float = 0.0
    query_type: str = "natural_language"
    confidence_gap: float = 0.0
    """top1_dense_cosine − top2_dense_cosine. Big gap = clear winner;
    near zero = top results are tied / a coin flip and the agent
    should treat ranking as low-information. 0.0 when fewer than two
    chunks were considered."""


@dataclass(frozen=True)
class MultiSearchResult:
    """Response from the searcher's multi-question API
    (``Searcher.search_multi`` / ``search_codebase`` with
    ``query: list[str]``).

    Each element of ``groups`` is one full ``SearchResult`` for one
    sub-query. Order matches the input list. Results are NOT merged
    or de-duplicated across groups — the caller needs to know which
    chunks came from which sub-question.

    Shared fields (scope_decision, freshness, total latency) live at
    the top level so the per-call MCP trace stays compact.
    """
    queries: tuple[str, ...]
    groups: tuple[SearchResult, ...]
    scope_decision: ScopeDecision
    freshness: SearchFreshness
    latency_ms: LatencyBreakdown


# --------------------------------------------------------------- scope types

@dataclass(frozen=True)
class ScopeDecision:
    """How the daemon decided what to search for one query.

    primary_source values:
      - "explicit_project"            : caller passed project=
      - "cross_project_pattern"       : query mentioned a project name
      - "active_project_sticky"       : set_active_project earlier in session
      - "cwd_walk_to_sentinel"        : walked up from cwd
      - "fanout_no_primary"           : no signal → fanned across all
    """
    primary_project: Optional[str]
    primary_source: str
    fanout_projects: tuple[str, ...]
    cross_project_pattern_matched: Optional[str]
    active_project_sticky: Optional[str]


# --------------------------------------------------------------- timing types

@dataclass(frozen=True)
class LatencyBreakdown:
    """Per-stage wall-clock timings in milliseconds. Stages map to the
    §14.4 logging schema; missing stages are 0.0 (e.g. wait_quiescence
    when the queue is empty)."""
    wait_quiescence: float = 0.0
    embed_query: float = 0.0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_fuse: float = 0.0
    fetch_chunks: float = 0.0
    total: float = 0.0


# --------------------------------------------------------------- index status

@dataclass(frozen=True)
class IndexStatus:
    """Snapshot of daemon health, returned by index_status MCP tool."""
    status: str   # "ready" | "indexing" | "error"
    total_chunks: int
    pending_updates: int
    embedder_model: str
    embedder_sha256: str
    last_full_scan: Optional[str]   # ISO timestamp
    error: Optional[str] = None
    projects: tuple[Project, ...] = field(default_factory=tuple)
