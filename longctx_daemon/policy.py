"""Auto-policy router — picks retrieval config based on corpus size and
query shape.

The policy table comes from the cross-bin MRCR e2e measurements in
``benchmark/mrcr_e2e/RESULTS.md``. Each cell is a measured operating
point — short bin / mid bin / long bin / extra-long bin × symbolic /
prose / mixed query shape.

Why this exists: a single global default for ``bm25_weight``,
``chunk_tokens``, ``rerank_pre_k``, and embedder choice can't be right
across both code-symbol queries on a 50K-token repo and prose
disambiguation queries on a 1M-token corpus. The MRCR data shows
specific switches:

  * Dense-only beats BM25+dense+RRF on prose disambiguation by ~21
    abs pts at 32K. BM25's lexical-token match promotes wrong-but-
    similar candidates into the fused top-K when essays share topic
    vocabulary.
  * Hierarchical chunking lifts long bins (1M cell: 0.523 → 0.555
    with chunk_tokens=300) but is a no-op at short bins.
  * Cross-encoder rerank lifts long bins (1M cell: +5 abs pts) but
    adds latency without much gain at short bins.
  * bge-m3 wins at 32K (full essay fits in 8K maxlen) but LOSES at
    256K (long candidates dilute embedding). MiniLM's 256-token
    truncation acts as a "first-paragraph anchor" that's more
    discriminative when essays exceed that length.
  * 32B generator wins at ≤32K (its native window); for longer bins
    you need the 14B-1M variant whose 1M context fits the prompt.

The router below encodes those switches as a (size_bucket, query_shape)
→ RetrievalPolicy mapping. ``embedder_hint`` and ``generator_hint``
are advisory: the daemon can't swap them on the fly without
re-indexing / restarting the LLM server, so they surface as
recommendations the CLI / agent can act on at the orchestration
layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryShape(Enum):
    """Coarse shape of the user's query, drives retrieval-side knobs."""
    UNKNOWN = "unknown"      # fall back to production-safe defaults
    SYMBOLIC = "symbolic"    # identifier-shaped (camelCase / snake_case / dotted)
    PROSE = "prose"          # natural-language description
    MIXED = "mixed"          # both signals present


@dataclass(frozen=True)
class RetrievalPolicy:
    """Configuration choices for one search call.

    All fields are advisory — callers merge them into the active
    ``SearcherConfig`` plus orchestration-layer choices (which
    embedder loaded, which generator served).

    ``embedder_hint`` and ``generator_hint`` are NOT applied
    automatically; they surface as recommendations because changing
    them requires re-indexing or restarting the LLM server.
    """
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    chunk_tokens: Optional[int] = None
    rerank_pre_k: Optional[int] = None
    embedder_hint: Optional[str] = None
    generator_hint: Optional[str] = None
    # Free-form rationale — useful for surfacing why this stack was
    # picked when the agent debugs an unexpected result.
    rationale: str = ""


# Size buckets in chars. The thresholds approximately mirror the
# MRCR bin labels but in source-corpus terms (we lose ~3-4× compression
# from chars → tokens for the MRCR conversation format).
SHORT = "short"          # ≤64K chars  (≤16K tokens-ish)
MID = "mid"              # 64K-256K   (16K-64K tokens)
LONG = "long"            # 256K-1M    (64K-256K tokens)
EXTRA_LONG = "xlong"     # >1M        (>256K tokens)


def _size_bucket(corpus_size_chars: int) -> str:
    if corpus_size_chars < 64_000:
        return SHORT
    if corpus_size_chars < 256_000:
        return MID
    if corpus_size_chars < 1_000_000:
        return LONG
    return EXTRA_LONG


# Production-safe default: BM25+dense+RRF, no chunking, no rerank.
# Used when query_shape is UNKNOWN at any size, or when no entry
# matches the (bucket, shape) key. Conservative to avoid silently
# breaking existing callers.
_DEFAULT_POLICY = RetrievalPolicy(
    bm25_weight=1.0, dense_weight=1.0,
    chunk_tokens=None, rerank_pre_k=None,
    rationale="default — BM25+dense+RRF, safe across workloads",
)


# Policy table from MRCR e2e measurements + Phase 2.0.1 hybrid-retrieval
# observations. Update keys when adding a new measured operating point.
_POLICY_TABLE: dict[tuple[str, QueryShape], RetrievalPolicy] = {
    # ───────── SYMBOLIC: identifier-shaped (rename, find references)
    # BM25 lexical match is exactly what wins here. Keep dense too as
    # a tiebreaker for partial-match cases.
    (SHORT, QueryShape.SYMBOLIC): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        rationale="symbolic short — BM25 lexical match dominates",
    ),
    (MID, QueryShape.SYMBOLIC): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        rationale="symbolic mid — BM25+dense, no chunking needed",
    ),
    (LONG, QueryShape.SYMBOLIC): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        chunk_tokens=500, rerank_pre_k=32,
        rationale="symbolic long — sub-chunk + rerank surface "
                  "deep-buried symbol mentions",
    ),
    (EXTRA_LONG, QueryShape.SYMBOLIC): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        chunk_tokens=300, rerank_pre_k=64,
        rationale="symbolic xlong — finer chunks + wider rerank pool",
    ),

    # ───────── PROSE: natural-language description (essay disambiguation,
    # explain-this, summarize-this).
    # MRCR data: BM25 actively HURTS by 21 abs pts at 32K when essays
    # share topic vocabulary. Use dense-only.
    (SHORT, QueryShape.PROSE): RetrievalPolicy(
        bm25_weight=0.0, dense_weight=1.0,
        embedder_hint="BAAI/bge-m3",
        rationale="prose short — dense-only with bge-m3 (essays fit "
                  "in 8K maxlen, finer semantic disambiguation)",
    ),
    (MID, QueryShape.PROSE): RetrievalPolicy(
        bm25_weight=0.0, dense_weight=1.0,
        embedder_hint="BAAI/bge-m3",
        rerank_pre_k=32,
        rationale="prose mid — dense+rerank, bge-m3 still wins at "
                  "this size",
    ),
    (LONG, QueryShape.PROSE): RetrievalPolicy(
        bm25_weight=0.0, dense_weight=1.0,
        embedder_hint="sentence-transformers/all-MiniLM-L6-v2",
        chunk_tokens=300, rerank_pre_k=64,
        rationale="prose long — MiniLM's 256-token first-paragraph "
                  "anchor beats bge-m3 here; chunked + reranked",
    ),
    (EXTRA_LONG, QueryShape.PROSE): RetrievalPolicy(
        bm25_weight=0.0, dense_weight=1.0,
        embedder_hint="sentence-transformers/all-MiniLM-L6-v2",
        chunk_tokens=300, rerank_pre_k=64,
        rationale="prose xlong — same as long; MiniLM holds up at "
                  "1M+ when bge-m3's full-essay embedding dilutes",
    ),

    # ───────── MIXED: query has both identifier-shape and prose tokens.
    # Default to BM25-on (symbolic component matters) but add rerank at
    # scale to handle the prose component too.
    (SHORT, QueryShape.MIXED): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        rationale="mixed short — full hybrid, no extras",
    ),
    (MID, QueryShape.MIXED): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        rerank_pre_k=32,
        rationale="mixed mid — hybrid + rerank for prose disambig",
    ),
    (LONG, QueryShape.MIXED): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        chunk_tokens=500, rerank_pre_k=32,
        rationale="mixed long — chunked + reranked hybrid",
    ),
    (EXTRA_LONG, QueryShape.MIXED): RetrievalPolicy(
        bm25_weight=1.0, dense_weight=1.0,
        chunk_tokens=300, rerank_pre_k=64,
        rationale="mixed xlong — finer chunks + wider rerank",
    ),
}


def select_policy(
    *,
    corpus_size_chars: int,
    query_shape: QueryShape = QueryShape.UNKNOWN,
) -> RetrievalPolicy:
    """Look up the policy for a (corpus-size, query-shape) cell.

    UNKNOWN query_shape returns the production-safe default at every
    size — when the caller doesn't know the shape, we don't risk
    silently breaking BM25-on callers by switching to dense-only.
    """
    if query_shape is QueryShape.UNKNOWN:
        return _DEFAULT_POLICY
    bucket = _size_bucket(corpus_size_chars)
    return _POLICY_TABLE.get((bucket, query_shape), _DEFAULT_POLICY)


# ─────────────────────────────────────────────── query-shape heuristic

# Identifier-shape patterns:
#   camelCase    e.g. getUserById
#   snake_case   e.g. process_payment
#   SCREAMING    e.g. SECRET_KEY
#   dotted.path  e.g. app.config.SECRET
#   path/like    e.g. auth/login (debatable; lean SYMBOLIC if rest is
#                identifier-shape)
_IDENTIFIER_RE = re.compile(
    r"\b("
    r"[a-z][a-z0-9]*[A-Z][A-Za-z0-9]+"          # camelCase
    r"|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+"         # PascalCase
    r"|[a-z]+_[a-z_0-9]+"                        # snake_case
    r"|[A-Z][A-Z0-9]+_[A-Z0-9_]+"                # SCREAMING_SNAKE
    r"|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*"  # dotted.path
    r")\b"
)

# Cheap natural-language signal: presence of common interrogative /
# directive function words. Robust enough as a tiebreaker — not
# authoritative.
_NL_SIGNAL_RE = re.compile(
    r"\b(the|a|an|how|why|what|where|who|when|"
    r"explain|describe|find|reproduce|show|tell|"
    r"summarize|summarise|list|give|fetch)\b",
    re.IGNORECASE,
)


def detect_query_shape(query: str) -> QueryShape:
    """Heuristic classification — NOT authoritative.

    The agent / CLI should override this when intent is known
    (e.g., a rename refactor calls this with SYMBOLIC explicitly).

    Returns:
        QueryShape.SYMBOLIC if identifier-shape tokens dominate and no
            natural-language signal,
        QueryShape.PROSE if NL words present and no identifiers,
        QueryShape.MIXED if both,
        QueryShape.UNKNOWN if too short / ambiguous to classify.
    """
    q = (query or "").strip()
    if not q or len(q) < 5:
        return QueryShape.UNKNOWN
    has_identifier = bool(_IDENTIFIER_RE.search(q))
    has_natural = bool(_NL_SIGNAL_RE.search(q))
    if has_identifier and not has_natural:
        return QueryShape.SYMBOLIC
    if has_natural and not has_identifier:
        return QueryShape.PROSE
    if has_identifier and has_natural:
        return QueryShape.MIXED
    # No identifier, no NL function words. Likely a bag-of-keywords
    # query — treat as PROSE if it has 4+ words, else UNKNOWN.
    if len(q.split()) >= 4:
        return QueryShape.PROSE
    return QueryShape.UNKNOWN


__all__ = [
    "QueryShape",
    "RetrievalPolicy",
    "select_policy",
    "detect_query_shape",
    "SHORT",
    "MID",
    "LONG",
    "EXTRA_LONG",
]
