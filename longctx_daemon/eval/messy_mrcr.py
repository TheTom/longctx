"""Messy MRCR — robustness × context-length retrieval benchmark.

This module composes two axes the literature usually conflates:

1. **Context length** (the MRCR axis): how does retrieval recall degrade
   as the haystack grows? OpenAI's MRCR v2 8-needle samples span 8K to
   1M tokens of conversation transcript with 8 plausibly-similar
   "needle" assistant messages, and the model is asked to reproduce
   the Nth one verbatim, prefixed with a random string. The clean
   query is precise: "Please write me the 5th poem about a tabby
   cat verbatim, prepended with QXR-".

2. **Query messiness** (the messy-queries axis): how does retrieval
   degrade when the user's question is malformed — typos, run-on
   prose, multi-question dumps, code-blob-prefixed, fragment-only,
   or off-corpus entirely? The messy-query rig in
   ``messy_queries_real.py`` proved this matters at small scale on
   a synthetic corpus.

Messy MRCR crosses these two axes. Per (transform, context_bin) cell
we compute Recall@K — did the chunk containing the gold answer
(``random_string_to_prepend + answer``) land in top-K?

The result is a matrix the long-context retrieval community can use
to design honestly: short-context-tuned heuristics may collapse at
1M; messiness penalties may grow super-linearly with haystack size.
Nobody has published this measurement before — that's the point.

## Why retrieval-only

We deliberately do NOT call a generator here. The MRCR generation
metric (prefix-gated SequenceMatcher) conflates retrieval failure
with generation failure. By scoring **only** whether the right
chunk surfaces, this benchmark isolates the retrieval-robustness
question. Generator integration is a separate axis.

## The transformations

Each transform is a pure function ``(query: str, seed: int) -> str``.
Document each algorithm in :func:`_transform_*` so others can
reproduce them exactly.

1. **clean** — control. Identity. Returns input unchanged.
2. **typo** — adjacent-letter swap in 3-5% of words; trailing ``?``
   dropped. Char-level perturbation with low Levenshtein distance
   to the clean form.
3. **multi-question** — prepend 2-3 unrelated questions before the
   real query. Tests dilution of intent across multiple topics.
4. **run-on** — strip all punctuation, lowercase, collapse
   whitespace. Tests robustness to unstructured natural prose.
5. **fragments** — extract content tokens (drop stopwords,
   articles, modal verbs, punctuation). Tests robustness to
   keyword-only queries.
6. **blob+question** — prepend a fake stack-trace before the real
   query. Tests dilution from leading non-query text.
7. **off-corpus** — replace with a sanity-check abstention query
   (e.g. "what is the capital of france"). Should always trigger
   ``no_relevant_results=True``; the row serves as a calibration
   line for the abstention path.

## CLI

::

    python3 -m longctx_daemon.eval.messy_mrcr --needles 8 --samples 20

Common flags:

* ``--bins 8K,32K,64K,128K,256K`` — context bins to scan. Add ``1M``
  explicitly only when you mean it; that bin is opt-in because each
  cell can take 30+ minutes at scale.
* ``--transforms clean,typo,run-on`` — restrict to a subset.
* ``--samples N`` — samples per cell. Default 20 for fast turn-arounds;
  100 for headline numbers.
* ``--json`` — emit structured JSON instead of the markdown table.

Outputs:

* ``benchmark/messy_mrcr/RESULTS.md`` — markdown writeup with the
  matrix + methodology. Postable as a thread.
* ``benchmark/messy_mrcr/messy_mrcr_<date>.json`` — raw per-sample
  records for replay / re-scoring.

## Reproducibility

All transformations are seeded; the per-sample seed is the MRCR
sample index. Re-running with the same ``--seed`` should produce
the same transformed queries, so re-running mostly only varies the
recall numbers (which depend on the embedder + corpus mix).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


# ─────────────────────────────────────────── bin definitions

# Bin boundaries are MRCR's published splits. Tokens are estimated via
# the standard 4-chars/token heuristic; the actual filter uses
# ``n_chars`` from the parquet metadata so we don't need a tokenizer
# server here. ``(min_chars, max_chars)`` ranges. The 1M bin is opt-in
# because each cell can take 30+ minutes — running the full matrix at
# 1M is a deliberate decision, not the default.
BIN_RANGES_CHAR: dict[str, tuple[int, int]] = {
    "8K": (16_000, 32_000),
    "32K": (64_000, 128_000),
    "64K": (128_000, 256_000),
    "128K": (256_000, 512_000),
    "256K": (512_000, 1_000_000),
    "1M": (2_000_000, 5_000_000),
}

# Default bin order presented in the matrix. 1M is intentionally OUT —
# callers must opt in by passing ``--bins ...,1M`` explicitly.
DEFAULT_BINS: tuple[str, ...] = ("8K", "32K", "64K", "128K", "256K")

# 30-query abstention pool. These are off-corpus questions a code/
# conversation retrieval system has no business answering. Picked by
# ``seed % len(pool)`` so the off-corpus row is deterministic per
# seed but covers a range of topics across cells.
ABSTENTION_POOL: tuple[str, ...] = (
    "what is the capital of france",
    "how do i make sourdough bread",
    "who wrote pride and prejudice",
    "what year did the berlin wall fall",
    "how tall is mount everest in meters",
    "what are the rules of cricket",
    "explain the difference between html and css",
    "how do i tie a windsor knot",
    "what is photosynthesis",
    "list the colors of the rainbow",
    "what causes the northern lights",
    "how does a nuclear reactor work",
    "what is the population of tokyo",
    "who invented the telephone",
    "what is the boiling point of water in celsius",
    "describe the plot of hamlet",
    "what languages are spoken in switzerland",
    "how do i bake a chocolate cake",
    "what is the speed of light",
    "name the planets in our solar system",
    "what is the meaning of carpe diem",
    "how does compound interest work",
    "who painted the mona lisa",
    "what are the symptoms of the common cold",
    "explain the theory of relativity",
    "how do you say hello in japanese",
    "what is the largest ocean on earth",
    "who was the first person on the moon",
    "what is the chemical formula for water",
    "list five mediterranean countries",
)


# ─────────────────────────────────────────── transform algorithms

# Stopword set used by the ``fragments`` transform. Conservative — we
# strip articles, common modal verbs, prepositions, and the literal
# words that wrap the typical MRCR query frame ("write me the ...
# verbatim, prepended with ..."). The goal is to leave only content
# tokens. Tuned so the standard MRCR query "please write me the 5th
# poem about a tabby cat verbatim, prepended with QXR-" reduces to
# something like "5th poem tabby cat verbatim QXR-".
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "should", "could", "may",
    "might", "must", "can", "shall",
    # MRCR query-frame words. Drop them so the noun phrases pop.
    "please", "write", "me", "give", "show", "tell", "find",
    "verbatim", "prepended", "this",  # NOTE: keep "verbatim" out of
    # _STOPWORDS for ``fragments`` to keep the verbatim-anchor token.
    "that", "those", "these",
})

# Re-add ``verbatim`` because it's a useful retrieval anchor for
# MRCR (the assistant-message text often doesn't repeat it, but the
# user query does and BM25 picks it up cheaply).
_STOPWORDS = _STOPWORDS - {"verbatim"}


def _transform_clean(query: str, seed: int) -> str:
    """Identity transform. The CONTROL row.

    Returns ``query`` unchanged. Critical that this is a true no-op
    so the control column is unambiguous; do not even strip
    whitespace here.
    """
    return query


def _transform_typo(query: str, seed: int) -> str:
    """Swap two adjacent letters in 3-5% of words; drop trailing ``?``.

    Algorithm (deterministic given ``seed``):

      1. Tokenize on whitespace (preserve punctuation in tokens).
      2. Pick a swap rate in [0.03, 0.05] using ``random.Random(seed)``.
      3. For each eligible word (≥4 alphabetic chars), with probability
         equal to that rate, pick an interior position ``i`` and swap
         ``word[i]`` with ``word[i+1]``.
      4. Strip trailing ``?`` from the joined string.

    The 4-char-min protects short tokens (which a typo would mangle
    beyond bge-small's char-bigram robustness). The interior-only
    swap protects the leading/trailing characters which are
    disproportionately important for retrieval.
    """
    rng = random.Random(seed)
    rate = rng.uniform(0.03, 0.05)
    words = query.split()
    out_words: list[str] = []
    for w in words:
        # Need at least 4 chars: positions [1..len-3] for i, swapping
        # i with i+1, keeps both first AND last chars in place.
        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            # i ∈ [1, len-3] inclusive so i+1 ≤ len-2; both first and
            # last char are protected (they're disproportionately
            # important for retrieval / first-letter indexing).
            i = rng.randint(1, len(w) - 3)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out_words.append(w)
    out = " ".join(out_words)
    # Drop the question mark — typos correlate with sloppy
    # punctuation, and MRCR clean queries usually end without one
    # already, so this is a small additional perturbation, not a
    # dominant one.
    if out.endswith("?"):
        out = out[:-1]
    return out


# A pool of unrelated questions for the multi-question transform.
# Picked so they don't accidentally hit the MRCR corpus surface.
_DISTRACTOR_QUESTIONS: tuple[str, ...] = (
    "what is the capital of france",
    "how does python list comprehension work",
    "what is the difference between tcp and udp",
    "explain how decision trees split nodes",
    "what is dependency injection",
    "how do i deploy a docker container",
    "what is the time complexity of quicksort",
    "describe how http cookies work",
    "what is functional programming",
    "explain how a mutex differs from a semaphore",
)


def _transform_multi_question(query: str, seed: int) -> str:
    """Concatenate 2-3 distractor questions before the real query.

    Algorithm (deterministic given ``seed``):

      1. ``rng = random.Random(seed)``.
      2. ``k = rng.choice([2, 3])`` — 2 or 3 distractors.
      3. Sample ``k`` questions from ``_DISTRACTOR_QUESTIONS`` without
         replacement.
      4. Join distractors with ``". "`` and append ``". " + query``.

    The distractors come BEFORE the real query so that lexical
    rankers see them as part of the query's BoW; semantic rankers
    have to disentangle the topic. Empirically, this is the most
    common real-world failure mode (user pastes their actual
    question after a paragraph of context).
    """
    rng = random.Random(seed)
    k = rng.choice([2, 3])
    distractors = rng.sample(_DISTRACTOR_QUESTIONS, k=k)
    return ". ".join(distractors) + ". " + query


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _transform_run_on(query: str, seed: int) -> str:
    """Strip all punctuation, lowercase, collapse whitespace.

    Algorithm (deterministic; ``seed`` unused but kept for signature
    consistency):

      1. Lowercase the query.
      2. Replace any non-word, non-whitespace char with a single
         space.
      3. Collapse runs of whitespace to single spaces.
      4. Strip leading/trailing whitespace.

    This is the "voice transcript with no punctuation" mode. Run-on
    prose loses sentence-boundary signal, so retrieval has to lean
    harder on bag-of-words match.
    """
    s = query.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _transform_fragments(query: str, seed: int) -> str:
    """Reduce to content noun phrases by dropping stopwords + punctuation.

    Algorithm (deterministic; ``seed`` unused):

      1. Run :func:`_transform_run_on` to lowercase + strip
         punctuation.
      2. Tokenize on whitespace.
      3. Drop tokens in :data:`_STOPWORDS`.
      4. Re-join with spaces.

    The output is keyword-only — no syntax, no question structure.
    Tests how well the embedder can match on bag-of-content alone.
    """
    cleaned = _transform_run_on(query, seed)
    tokens = [t for t in cleaned.split() if t not in _STOPWORDS]
    return " ".join(tokens)


_FAKE_STACK_TRACE = (
    "Traceback (most recent call last):\n"
    "  File '/usr/lib/python3.13/foo/bar.py', line 142, in baz\n"
    "    raise RuntimeError('thing went wrong')\n"
    "  File 'other/path.py', line 99, in qux\n"
    "    process(buf, mode='rw')\n"
    "  File 'lib/handler.py', line 27, in handle\n"
    "    return self._dispatch(req)\n"
)


def _transform_blob_question(query: str, seed: int) -> str:
    """Prepend a fake stack-trace blob before the real query.

    Algorithm (deterministic; ``seed`` unused):

      1. Concatenate :data:`_FAKE_STACK_TRACE` and ``query``,
         separated by ``"\\n\\n"``.

    The blob is constant across samples by design — varying it adds
    noise without changing the dilution mechanism we're trying to
    measure. This tests the case where the user pastes an error
    message before asking their actual question.
    """
    return _FAKE_STACK_TRACE + "\n\n" + query


def _transform_off_corpus(query: str, seed: int) -> str:
    """Replace the real query with an abstention question.

    Algorithm (deterministic given ``seed``):

      1. ``i = seed % len(ABSTENTION_POOL)``.
      2. Return ``ABSTENTION_POOL[i]`` unchanged.

    Used as the calibration row: the off-corpus query has zero
    business hitting any conversation needle, so its recall should
    be near-zero across all bins. If it isn't, the threshold is
    miscalibrated and the abstention path is broken.
    """
    return ABSTENTION_POOL[seed % len(ABSTENTION_POOL)]


# ─────────────────────────────────────────── transform registry

@dataclass(frozen=True)
class MessyTransform:
    """One messiness transform. Pure function over a query string."""
    name: str
    transform: Callable[[str, int], str]
    description: str


TRANSFORMS: dict[str, MessyTransform] = {
    "clean": MessyTransform(
        name="clean",
        transform=_transform_clean,
        description="control: identity transform; the unmodified query.",
    ),
    "typo": MessyTransform(
        name="typo",
        transform=_transform_typo,
        description="adjacent-letter swaps in 3-5% of words; drops trailing ?",
    ),
    "multi-question": MessyTransform(
        name="multi-question",
        transform=_transform_multi_question,
        description="prepend 2-3 unrelated questions before the real query.",
    ),
    "run-on": MessyTransform(
        name="run-on",
        transform=_transform_run_on,
        description="strip punctuation, lowercase, collapse whitespace.",
    ),
    "fragments": MessyTransform(
        name="fragments",
        transform=_transform_fragments,
        description="content tokens only: drop stopwords, drop syntax.",
    ),
    "blob+question": MessyTransform(
        name="blob+question",
        transform=_transform_blob_question,
        description="prepend a fake stack-trace blob before the real query.",
    ),
    "off-corpus": MessyTransform(
        name="off-corpus",
        transform=_transform_off_corpus,
        description="replace with an abstention query (sanity-check row).",
    ),
}


def apply_transform(transform_name: str, query: str, seed: int) -> str:
    """Look up ``transform_name`` in :data:`TRANSFORMS` and apply it."""
    if transform_name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform {transform_name!r}. "
            f"Known: {sorted(TRANSFORMS)}"
        )
    return TRANSFORMS[transform_name].transform(query, seed)


# ─────────────────────────────────────────── result types

@dataclass
class MessyMRCRSample:
    """One per-cell sample record; one row per (sample, transform).

    Stored verbatim in the JSON output for replay / re-scoring.
    """
    transform: str
    bin: str
    sample_idx: int
    n_chars: int
    n_needles: int
    clean_query: str
    transformed_query: str
    gold_chunk_idx: int
    top_k_chunk_indices: tuple[int, ...]
    top_k_scores: tuple[float, ...]
    hit: bool
    no_relevant_results: bool
    top1_dense_cosine: float
    confidence_gap: float
    query_type: str


@dataclass
class MessyMRCRReport:
    """Top-level report shape — matrix + per-sample rows + metadata."""
    transforms: tuple[str, ...]
    bins: tuple[str, ...]
    n_samples_per_cell: int
    needles: int
    embedder_id: str
    seed: int
    matrix: dict[tuple[str, str], float]
    per_sample: list[MessyMRCRSample]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────── MRCR loader

NEEDLE_SHARDS = {
    2: "2needle/2needle_0.parquet",
    4: "4needle/4needle_0.parquet",
    8: "8needle/8needle_0.parquet",
}
DATASET_REPO = "openai/mrcr"


def _load_mrcr_samples_for_bin(
    needles: int,
    bin_name: str,
    n_samples: int,
    seed: int,
) -> list[dict]:
    """Stream MRCR samples whose ``n_chars`` falls in the bin range.

    Returns up to ``n_samples`` records with shape::

        {
            "messages": list[dict],       # full conversation
            "answer": str,                # gold assistant message text
            "random_string_to_prepend": str,
            "n_needles": int,
            "n_chars": int,
            "final_user_query": str,      # the clean query we transform
        }

    The MRCR ``answer`` field already includes the random prefix; we
    surface it separately because retrieval scoring keys on the
    chunk that contains the gold text, not on whether the prefix
    appears in the response (that's a generation metric).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Messy MRCR needs `datasets`. Install with: pip install datasets"
        ) from e

    if needles not in NEEDLE_SHARDS:
        raise ValueError(f"Unsupported needle count {needles}")
    if bin_name not in BIN_RANGES_CHAR:
        raise ValueError(f"Unknown bin {bin_name!r}")
    lo, hi = BIN_RANGES_CHAR[bin_name]

    ds = load_dataset(
        DATASET_REPO,
        data_files=NEEDLE_SHARDS[needles],
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=64)

    out: list[dict] = []
    for row in ds:
        nc = int(row.get("n_chars", 0))
        if not (lo <= nc < hi):
            continue
        prompt = row["prompt"]
        messages = json.loads(prompt) if isinstance(prompt, str) else list(prompt)
        # Final user message is the query we transform.
        final_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if not final_user:
            continue
        out.append({
            "messages": messages,
            "answer": row["answer"],
            "random_string_to_prepend": row["random_string_to_prepend"],
            "n_needles": int(row["n_needles"]),
            "n_chars": nc,
            "final_user_query": final_user,
        })
        if len(out) >= n_samples:
            break
    return out


# ─────────────────────────────────────────── per-sample retrieval

def _build_corpus_for_sample(root: Path, sample: dict) -> tuple[Path, int]:
    """Write each conversation message to disk as its own file.

    Each assistant message becomes one file ``msg_NNNN_assistant.txt``;
    each user message becomes ``msg_NNNN_user.txt``. The files are
    numbered sequentially so we can locate the gold message by its
    position in the conversation.

    Returns
    -------
    (corpus_root, gold_msg_index): the project root path and the
    1-indexed message position of the gold assistant message
    (the first assistant message whose content equals
    ``sample["answer"]``). The chunk that contains this message
    will be flagged as the gold chunk by :func:`_is_gold_chunk`.
    """
    corpus = root / "mrcr"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / ".git").mkdir(exist_ok=True)
    gold_index = -1
    answer = sample["answer"]
    for i, msg in enumerate(sample["messages"]):
        role = msg["role"]
        content = msg["content"]
        fname = f"msg_{i:04d}_{role}.txt"
        (corpus / fname).write_text(content)
        if gold_index == -1 and role == "assistant" and content == answer:
            gold_index = i
    return corpus, gold_index


def _is_gold_chunk(chunk_text: str, gold_text: str, prefix: str) -> bool:
    """True iff ``chunk_text`` contains the gold answer.

    Two acceptance criteria (gold text in MRCR is up to thousands of
    chars, often longer than one chunk):

      1. The chunk contains the random prefix (rare in the corpus
         elsewhere, so a strong signal).
      2. The chunk contains the first 200 chars of the gold answer
         (sufficient for the chunker's typical 64-256 token chunks).

    Either suffices. We don't require BOTH because chunk boundaries
    can split the prefix off from the answer body.
    """
    if prefix and prefix in chunk_text:
        return True
    head = gold_text[:200]
    return bool(head) and head in chunk_text


def _score_one_sample(
    *,
    sample: dict,
    transform_name: str,
    transformed_query: str,
    bin_name: str,
    sample_idx: int,
    embedder,
    embed_dim: int,
    embedder_sha: str,
    embedder_id: str,
    chunk_store_factory,
    embed_store_factory,
    chunker,
    indexer_factory,
    searcher_factory,
    top_k: int,
) -> MessyMRCRSample:
    """Index one MRCR sample, run one search, return the row record.

    Each call gets its OWN chunk_store + embed_store (we use
    temp directories so cells are independent and we don't carry
    state across samples). The factories are passed in so tests
    can swap fakes in.
    """
    with tempfile.TemporaryDirectory(prefix="messy-mrcr-") as tmp:
        tmp_path = Path(tmp)
        corpus, gold_msg_idx = _build_corpus_for_sample(tmp_path, sample)
        cache = tmp_path / "cache"
        cache.mkdir()

        chunk_store = chunk_store_factory(cache)
        embed_store = embed_store_factory(
            cache, embedder_id, embedder_sha, embed_dim,
        )
        indexer = indexer_factory(chunk_store, embed_store, embedder, chunker)
        indexer.add_project("mrcr", corpus)
        indexer.full_scan("mrcr")

        searcher = searcher_factory(chunk_store, embed_store, embedder)

        result = searcher.search(query=transformed_query, cwd=str(corpus))

        # Find rank of gold chunk (1-indexed). For the off-corpus row
        # we expect this to be missing — the abstention path returns
        # empty chunks when the dense floor isn't met.
        gold_text = sample["answer"]
        prefix = sample["random_string_to_prepend"]
        hit_chunks = result.chunks[:top_k]

        # Map each top-K chunk back to which message file it came from
        # so the per-sample record carries useful debug info.
        msg_indices: list[int] = []
        for c in hit_chunks:
            fp = c.citation.file_path
            # File path is "mrcr/msg_NNNN_role.txt" — pull the NNNN.
            try:
                base = fp.split("/")[-1]
                idx = int(base.split("_")[1])
            except (IndexError, ValueError):
                idx = -1
            msg_indices.append(idx)

        hit = any(
            _is_gold_chunk(c.text, gold_text, prefix)
            for c in hit_chunks
        )

        chunk_store.close()
        embed_store.close()

        return MessyMRCRSample(
            transform=transform_name,
            bin=bin_name,
            sample_idx=sample_idx,
            n_chars=sample["n_chars"],
            n_needles=sample["n_needles"],
            clean_query=sample["final_user_query"][:500],
            transformed_query=transformed_query[:500],
            gold_chunk_idx=gold_msg_idx,
            top_k_chunk_indices=tuple(msg_indices),
            top_k_scores=tuple(c.relevance_score for c in hit_chunks),
            hit=hit,
            no_relevant_results=getattr(result, "no_relevant_results", False),
            top1_dense_cosine=getattr(result, "top1_dense_cosine", 0.0),
            confidence_gap=getattr(result, "confidence_gap", 0.0),
            query_type=getattr(result, "query_type", "natural_language"),
        )


# ─────────────────────────────────────────── runner

def _default_chunk_store_factory(cache: Path):
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore
    return SqliteChunkStore(cache / "index.db")


def _default_embed_store_factory(
    cache: Path, model_id: str, model_sha: str, dim: int,
):
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    return MemmapEmbedStore(
        cache / "embeddings",
        model_name=model_id, model_sha256=model_sha, dim=dim,
    )


def _default_indexer_factory(chunk_store, embed_store, embedder, chunker):
    from longctx_daemon.indexer import Indexer, IndexerConfig
    return Indexer(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, chunker=chunker,
        config=IndexerConfig(),
    )


def _default_searcher_factory(chunk_store, embed_store, embedder):
    from longctx_daemon.searcher import Searcher, SearcherConfig
    return Searcher(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, config=SearcherConfig(),
    )


def run_messy_mrcr(
    needles: int = 8,
    samples_per_cell: int = 20,
    bins: tuple[str, ...] = DEFAULT_BINS,
    transforms: tuple[str, ...] = tuple(TRANSFORMS),
    embedder_id: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    seed: int = 0,
    top_k: int = 5,
    *,
    sample_loader: Optional[Callable[[int, str, int, int], list[dict]]] = None,
    embedder_factory: Optional[Callable[[str, str], tuple]] = None,
    chunk_store_factory: Optional[Callable[[Path], object]] = None,
    embed_store_factory=None,
    indexer_factory=None,
    searcher_factory=None,
    progress: Callable[[str], None] = lambda _msg: None,
) -> MessyMRCRReport:
    """Run the full Messy MRCR matrix and return a :class:`MessyMRCRReport`.

    The factory parameters exist so tests can swap in fakes without
    standing up the full storage stack. In production, the defaults
    use the daemon's real ``SqliteChunkStore`` / ``MemmapEmbedStore``
    and a real ``SentenceTransformer`` embedder.

    Parameters
    ----------
    needles : int
        MRCR needle count (2, 4, or 8). 8 is hardest and the
        published-paper default.
    samples_per_cell : int
        How many MRCR samples per (transform, bin) cell.
    bins : tuple[str, ...]
        Context bins to scan, e.g. ``("8K", "32K", "64K")``.
    transforms : tuple[str, ...]
        Transform names to evaluate; subset of :data:`TRANSFORMS`.
    embedder_id : str
        SentenceTransformer model id.
    device : str
        Embedder device — ``"auto"`` (mps/cuda/cpu autodetect),
        ``"mps"``, ``"cuda"``, or ``"cpu"``.
    seed : int
        Top-level seed; per-sample seeds are derived as
        ``seed + sample_idx`` so re-runs are deterministic.
    top_k : int
        Cutoff for Recall@K.

    Returns
    -------
    MessyMRCRReport with the matrix, per-sample rows, and metadata.
    """
    # Resolve factories.
    sample_loader = sample_loader or _load_mrcr_samples_for_bin
    chunk_store_factory = chunk_store_factory or _default_chunk_store_factory
    embed_store_factory = embed_store_factory or _default_embed_store_factory
    indexer_factory = indexer_factory or _default_indexer_factory
    searcher_factory = searcher_factory or _default_searcher_factory

    # Instantiate the embedder once for the whole run; each cell
    # reuses it. The chunker is also stateless after construction.
    if embedder_factory is None:
        from sentence_transformers import SentenceTransformer

        from longctx.rag.pipeline import _resolve_device
        embedder = SentenceTransformer(
            embedder_id, device=_resolve_device(device),
        )
        embed_dim = (
            embedder.get_embedding_dimension()
            if hasattr(embedder, "get_embedding_dimension")
            else embedder.get_sentence_embedding_dimension()
        )
        from longctx_daemon.indexer import _embedder_sha256
        embedder_sha = _embedder_sha256(embedder)
    else:
        embedder, embed_dim, embedder_sha = embedder_factory(embedder_id, device)

    from longctx.rag.chunker import Chunker
    chunker = Chunker(tokens_per_chunk=128, respect_sentences=False)

    # Cache MRCR samples per bin so we re-use them across transforms
    # within one bin (so each transform sees the SAME conversations,
    # which is what we want — we're measuring the marginal effect of
    # the transform, not noise from sampling different conversations).
    samples_by_bin: dict[str, list[dict]] = {}
    for b in bins:
        progress(f"loading {samples_per_cell} samples for bin={b}")
        samples_by_bin[b] = sample_loader(
            needles, b, samples_per_cell, seed,
        )
        progress(f"  got {len(samples_by_bin[b])} samples for bin={b}")

    matrix: dict[tuple[str, str], float] = {}
    per_sample_rows: list[MessyMRCRSample] = []

    for transform_name in transforms:
        for b in bins:
            samples = samples_by_bin[b]
            if not samples:
                matrix[(transform_name, b)] = 0.0
                continue
            hits = 0
            for i, s in enumerate(samples):
                tq = apply_transform(
                    transform_name, s["final_user_query"], seed + i,
                )
                row = _score_one_sample(
                    sample=s,
                    transform_name=transform_name,
                    transformed_query=tq,
                    bin_name=b,
                    sample_idx=i,
                    embedder=embedder,
                    embed_dim=embed_dim,
                    embedder_sha=embedder_sha,
                    embedder_id=embedder_id,
                    chunk_store_factory=chunk_store_factory,
                    embed_store_factory=embed_store_factory,
                    chunker=chunker,
                    indexer_factory=indexer_factory,
                    searcher_factory=searcher_factory,
                    top_k=top_k,
                )
                per_sample_rows.append(row)
                if row.hit:
                    hits += 1
            recall = hits / len(samples)
            matrix[(transform_name, b)] = recall
            progress(
                f"  {transform_name:<14s} {b:>5s}  recall@{top_k}={recall:.3f} "
                f"({hits}/{len(samples)})"
            )

    return MessyMRCRReport(
        transforms=tuple(transforms),
        bins=tuple(bins),
        n_samples_per_cell=samples_per_cell,
        needles=needles,
        embedder_id=embedder_id,
        seed=seed,
        matrix=matrix,
        per_sample=per_sample_rows,
    )


# ─────────────────────────────────────────── rendering

def render_matrix(report: MessyMRCRReport, top_k: int = 5) -> str:
    """Render the matrix as a markdown table — rows × bins.

    Cell value is recall@top_k formatted to 3 decimals; ``--`` for
    cells with no samples (typically a bin where MRCR has no
    in-range data).
    """
    name_w = max(len(t) for t in report.transforms)
    name_w = max(name_w, len("query_type"))
    head = "| " + "query_type".ljust(name_w) + " |"
    sep = "|" + "-" * (name_w + 2) + "|"
    for b in report.bins:
        head += f" {b:>6s} |"
        sep += "-" * 8 + "|"
    body = []
    for t in report.transforms:
        line = "| " + t.ljust(name_w) + " |"
        for b in report.bins:
            v = report.matrix.get((t, b))
            cell = "  --  " if v is None else f"{v:.3f}"
            line += f" {cell:>6s} |"
        body.append(line)
    return head + "\n" + sep + "\n" + "\n".join(body) + "\n"


def render_results_md(report: MessyMRCRReport, top_k: int = 5) -> str:
    """Build the full RESULTS.md writeup, methodology + matrix.

    Tom's voice: lowercase, direct, no em-dashes, "I" not "we".
    """
    matrix_table = render_matrix(report, top_k=top_k)
    lines = [
        f"# messy MRCR: robustness x context-length retrieval matrix",
        "",
        f"Generated: {report.timestamp}  ",
        f"Embedder: `{report.embedder_id}`  ",
        f"Needles: {report.needles}  ",
        f"Samples per cell: {report.n_samples_per_cell}  ",
        f"Top-K: {top_k}  ",
        f"Seed: {report.seed}",
        "",
        "## what this measures",
        "",
        "long-context retrieval has two axes that the literature usually conflates:",
        "",
        "1. **context length**: how does recall degrade as the haystack grows?",
        "2. **query messiness**: how does recall degrade when the user's",
        "   question is malformed (typos, run-on, multi-question, off-corpus)?",
        "",
        "messy MRCR crosses them. each cell of the matrix is recall@K for one",
        "(messiness, context-bin) pair. that's enough to see if the messiness",
        "penalty scales linearly with context, super-linearly, or saturates.",
        "",
        "## why it's novel",
        "",
        "MRCR measures retrieval recall at long context but with clean queries.",
        "messy-query work measures messiness penalty but on small corpora.",
        "nobody has crossed the axes. this matrix is the crossing.",
        "",
        "the off-corpus row also gives a calibration line. if my abstention",
        "path is honest, those cells should be near-zero across all bins.",
        "",
        "## method",
        "",
        f"* baseline: openai/mrcr {report.needles}-needle samples streamed",
        "  from huggingface, filtered into char-range bins.",
        "* corpus: each conversation message is written as its own file,",
        "  indexed via the longctx daemon's BM25 + dense + RRF stack.",
        f"* query: each MRCR sample's final user message is fed through one",
        "  of the messiness transforms.",
        f"* score: recall@{top_k}. did any chunk in the top-{top_k} retrieved",
        "  results contain the gold answer text (or its random-string prefix)?",
        f"* embedder: {report.embedder_id}, cosine on L2-normalized vectors.",
        "",
        "## algorithms",
        "",
    ]
    for name in report.transforms:
        if name in TRANSFORMS:
            t = TRANSFORMS[name]
            lines.append(f"* **{t.name}**: {t.description}")
    lines += [
        "",
        "see `longctx_daemon/eval/messy_mrcr.py` for the exact algorithm of",
        "each transform with line-by-line pseudocode.",
        "",
        "## results",
        "",
        matrix_table,
        "## reading guide",
        "",
        "* read the **clean** row left-to-right to see how recall degrades",
        "  with context length on perfectly-formed queries.",
        "* read each **messy** row against the clean row at the same column",
        "  to see how much that specific messiness costs at that context size.",
        "* read the **off-corpus** row across the board to confirm the",
        "  abstention path holds. those cells should be near-zero. if they're",
        "  not, the relevance floor is mis-calibrated for that bin.",
        "* compare deltas diagonally to spot compounding effects: does the",
        "  typo penalty at 256K eat more recall than at 8K?",
        "",
        "## caveats",
        "",
        f"* embedder-specific. all numbers are {report.embedder_id} on cosine.",
        "  bigger embedders (bge-large, e5-mistral) will move the absolute",
        "  numbers but the SHAPE of the matrix should hold.",
        "* retrieval-only. the generator is not in the loop. these are pure",
        "  retrieval recall numbers, not MRCR's published prefix-gated",
        "  SequenceMatcher score (which mixes retrieval and generation).",
        f"* sample size: {report.n_samples_per_cell} per cell. fast-iteration",
        "  default. 100/cell is the headline-numbers setting.",
        "* the 1M bin is opt-in via the CLI. each cell at 1M can take 30+",
        "  minutes; default bins stop at 256K.",
        "* MRCR's 8-needle split has no samples below ~32K char context, so",
        "  the 8K column is genuinely empty for the 8-needle matrix. use 2-",
        "  or 4-needle if you need that column populated.",
        "",
        "## reproduce",
        "",
        "```bash",
        f"python3 -m longctx_daemon.eval.messy_mrcr \\",
        f"    --needles {report.needles} \\",
        f"    --samples {report.n_samples_per_cell} \\",
        f"    --bins {','.join(report.bins)} \\",
        f"    --transforms {','.join(report.transforms)} \\",
        f"    --seed {report.seed}",
        "```",
        "",
        "raw per-sample records live alongside this file in",
        f"`messy_mrcr_<date>.json` for replay or alternate scoring.",
        "",
    ]
    return "\n".join(lines)


def report_to_dict(report: MessyMRCRReport) -> dict:
    """Make :class:`MessyMRCRReport` JSON-serialisable."""
    return {
        "timestamp": report.timestamp,
        "embedder_id": report.embedder_id,
        "needles": report.needles,
        "seed": report.seed,
        "transforms": list(report.transforms),
        "bins": list(report.bins),
        "n_samples_per_cell": report.n_samples_per_cell,
        "matrix": [
            {"transform": t, "bin": b, "recall": v}
            for (t, b), v in report.matrix.items()
        ],
        "per_sample": [asdict(r) for r in report.per_sample],
    }


# ─────────────────────────────────────────── CLI

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI args. Exposed for testing."""
    ap = argparse.ArgumentParser(
        description="Messy MRCR — robustness × context-length retrieval matrix",
    )
    ap.add_argument(
        "--needles", type=int, default=8, choices=[2, 4, 8],
        help="MRCR needle count. 8 = hardest = paper default.",
    )
    ap.add_argument(
        "--samples", type=int, default=20,
        help="Samples per (transform, bin) cell. 20 for fast runs, 100 for headline numbers.",
    )
    ap.add_argument(
        "--bins", type=str, default=",".join(DEFAULT_BINS),
        help="Comma-separated context bins. 1M is opt-in (slow).",
    )
    ap.add_argument(
        "--transforms", type=str, default=",".join(TRANSFORMS),
        help="Comma-separated messiness transform names.",
    )
    ap.add_argument(
        "--top-k", type=int, default=5,
        help="Recall@K cutoff.",
    )
    ap.add_argument(
        "--embedder", type=str, default="BAAI/bge-small-en-v1.5",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--json", action="store_true",
        help="Emit JSON to stdout instead of the markdown matrix.",
    )
    ap.add_argument(
        "--out-dir", type=str,
        default=str(Path("benchmark/messy_mrcr").resolve()),
        help="Directory to write RESULTS.md + json to.",
    )
    ap.add_argument(
        "--no-write", action="store_true",
        help="Skip writing RESULTS.md / json to disk; print only.",
    )
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    bins = tuple(b.strip() for b in args.bins.split(",") if b.strip())
    for b in bins:
        if b not in BIN_RANGES_CHAR:
            print(f"error: unknown bin {b!r}. known: {list(BIN_RANGES_CHAR)}",
                  file=sys.stderr)
            return 2

    transforms = tuple(t.strip() for t in args.transforms.split(",") if t.strip())
    for t in transforms:
        if t not in TRANSFORMS:
            print(f"error: unknown transform {t!r}. known: {list(TRANSFORMS)}",
                  file=sys.stderr)
            return 2

    def _progress(msg: str) -> None:
        if not args.quiet:
            print(f"[messy-mrcr] {msg}", file=sys.stderr)

    t0 = time.time()
    report = run_messy_mrcr(
        needles=args.needles,
        samples_per_cell=args.samples,
        bins=bins,
        transforms=transforms,
        embedder_id=args.embedder,
        device=args.device,
        seed=args.seed,
        top_k=args.top_k,
        progress=_progress,
    )
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(render_matrix(report, top_k=args.top_k))
        print(f"\n[messy-mrcr] done in {elapsed:.1f}s")

    if not args.no_write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        json_path = out / f"messy_mrcr_{date}.json"
        md_path = out / "RESULTS.md"
        json_path.write_text(json.dumps(report_to_dict(report), indent=2) + "\n")
        md_path.write_text(render_results_md(report, top_k=args.top_k))
        if not args.quiet:
            print(f"[messy-mrcr] wrote {md_path}", file=sys.stderr)
            print(f"[messy-mrcr] wrote {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
