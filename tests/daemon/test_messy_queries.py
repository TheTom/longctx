"""Messy-query observation suite for Phase 2.0.

These tests don't ASSERT optimal behavior — they DOCUMENT current
behavior on hostile inputs so we can see exactly what Phase 2.0
does, then design the 2.0.1 messy-query handling against real
observed failure modes instead of imagined ones.

Each test plants known answers in a corpus, throws a hostile query
at the searcher, and records:
  - did the right chunk surface in top-K?
  - what was its rank?
  - what was the top-1 score (for relevance-floor calibration)?

Failures are EXPECTED for some of these. The point is the data,
not the green checks. The few hard asserts are basic
"didn't-crash" guards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from longctx.rag.chunker import Chunker
from longctx_daemon.indexer import Indexer, IndexerConfig
from longctx_daemon.searcher import Searcher, SearcherConfig
from longctx_daemon.storage.memmap_store import MemmapEmbedStore
from longctx_daemon.storage.sqlite_store import SqliteChunkStore


# --------------------------------------------------------------- fixtures

class _DeterministicEmbedder:
    """Same shape as the integration-test embedder — keyword-keyed
    4-dim vectors so we can hand-reason about cosine outcomes."""

    fake_sha = "0" * 64
    model_name = "deterministic-test-embedder"
    dim = 4

    @property
    def max_seq_length(self) -> int:
        return 128

    @max_seq_length.setter
    def max_seq_length(self, _v: int) -> None:
        pass

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, convert_to_numpy=True,
               normalize_embeddings=True, **_) -> np.ndarray:
        out = []
        for t in texts:
            tl = t.lower()
            v = np.array([
                1.0 if "needle" in tl else 0.0,
                1.0 if "auth" in tl else 0.0,
                1.0 if "billing" in tl else 0.0,
                0.5,
            ], dtype=np.float32)
            n = float(np.linalg.norm(v)) + 1e-8
            out.append(v / n)
        return np.stack(out)


def _build_planted_corpus(tmp_path: Path) -> Path:
    """Tiny corpus with 3 well-known files: an auth file, a billing
    file, and a notes doc with a planted needle."""
    proj = tmp_path / "myapp"
    proj.mkdir()
    (proj / ".git").mkdir()
    (proj / "src").mkdir()
    (proj / "src/auth.py").write_text(
        "def auth_middleware(token: str) -> bool:\n"
        "    \"\"\"Validates session tokens.\"\"\"\n"
        "    return token != ''\n"
    )
    (proj / "src/billing.py").write_text(
        "def charge_card(card: str, amount: int) -> bool:\n"
        "    \"\"\"Charges a card via the billing service.\"\"\"\n"
        "    return amount > 0\n"
    )
    (proj / "docs").mkdir()
    (proj / "docs/notes.md").write_text(
        "# Engineering notes\n\n"
        "The secret needle phrase is planted here so retrieval tests "
        "can find it. nothing else in this corpus mentions needle.\n"
    )
    return proj


@pytest.fixture
def stack(tmp_path: Path):
    chunk_store = SqliteChunkStore(tmp_path / "index.db")
    embedder = _DeterministicEmbedder()
    embed_store = MemmapEmbedStore(
        tmp_path / "embeddings",
        model_name=embedder.model_name,
        model_sha256=embedder.fake_sha,
        dim=embedder.dim,
    )
    chunker = Chunker(tokens_per_chunk=64, respect_sentences=False)
    indexer = Indexer(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, chunker=chunker,
        config=IndexerConfig(),
    )
    searcher = Searcher(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, config=SearcherConfig(),
    )
    yield {
        "chunk_store": chunk_store, "embed_store": embed_store,
        "indexer": indexer, "searcher": searcher,
    }
    chunk_store.close()
    embed_store.close()


def _index(stack: dict[str, Any], proj: Path) -> None:
    stack["indexer"].add_project("myapp", proj)
    stack["indexer"].full_scan("myapp")


def _rank_of(result, predicate) -> int | None:
    """Find the first chunk satisfying ``predicate``; return its 1-indexed
    rank or None if absent from the result."""
    for i, c in enumerate(result.chunks, start=1):
        if predicate(c):
            return i
    return None


def _has_needle(c) -> bool:
    return "needle" in c.text.lower() or "notes.md" in c.citation.file_path


# ============================================================ multi-question

def test_OBSERVE_multi_question_concatenated(stack, tmp_path):
    """User pastes 3 distinct questions in one query string. Document
    whether the planted ``needle`` chunk still surfaces."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    query = (
        "how many times does ls appear in this dir "
        "where does the function taco_libre appear "
        "what is the capital of france "
        "and where is the secret needle hidden"
    )
    result = stack["searcher"].search(query=query, cwd=str(proj))
    rank = _rank_of(result, _has_needle)
    print(
        f"\n[OBSERVE multi-question] needle rank: {rank} / "
        f"{len(result.chunks)} returned (top score: "
        f"{result.chunks[0].relevance_score:.4f} if any)"
    )
    # Hard asserts: doesn't crash, returns SOMETHING.
    assert isinstance(result.chunks, tuple)


def test_OBSERVE_multi_question_no_punctuation(stack, tmp_path):
    """Run-on style — no question marks, no periods between questions."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    query = (
        "how many ls in dir and where does taco_libre appear "
        "and france capital and find me the needle"
    )
    result = stack["searcher"].search(query=query, cwd=str(proj))
    rank = _rank_of(result, _has_needle)
    print(f"\n[OBSERVE multi-question no-punc] needle rank: {rank}")
    assert isinstance(result.chunks, tuple)


# ============================================================ pasted blob

def test_OBSERVE_pasted_stack_trace_with_question(stack, tmp_path):
    """User pastes a fake stack trace + asks where it comes from. The
    blob's tokens dominate BM25; the question's signal dilutes."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    blob = "\n".join([
        "Traceback (most recent call last):",
        "  File '/usr/local/lib/python3.14/foo/bar.py', line 42, in baz",
        "    raise RuntimeError('thing went wrong')",
        "  File 'other/path.py', line 99, in qux",
        "    process(buf, mode='rw')",
    ] * 5)
    query = blob + "\n\nwhere does the secret needle appear in our codebase?"

    result = stack["searcher"].search(query=query, cwd=str(proj))
    rank = _rank_of(result, _has_needle)
    print(f"\n[OBSERVE pasted-blob+question] needle rank: {rank}")
    assert isinstance(result.chunks, tuple)


def test_OBSERVE_pasted_blob_only_no_question(stack, tmp_path):
    """User pastes code without an explicit question — agent expects
    'find similar' behavior. No question marker; the corpus has no
    matching code, so the result is what we get."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    pasted = (
        "def auth_middleware(token: str) -> bool:\n"
        "    return token != ''\n"
    )
    result = stack["searcher"].search(query=pasted, cwd=str(proj))
    print(
        f"\n[OBSERVE pasted-blob only] top-1 file: "
        f"{result.chunks[0].citation.file_path if result.chunks else 'none'} "
        f"score={result.chunks[0].relevance_score if result.chunks else 0:.4f}"
    )
    # The user pasted auth_middleware code — the auth.py file should
    # plausibly rank top-1 in a 'find similar' context.
    assert isinstance(result.chunks, tuple)


# ============================================================ grammar / typos

def test_OBSERVE_fragments_no_grammar(stack, tmp_path):
    """Keyword-soup query — no grammar, just bag of words."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    result = stack["searcher"].search(
        query="needle find secret hidden where",
        cwd=str(proj),
    )
    rank = _rank_of(result, _has_needle)
    print(f"\n[OBSERVE fragments] needle rank: {rank}")
    assert isinstance(result.chunks, tuple)


def test_OBSERVE_typo_in_keyword(stack, tmp_path):
    """Single-character typo: 'neddle' instead of 'needle'. BM25 won't
    match (no exact term hit); dense embedder might still surface it
    via subword similarity (or might not)."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    result = stack["searcher"].search(
        query="where is the secret neddle hidden",
        cwd=str(proj),
    )
    rank = _rank_of(result, _has_needle)
    print(f"\n[OBSERVE typo] needle rank: {rank}")
    assert isinstance(result.chunks, tuple)


# ============================================================ off-corpus

def test_OBSERVE_off_corpus_question(stack, tmp_path):
    """The classic 'capital of france' that has nothing to do with
    the codebase. We want to see what scores come back so we can
    calibrate the relevance floor for 2.0.1."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    result = stack["searcher"].search(
        query="what is the capital of france",
        cwd=str(proj),
    )
    top_score = result.chunks[0].relevance_score if result.chunks else 0.0
    print(
        f"\n[OBSERVE off-corpus] returned {len(result.chunks)} chunks, "
        f"top score {top_score:.4f} — calibrate relevance_floor near this"
    )
    # The honest API: today, longctx returns chunks anyway. The 2.0.1
    # work will surface no_relevant_results=True when top_score is low.
    assert isinstance(result.chunks, tuple)


# ============================================================ summary

def test_OBSERVE_messy_query_summary(stack, tmp_path, capsys):
    """One row per messy case in a single readable table — what
    Phase 2.0 actually does on each input shape."""
    proj = _build_planted_corpus(tmp_path)
    _index(stack, proj)

    cases = [
        ("clean (control)", "where is the secret needle hidden"),
        ("multi-question", "where does ls appear and where is "
                            "the secret needle and capital of france"),
        ("run-on no-punc", "where does ls appear and where needle "
                            "and france capital"),
        ("blob+question", "Traceback ...\n  File 'a.py', line 1\n    raise X\n"
                          "where is the secret needle"),
        ("blob only", "def auth_middleware(token): return token != ''"),
        ("fragments", "needle find secret hidden"),
        ("typo", "where is the secret neddle"),
        ("off-corpus", "what is the capital of france"),
    ]

    rows = []
    for label, q in cases:
        r = stack["searcher"].search(query=q, cwd=str(proj))
        needle_rank = _rank_of(r, _has_needle)
        top_score = r.chunks[0].relevance_score if r.chunks else 0.0
        rows.append((label, needle_rank, top_score, len(r.chunks)))

    # Pretty-print to stdout for human inspection. Surfaces what the
    # current pipeline ACTUALLY does on each shape so we can decide
    # what 2.0.1's messy-query handling needs to address.
    with capsys.disabled():
        print()
        print(f"  {'case':<20s} {'needle_rank':>12s} {'top_score':>12s} "
              f"{'returned':>10s}")
        print("  " + "-" * 60)
        for label, rank, score, n in rows:
            rank_s = str(rank) if rank is not None else "MISS"
            print(f"  {label:<20s} {rank_s:>12s} {score:>12.4f} {n:>10d}")
    assert isinstance(rows, list)
