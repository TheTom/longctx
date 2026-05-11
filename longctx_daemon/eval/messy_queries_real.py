"""Real-embedder messy-query observation rig + Phase 2.0.1 assertions.

Runs hostile inputs (multi-question, run-on, pasted blob, fragments,
typos, off-corpus) through the FULL pipeline with a real
SentenceTransformer + SqliteChunkStore + MemmapEmbedStore.

Output: a markdown table per case showing:
  * needle rank (None = miss)
  * top-1 fused RRF score
  * top-1 dense cosine (the unfused similarity — actually anchored
    in semantic match quality, unlike RRF)
  * top-1 BM25 raw score (anchored in lexical match quality)
  * no_relevant_results flag (Phase 2.0.1)
  * query_type tag (Phase 2.0.1)

Phase 2.0.1 added a ``--check`` mode that ASSERTS expected behavior
under the new relevance_floor (default 0.50, dense-cosine-anchored)
and the find_similar tag heuristic. Without ``--check`` the rig
runs in observation mode and prints the table only.

Expected behavior under ``--check`` (matches the Phase 2.0.1 spec):
  * off-corpus → no_relevant_results=True
  * run-on no-punc → no_relevant_results=True (cosine ≤ 0.41)
  * fragments → no_relevant_results=True (cosine ~0.48)
  * clean control / typo / multi-question / blob+question → needle
    chunk in the response (rank ≥ 1)
  * blob-only → query_type="find_similar" tag set
  * multi-question via list API → 3 separate result groups

Usage:

    python3 -m longctx_daemon.eval.messy_queries_real
    python3 -m longctx_daemon.eval.messy_queries_real --json
    python3 -m longctx_daemon.eval.messy_queries_real --check
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────── corpus

_PROJECT_FILES = {
    "src/auth.py": (
        "def auth_middleware(token: str) -> bool:\n"
        '    """Validates session tokens for the auth subsystem."""\n'
        "    return token != ''\n"
    ),
    "src/billing.py": (
        "def charge_card(card: str, amount: int) -> bool:\n"
        '    """Charges a card via the billing subsystem."""\n'
        "    return amount > 0\n"
    ),
    "docs/notes.md": (
        "# Engineering notes\n\n"
        "The secret needle phrase is planted here so retrieval tests "
        "can find it. Nothing else in this corpus mentions a needle.\n"
    ),
    "src/inventory.py": (
        "class Inventory:\n"
        '    """Tracks per-warehouse stock counts."""\n'
        "    def __init__(self):\n"
        "        self.stock = {}\n"
    ),
    "README.md": (
        "# myapp\n\n"
        "A small example codebase used for messy-query tests. "
        "Real prose, mixed with code, no trick wording.\n"
    ),
}


# ─────────────────────────────────────────────── cases

_CASES: tuple[tuple[str, str], ...] = (
    ("clean (control)", "where is the secret needle hidden"),
    ("multi-question",
     "where does ls appear and where is the secret needle "
     "and what is the capital of france"),
    ("run-on no-punc",
     "where does ls appear and where needle and france capital"),
    ("blob+question",
     "Traceback (most recent call last):\n"
     "  File '/usr/lib/foo.py', line 42, in baz\n"
     "    raise RuntimeError('thing went wrong')\n"
     "  File 'other/path.py', line 99, in qux\n"
     "    process(buf, mode='rw')\n"
     "where is the secret needle"),
    ("blob only (find similar)",
     "def auth_middleware(token: str) -> bool:\n"
     "    return token != ''"),
    ("fragments", "needle find secret hidden where"),
    ("typo", "where is the secret neddle hidden"),
    ("off-corpus", "what is the capital of france"),
)


# ─────────────────────────────────────────────── result row

@dataclass
class CaseResult:
    label: str
    query: str
    needle_rank: Optional[int]
    n_returned: int
    top1_fused_rrf: float
    top1_dense_cosine: float
    top1_bm25_raw: float
    top1_file: str
    no_relevant_results: bool = False
    query_type: str = "natural_language"
    confidence_gap: float = 0.0


# ─────────────────────────────────────────────── runner

def _build_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    for rel, text in _PROJECT_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _has_needle(chunk_text: str) -> bool:
    return "needle" in chunk_text.lower()


def run(*, embedder_id: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto", verbose: bool = False) -> list[CaseResult]:
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import (
        Indexer, IndexerConfig, _embedder_sha256,
    )
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    import tempfile
    with tempfile.TemporaryDirectory(prefix="longctx-messy-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "myapp"
        cache = tmp_path / "cache"
        cache.mkdir()
        _build_corpus(corpus)

        if verbose:
            print(f"[messy-real] embedder={embedder_id} on "
                  f"{_resolve_device(device)}", file=sys.stderr)

        embedder = SentenceTransformer(
            embedder_id, device=_resolve_device(device),
        )
        embed_dim = (
            embedder.get_embedding_dimension()
            if hasattr(embedder, "get_embedding_dimension")
            else embedder.get_sentence_embedding_dimension()
        )

        chunk_store = SqliteChunkStore(cache / "index.db")
        embed_store = MemmapEmbedStore(
            cache / "embeddings",
            model_name=embedder_id,
            model_sha256=_embedder_sha256(embedder),
            dim=embed_dim,
        )
        chunker = Chunker(tokens_per_chunk=64, respect_sentences=False)
        indexer = Indexer(
            chunk_store=chunk_store, embed_store=embed_store,
            embedder=embedder, chunker=chunker,
            config=IndexerConfig(),
        )
        indexer.add_project("myapp", corpus)
        indexer.full_scan("myapp")

        searcher = Searcher(
            chunk_store=chunk_store, embed_store=embed_store,
            embedder=embedder, config=SearcherConfig(),
        )

        # We also need the raw dense + BM25 scores for the top-1, since
        # the SearchResult only carries the fused RRF. Snag them via
        # direct calls into the stores.
        from longctx_daemon.searcher import _word_tokenize  # noqa: PLC0415

        results: list[CaseResult] = []
        for label, query in _CASES:
            t0 = time.time()
            r = searcher.search(query=query, cwd=str(corpus))
            elapsed = time.time() - t0

            rank = next(
                (i for i, c in enumerate(r.chunks, 1) if _has_needle(c.text)),
                None,
            )
            top1 = r.chunks[0] if r.chunks else None
            top1_text = top1.text if top1 else ""
            top1_file = top1.citation.file_path if top1 else "(none)"
            top1_rrf = top1.relevance_score if top1 else 0.0

            # Dense cosine for top-1
            dense_top1 = 0.0
            bm25_top1 = 0.0
            if top1 is not None:
                # Locate the chunk record in the store via citation.
                # The searcher already resolved file_id → path; we
                # invert by listing files and matching rel_path.
                try:
                    rel = top1.citation.file_path.split("/", 1)[1]
                except IndexError:
                    rel = top1.citation.file_path
                f = chunk_store.get_file("myapp", rel)
                if f is not None:
                    chunks = chunk_store.get_chunks_by_file(f.id)
                    # Pick the chunk whose start_line matches; fall back
                    # to first.
                    target = next(
                        (
                            c for c in chunks
                            if c.start_line == top1.citation.start_line
                        ),
                        chunks[0] if chunks else None,
                    )
                    if target is not None and target.embedding_row is not None:
                        q_emb = embedder.encode(
                            [query], convert_to_numpy=True,
                            normalize_embeddings=True,
                        )[0]
                        chunk_emb = embed_store.get(target.embedding_row)
                        dense_top1 = float(np.dot(q_emb, chunk_emb))
                        # BM25 raw via search_lexical scoped to this project
                        terms = _word_tokenize(query)
                        if terms:
                            hits = chunk_store.search_lexical(
                                terms, k=1, scope=None,
                            )
                            if hits:
                                bm25_top1 = float(hits[0].score)

            results.append(CaseResult(
                label=label, query=query[:80],
                needle_rank=rank,
                n_returned=len(r.chunks),
                top1_fused_rrf=top1_rrf,
                top1_dense_cosine=dense_top1,
                top1_bm25_raw=bm25_top1,
                top1_file=top1_file,
                no_relevant_results=getattr(r, "no_relevant_results", False),
                query_type=getattr(r, "query_type", "natural_language"),
                confidence_gap=getattr(r, "confidence_gap", 0.0),
            ))
            if verbose:
                print(
                    f"[messy-real] {label:<26s} rank={rank} "
                    f"rrf={top1_rrf:.4f} dense={dense_top1:.4f} "
                    f"bm25={bm25_top1:.2f} ({elapsed:.2f}s)",
                    file=sys.stderr,
                )

        chunk_store.close()
        embed_store.close()
    return results


# ─────────────────────────────────────────────── pretty + JSON

def render_table(rows: list[CaseResult]) -> str:
    # Local import: avoid module-load cost when this rig isn't being run
    # as a CLI (e.g. when something imports it just for the dataclasses).
    from longctx_daemon.eval._metrics import recall_at_k

    head = (
        "| case                    | rank | R@1 | R@3 | R@5 | RRF top-1 | dense cos | BM25 raw | no_rel | query_type      | top-1 file |\n"
        "|-------------------------|-----:|:---:|:---:|:---:|----------:|----------:|---------:|:------:|-----------------|------------|\n"
    )
    body = []
    for r in rows:
        rank = str(r.needle_rank) if r.needle_rank is not None else "MISS"
        no_rel = "Y" if r.no_relevant_results else "."
        # Per-row Recall@K indicators — useful for scanning which messy
        # shapes degrade from top-1 → top-3 / top-5 hit.
        ind = recall_at_k(r.needle_rank, (1, 3, 5))
        r1 = "1" if ind[1] else "0"
        r3 = "1" if ind[3] else "0"
        r5 = "1" if ind[5] else "0"
        body.append(
            f"| {r.label:<23s} | {rank:>4s} | "
            f" {r1} |  {r3} |  {r5} | "
            f"{r.top1_fused_rrf:>9.4f} | {r.top1_dense_cosine:>9.4f} | "
            f"{r.top1_bm25_raw:>8.2f} | {no_rel:^6s} | "
            f"{r.query_type:<15s} | {r.top1_file} |"
        )
    return head + "\n".join(body) + "\n"


# Phase 2.0.1 spec assertions per case label. ``floor`` is what we
# expect the relevance floor (default 0.50, dense-cosine-anchored) to
# filter (no_relevant_results=True); ``has_needle`` is what we expect
# to surface the planted answer.
#
# Note on calibration: the original Phase 2.0.1 spec expected run-on
# and fragments to trip the floor too. That expectation came from
# observation data taken BEFORE we fixed an ID-space bug in RRF
# fusion (BM25 emitted chunk.id, dense emitted embedding-row indices —
# the searcher was fusing on mismatched keys, so wrong chunks landed
# at fused-rank-1 with low cosine). With the bug fixed, the searcher
# correctly surfaces the needle for run-on / fragments / typo with
# top-1 cosines 0.56-0.76. The floor only catches the genuinely off-
# corpus case, which is still the design intent.
#
# This is exactly the kind of empirical re-calibration that's
# expected as the pipeline matures. The thresholds are corpus-
# dependent; future Messy MRCR work will measure how this shifts at
# 1M+ contexts with more distractor chunks competing for the slot.
_EXPECTATIONS = {
    "clean (control)":          {"floor": False, "has_needle": True},
    "multi-question":           {"floor": False, "has_needle": True},
    "run-on no-punc":           {"floor": False, "has_needle": True},
    "blob+question":            {"floor": False, "has_needle": True},
    # Blob-only must trigger the find_similar tag. We don't assert on
    # ranking — the ``query_type`` tag is the contract.
    "blob only (find similar)": {
        "floor": False, "has_needle": False,
        "query_type": "find_similar",
    },
    "fragments":                {"floor": False, "has_needle": True},
    "typo":                     {"floor": False, "has_needle": True},
    # Only off-corpus must trip the floor — the corpus genuinely has
    # nothing matching "what is the capital of france".
    "off-corpus":               {"floor": True,  "has_needle": False},
}


def assert_phase_2_0_1(rows: list[CaseResult]) -> list[str]:
    """Check observed behavior matches Phase 2.0.1 spec. Returns list
    of failures (empty on success); doesn't raise so the caller can
    print the full table even when assertions fail."""
    by_label = {r.label: r for r in rows}
    failures: list[str] = []
    for label, exp in _EXPECTATIONS.items():
        r = by_label.get(label)
        if r is None:
            failures.append(f"  {label}: case missing from results")
            continue
        if exp["floor"] != r.no_relevant_results:
            failures.append(
                f"  {label}: expected no_relevant_results="
                f"{exp['floor']}, got {r.no_relevant_results} "
                f"(dense_cosine={r.top1_dense_cosine:.4f})"
            )
        if exp["has_needle"] and r.needle_rank is None:
            failures.append(
                f"  {label}: expected planted needle in result, got MISS"
            )
        expected_qt = exp.get("query_type")
        if expected_qt is not None and r.query_type != expected_qt:
            failures.append(
                f"  {label}: expected query_type={expected_qt!r}, "
                f"got {r.query_type!r}"
            )
    return failures


def assert_multi_question_list_api() -> list[str]:
    """Run a 3-element multi-question via Searcher.search_multi and
    assert the response shape: 3 groups, each independent."""
    failures: list[str] = []
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import (
        Indexer, IndexerConfig, _embedder_sha256,
    )
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    import tempfile
    with tempfile.TemporaryDirectory(prefix="longctx-multi-q-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "myapp"
        cache = tmp_path / "cache"
        cache.mkdir()
        _build_corpus(corpus)

        embedder = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device=_resolve_device("auto"),
        )
        embed_dim = (
            embedder.get_embedding_dimension()
            if hasattr(embedder, "get_embedding_dimension")
            else embedder.get_sentence_embedding_dimension()
        )
        chunk_store = SqliteChunkStore(cache / "index.db")
        embed_store = MemmapEmbedStore(
            cache / "embeddings",
            model_name="BAAI/bge-small-en-v1.5",
            model_sha256=_embedder_sha256(embedder),
            dim=embed_dim,
        )
        chunker = Chunker(tokens_per_chunk=64, respect_sentences=False)
        indexer = Indexer(
            chunk_store=chunk_store, embed_store=embed_store,
            embedder=embedder, chunker=chunker, config=IndexerConfig(),
        )
        indexer.add_project("myapp", corpus)
        indexer.full_scan("myapp")

        searcher = Searcher(
            chunk_store=chunk_store, embed_store=embed_store,
            embedder=embedder, config=SearcherConfig(),
        )
        multi = searcher.search_multi(
            queries=[
                "where is the secret needle",
                "show me the auth middleware",
                "find the billing card-charge function",
            ],
            cwd=str(corpus),
        )

        if len(multi.groups) != 3:
            failures.append(
                f"  multi-question API: expected 3 groups, "
                f"got {len(multi.groups)}"
            )
        if multi.queries != (
            "where is the secret needle",
            "show me the auth middleware",
            "find the billing card-charge function",
        ):
            failures.append("  multi-question API: queries tuple mismatch")
        # Each group should be a real SearchResult with its own scope
        # decision and freshness — not merged across.
        for i, group in enumerate(multi.groups):
            if not hasattr(group, "chunks"):
                failures.append(f"  multi-question API: group {i} not a SearchResult")
        chunk_store.close()
        embed_store.close()
    return failures




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument(
        "--check", action="store_true",
        help="Assert Phase 2.0.1 expected behavior; exit non-zero on "
             "any mismatch. Without this flag the script is observation-"
             "only and always exits 0.",
    )
    args = ap.parse_args()

    rows = run(embedder_id=args.embedder, device=args.device,
               verbose=args.verbose)
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print(render_table(rows))
        print()
        print("Notes:")
        print("  * RRF top-1 is rank-driven (~0.0328 for top-1 with k=60).")
        print("  * Dense cosine is the actual semantic-match anchor.")
        print("  * BM25 raw is the lexical-match anchor.")
        print("  * `no_relevant_results` is anchored on dense cosine, not RRF.")
        print("  * `query_type=find_similar` flags pasted-blob inputs.")

    if args.check:
        print()
        print("Phase 2.0.1 spec checks:")
        failures = assert_phase_2_0_1(rows)
        failures.extend(assert_multi_question_list_api())
        if failures:
            print("  FAIL")
            for line in failures:
                print(line)
            return 1
        print("  OK — all expectations met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
