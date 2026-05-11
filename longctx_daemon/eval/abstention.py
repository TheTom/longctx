"""Abstention-rate eval + per-corpus relevance-floor calibration.

Phase 2.0.1 shipped a ``relevance_floor`` (default 0.50, dense-cosine-
anchored) that returns ``no_relevant_results=True`` when the top-1 dense
cosine falls below the threshold. The whole point is "honest retrieval":
don't cite garbage when the corpus has nothing.

This module measures HOW WELL that floor works on a real corpus and,
given the trade-off, recommends a per-corpus optimal threshold. The
suite is intentionally split into two halves:

  * **off-corpus suite** — 30 queries spanning history, geography, pop
    culture, math, sports, science, food, etc. None of these would
    plausibly appear in a typical software-codebase corpus. We MEASURE
    what fraction the floor correctly suppresses (``suppression_rate``).
  * **in-corpus suite** — 12 queries paired with the synthetic 5-file
    test project from ``messy_queries_real`` (the auth/billing/notes
    needle corpus). We MEASURE the false-negative rate — how many real
    questions the floor wrongly suppresses (``false_negative_rate``).

Both matter — a floor that suppresses everything is just as bad as one
that suppresses nothing. The recommendation rule for ``sweep_floor``:
pick the highest floor whose FNR is at most ``max_fnr`` (default 0.05).
This is the standard precision-recall framing — we want the tightest
possible suppression that still doesn't reject genuine corpus matches.

Usage:

    # Eval a single floor against the synthetic test corpus:
    python3 -m longctx_daemon.eval.abstention \\
        --corpus-dir <synthetic-or-real-codebase> \\
        --floor 0.50

    # Or sweep:
    python3 -m longctx_daemon.eval.abstention \\
        --corpus-dir <…> --sweep
"""
# TODO(future): expand the in-corpus suite to additionally cover real-
# project queries by cloning a small chosen open-source repo + a
# hand-curated query file. Today we're scoped to the synthetic 5-file
# corpus so the harness stays embedder-only and quick.
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ─────────────────────────────────────────────── query suites

# 30 off-corpus questions. Diversity is the point — if the embedder /
# floor accidentally lights up on one domain (say, every math fact),
# we want the metric to surface it. Domains intentionally span:
#   - world history          (#1-4)
#   - geography              (#5-8)
#   - pop culture / arts     (#9-12)
#   - science / math         (#13-18)
#   - sports                 (#19-22)
#   - food / cooking         (#23-26)
#   - everyday how-to        (#27-30)
_OFF_CORPUS_QUERIES: tuple[str, ...] = (
    # World history
    "when did the berlin wall fall",
    "who was the first president of the united states",
    "what year did world war 2 end",
    "what caused the fall of the roman empire",
    # Geography
    "what is the capital of france",
    "what is the longest river in africa",
    "how tall is mount everest",
    "what is the largest ocean on earth",
    # Pop culture / arts
    "who painted the mona lisa",
    "who wrote hamlet",
    "what year was the beatles formed",
    "who directed the movie inception",
    # Science / math
    "what is the speed of light",
    "what is the chemical formula for water",
    "what is the value of pi to five decimal places",
    "what is the boiling point of mercury",
    "how does photosynthesis work in plants",
    "what is the square root of 144",
    # Sports
    "who won the 2018 fifa world cup",
    "how many players are on a baseball team",
    "what is the diameter of a basketball hoop",
    "when did michael jordan retire from the nba",
    # Food / cooking
    "how do you bake a sourdough bread",
    "what is the difference between baking soda and baking powder",
    "how long does it take to boil an egg",
    "what is the main ingredient in hummus",
    # Everyday how-to
    "how do you tie a tie",
    "how do you change a flat tire on a car",
    "how do you whistle with your fingers",
    "what is the best way to remove a coffee stain from a shirt",
)
assert len(_OFF_CORPUS_QUERIES) == 30, "spec says 30 off-corpus queries"


# 12 in-corpus questions paired with the synthetic 5-file project from
# ``messy_queries_real`` (auth.py / billing.py / inventory.py /
# notes.md / README.md). Each query is ANSWERABLE within that corpus —
# the floor should NOT suppress them. Mix of needle/auth/billing
# targets so we don't accidentally measure FNR on a single domain.
_IN_CORPUS_QUERIES: tuple[str, ...] = (
    # Needle (the explicit planted target in notes.md)
    "where is the secret needle hidden",
    "find the planted needle phrase",
    # Auth subsystem
    "how does the auth middleware validate tokens",
    "show me the function that validates session tokens",
    "where is the auth_middleware function defined",
    # Billing subsystem
    "where is the function that charges a card",
    "how does the billing subsystem charge cards",
    "show me the charge_card function",
    # Inventory subsystem
    "where is the inventory class defined",
    "how are warehouse stock counts tracked",
    # README / project-level
    "what is this project about",
    "what is the example codebase used for",
)
assert len(_IN_CORPUS_QUERIES) >= 10, "spec asks for 10-20 in-corpus queries"


# ─────────────────────────────────────────────── result types

@dataclass
class AbstentionReport:
    """Single-floor abstention metrics over the suite.

    Attributes:
        floor: the dense-cosine threshold under test.
        n_off_corpus: count of off-corpus queries (always 30 today).
        n_in_corpus: count of in-corpus queries (always 12 today).
        suppression_rate: off-corpus correctly suppressed / total. 1.0
            means every garbage query was honestly refused; 0.0 means
            the corpus pretended to have answers for everything.
        false_negative_rate: in-corpus wrongly suppressed / total. 0.0
            means the corpus never refused a real question; 1.0 means
            it refused every real question (floor too tight).
        per_query: full per-query trace, keyed by ``query``, ``label``
            (off_corpus or in_corpus), ``top1_dense_cosine``,
            ``no_relevant_results``, ``correct`` (matches expected
            behavior). Useful for replay and debugging false negatives.
    """
    floor: float
    n_off_corpus: int
    n_in_corpus: int
    suppression_rate: float
    false_negative_rate: float
    per_query: list[dict] = field(default_factory=list)


@dataclass
class SweepReport:
    """Floor-sweep result. ``points`` is one row per swept threshold."""
    points: list[tuple[float, float, float]]  # (floor, suppression, FNR)
    recommended_floor: float
    max_fnr: float = 0.05


# ─────────────────────────────────────────────── runner core

# We hide the heavy embedder import inside ``_make_searcher`` so unit
# tests (which mock ``run_abstention`` callees) don't need to actually
# import sentence-transformers / torch.
def _make_searcher(
    corpus_dir: Path,
    *,
    embedder_id: str,
    device: str,
    cache_dir: Optional[Path] = None,
):
    """Build the (chunk_store, embed_store, searcher) trio over
    ``corpus_dir``.

    Returns a tuple ``(chunk_store, embed_store, searcher, project)``
    plus a teardown ``close()`` callable. The caller is responsible
    for invoking ``close()`` when finished — both stores hold OS
    resources (sqlite + memmap).
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import (
        Indexer,
        IndexerConfig,
        _embedder_sha256,
    )
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    cache = cache_dir if cache_dir is not None else Path(
        tempfile.mkdtemp(prefix="longctx-abst-")
    )
    cache.mkdir(parents=True, exist_ok=True)

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
        on_mismatch="warn",
    )
    chunker = Chunker(tokens_per_chunk=64, respect_sentences=False)
    indexer = Indexer(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, chunker=chunker,
        config=IndexerConfig(),
    )
    project = corpus_dir.name or "corpus"
    indexer.add_project(project, corpus_dir)
    indexer.full_scan(project)

    # Construct the searcher with floor=0 baseline so we can override
    # per-call. The floor on each ``search`` call is what drives the
    # measurement — we want a clean slate, not a config that biases.
    searcher = Searcher(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, config=SearcherConfig(relevance_floor=0.0),
    )

    def close() -> None:
        chunk_store.close()
        embed_store.close()

    return chunk_store, embed_store, searcher, project, close


def _run_query_suite(
    searcher,
    corpus_dir: Path,
    queries: Sequence[str],
    floor: float,
    label: str,
) -> list[dict]:
    """Run ``queries`` through ``searcher`` with ``floor`` and capture
    per-query traces.

    ``label`` is "off_corpus" or "in_corpus"; we use it to flip the
    "correct" sense:
      * off_corpus → correct == no_relevant_results (we WANT it suppressed)
      * in_corpus  → correct == not no_relevant_results (we want it surfaced)
    """
    out: list[dict] = []
    for q in queries:
        t0 = time.time()
        r = searcher.search(
            query=q, cwd=str(corpus_dir),
            relevance_floor=floor,
        )
        elapsed = time.time() - t0
        no_rel = bool(getattr(r, "no_relevant_results", False))
        cos = float(getattr(r, "top1_dense_cosine", 0.0))
        if label == "off_corpus":
            correct = no_rel
        else:
            correct = not no_rel
        out.append({
            "query": q,
            "label": label,
            "no_relevant_results": no_rel,
            "top1_dense_cosine": cos,
            "elapsed_s": elapsed,
            "correct": correct,
        })
    return out


def _compute_metrics(per_query: list[dict]) -> tuple[float, float]:
    """Compute (suppression_rate, false_negative_rate) from per-query
    traces.

    Suppression = correctly-suppressed off-corpus / total off-corpus
    FNR        = wrongly-suppressed in-corpus / total in-corpus

    Both default to 0.0 when their respective denominator is zero so
    callers don't trip on divide-by-zero in degenerate suites. (Today
    the suites are non-empty so this only matters in tests.)
    """
    off = [r for r in per_query if r["label"] == "off_corpus"]
    inc = [r for r in per_query if r["label"] == "in_corpus"]
    if off:
        suppression = sum(1 for r in off if r["no_relevant_results"]) / len(off)
    else:
        suppression = 0.0
    if inc:
        fnr = sum(1 for r in inc if r["no_relevant_results"]) / len(inc)
    else:
        fnr = 0.0
    return suppression, fnr


def run_abstention(
    corpus_dir: Path,
    floor: float = 0.50,
    *,
    embedder_id: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
) -> AbstentionReport:
    """Run the abstention eval against ``corpus_dir`` at ``floor``.

    Indexes the corpus once, runs the off-corpus + in-corpus suites,
    returns metrics + per-query traces. Side effects: creates a temp
    cache directory and tears it down on exit.

    Example:
        >>> report = run_abstention(Path("~/dev/myapp"), floor=0.50)
        >>> assert report.suppression_rate >= 0.6
        >>> assert report.false_negative_rate <= 0.10
    """
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus_dir not a directory: {corpus_dir}")

    with tempfile.TemporaryDirectory(prefix="longctx-abst-cache-") as tmp:
        _, _, searcher, _, close = _make_searcher(
            corpus_dir,
            embedder_id=embedder_id,
            device=device,
            cache_dir=Path(tmp),
        )
        try:
            off = _run_query_suite(
                searcher, corpus_dir,
                _OFF_CORPUS_QUERIES, floor, "off_corpus",
            )
            inc = _run_query_suite(
                searcher, corpus_dir,
                _IN_CORPUS_QUERIES, floor, "in_corpus",
            )
        finally:
            close()

    per_query = off + inc
    suppression, fnr = _compute_metrics(per_query)
    return AbstentionReport(
        floor=floor,
        n_off_corpus=len(off),
        n_in_corpus=len(inc),
        suppression_rate=suppression,
        false_negative_rate=fnr,
        per_query=per_query,
    )


def sweep_floor(
    corpus_dir: Path,
    floors: Sequence[float] = (
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    ),
    *,
    embedder_id: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    max_fnr: float = 0.05,
) -> SweepReport:
    """Sweep ``floors`` against ``corpus_dir`` and recommend the best.

    Indexes the corpus ONCE and reuses the searcher across all
    threshold samples (the floor is applied per-call, not at index
    time, so this is correct and 9× cheaper than re-indexing).

    Recommendation rule: pick the HIGHEST floor whose false-negative
    rate is ≤ ``max_fnr`` (default 0.05). Ties broken by floor value
    (we always prefer the tightest one). If no floor satisfies the
    constraint, fall back to the lowest swept floor — caller decides
    whether that means "use this anyway" or "you need a different
    embedder".
    """
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus_dir not a directory: {corpus_dir}")
    if not floors:
        raise ValueError("floors must be non-empty")

    with tempfile.TemporaryDirectory(prefix="longctx-sweep-") as tmp:
        _, _, searcher, _, close = _make_searcher(
            corpus_dir,
            embedder_id=embedder_id,
            device=device,
            cache_dir=Path(tmp),
        )
        try:
            points: list[tuple[float, float, float]] = []
            for f in floors:
                off = _run_query_suite(
                    searcher, corpus_dir,
                    _OFF_CORPUS_QUERIES, f, "off_corpus",
                )
                inc = _run_query_suite(
                    searcher, corpus_dir,
                    _IN_CORPUS_QUERIES, f, "in_corpus",
                )
                supp, fnr = _compute_metrics(off + inc)
                points.append((float(f), float(supp), float(fnr)))
        finally:
            close()

    recommended = recommend_floor(points, max_fnr=max_fnr)
    return SweepReport(points=points, recommended_floor=recommended,
                       max_fnr=max_fnr)


def recommend_floor(
    points: Sequence[tuple[float, float, float]],
    *,
    max_fnr: float = 0.05,
) -> float:
    """Pick the highest floor whose FNR is ≤ ``max_fnr``.

    Pure function (no I/O) so the recommendation logic is unit-testable
    independent of the corpus-indexing step. Falls back to the lowest
    swept floor when nothing satisfies the constraint — better to
    return something than crash, and the caller can re-decide based on
    the metrics.
    """
    if not points:
        raise ValueError("cannot recommend from empty sweep")
    candidates = [(f, s, fnr) for f, s, fnr in points if fnr <= max_fnr]
    if candidates:
        # Highest floor among those satisfying the constraint
        return max(candidates, key=lambda r: r[0])[0]
    # Nobody satisfied; bail to the lowest swept floor (most permissive)
    return min(points, key=lambda r: r[0])[0]


# ─────────────────────────────────────────────── pretty + CLI

def render_sweep_table(
    report: SweepReport,
    *,
    show_recommended: bool = True,
) -> str:
    """ASCII table of the sweep result; prints from stdout in CLI mode.

    Format mirrors the docstring at the top of the module so users see
    the expected shape:

        floor    suppression  FNR    recommended
        0.30        0.42      0.00
        0.50        0.68      0.05    ← recommended (FNR ≤ 5%)
    """
    lines = ["floor    suppression  FNR    recommended"]
    for f, s, fnr in report.points:
        marker = ""
        if show_recommended and abs(f - report.recommended_floor) < 1e-9:
            marker = f"    ← recommended (FNR ≤ {report.max_fnr:.0%})"
        lines.append(f"{f:>4.2f}        {s:.2f}      {fnr:.2f}{marker}")
    return "\n".join(lines) + "\n"


def render_abstention_report(report: AbstentionReport) -> str:
    """Human-readable summary of a single-floor eval. Used by the CLI
    when ``--sweep`` is NOT set (the default observation mode)."""
    return (
        f"floor={report.floor:.2f}\n"
        f"  off-corpus queries: {report.n_off_corpus}\n"
        f"  in-corpus queries:  {report.n_in_corpus}\n"
        f"  suppression rate:   {report.suppression_rate:.2%}  "
        f"(off-corpus correctly suppressed)\n"
        f"  false-negative:     {report.false_negative_rate:.2%}  "
        f"(in-corpus wrongly suppressed)\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--corpus-dir", required=True,
        help="Directory to index + evaluate.",
    )
    ap.add_argument(
        "--floor", type=float, default=0.50,
        help="Single-floor threshold for abstention eval (default 0.50).",
    )
    ap.add_argument(
        "--sweep", action="store_true",
        help="Sweep a range of floors and print the recommendation table.",
    )
    ap.add_argument(
        "--floor-min", type=float, default=0.30,
        help="Sweep min (default 0.30).",
    )
    ap.add_argument(
        "--floor-max", type=float, default=0.70,
        help="Sweep max (default 0.70).",
    )
    ap.add_argument(
        "--floor-step", type=float, default=0.05,
        help="Sweep step (default 0.05).",
    )
    ap.add_argument(
        "--max-fnr", type=float, default=0.05,
        help="Max FNR for recommendation (default 0.05).",
    )
    ap.add_argument(
        "--embedder", default="BAAI/bge-small-en-v1.5",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus_dir).expanduser().resolve()
    if not corpus.is_dir():
        print(
            f"error: --corpus-dir not a directory: {corpus}",
            file=sys.stderr,
        )
        return 2

    if args.sweep:
        floors = []
        f = args.floor_min
        # Use a tight epsilon so a request like 0.30..0.70 step 0.05
        # actually emits 0.70 (avoiding 0.6999… cutoff).
        while f <= args.floor_max + 1e-9:
            floors.append(round(f, 6))
            f += args.floor_step
        report = sweep_floor(
            corpus,
            floors=floors,
            embedder_id=args.embedder,
            device=args.device,
            max_fnr=args.max_fnr,
        )
        if args.json:
            print(json.dumps({
                "points": report.points,
                "recommended_floor": report.recommended_floor,
                "max_fnr": report.max_fnr,
            }, indent=2))
        else:
            print(render_sweep_table(report))
            print(
                f"recommended floor: {report.recommended_floor:.2f} "
                f"(highest floor with FNR ≤ {report.max_fnr:.0%})"
            )
        return 0

    report = run_abstention(
        corpus,
        floor=args.floor,
        embedder_id=args.embedder,
        device=args.device,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(render_abstention_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
