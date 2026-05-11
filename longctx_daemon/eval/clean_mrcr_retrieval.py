"""Clean-MRCR retrieval-only baseline through the longctx daemon pipeline.

What this measures:
    For each MRCR sample, treat the prompt's prior assistant messages
    as a synthetic corpus, the user's final question as the QUERY, and
    measure: did the daemon's BM25 + dense + RRF retrieval rank the
    correct assistant message in top-K?

What this does NOT measure:
    End-to-end answer correctness. We never call an LLM. The published
    SubQ + LongCtx end-to-end MRCR numbers (0.659 / 0.601 / 0.688) live
    on a different metric — verbatim answer match through a generator.
    Retrieval-only Recall@K is a NECESSARY upper bound: if the right
    chunk doesn't surface, no generator on earth recovers it. That
    makes high R@8 a precondition for matching SubQ end-to-end, not a
    direct comparison.

Why retrieval-only:
    Generator-bound MRCR runs depend on the model, the chat template,
    and the LLM's prefix-following behavior. Retrieval-only is hardware-
    cheap, deterministic, and isolates the part of the stack we own
    (chunker + BM25 + dense + RRF). Easy to iterate on without spinning
    up a vLLM endpoint.

Bin definitions match ``longctx.eval.mrcr.BINS_BY_CHAR`` so the
retrieval and end-to-end pipelines share a vocabulary:

    8K     16K - 32K chars
    32K    64K - 128K chars
    64K    128K - 256K chars
    128K   256K - 512K chars
    256K   512K - 1M chars
    1M     2M - 5M chars

Usage::

    python3 -m longctx_daemon.eval.clean_mrcr_retrieval \\
        --needles 8 --samples 30 --bins 8K,32K,64K,128K,256K,1M

    # Fast 8K-only smoke
    python3 -m longctx_daemon.eval.clean_mrcr_retrieval \\
        --needles 8 --samples 5 --bins 8K --json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


# ────────────────────────────────────────────────────────────── bin spec

# Bin label → (lo_chars_inclusive, hi_chars_exclusive).
# Mirrors ``longctx.eval.mrcr.BINS_BY_CHAR`` but in the case canonical
# to MRCR v2 published tables (uppercase 8K/32K/...). Values match.
BINS_BY_CHAR: dict[str, tuple[int, int]] = {
    "8K":   (16_000,    32_000),
    "16K":  (32_000,    64_000),
    "32K":  (64_000,   128_000),
    "64K":  (128_000,  256_000),
    "128K": (256_000,  512_000),
    "256K": (512_000, 1_000_000),
    "512K": (1_000_000, 2_000_000),
    "1M":   (2_000_000, 5_000_000),
}

# Recall depths reported in the table. K=8 covers the canonical "1 of 8
# needles" bound; smaller K's tell us how peaky the ranking is.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 8)

# HF dataset shards.
DATASET_REPO = "openai/mrcr"
NEEDLE_SHARDS: dict[int, str] = {
    2: "2needle/2needle_0.parquet",
    4: "4needle/4needle_0.parquet",
    8: "8needle/8needle_0.parquet",
}


# ────────────────────────────────────────────────────────────── result types

@dataclass
class SampleRecord:
    """One MRCR sample's retrieval outcome. Surfaced for replay/debug."""
    bin: str
    n_chars: int
    n_needles: int
    needle_rank: Optional[int]   # 1-indexed; None = miss
    n_returned: int
    top1_dense_cosine: float
    no_relevant_results: bool
    query_type: str
    query_head: str              # first 120 chars
    answer_head: str             # first 120 chars
    elapsed_s: float


@dataclass
class RecallReport:
    """Aggregated Recall@K across bins. JSON-serializable."""
    bins: tuple[str, ...]
    needles: int
    n_samples_per_bin: int
    recall_at_1: dict[str, float]
    recall_at_3: dict[str, float]
    recall_at_5: dict[str, float]
    recall_at_8: dict[str, float]
    sample_records: list[dict] = field(default_factory=list)
    embedder_id: str = ""
    floor_tripped_count: dict[str, int] = field(default_factory=dict)
    floor_tripped_with_correct_chunk: dict[str, int] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────── data loading

def _stream_mrcr_samples(
    *,
    needles: int,
    bin_label: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stream up to ``n_samples`` MRCR rows that fall inside ``bin_label``.

    Filters by ``n_chars`` against ``BINS_BY_CHAR``. Returns dicts with
    ``messages`` (parsed list), ``answer``, ``random_string_to_prepend``,
    ``n_chars``, ``n_needles``. Falls back to an in-memory synthetic
    fixture if the HF download fails (offline / network-locked CI).
    """
    if bin_label not in BINS_BY_CHAR:
        raise ValueError(
            f"unknown bin label: {bin_label!r}; valid: {sorted(BINS_BY_CHAR)}"
        )
    if needles not in NEEDLE_SHARDS:
        raise ValueError(
            f"unsupported needle count: {needles}; valid: {sorted(NEEDLE_SHARDS)}"
        )
    lo, hi = BINS_BY_CHAR[bin_label]

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        return _synthetic_fixture(
            needles=needles, bin_label=bin_label,
            n_samples=n_samples,
        )

    try:
        ds = load_dataset(
            DATASET_REPO,
            data_files=NEEDLE_SHARDS[needles],
            split="train",
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 — network/auth errors all funnel here
        print(
            f"[mrcr] dataset load failed ({exc}); falling back to synthetic",
            file=sys.stderr,
        )
        return _synthetic_fixture(
            needles=needles, bin_label=bin_label,
            n_samples=n_samples,
        )

    samples: list[dict[str, Any]] = []
    for row in ds:
        n_chars = int(row.get("n_chars", 0))
        if n_chars < lo or n_chars >= hi:
            continue
        prompt = row["prompt"]
        try:
            messages = (
                json.loads(prompt) if isinstance(prompt, str)
                else list(prompt)
            )
        except json.JSONDecodeError:
            continue
        samples.append({
            "messages": messages,
            "answer": row["answer"],
            "random_string_to_prepend": row["random_string_to_prepend"],
            "n_chars": n_chars,
            "n_needles": int(row.get("n_needles", needles)),
        })
        if len(samples) >= n_samples:
            break
    if not samples:
        # No rows in this bin from the streaming pass — fall back so the
        # eval still produces a number for the report instead of a hard
        # failure (handy when HF rate-limits or the bin is sparse in
        # the first shard).
        print(
            f"[mrcr] no rows found in bin={bin_label} from HF; "
            "using synthetic",
            file=sys.stderr,
        )
        return _synthetic_fixture(
            needles=needles, bin_label=bin_label,
            n_samples=n_samples,
        )
    return samples


def _synthetic_fixture(
    *,
    needles: int,
    bin_label: str,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Minimal MRCR-shaped fixture — used when HF is unreachable.

    Each sample has 2*N+1 messages: N user/assistant pairs, then a
    final user question that asks for the Kth assistant message
    prepended with a random string. Char counts are fudged toward the
    bin's lo bound so filter logic still passes when an integration test
    points at this fallback.
    """
    lo, _hi = BINS_BY_CHAR[bin_label]
    samples: list[dict[str, Any]] = []
    # Pad assistant messages to roughly hit the bin's lower bound.
    n_msgs = max(needles * 2, 8)
    pad_target = max(lo // n_msgs, 200)
    for i in range(n_samples):
        messages: list[dict[str, str]] = []
        bare_answers: list[str] = []
        for k in range(n_msgs):
            messages.append({"role": "user", "content": f"sample {i} question {k}"})
            bare = f"answer-{i}-{k} " + ("filler " * (pad_target // 7))
            bare_answers.append(bare)
            messages.append({"role": "assistant", "content": bare})
        target_idx = needles - 1  # pick a known assistant index
        rsp = f"PFX{i:03d}"
        bare_target = bare_answers[target_idx]
        question = (
            f"Prepend {rsp} to assistant message #{target_idx + 1}. "
            "Return verbatim."
        )
        messages.append({"role": "user", "content": question})
        samples.append({
            "messages": messages,
            "answer": rsp + bare_target,
            "random_string_to_prepend": rsp,
            "n_chars": sum(len(m["content"]) for m in messages),
            "n_needles": needles,
        })
    return samples


# ────────────────────────────────────────────────────────────── per-sample run

def _final_user_query(messages: list[dict[str, str]]) -> str:
    """Return the last user-role message's content — the MRCR question."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _bare_answer(answer: str, prefix: str) -> str:
    """Strip the random-string prefix from the gold answer."""
    if answer.startswith(prefix):
        return answer[len(prefix):]
    return answer


def _materialize_corpus(messages: list[dict[str, str]], root: Path) -> list[str]:
    """Write each prior assistant message to its own file.

    Returns the list of relative paths in message order. Filenames are
    zero-padded so glob ordering matches insertion order — handy when
    debugging which slot an assistant message occupies.

    NOTE: writing N files does cost N stat() + N write() calls on tmpfs.
    At ~378 assistants/sample for 8-needle this is ~30ms in practice.
    Cheap enough vs the embedding step that we don't bother batching
    into a single large file.
    """
    root.mkdir(parents=True, exist_ok=True)
    # Sentinel so the indexer's project root walk treats this as a
    # legitimate project boundary (mirrors messy_queries_real.py).
    (root / ".git").mkdir(exist_ok=True)
    rel_paths: list[str] = []
    asst_idx = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        # Use .md so the indexer's TEXT_EXTENSIONS check fast-paths it.
        rel = f"asst_{asst_idx:05d}.md"
        (root / rel).write_text(m.get("content") or "")
        rel_paths.append(rel)
        asst_idx += 1
    return rel_paths


def _find_needle_rank(
    chunks: tuple,
    bare_answer: str,
    needle_rel_path: Optional[str],
) -> Optional[int]:
    """First chunk whose text or citation matches the gold needle.

    Two-step probe:
      1. file-path match: did the chunk come from the assistant
         message we know contains the answer? Robust to the chunker
         splitting an assistant message into multiple chunks.
      2. text-prefix match: chunk text starts with the bare answer.
         Fallback when file-path detection misses (e.g. a fixture
         doesn't expose rel_path on citations).
    """
    needle_basename = (
        Path(needle_rel_path).name if needle_rel_path else None
    )
    head = bare_answer[:80].strip()
    for i, c in enumerate(chunks, start=1):
        # Citation file_path is "<project>/<rel_path>" — match by basename
        # so we don't have to mirror the project prefix here.
        cit_basename = Path(c.citation.file_path).name
        if needle_basename and cit_basename == needle_basename:
            return i
        if head and head in c.text:
            return i
    return None


def _run_one_sample(
    *,
    sample: dict[str, Any],
    bin_label: str,
    embedder,
    embedder_id: str,
    workspace: Path,
) -> SampleRecord:
    """Index one MRCR sample's corpus + run one search. Returns
    a ``SampleRecord``.

    Uses the same Indexer + Searcher + storage as production — this
    is the whole point of the eval. Each sample gets a fresh
    SqliteChunkStore + MemmapEmbedStore inside ``workspace`` so we
    never carry chunks across samples.
    """
    from longctx.rag.chunker import Chunker
    from longctx_daemon.indexer import (
        Indexer, IndexerConfig, _embedder_sha256,
    )
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    messages = sample["messages"]
    bare = _bare_answer(sample["answer"], sample["random_string_to_prepend"])
    query = _final_user_query(messages)

    # Corpus + cache for THIS sample only — wiped on context exit.
    corpus = workspace / "corpus"
    cache = workspace / "cache"
    cache.mkdir()
    rel_paths = _materialize_corpus(messages, corpus)

    # Find which assistant rel_path actually contains the bare answer.
    # Match by string presence; same heuristic as the bare-answer probe
    # in scoring. We do this on disk-text directly so we don't depend
    # on indexer chunking succeeding.
    needle_rel_path: Optional[str] = None
    head = bare[:80].strip()
    if head:
        for rel in rel_paths:
            try:
                content = (corpus / rel).read_text()
            except OSError:
                continue
            if head in content:
                needle_rel_path = rel
                break

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

    # Token budget tuned so each assistant message generally stays in a
    # single chunk. 256 tokens × 4 chars/token ≈ 1024 chars/chunk; most
    # assistant messages in MRCR are short (200-2000 chars) so this
    # avoids over-splitting. Long messages will still split — that's
    # fine, the chunk-text or file-path probe still finds the right
    # chunk for needle attribution.
    chunker = Chunker(tokens_per_chunk=256, respect_sentences=False)
    indexer = Indexer(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, chunker=chunker, config=IndexerConfig(),
    )
    indexer.add_project("mrcr", corpus)
    indexer.full_scan("mrcr")

    searcher = Searcher(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(),  # production defaults — incl. floor=0.50
    )

    t0 = time.perf_counter()
    # Pull a generous max_results so smaller K cutoffs are valid; the
    # retrieval depth is the experiment, not the token budget.
    result = searcher.search(
        query=query, cwd=str(corpus),
        max_results=max(DEFAULT_K_VALUES) * 4,
        max_tokens=1_000_000,  # effectively unlimited; we want recall
    )
    elapsed = time.perf_counter() - t0

    rank = _find_needle_rank(result.chunks, bare, needle_rel_path)
    record = SampleRecord(
        bin=bin_label,
        n_chars=int(sample["n_chars"]),
        n_needles=int(sample["n_needles"]),
        needle_rank=rank,
        n_returned=len(result.chunks),
        top1_dense_cosine=float(result.top1_dense_cosine),
        no_relevant_results=bool(result.no_relevant_results),
        query_type=str(result.query_type),
        query_head=query[:120],
        answer_head=bare[:120],
        elapsed_s=elapsed,
    )

    chunk_store.close()
    embed_store.close()
    return record


# ────────────────────────────────────────────────────────────── runner

def run(
    needles: int = 8,
    n_samples: int = 30,
    bins: tuple[str, ...] = ("8K", "32K", "64K", "128K", "256K", "1M"),
    embedder_id: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    *,
    verbose: bool = False,
    samples_per_bin_override: Optional[dict[str, int]] = None,
) -> RecallReport:
    """Stream N samples per bin, run retrieval-only, return Recall@K.

    Args:
        needles: 2 / 4 / 8 — which MRCR shard.
        n_samples: per-bin sample count. The 1M bin is heavy
            (~30s indexing + ~1s search per sample) so callers can
            override individual bins via ``samples_per_bin_override``.
        bins: which bins to evaluate. Strings keyed against
            :data:`BINS_BY_CHAR`.
        embedder_id: HF model id passed to ``SentenceTransformer``.
            Production default ``BAAI/bge-small-en-v1.5`` mirrors
            messy_queries_real.py.
        device: ``auto`` / ``cuda`` / ``mps`` / ``cpu``. ``auto`` walks
            cuda → mps → cpu.
        verbose: print per-sample progress to stderr.
        samples_per_bin_override: e.g. ``{"1M": 10}`` to run fewer
            samples in the slow bin.

    Returns:
        :class:`RecallReport` with R@1/3/5/8 per bin + per-sample records.

    Imports the embedder lazily so callers that pass synthetic samples
    (tests) and a fake embedder don't pay the SentenceTransformer load
    cost. Real-embedder path always loads ``sentence_transformers``.
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.eval._metrics import (
        aggregate_recall, recall_at_k,
    )

    bins_t = tuple(bins)
    invalid = [b for b in bins_t if b not in BINS_BY_CHAR]
    if invalid:
        raise ValueError(
            f"unknown bin labels: {invalid}; valid: {sorted(BINS_BY_CHAR)}"
        )

    resolved_device = _resolve_device(device)
    if verbose:
        print(
            f"[mrcr] embedder={embedder_id} on {resolved_device} | "
            f"needles={needles} bins={bins_t}",
            file=sys.stderr,
        )
    embedder = SentenceTransformer(embedder_id, device=resolved_device)

    return _run_with_embedder(
        needles=needles,
        n_samples=n_samples,
        bins=bins_t,
        embedder=embedder,
        embedder_id=embedder_id,
        verbose=verbose,
        samples_per_bin_override=samples_per_bin_override,
    )


def _run_with_embedder(
    *,
    needles: int,
    n_samples: int,
    bins: tuple[str, ...],
    embedder,
    embedder_id: str,
    verbose: bool,
    samples_per_bin_override: Optional[dict[str, int]],
    sample_loader=None,
) -> RecallReport:
    """Embedder-decoupled inner loop. Used by tests with a fake embedder
    + synthetic loader. Public API goes through :func:`run`."""
    from longctx_daemon.eval._metrics import (
        aggregate_recall, recall_at_k,
    )

    invalid = [b for b in bins if b not in BINS_BY_CHAR]
    if invalid:
        raise ValueError(
            f"unknown bin labels: {invalid}; valid: {sorted(BINS_BY_CHAR)}"
        )

    sample_loader = sample_loader or _stream_mrcr_samples
    overrides = samples_per_bin_override or {}

    per_sample_by_bin: dict[str, list[dict[int, int]]] = {}
    sample_records: list[SampleRecord] = []
    floor_tripped_count: dict[str, int] = {}
    floor_tripped_with_correct: dict[str, int] = {}

    for bin_label in bins:
        target = overrides.get(bin_label, n_samples)
        if target <= 0:
            continue
        if verbose:
            print(
                f"[mrcr] bin={bin_label} loading up to {target} samples",
                file=sys.stderr,
            )
        samples = sample_loader(
            needles=needles, bin_label=bin_label, n_samples=target,
        )
        per_sample_by_bin[bin_label] = []
        for j, sample in enumerate(samples, start=1):
            with tempfile.TemporaryDirectory(
                prefix=f"longctx-mrcr-{bin_label}-",
            ) as tmp:
                rec = _run_one_sample(
                    sample=sample,
                    bin_label=bin_label,
                    embedder=embedder,
                    embedder_id=embedder_id,
                    workspace=Path(tmp),
                )
            sample_records.append(rec)
            per_sample = recall_at_k(rec.needle_rank, DEFAULT_K_VALUES)
            per_sample_by_bin[bin_label].append(per_sample)
            if rec.no_relevant_results:
                floor_tripped_count[bin_label] = (
                    floor_tripped_count.get(bin_label, 0) + 1
                )
                if rec.needle_rank is not None:
                    # Floor tripped but the right chunk WAS in the
                    # ranking pre-floor — this is a "the floor cost
                    # us a real hit" diagnostic we surface in the
                    # report. needle_rank=None means a true miss
                    # regardless of the floor.
                    floor_tripped_with_correct[bin_label] = (
                        floor_tripped_with_correct.get(bin_label, 0) + 1
                    )
            if verbose:
                tag = "miss" if rec.needle_rank is None else f"r{rec.needle_rank}"
                floor_tag = " floor" if rec.no_relevant_results else ""
                print(
                    f"  [{bin_label} {j}/{len(samples)}] {tag}{floor_tag} "
                    f"cos={rec.top1_dense_cosine:.3f} "
                    f"({rec.elapsed_s:.1f}s)",
                    file=sys.stderr,
                )

    # Aggregate per bin.
    r_at: dict[int, dict[str, float]] = {k: {} for k in DEFAULT_K_VALUES}
    for bin_label, rows in per_sample_by_bin.items():
        agg = aggregate_recall(rows)
        for k in DEFAULT_K_VALUES:
            r_at[k][bin_label] = float(agg.get(k, 0.0))

    return RecallReport(
        bins=tuple(bins),
        needles=needles,
        n_samples_per_bin=n_samples,
        recall_at_1=r_at[1],
        recall_at_3=r_at[3],
        recall_at_5=r_at[5],
        recall_at_8=r_at[8],
        sample_records=[asdict(r) for r in sample_records],
        embedder_id=embedder_id,
        floor_tripped_count=floor_tripped_count,
        floor_tripped_with_correct_chunk=floor_tripped_with_correct,
    )


# ────────────────────────────────────────────────────────────── pretty print

def render_table(report: RecallReport) -> str:
    """Markdown Recall@K table — one row per bin."""
    lines = [
        "| bin   | n  | R@1   | R@3   | R@5   | R@8   |",
        "|-------|----|-------|-------|-------|-------|",
    ]
    # Tally per-bin sample counts from records (overrides may differ
    # from n_samples_per_bin, so trust what we actually ran).
    counts: dict[str, int] = {}
    for rec in report.sample_records:
        counts[rec["bin"]] = counts.get(rec["bin"], 0) + 1
    for b in report.bins:
        n = counts.get(b, 0)
        r1 = report.recall_at_1.get(b, 0.0)
        r3 = report.recall_at_3.get(b, 0.0)
        r5 = report.recall_at_5.get(b, 0.0)
        r8 = report.recall_at_8.get(b, 0.0)
        lines.append(
            f"| {b:<5s} | {n:>2d} | {r1:.3f} | {r3:.3f} | {r5:.3f} | {r8:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────── cli

def _parse_bins(arg: str) -> tuple[str, ...]:
    """Comma- or space-separated bin labels, normalized to uppercase."""
    parts = [p.strip().upper() for p in arg.replace(" ", ",").split(",")]
    return tuple(p for p in parts if p)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--needles", type=int, default=8, choices=[2, 4, 8])
    ap.add_argument("--samples", type=int, default=30,
                    help="per-bin sample count (override slow bins via "
                         "--samples-per-bin)")
    ap.add_argument(
        "--bins", default="8K,32K,64K,128K,256K,1M",
        help="comma-separated bin labels, e.g. '8K,32K'",
    )
    ap.add_argument(
        "--samples-per-bin", default=None,
        help='JSON dict overriding sample counts per bin, '
             'e.g. \'{"1M": 10}\' to run fewer in the slow bin',
    )
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", action="store_true",
                    help="print the full RecallReport as JSON instead "
                         "of the markdown table")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    overrides = json.loads(args.samples_per_bin) if args.samples_per_bin else None
    bins = _parse_bins(args.bins)

    report = run(
        needles=args.needles,
        n_samples=args.samples,
        bins=bins,
        embedder_id=args.embedder,
        device=args.device,
        verbose=args.verbose,
        samples_per_bin_override=overrides,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(render_table(report))
        print()
        print(
            "Notes: Retrieval-only Recall@K. R@K=1.0 means every "
            "sample's needle landed in top-K. SubQ's published 0.659 "
            "and prior LongCtx 0.601/0.688 are end-to-end (RAG → "
            "generator) — high R@8 here is a NECESSARY upper bound, "
            "not a 1:1 comparison."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
