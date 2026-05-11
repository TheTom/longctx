"""Advanced NIAH evals over the daemon pipeline (Phase 2.0.1).

Three NIAH variants, all driven through the production-shaped retrieval
path: real ``Indexer`` (filesystem walk → chunk → embed → SqliteChunkStore
+ MemmapEmbedStore) + real ``Searcher`` (BM25 + dense + RRF). The Phase 1
``bench_coarse_filter`` exercises the in-memory ``CoarseFilter`` only;
this rig validates that the daemon-shaped pipeline (persistent storage,
scope routing, freshness signaling) ALSO survives at long context.

Variants
--------

1. **Multi-needle**. Plant N facts (3-5) at evenly spaced positions of a
   synthetic haystack and ask one question per fact. Reports per-fact
   Recall@K AND a stricter "ALL-OF-N hit rate" — the fraction of samples
   where every needle landed in top-K. Partial recall is dangerous in
   production: it lets the agent cite a stale or wrong fact when the
   freshest one falls off the list.

2. **Needle-near-needle**. Plant two semantically similar phrases — a
   real needle (the right answer) and a distractor (similar surrounding
   context, wrong value). Score: ``real_rank < distractor_rank``. This
   is the hard ambiguity case; bge-small either nails it or silently
   prefers the wrong chunk.

3. **Depth sweep**. Plant one needle at varying corpus depths
   (10/30/50/70/90%) crossed with corpus sizes (100K / 1M / 12M
   tokens). Strong depth bias (last position much worse) indicates a
   chunker positional issue — e.g. uneven chunk boundaries near tail.

Implementation notes
--------------------

* The synthetic haystack is split across multiple files so each file
  stays under ``max_file_size_bytes`` (5 MB) — that's the realistic
  shape (no real corpus is one massive blob) AND it dodges the
  indexer's per-file size cap. Total token count is approximate
  (4 chars/token heuristic, same as the chunker).
* ``relevance_floor=0.0`` is passed to ``search`` so the floor doesn't
  silently drop top-K matches in the synthetic regime where neither
  the haystack nor the query reaches the 0.50 production threshold.
* Per-sample timings are tracked so slow runs can be correlated with
  quality regressions. Total wall-time budget per variant is kept
  under 10 minutes by default; 12M depth-sweep cells require
  ``--include-12m``.

Usage
-----

    python3 -m longctx_daemon.eval.niah_advanced multi-needle \\
        --n-facts 3 --tokens 100000 --samples 20
    python3 -m longctx_daemon.eval.niah_advanced near-needle \\
        --tokens 100000 --samples 20
    python3 -m longctx_daemon.eval.niah_advanced depth \\
        --tokens 100000,1000000 --samples 10
    python3 -m longctx_daemon.eval.niah_advanced all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence


# ─────────────────────────────────────────────── filler + needle templates

# Topically-bland filler — same shape as ``bench_coarse_filter.FILLER_SENTENCES``
# so per-fact dense signal isn't drowned by accidental overlap with the
# generic prose. Re-declared here to avoid the cross-package import the
# Phase 2 spec explicitly forbids (daemon shouldn't reach into longctx.eval).
FILLER_SENTENCES = (
    "The routine quarterly procedure required careful review.",
    "Documented operational records were checked during the cycle.",
    "The technicians reviewed standard checklists.",
    "All routine items were confirmed in the assessment.",
    "Logistics teams coordinated supplies for the next phase.",
    "Daily reports were filed without irregularity.",
    "Compliance review noted no exceptions for the period.",
    "Inventory checks aligned with the master schedule.",
)

# Multi-needle facts use distinct project codenames so each fact has its
# own retrievable surface. Codes baked at runtime so different samples
# get different needles (avoids embed-cache hits masking real recall).
MULTI_NEEDLE_NAMES = (
    "ATLAS", "ORION", "TITAN", "HYDRA", "PEGASUS",
    "ZEPHYR", "KESTREL", "GEMINI", "VIRGO", "LYRA",
)

# Real / distractor templates for needle-near-needle. They share the
# project name + "access code" surface so dense embedding sees
# legitimately-similar context; only the value differs and the
# distractor explicitly says "deprecated" / "legacy" so the agent has a
# textual cue to ignore it.
REAL_NEEDLE_TEMPLATE = (
    "IMPORTANT: Project NOVA was provisioned with access code "
    "{code}. All technicians must memorise this code prior to deployment."
)
DISTRACTOR_TEMPLATE = (
    "Project NOVA's training scenarios reference legacy access code "
    "{code} from the deprecated system. Do not use this code in production."
)
NEAR_NEEDLE_QUERY = "What is the active access code for Project NOVA?"

# Single-needle (depth sweep) reuses the standard NIAH form.
SINGLE_NEEDLE_TEMPLATE = (
    "IMPORTANT: Project NOVA was provisioned with access code "
    "{code}. All technicians must memorise this code prior to deployment."
)
SINGLE_NEEDLE_QUERY = "What is the access code for Project NOVA?"


# ─────────────────────────────────────────────── result dataclasses

@dataclass
class MultiNeedleSample:
    """One multi-needle sample's outcome.

    Attributes:
        seed: per-sample seed (so failures can be replayed).
        per_fact_hits: list of bools, one per planted fact; True iff the
            fact's chunk was in the top-K result list for ITS query.
        per_fact_ranks: 1-indexed rank for each fact (None on miss).
        all_of_n_hit: True iff every fact landed in top-K.
        wall_secs: total wall time for the sample (index + N searches).
    """
    seed: int
    per_fact_hits: tuple[bool, ...]
    per_fact_ranks: tuple[Optional[int], ...]
    all_of_n_hit: bool
    wall_secs: float


@dataclass
class MultiNeedleReport:
    """Aggregate of multiple multi-needle samples."""
    n_facts: int
    corpus_tokens: int
    n_samples: int
    top_k: int
    seed: int
    embedder_id: str
    device: str
    samples: list[MultiNeedleSample] = field(default_factory=list)
    per_fact_recall: float = 0.0   # mean over (sample × fact) pairs
    all_of_n_rate: float = 0.0     # fraction where every fact hit
    mean_wall_secs: float = 0.0


@dataclass
class NearNeedleSample:
    """One near-needle sample's outcome.

    Attributes:
        seed: per-sample seed.
        real_rank: 1-indexed rank of the real needle (None = miss).
        distractor_rank: 1-indexed rank of the distractor (None = miss).
        real_beats_distractor: True iff real ranked above distractor;
            False if distractor outranked real OR real missed entirely.
            A miss-on-real even when distractor also missed counts as
            False — partial credit doesn't make sense here.
        wall_secs: total wall time for the sample.
    """
    seed: int
    real_rank: Optional[int]
    distractor_rank: Optional[int]
    real_beats_distractor: bool
    wall_secs: float


@dataclass
class NearNeedleReport:
    corpus_tokens: int
    n_samples: int
    top_k: int
    seed: int
    embedder_id: str
    device: str
    samples: list[NearNeedleSample] = field(default_factory=list)
    real_beats_distractor_rate: float = 0.0
    real_recall: float = 0.0
    mean_wall_secs: float = 0.0


@dataclass
class DepthCellResult:
    """One cell of the depth × size matrix."""
    depth: float
    corpus_tokens: int
    n_samples: int
    hits: int
    recall: float
    mean_rank: Optional[float]
    mean_wall_secs: float


@dataclass
class DepthSweepReport:
    depths: tuple[float, ...]
    corpus_tokens_list: tuple[int, ...]
    n_samples_per_cell: int
    top_k: int
    seed: int
    embedder_id: str
    device: str
    cells: list[DepthCellResult] = field(default_factory=list)


# ─────────────────────────────────────────────── synthetic haystack

# Per-file size cap for the multi-file haystack. Stays below the
# indexer's default ``max_file_size_bytes=5 MiB`` so we don't trip the
# filter chain.
_FILE_BYTES_CAP = 3 * 1024 * 1024  # 3 MiB
# 4 chars/token, matches the chunker's char-proxy heuristic.
_CHARS_PER_TOKEN = 4


def _build_filler(target_chars: int, rng: random.Random) -> str:
    """Generate ``target_chars`` of bland filler from ``FILLER_SENTENCES``.

    Sentence pool is small (8 entries); the chunker hashes chunk text
    so identical slices across runs reuse cached embeddings — fine for
    the bench, deliberate at corpus level.
    """
    parts: list[str] = []
    so_far = 0
    while so_far < target_chars:
        s = rng.choice(FILLER_SENTENCES)
        parts.append(s)
        so_far += len(s) + 1
    return " ".join(parts)


def _split_into_files(
    text: str, root: Path, prefix: str = "haystack",
) -> list[tuple[Path, int, int]]:
    """Split ``text`` into multiple files under ``root``, each ≤ 3 MiB.

    Returns a list of ``(path, char_start, char_end)`` tuples in order.
    Used so we can map a "global" character offset (multi/depth tests
    pick a haystack-relative position) to the right file + relative
    offset for in-place needle insertion.
    """
    out: list[tuple[Path, int, int]] = []
    n = len(text)
    pos = 0
    idx = 0
    while pos < n:
        end = min(pos + _FILE_BYTES_CAP, n)
        # Round to nearest space so we don't slice mid-word.
        if end < n:
            sp = text.rfind(" ", pos, end)
            if sp > pos:
                end = sp
        path = root / f"{prefix}_{idx:04d}.txt"
        path.write_text(text[pos:end])
        out.append((path, pos, end))
        pos = end
        idx += 1
    if not out:
        # Empty corpus edge case: still write one empty file so the
        # indexer sees a valid project. (The indexer skips zero-byte
        # files — write a single space so the file is non-empty.)
        path = root / f"{prefix}_0000.txt"
        path.write_text(" ")
        out.append((path, 0, 1))
    return out


def _insert_needle_at_offset(
    files: list[tuple[Path, int, int]],
    char_offset: int,
    needle: str,
) -> tuple[Path, int]:
    """Insert ``needle`` at the global ``char_offset`` of the haystack.

    Mutates the targeted file in place — appends a separator + needle +
    separator and rewrites. Returns ``(file_path, file_relative_offset)``
    so the caller can track where the needle landed.

    The needle gets newline padding around it so the chunker's sentence
    boundary heuristic doesn't smear it into surrounding filler. That
    matters for multi-needle: each fact must land in its own chunk, not
    share one with a neighbor.
    """
    # Find the file whose char range covers ``char_offset``.
    target_path: Optional[Path] = None
    target_rel = 0
    for path, start, end in files:
        if start <= char_offset < end:
            target_path = path
            target_rel = char_offset - start
            break
    if target_path is None:
        # Offset past the end (e.g. depth=1.0 + rounding) — append to
        # last file.
        target_path = files[-1][0]
        target_rel = files[-1][2] - files[-1][1]

    text = target_path.read_text()
    # Snap to nearest space to avoid splitting a word (which would
    # produce a chunk like "thequickIMPORTANT:" that BM25 wouldn't tokenize
    # cleanly).
    if 0 < target_rel < len(text):
        sp = text.rfind(" ", 0, target_rel)
        if sp >= 0:
            target_rel = sp + 1
    padded = "\n\n" + needle + "\n\n"
    new_text = text[:target_rel] + padded + text[target_rel:]
    target_path.write_text(new_text)
    return target_path, target_rel


def _build_haystack(
    corpus_tokens: int,
    root: Path,
    seed: int,
    *,
    needles: Sequence[tuple[float, str]],
) -> list[tuple[Path, int]]:
    """Build a multi-file haystack with ``needles`` planted at given depths.

    Args:
        corpus_tokens: approximate target token count (4 chars/token).
        root: directory to write the haystack into.
        seed: filler RNG seed.
        needles: sequence of ``(depth, needle_text)``. Depth in [0, 1].

    Returns:
        ``[(file_path, file_relative_offset), ...]`` in needles-input
        order — useful for tests asserting the planted positions.

    Insertion order: needles are inserted from LOWEST to HIGHEST depth
    so each insertion's offset stays valid (later insertions are at
    larger offsets, so earlier insertions' shifts don't affect them).
    The returned list preserves the input ordering.
    """
    rng = random.Random(seed)
    target_chars = max(corpus_tokens * _CHARS_PER_TOKEN, 1024)
    filler = _build_filler(target_chars, rng)
    files = _split_into_files(filler, root)

    # Stable-sort by depth ascending; remember original positions so we
    # can return outputs in the caller's order.
    order = sorted(range(len(needles)), key=lambda i: needles[i][0])

    placements: dict[int, tuple[Path, int]] = {}
    for orig_idx in order:
        depth, needle_text = needles[orig_idx]
        offset = int(round(depth * len(filler)))
        # After previous insertions, ``filler`` length is no longer the
        # source of truth for offsets in the on-disk text. We compute
        # the offset once against the ORIGINAL filler length so depth=
        # 0.5 always lands near the median character position of the
        # filler-only text. That's the documented contract: depth
        # measures into the *bland* haystack, not into a corpus
        # already populated with earlier needles.
        # _insert_needle_at_offset does its own re-stat of the file.
        # Recompute the file ranges by re-listing the directory each
        # iteration so we always shift onto the on-disk text.
        files = _refresh_file_ranges(root)
        offset = min(offset, sum(end - start for _, start, end in files) - 1)
        path, rel = _insert_needle_at_offset(files, offset, needle_text)
        placements[orig_idx] = (path, rel)

    return [placements[i] for i in range(len(needles))]


def _refresh_file_ranges(root: Path) -> list[tuple[Path, int, int]]:
    """Re-stat the files in ``root`` and rebuild ``(path, start, end)`` tuples.

    Mutating insertions change file sizes, so ranges stored from the
    initial split go stale. This rescans the directory so the next
    insertion sees current offsets.
    """
    out: list[tuple[Path, int, int]] = []
    pos = 0
    for path in sorted(root.glob("*.txt")):
        size = path.stat().st_size
        out.append((path, pos, pos + size))
        pos += size
    return out


# ─────────────────────────────────────────────── daemon pipeline harness

def _build_pipeline(
    corpus_root: Path,
    cache_root: Path,
    embedder_id: str,
    device: str,
):
    """Wire up Indexer + Searcher against a real corpus directory.

    Returns ``(indexer, searcher, chunk_store, embed_store, embedder)``
    so the caller can run the full pipeline + close stores. Side-effect:
    runs ``indexer.full_scan`` so the project is queryable on return.

    The indexer config uses a SMALL chunk size (256 tokens / ≈1024 chars)
    so multi-needle facts at adjacent depths land in distinct chunks.
    Production defaults are larger (2048 tokens / chunk) but those would
    smear the needles together at small corpus sizes — bench artefact,
    not a production decision.
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import (
        Indexer, IndexerConfig, _embedder_sha256,
    )
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    resolved = _resolve_device(device)
    embedder = SentenceTransformer(embedder_id, device=resolved)
    embed_dim = (
        embedder.get_embedding_dimension()
        if hasattr(embedder, "get_embedding_dimension")
        else embedder.get_sentence_embedding_dimension()
    )

    chunk_store = SqliteChunkStore(cache_root / "index.db")
    embed_store = MemmapEmbedStore(
        cache_root / "embeddings",
        model_name=embedder_id,
        model_sha256=_embedder_sha256(embedder),
        dim=embed_dim,
    )
    chunker = Chunker(tokens_per_chunk=256, respect_sentences=False)
    indexer = Indexer(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        chunker=chunker,
        config=IndexerConfig(),
    )
    indexer.add_project("haystack", corpus_root)
    indexer.full_scan("haystack")

    searcher = Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(),
    )
    return indexer, searcher, chunk_store, embed_store, embedder, resolved


def _search_with_floor_off(searcher, query: str, top_k: int, cwd: str):
    """Run ``searcher.search`` with the relevance floor disabled.

    The synthetic-haystack regime can drive top-1 dense cosine below
    the production 0.50 threshold (BM25-only fallbacks, very-low-IDF
    queries, etc.). For these benches we want raw retrieval behavior,
    not a guarded production response. ``relevance_floor=0.0`` disables
    the gate entirely.
    """
    return searcher.search(
        query=query,
        cwd=cwd,
        max_tokens=10**9,         # don't let token budget truncate
        max_results=top_k,
        relevance_floor=0.0,
    )


def _find_needle_rank(chunks: Sequence, needle_marker: str) -> Optional[int]:
    """Return 1-indexed rank of ``needle_marker`` in ``chunks``, or None.

    ``needle_marker`` is a per-needle unique substring (the access code
    or project name + code) — we search the chunk text for it. If it
    appears in multiple chunks (chunk overlap edge case), the first
    matching rank wins.
    """
    for i, ch in enumerate(chunks, start=1):
        if needle_marker in ch.text:
            return i
    return None


# ─────────────────────────────────────────────── variant: multi-needle

def run_multi_needle(
    n_facts: int,
    corpus_tokens: int,
    n_samples: int,
    seed: int,
    embedder_id: str,
    device: str,
    *,
    top_k: int = 50,
    verbose: bool = False,
) -> MultiNeedleReport:
    """Plant ``n_facts`` distinct needles, query each, score per-fact +
    all-of-N recall.

    Each sample uses a fresh corpus + fresh codes so embed cache hits
    don't mask retrieval. Per-fact recall is the mean over
    (sample × fact) pairs; all-of-N is the fraction of samples where
    EVERY needle landed in top-K.
    """
    if n_facts < 1:
        raise ValueError("n_facts must be >= 1")
    if n_facts > len(MULTI_NEEDLE_NAMES):
        raise ValueError(
            f"n_facts={n_facts} exceeds the {len(MULTI_NEEDLE_NAMES)} "
            "named project codenames"
        )

    report = MultiNeedleReport(
        n_facts=n_facts, corpus_tokens=corpus_tokens, n_samples=n_samples,
        top_k=top_k, seed=seed, embedder_id=embedder_id, device=device,
    )

    rng = random.Random(seed)
    for sample_i in range(n_samples):
        sample_seed = rng.randint(0, 2**31 - 1)
        if verbose:
            print(
                f"[multi-needle] sample {sample_i+1}/{n_samples} "
                f"seed={sample_seed}",
                file=sys.stderr,
            )
        sample = _run_one_multi_needle(
            n_facts=n_facts, corpus_tokens=corpus_tokens,
            sample_seed=sample_seed, embedder_id=embedder_id,
            device=device, top_k=top_k,
        )
        report.samples.append(sample)

    if report.samples:
        total_facts = n_facts * len(report.samples)
        total_hits = sum(
            sum(1 for h in s.per_fact_hits if h) for s in report.samples
        )
        report.per_fact_recall = total_hits / total_facts
        report.all_of_n_rate = (
            sum(1 for s in report.samples if s.all_of_n_hit)
            / len(report.samples)
        )
        report.mean_wall_secs = (
            sum(s.wall_secs for s in report.samples) / len(report.samples)
        )
    return report


def _run_one_multi_needle(
    n_facts: int,
    corpus_tokens: int,
    sample_seed: int,
    embedder_id: str,
    device: str,
    top_k: int,
) -> MultiNeedleSample:
    """One multi-needle sample: build corpus, plant N needles, run N
    searches, compute per-fact + all-of-N hits."""
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="niah-multi-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        # Plant n_facts at evenly spaced depths between 0.1 and 0.9.
        if n_facts == 1:
            depths = (0.5,)
        else:
            depths = tuple(
                0.1 + i * (0.8 / (n_facts - 1)) for i in range(n_facts)
            )
        names = MULTI_NEEDLE_NAMES[:n_facts]
        codes = [f"{sample_seed % 1000:03d}{i:03d}" for i in range(n_facts)]
        needles = [
            (depths[i],
             f"IMPORTANT: Project {names[i]} was provisioned with "
             f"access code {codes[i]}. All technicians must memorise this "
             f"code prior to deployment.")
            for i in range(n_facts)
        ]
        _build_haystack(
            corpus_tokens=corpus_tokens, root=corpus, seed=sample_seed,
            needles=needles,
        )

        _, searcher, chunk_store, embed_store, _, _ = _build_pipeline(
            corpus_root=corpus, cache_root=cache,
            embedder_id=embedder_id, device=device,
        )
        try:
            per_fact_hits: list[bool] = []
            per_fact_ranks: list[Optional[int]] = []
            for i in range(n_facts):
                query = (
                    f"What is the access code for Project {names[i]}? "
                    f"access code {codes[i]}"
                )
                r = _search_with_floor_off(
                    searcher, query, top_k=top_k, cwd=str(corpus),
                )
                rank = _find_needle_rank(r.chunks, codes[i])
                per_fact_ranks.append(rank)
                per_fact_hits.append(rank is not None)
        finally:
            chunk_store.close()
            embed_store.close()

    return MultiNeedleSample(
        seed=sample_seed,
        per_fact_hits=tuple(per_fact_hits),
        per_fact_ranks=tuple(per_fact_ranks),
        all_of_n_hit=all(per_fact_hits),
        wall_secs=time.perf_counter() - t_total,
    )


# ─────────────────────────────────────────────── variant: needle-near-needle

def run_needle_near_needle(
    corpus_tokens: int,
    n_samples: int,
    seed: int,
    embedder_id: str,
    device: str,
    *,
    top_k: int = 50,
    verbose: bool = False,
) -> NearNeedleReport:
    """Plant a real needle + a semantically-similar distractor, score
    whether the real needle outranks the distractor."""
    report = NearNeedleReport(
        corpus_tokens=corpus_tokens, n_samples=n_samples, top_k=top_k,
        seed=seed, embedder_id=embedder_id, device=device,
    )
    rng = random.Random(seed)
    for sample_i in range(n_samples):
        sample_seed = rng.randint(0, 2**31 - 1)
        if verbose:
            print(
                f"[near-needle] sample {sample_i+1}/{n_samples} "
                f"seed={sample_seed}",
                file=sys.stderr,
            )
        sample = _run_one_near_needle(
            corpus_tokens=corpus_tokens, sample_seed=sample_seed,
            embedder_id=embedder_id, device=device, top_k=top_k,
        )
        report.samples.append(sample)

    if report.samples:
        report.real_beats_distractor_rate = (
            sum(1 for s in report.samples if s.real_beats_distractor)
            / len(report.samples)
        )
        report.real_recall = (
            sum(1 for s in report.samples if s.real_rank is not None)
            / len(report.samples)
        )
        report.mean_wall_secs = (
            sum(s.wall_secs for s in report.samples) / len(report.samples)
        )
    return report


def _run_one_near_needle(
    corpus_tokens: int,
    sample_seed: int,
    embedder_id: str,
    device: str,
    top_k: int,
) -> NearNeedleSample:
    """One near-needle sample: real needle at depth 0.5, distractor at
    0.3, query NEAR_NEEDLE_QUERY, compute relative ranks."""
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="niah-near-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        # Distinct codes per sample so ranks don't collide via cache.
        real_code = f"REAL{sample_seed % 1_000_000:06d}"
        distractor_code = f"FAKE{sample_seed % 1_000_000:06d}"
        real_text = REAL_NEEDLE_TEMPLATE.format(code=real_code)
        distractor_text = DISTRACTOR_TEMPLATE.format(code=distractor_code)

        # Distractor at 0.3, real at 0.6 — far enough that they land in
        # distinct chunks at the small chunk size.
        _build_haystack(
            corpus_tokens=corpus_tokens, root=corpus, seed=sample_seed,
            needles=[(0.3, distractor_text), (0.6, real_text)],
        )

        _, searcher, chunk_store, embed_store, _, _ = _build_pipeline(
            corpus_root=corpus, cache_root=cache,
            embedder_id=embedder_id, device=device,
        )
        try:
            r = _search_with_floor_off(
                searcher, NEAR_NEEDLE_QUERY, top_k=top_k, cwd=str(corpus),
            )
            real_rank = _find_needle_rank(r.chunks, real_code)
            distractor_rank = _find_needle_rank(r.chunks, distractor_code)
            real_beats = (
                real_rank is not None
                and (distractor_rank is None or real_rank < distractor_rank)
            )
        finally:
            chunk_store.close()
            embed_store.close()

    return NearNeedleSample(
        seed=sample_seed,
        real_rank=real_rank,
        distractor_rank=distractor_rank,
        real_beats_distractor=real_beats,
        wall_secs=time.perf_counter() - t_total,
    )


# ─────────────────────────────────────────────── variant: depth sweep

def run_depth_sweep(
    depths: Sequence[float],
    corpus_tokens_list: Sequence[int],
    n_samples_per_cell: int,
    seed: int,
    embedder_id: str,
    device: str,
    *,
    top_k: int = 50,
    verbose: bool = False,
) -> DepthSweepReport:
    """Plant a single needle at each ``(depth, corpus_tokens)`` cell and
    measure recall + mean rank.

    A flat matrix indicates no positional bias — healthy. A degrading
    last column or last row points at the chunker's tail handling or
    the embed-store search ordering.
    """
    report = DepthSweepReport(
        depths=tuple(depths),
        corpus_tokens_list=tuple(corpus_tokens_list),
        n_samples_per_cell=n_samples_per_cell,
        top_k=top_k, seed=seed, embedder_id=embedder_id, device=device,
    )
    cell_rng = random.Random(seed)
    for depth in depths:
        for corpus_tokens in corpus_tokens_list:
            cell_hits = 0
            cell_ranks: list[int] = []
            cell_wall_secs: list[float] = []
            for sample_i in range(n_samples_per_cell):
                sample_seed = cell_rng.randint(0, 2**31 - 1)
                if verbose:
                    print(
                        f"[depth] depth={depth:.2f} tokens={corpus_tokens:,} "
                        f"sample {sample_i+1}/{n_samples_per_cell} "
                        f"seed={sample_seed}",
                        file=sys.stderr,
                    )
                rank, wall = _run_one_depth(
                    depth=depth, corpus_tokens=corpus_tokens,
                    sample_seed=sample_seed,
                    embedder_id=embedder_id, device=device, top_k=top_k,
                )
                cell_wall_secs.append(wall)
                if rank is not None:
                    cell_hits += 1
                    cell_ranks.append(rank)
            recall = cell_hits / max(n_samples_per_cell, 1)
            mean_rank = (
                sum(cell_ranks) / len(cell_ranks) if cell_ranks else None
            )
            mean_wall = (
                sum(cell_wall_secs) / len(cell_wall_secs)
                if cell_wall_secs else 0.0
            )
            report.cells.append(DepthCellResult(
                depth=depth, corpus_tokens=corpus_tokens,
                n_samples=n_samples_per_cell,
                hits=cell_hits, recall=recall, mean_rank=mean_rank,
                mean_wall_secs=mean_wall,
            ))
    return report


def _run_one_depth(
    depth: float,
    corpus_tokens: int,
    sample_seed: int,
    embedder_id: str,
    device: str,
    top_k: int,
) -> tuple[Optional[int], float]:
    """One depth-sweep sample. Returns ``(needle_rank, wall_secs)``."""
    t_total = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="niah-depth-") as tmp:
        tmp_path = Path(tmp)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        code = f"DEPTH{sample_seed % 1_000_000:06d}"
        needle = SINGLE_NEEDLE_TEMPLATE.format(code=code)
        _build_haystack(
            corpus_tokens=corpus_tokens, root=corpus, seed=sample_seed,
            needles=[(depth, needle)],
        )
        _, searcher, chunk_store, embed_store, _, _ = _build_pipeline(
            corpus_root=corpus, cache_root=cache,
            embedder_id=embedder_id, device=device,
        )
        try:
            r = _search_with_floor_off(
                searcher,
                f"{SINGLE_NEEDLE_QUERY} access code {code}",
                top_k=top_k, cwd=str(corpus),
            )
            rank = _find_needle_rank(r.chunks, code)
        finally:
            chunk_store.close()
            embed_store.close()
    return rank, time.perf_counter() - t_total


# ─────────────────────────────────────────────── pretty-printing

def render_multi_needle(report: MultiNeedleReport) -> str:
    """Markdown table for a multi-needle report."""
    head = (
        "| n_facts | corpus_tokens | samples | R@K (per-fact) "
        "| All-of-N@K | mean wall (s) |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    body = (
        f"| {report.n_facts} | {report.corpus_tokens:,} "
        f"| {report.n_samples} "
        f"| {report.per_fact_recall:.2f} "
        f"| {report.all_of_n_rate:.2f} "
        f"| {report.mean_wall_secs:.1f} |\n"
    )
    return head + body


def render_near_needle(report: NearNeedleReport) -> str:
    head = (
        "| corpus_tokens | samples | real recall | "
        "real_beats_distractor | mean wall (s) |\n"
        "|---:|---:|---:|---:|---:|\n"
    )
    body = (
        f"| {report.corpus_tokens:,} | {report.n_samples} "
        f"| {report.real_recall:.2f} "
        f"| {report.real_beats_distractor_rate:.2f} "
        f"| {report.mean_wall_secs:.1f} |\n"
    )
    return head + body


def render_depth_sweep(report: DepthSweepReport) -> str:
    """Pivot the ``cells`` list into a depth × size matrix."""
    cells_by_key = {
        (c.depth, c.corpus_tokens): c for c in report.cells
    }
    sizes = report.corpus_tokens_list
    head_cells = " | ".join(f"{s:,}" for s in sizes)
    head = f"| depth | {head_cells} |\n"
    sep = "|---:" + "|---:" * len(sizes) + "|\n"
    body = []
    for depth in report.depths:
        row = [f"{depth*100:.0f}%"]
        for size in sizes:
            cell = cells_by_key.get((depth, size))
            row.append(
                f"{cell.recall:.2f}" if cell is not None else "—"
            )
        body.append("| " + " | ".join(row) + " |")
    return head + sep + "\n".join(body) + "\n"


# ─────────────────────────────────────────────── output snapshots

def _output_dir() -> Path:
    """Where to drop ``benchmark/niah_advanced/...`` snapshots.

    Resolves relative to the repo root (parent of the package), so the
    file lands inside the worktree regardless of where the bench is
    invoked from.
    """
    here = Path(__file__).resolve()
    # longctx_daemon/eval/niah_advanced.py → repo root is two up.
    repo = here.parent.parent.parent
    out = repo / "benchmark" / "niah_advanced"
    return out


def _save_json_snapshot(name: str, payload) -> Path:
    """Drop a JSON snapshot at
    ``benchmark/niah_advanced/<name>_<date>.json``.

    Date is the system date so consecutive runs replace cleanly.
    Returns the path written.
    """
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y-%m-%d")
    p = out / f"{name}_{date}.json"
    p.write_text(json.dumps(payload, indent=2, default=_json_default))
    return p


def _json_default(obj):
    """``json.dumps`` fallback for our dataclass-of-dataclass reports."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# ─────────────────────────────────────────────── CLI

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Args every variant subcommand accepts."""
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true",
                   help="emit report as JSON instead of table")
    p.add_argument("--save", action="store_true",
                   help="also write a JSON snapshot to "
                   "benchmark/niah_advanced/")
    p.add_argument("--verbose", "-v", action="store_true")


def _parse_tokens_list(s: str) -> list[int]:
    """Parse ``"100000,1000000"`` → [100000, 1000000]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_depths(s: str) -> list[float]:
    """Parse ``"0.1,0.5,0.9"`` → [0.1, 0.5, 0.9]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    # multi-needle
    p_multi = sp.add_parser("multi-needle")
    _add_common_args(p_multi)
    p_multi.add_argument("--n-facts", type=int, default=3)
    p_multi.add_argument("--tokens", type=int, default=100_000)

    # near-needle
    p_near = sp.add_parser("near-needle")
    _add_common_args(p_near)
    p_near.add_argument("--tokens", type=int, default=100_000)

    # depth
    p_depth = sp.add_parser("depth")
    _add_common_args(p_depth)
    p_depth.add_argument(
        "--tokens", default="100000,1000000",
        help="comma-separated corpus sizes",
    )
    p_depth.add_argument(
        "--depths", default="0.1,0.3,0.5,0.7,0.9",
        help="comma-separated depths in [0,1]",
    )
    p_depth.add_argument(
        "--include-12m", action="store_true",
        help="include the 12M-token cell (slow; opt-in only)",
    )

    # all
    p_all = sp.add_parser(
        "all", help="run multi-needle + near-needle + depth with "
        "sensible smoke defaults",
    )
    _add_common_args(p_all)

    args = ap.parse_args()

    if args.cmd == "multi-needle":
        return _cmd_multi_needle(args)
    if args.cmd == "near-needle":
        return _cmd_near_needle(args)
    if args.cmd == "depth":
        return _cmd_depth(args)
    if args.cmd == "all":
        return _cmd_all(args)
    return 1


def _cmd_multi_needle(args) -> int:
    report = run_multi_needle(
        n_facts=args.n_facts, corpus_tokens=args.tokens,
        n_samples=args.samples, seed=args.seed,
        embedder_id=args.embedder, device=args.device,
        top_k=args.top_k, verbose=args.verbose,
    )
    _emit(args, report, render_multi_needle, "multi_needle")
    return 0


def _cmd_near_needle(args) -> int:
    report = run_needle_near_needle(
        corpus_tokens=args.tokens, n_samples=args.samples,
        seed=args.seed, embedder_id=args.embedder, device=args.device,
        top_k=args.top_k, verbose=args.verbose,
    )
    _emit(args, report, render_near_needle, "near_needle")
    return 0


def _cmd_depth(args) -> int:
    sizes = _parse_tokens_list(args.tokens)
    if not args.include_12m:
        sizes = [s for s in sizes if s < 12_000_000]
    report = run_depth_sweep(
        depths=_parse_depths(args.depths),
        corpus_tokens_list=sizes,
        n_samples_per_cell=args.samples,
        seed=args.seed, embedder_id=args.embedder, device=args.device,
        top_k=args.top_k, verbose=args.verbose,
    )
    _emit(args, report, render_depth_sweep, "depth")
    return 0


def _cmd_all(args) -> int:
    """Smoke run across all three variants. Sized to fit in ~10 min total.

    Defaults: 5 samples per variant, 100K tokens. Each variant's
    output is rendered + optionally saved.
    """
    args.tokens = 100_000
    args.n_facts = 3
    if args.samples > 10:
        # Cap to keep ``all`` runs reasonable.
        args.samples = 5
    rc = _cmd_multi_needle(args)
    if rc:
        return rc
    rc = _cmd_near_needle(args)
    if rc:
        return rc
    # Depth-sweep with the same default size grid.
    args.tokens = "100000,1000000"
    args.depths = "0.1,0.3,0.5,0.7,0.9"
    args.include_12m = False
    return _cmd_depth(args)


def _emit(args, report, renderer, name: str) -> None:
    """Print JSON or markdown, optionally save a snapshot."""
    if args.json:
        print(json.dumps(asdict(report), indent=2, default=_json_default))
    else:
        print(renderer(report))
    if args.save:
        path = _save_json_snapshot(name, report)
        print(f"# saved snapshot: {path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
