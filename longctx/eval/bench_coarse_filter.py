"""Self-contained NIAH-style bench for the 12M coarse-filter pipeline.

Validates the **retrieval** stage in isolation — no generator server, no
model tokens. Builds a synthetic haystack of N tokens, plants a needle
at a known position, then runs Chunker → CoarseFilter and reports:

  - per-stage wall time (chunk, BM25, dense embed, fuse)
  - top-K recall (did the needle land in the top-K result?)
  - peak chunk count and the fraction kept by the prefilter

This is the *cheapest possible* end-to-end check that the coarse stage
stays useful at huge inputs. Pair with the full v0.3 svc + a generator
to measure end-task accuracy; that lives elsewhere.

Usage:
    python -m longctx.eval.bench_coarse_filter --tokens 100_000
    python -m longctx.eval.bench_coarse_filter --tokens 12_000_000 --top-k 1000

Defaults target a fast smoke run. ``--quiet`` suppresses progress prints
for use in tests.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass

from longctx.rag.chunker import Chunker
from longctx.rag.coarse_filter import CoarseFilter

# A short, memorable needle phrase + a unique digit code. The phrase is
# chosen to share zero high-IDF terms with the filler, so BM25 + dense
# both have a clean shot at retrieving it.
NEEDLE_TEMPLATE = (
    "IMPORTANT: Project NOVA was provisioned with access code "
    "{code}. All technicians must memorise this code prior to deployment."
)
NEEDLE_QUERY = "What is the access code for Project NOVA?"

FILLER_SENTENCES = [
    "The routine quarterly procedure required careful review.",
    "Documented operational records were checked during the cycle.",
    "The technicians reviewed standard checklists.",
    "All routine items were confirmed in the assessment.",
    "Logistics teams coordinated supplies for the next phase.",
    "Daily reports were filed without irregularity.",
    "Compliance review noted no exceptions for the period.",
    "Inventory checks aligned with the master schedule.",
]

# A "hard mode" filler pool that overlaps topically with the needle.
# Mixes plausible-sounding talk about projects, codes, and access
# controls so the needle's semantic signal has actual competition. Used
# when ``--hard`` is passed on the CLI.
HARD_FILLER_SENTENCES = [
    "Project ATLAS shipped its phase-two milestones on schedule.",
    "Project HYDRA used a rotating credential scheme during deployment.",
    "All access codes for Project ORION were retired last quarter.",
    "Project TITAN technicians completed onboarding without incident.",
    "The deprecated Project GEMINI codes were archived for compliance.",
    "Project PEGASUS ran a code-rotation drill ahead of the release.",
    "A prior memo flagged Project NOVA as decommissioned in 2024.",
    "The training manual lists access codes only as historical examples.",
    "Project ZEPHYR replaced static codes with hardware tokens.",
    "Project KESTREL deployment notes referenced an unrelated 6-digit ID.",
    "Project NOVA training documents were superseded by the new SOP.",
    "All projects now use centrally provisioned credentials.",
]

# Paraphrase queries — semantically equivalent ways to ask the question
# that share *zero* high-IDF terms with the planted needle text. Forces
# the dense path to do real work.
PARAPHRASE_QUERIES = [
    "What is the access code for Project NOVA?",
    "Which numeric credential was issued to NOVA technicians?",
    "Tell me the secret string assigned to NOVA before deployment.",
    "Locate the digit sequence NOVA personnel must memorise.",
]


@dataclass
class BenchResult:
    target_tokens: int
    n_chunks: int
    n_kept: int
    needle_position_chars: int
    needle_in_topk: bool
    needle_rank: int | None
    chunk_secs: float
    filter_secs: float
    total_secs: float
    top_k: int
    embedder_model: str
    device: str

    def report(self) -> str:
        rank = f"#{self.needle_rank}" if self.needle_rank is not None else "absent"
        return (
            f"target~{self.target_tokens:,} tokens | "
            f"chunks={self.n_chunks:,} → kept={self.n_kept:,} | "
            f"needle={'HIT' if self.needle_in_topk else 'MISS'} "
            f"(rank {rank} in top-{self.top_k}) | "
            f"chunk={self.chunk_secs:.1f}s filter={self.filter_secs:.1f}s "
            f"total={self.total_secs:.1f}s"
        )


def _build_haystack(
    target_tokens: int,
    code: str,
    seed: int = 0,
    hard: bool = False,
) -> tuple[str, int]:
    """Return (haystack_text, needle_char_position).

    Generates filler from a sentence pool, joins with whitespace,
    inserts the needle near the midpoint. Token count is approximate (we
    use the chunker's char-proxy assumption of 4 chars/token); the bench
    isn't sensitive to ±10% on token count.

    ``hard=True`` swaps in a topically-overlapping filler pool that
    mentions projects, access codes, and credentials — the needle has
    actual semantic competition. ``hard=False`` (default) uses bland
    operational filler with no shared lexicon, which is the easy case
    used in the headline numbers.
    """
    rng = random.Random(seed)
    target_chars = target_tokens * 4
    needle = NEEDLE_TEMPLATE.format(code=code)
    pool = HARD_FILLER_SENTENCES if hard else FILLER_SENTENCES

    parts: list[str] = []
    so_far = 0
    while so_far < target_chars:
        s = rng.choice(pool)
        parts.append(s)
        so_far += len(s) + 1  # +1 for join space
    midpoint = len(parts) // 2
    parts.insert(midpoint, needle)

    text = " ".join(parts)
    needle_pos = text.find(needle)
    return text, needle_pos


def run_bench(
    target_tokens: int,
    top_k: int = 1000,
    tokens_per_chunk: int = 2048,
    embedder_model: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    seed: int = 0,
    quiet: bool = False,
    hard: bool = False,
    query_index: int = 0,
    multi_query: bool = False,
    batch_size: int = 64,
    max_seq_length: int | None = None,
) -> BenchResult:
    """Run one bench rung end-to-end. Returns a BenchResult.

    ``hard=True`` uses topically-overlapping filler. ``query_index``
    picks one of the paraphrase queries (0 = the literal NIAH question;
    higher indices share fewer surface tokens with the needle so the
    dense path has more work to do). ``multi_query=True`` ignores
    ``query_index`` and runs all four paraphrases through
    ``filter_multi_query`` — exercises the hard-mode-uplift feature."""
    code = f"{seed:06d}_NOVA_{target_tokens}"  # unique per rung
    if not quiet:
        mode = "hard" if hard else "easy"
        print(f"[bench] building {mode} haystack for ~{target_tokens:,} tokens",
              file=sys.stderr)
    text, needle_pos = _build_haystack(target_tokens, code, seed=seed, hard=hard)

    chunker = Chunker(tokens_per_chunk=tokens_per_chunk)
    t0 = time.time()
    chunks = chunker.chunk(text)
    chunk_secs = time.time() - t0
    if not quiet:
        print(f"[bench] chunked: {len(chunks):,} chunks in {chunk_secs:.1f}s",
              file=sys.stderr)

    if not quiet:
        print(f"[bench] loading CoarseFilter ({embedder_model} on {device})",
              file=sys.stderr)
    cf = CoarseFilter(
        embedder_model=embedder_model, device=device,
        batch_size=batch_size, max_seq_length=max_seq_length,
    )

    if multi_query:
        # All paraphrases. In easy mode tag each with the unique code so
        # the lexical signal still fires; in hard mode keep them clean
        # so they avoid the topical-overlap trap.
        suffix = "" if hard else f" access code {code}"
        queries = [q + suffix for q in PARAPHRASE_QUERIES]
        if not quiet:
            print(f"[bench] multi-query: {len(queries)} paraphrases",
                  file=sys.stderr)
        t0 = time.time()
        kept = cf.filter_multi_query(chunks, queries, top_k=top_k)
        filter_secs = time.time() - t0
    else:
        base_query = PARAPHRASE_QUERIES[query_index % len(PARAPHRASE_QUERIES)]
        query = base_query if hard else f"{base_query} access code {code}"
        if not quiet:
            print(f"[bench] query: {query!r}", file=sys.stderr)
        t0 = time.time()
        kept = cf.filter(chunks, query, top_k=top_k)
        filter_secs = time.time() - t0

    if not quiet:
        print(f"[bench] coarse-filter kept {len(kept):,} chunks "
              f"in {filter_secs:.1f}s", file=sys.stderr)

    needle_rank: int | None = None
    for i, (c, _score) in enumerate(kept, start=1):
        if code in c.text:
            needle_rank = i
            break

    return BenchResult(
        target_tokens=target_tokens,
        n_chunks=len(chunks),
        n_kept=len(kept),
        needle_position_chars=needle_pos,
        needle_in_topk=needle_rank is not None,
        needle_rank=needle_rank,
        chunk_secs=chunk_secs,
        filter_secs=filter_secs,
        total_secs=chunk_secs + filter_secs,
        top_k=top_k,
        embedder_model=embedder_model,
        device=cf.device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", type=int, default=100_000,
                    help="approximate haystack size in tokens (default 100K)")
    ap.add_argument("--top-k", type=int, default=1000,
                    help="how many chunks the coarse filter keeps")
    ap.add_argument("--tokens-per-chunk", type=int, default=2048,
                    help="chunker target chunk size (default 2K tokens)")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="emit BenchResult as JSON on stdout")
    ap.add_argument("--hard", action="store_true",
                    help="use topically-overlapping filler that mentions "
                    "projects + access codes (the realistic-haystack case)")
    ap.add_argument("--query-index", type=int, default=0,
                    help="paraphrase query selector (0 = literal, higher "
                    "indices share fewer tokens with the needle)")
    ap.add_argument("--multi-query", action="store_true",
                    help="fuse all 4 paraphrases via filter_multi_query "
                    "(overrides --query-index). Best on hard mode.")
    args = ap.parse_args()

    res = run_bench(
        target_tokens=args.tokens,
        top_k=args.top_k,
        tokens_per_chunk=args.tokens_per_chunk,
        embedder_model=args.embedder,
        device=args.device,
        seed=args.seed,
        quiet=args.quiet,
        hard=args.hard,
        query_index=args.query_index,
        multi_query=args.multi_query,
    )

    if args.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        print(res.report())


if __name__ == "__main__":
    main()
