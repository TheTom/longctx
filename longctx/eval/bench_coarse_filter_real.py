"""Real-corpus NIAH bench for the 12M coarse filter.

The synthetic bench (`bench_coarse_filter`) confirms the pipeline
works on a controlled haystack. This bench runs the same
Chunker → CoarseFilter pipeline against a real prose corpus walked
from disk so we have a non-toy data point: realistic vocabulary
distribution, real sentence variety, real topical coherence.

Privacy: only summary metrics are saved (token count, chunk count,
recall, latency). No corpus text is logged or written to disk by
this script. The corpus path is the user's responsibility.

Usage:
    python -m longctx.eval.bench_coarse_filter_real \\
        --corpus-dir ~/dev/longctx \\
        --extensions .py,.md \\
        --top-k 1000

Plants a synthetic needle in the corpus at a known offset, runs the
pipeline, reports timing + recall. The needle is generated each run
so cache hit rates aren't masked by a stale planted phrase.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from longctx.rag.chunker import Chunker
from longctx.rag.coarse_filter import CoarseFilter

NEEDLE_TEMPLATE = (
    "[REAL_CORPUS_NEEDLE] Project NOVA was provisioned with access "
    "code {code}. All technicians must memorise this code prior to "
    "deployment. [/REAL_CORPUS_NEEDLE]"
)

PARAPHRASE_QUERIES = [
    "What is the access code for Project NOVA?",
    "Which numeric credential was issued to NOVA technicians?",
    "Tell me the secret string assigned to NOVA before deployment.",
    "Locate the digit sequence NOVA personnel must memorise.",
]


@dataclass
class RealCorpusResult:
    corpus_dir: str
    extensions: list[str]
    n_files: int
    total_chars: int
    approx_tokens: int
    n_chunks: int
    n_kept: int
    needle_position_chars: int
    needle_in_topk: bool
    needle_rank: int | None
    walk_secs: float
    chunk_secs: float
    filter_secs: float
    total_secs: float
    top_k: int
    multi_query: bool
    embedder_model: str
    device: str

    def report(self) -> str:
        rank = f"#{self.needle_rank}" if self.needle_rank is not None else "absent"
        verdict = "HIT" if self.needle_in_topk else "MISS"
        return (
            f"corpus={self.corpus_dir} files={self.n_files} "
            f"~{self.approx_tokens:,} tokens chunks={self.n_chunks:,} "
            f"→ kept={self.n_kept:,} | {verdict} (rank {rank} in top-{self.top_k}) "
            f"| walk={self.walk_secs:.1f}s chunk={self.chunk_secs:.1f}s "
            f"filter={self.filter_secs:.1f}s total={self.total_secs:.1f}s"
        )


def _walk(corpus_dir: Path, extensions: tuple[str, ...],
          max_files: int | None,
          quiet: bool) -> tuple[str, int]:
    """Walk directory, concatenate matching files, return (text, n_files)."""
    parts: list[str] = []
    n_files = 0
    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        if extensions and p.suffix.lower() not in extensions:
            continue
        # Skip dot-dirs (.git, .obsidian, .venv, etc.) anywhere in path.
        if any(part.startswith(".") for part in p.parts):
            continue
        try:
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        n_files += 1
        if max_files is not None and n_files >= max_files:
            break
    if not quiet:
        print(f"[bench] walked {n_files} files from {corpus_dir}",
              file=sys.stderr)
    return "\n\n".join(parts), n_files


def run_real_bench(
    corpus_dir: Path,
    extensions: tuple[str, ...] = (".md", ".txt", ".rst"),
    max_files: int | None = None,
    top_k: int = 1000,
    tokens_per_chunk: int = 2048,
    embedder_model: str = "BAAI/bge-small-en-v1.5",
    device: str = "auto",
    seed: int = 0,
    multi_query: bool = True,
    quiet: bool = False,
    batch_size: int = 64,
    max_seq_length: int | None = None,
) -> RealCorpusResult:
    """Run the bench against a real corpus on disk."""
    rng = random.Random(seed)
    code = f"{seed:06d}_NOVA_real_{rng.randrange(10**9):09d}"
    needle = NEEDLE_TEMPLATE.format(code=code)

    t0 = time.time()
    text, n_files = _walk(corpus_dir, extensions, max_files, quiet)
    walk_secs = time.time() - t0

    if not text:
        raise ValueError(f"corpus_dir {corpus_dir} produced empty text — "
                         f"check --extensions and --max-files")

    # Plant needle at the midpoint between two file boundaries (the
    # \n\n joins) so it doesn't split a real document mid-sentence.
    joins = [i for i in range(len(text) - 1) if text[i:i+2] == "\n\n"]
    if joins:
        midpoint = joins[len(joins) // 2] + 2
    else:
        midpoint = len(text) // 2
    text = text[:midpoint] + needle + "\n\n" + text[midpoint:]
    needle_pos = midpoint

    chunker = Chunker(tokens_per_chunk=tokens_per_chunk)
    t0 = time.time()
    chunks = chunker.chunk(text)
    chunk_secs = time.time() - t0
    if not quiet:
        print(f"[bench] chunked {len(chunks):,} chunks in {chunk_secs:.1f}s",
              file=sys.stderr)

    cf = CoarseFilter(
        embedder_model=embedder_model, device=device,
        batch_size=batch_size, max_seq_length=max_seq_length,
    )
    if multi_query:
        queries = list(PARAPHRASE_QUERIES)
        t0 = time.time()
        kept = cf.filter_multi_query(chunks, queries, top_k=top_k)
        filter_secs = time.time() - t0
    else:
        t0 = time.time()
        kept = cf.filter(chunks, PARAPHRASE_QUERIES[0], top_k=top_k)
        filter_secs = time.time() - t0
    if not quiet:
        print(f"[bench] coarse-filter kept {len(kept):,} chunks "
              f"in {filter_secs:.1f}s", file=sys.stderr)

    needle_rank: int | None = None
    for i, (c, _score) in enumerate(kept, start=1):
        if code in c.text:
            needle_rank = i
            break

    return RealCorpusResult(
        corpus_dir=str(corpus_dir),
        extensions=list(extensions),
        n_files=n_files,
        total_chars=len(text),
        approx_tokens=len(text) // 4,
        n_chunks=len(chunks),
        n_kept=len(kept),
        needle_position_chars=needle_pos,
        needle_in_topk=needle_rank is not None,
        needle_rank=needle_rank,
        walk_secs=walk_secs,
        chunk_secs=chunk_secs,
        filter_secs=filter_secs,
        total_secs=walk_secs + chunk_secs + filter_secs,
        top_k=top_k,
        multi_query=multi_query,
        embedder_model=embedder_model,
        device=cf.device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--extensions", default=".md,.txt,.rst",
                    help="comma-separated list of file extensions to read")
    ap.add_argument("--max-files", type=int, default=None,
                    help="stop after N files (for partial-corpus runs)")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--tokens-per-chunk", type=int, default=2048)
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-multi-query", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="encode batch size; drop to 1-4 for heavy "
                    "embedders like bge-m3 to avoid device OOM")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="cap embedder seq length (e.g. 512) to bound "
                    "per-chunk activation memory on big BERTs")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    extensions = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())
    res = run_real_bench(
        corpus_dir=args.corpus_dir.expanduser().resolve(),
        extensions=extensions,
        max_files=args.max_files,
        top_k=args.top_k,
        tokens_per_chunk=args.tokens_per_chunk,
        embedder_model=args.embedder,
        device=args.device,
        seed=args.seed,
        multi_query=not args.no_multi_query,
        quiet=args.quiet,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    if args.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        print(res.report())


if __name__ == "__main__":
    main()
