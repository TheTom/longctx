"""longctx replay — replay captured interactions.jsonl against the
current index for regression testing.

Per spec §14.8: ``interactions.jsonl`` captures the FULL request +
response per MCP call. Replaying lets us answer "did this index/
embedder/chunker change improve or degrade what real agents asked
yesterday" — far more meaningful than synthetic NIAH alone.

For each captured ``search_codebase`` call:

  1. Re-issue the same query against the current index
  2. Diff: which top-K chunks changed? Did the relevance score change?
  3. Print a short per-call delta + summary stats

Phase 2.0 scope: read-only replay against a running daemon's index
(via direct ``Searcher`` import) OR against the captured replay log
file. No write side-effects on the index.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class ReplayCall:
    """One captured MCP call from interactions.jsonl."""
    trace_id: str
    tool: str
    args: dict
    result: dict


@dataclass(frozen=True)
class ReplayDelta:
    """Per-call diff between captured response and current response."""
    trace_id: str
    tool: str
    query: Optional[str]
    captured_top_files: tuple[str, ...]
    current_top_files: tuple[str, ...]
    rank_changes: int          # how many positions shifted in top-K
    score_drift: float         # max absolute relevance_score delta in top-K


# ---------------------------------------------------------- iter

def iter_replay_log(path: Path) -> Iterator[ReplayCall]:
    """Stream ``interactions.jsonl`` (or its gzipped shards) one
    record at a time. Tolerates malformed lines (skipped + warned)."""
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: path.open(encoding="utf-8")
    with opener() as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  warn: skip line {n}: {e}", file=sys.stderr)
                continue
            yield ReplayCall(
                trace_id=rec.get("trace_id", "?"),
                tool=rec.get("tool", "?"),
                args=rec.get("args", {}) or {},
                result=rec.get("result", {}) or {},
            )


# ----------------------------------------------------- compare two top-K

def diff_top_k(captured: dict, current: dict) -> ReplayDelta:
    """Compute per-call diff between two search responses."""
    cap_files = tuple(_files_from_result(captured))
    cur_files = tuple(_files_from_result(current))
    rank_changes = sum(
        1 for i, f in enumerate(cap_files)
        if i >= len(cur_files) or cur_files[i] != f
    )
    score_drift = _max_score_drift(captured, current)
    return ReplayDelta(
        trace_id=str(captured.get("_trace_id", "")),
        tool="search_codebase",
        query=captured.get("query"),
        captured_top_files=cap_files,
        current_top_files=cur_files,
        rank_changes=rank_changes,
        score_drift=score_drift,
    )


def _files_from_result(r: dict) -> list[str]:
    if "files" in r:
        return list(r["files"])
    chunks = r.get("chunks") or r.get("matches") or ()
    return [
        f"{c.get('file_path', '?')}:{c.get('start_line', 0)}-"
        f"{c.get('end_line', 0)}"
        for c in chunks
    ]


def _max_score_drift(captured: dict, current: dict) -> float:
    cap = {
        c.get("file_path"): float(c.get("relevance_score", 0.0))
        for c in (captured.get("chunks") or ())
    }
    cur = {
        c.get("file_path"): float(c.get("relevance_score", 0.0))
        for c in (current.get("chunks") or ())
    }
    if not cap or not cur:
        return 0.0
    overlap = set(cap) & set(cur)
    if not overlap:
        return 0.0
    return max(abs(cap[k] - cur[k]) for k in overlap)


# ------------------------------------------------------------- CLI

def cmd_replay(args: argparse.Namespace) -> int:
    """``longctx replay <path>`` entry point.

    Reads the captured log; for each ``search_codebase`` call, re-runs
    the query against the current index and prints a per-call delta
    plus aggregate stats (how many calls had any change, distribution
    of rank shifts, score drift histogram).

    Phase 2.0: prints a summary; doesn't write a regression report
    file. ``--json`` mode emits per-call deltas as JSONL for piping
    into a richer analyzer.
    """
    log_path = Path(args.log_path).expanduser()
    if not log_path.exists():
        print(f"replay log not found: {log_path}", file=sys.stderr)
        return 1

    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".cache" / "longctx"
    )
    db_path = cache_dir / "index.db"
    if not db_path.exists():
        print(f"no current index at {db_path}", file=sys.stderr)
        return 1

    # Lazy imports to avoid pulling sentence-transformers when this
    # module is loaded for argparse-only paths.
    from sentence_transformers import SentenceTransformer

    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import _embedder_sha256
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    chunk_store = SqliteChunkStore(db_path)
    try:
        embedder = SentenceTransformer(
            args.embedder, device=_resolve_device(args.device),
        )
        embed_dim = (
            embedder.get_embedding_dimension()
            if hasattr(embedder, "get_embedding_dimension")
            else embedder.get_sentence_embedding_dimension()
        )
        embed_store = MemmapEmbedStore(
            cache_dir / "embeddings",
            model_name=args.embedder,
            model_sha256=_embedder_sha256(embedder),
            dim=embed_dim,
            on_mismatch="warn",
        )
        searcher = Searcher(
            chunk_store=chunk_store, embed_store=embed_store,
            embedder=embedder, config=SearcherConfig(),
        )

        n_calls = 0
        n_changed = 0
        max_drift = 0.0

        for call in iter_replay_log(log_path):
            if call.tool != "search_codebase":
                continue
            n_calls += 1
            query = call.args.get("query")
            if not query:
                continue
            try:
                current = searcher.search(
                    query=query,
                    cwd=call.args.get("cwd"),
                    project=call.args.get("project"),
                    max_tokens=call.args.get("max_tokens", 4096),
                )
            except Exception as e:
                print(f"  trace {call.trace_id}: search failed: {e}",
                      file=sys.stderr)
                continue

            captured_for_diff = {
                "chunks": call.result.get("chunks", []),
                "_trace_id": call.trace_id,
                "query": query,
            }
            current_for_diff = _serialize_search_result(current)
            delta = diff_top_k(captured_for_diff, current_for_diff)
            if delta.rank_changes > 0:
                n_changed += 1
            if delta.score_drift > max_drift:
                max_drift = delta.score_drift

            if args.json:
                print(json.dumps({
                    "trace_id": delta.trace_id, "tool": delta.tool,
                    "query": delta.query,
                    "captured": list(delta.captured_top_files),
                    "current": list(delta.current_top_files),
                    "rank_changes": delta.rank_changes,
                    "score_drift": delta.score_drift,
                }))
            elif delta.rank_changes > 0 or delta.score_drift > 0.001:
                print(f"  ▲ {delta.trace_id}  rank_changes="
                      f"{delta.rank_changes}  drift="
                      f"{delta.score_drift:.4f}  q={query!r}")

        print()
        print(f"replayed {n_calls} search_codebase call(s)")
        print(f"  {n_changed} changed top-K ({n_changed / max(n_calls, 1) * 100:.1f}%)")
        print(f"  max relevance_score drift: {max_drift:.4f}")
        return 0
    finally:
        chunk_store.close()


def _serialize_search_result(result) -> dict:
    """Convert a SearchResult dataclass into the same dict shape we
    diff against in the replay log."""
    return {
        "chunks": [
            {
                "file_path": c.citation.file_path,
                "start_line": c.citation.start_line,
                "end_line": c.citation.end_line,
                "relevance_score": c.relevance_score,
            }
            for c in result.chunks
        ],
    }
