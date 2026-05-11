"""longctx daemon CLI.

Three subcommands ship in Phase 2.0:

  longctx ask "<question>" --corpus-dir <path>
      Zero-config in-process retrieval. No daemon, no MCP, no service
      install. First-time users verify the pipeline works on their
      data in 60 seconds before wiring it into an agent.

  longctx serve --corpus-dir <path>
      Foreground stdio MCP server over a single project. Suitable for
      Claude Desktop's "command" launch model. Multi-root + daemon mode
      land in 2.1.

  longctx version
      Print version + dependency info.

The command surface intentionally mirrors what Phase 2.1+ will keep —
``ask`` is the per-call use case, ``serve`` is the always-on use case.
Phase 2.1 adds ``init``, ``service install/start/stop``, ``status``,
``mcp-stdio``, ``port``, ``clean``, ``replay``, ``watch``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

from longctx_daemon import __version__


# --------------------------------------------------------------- ask command

def _cmd_ask(args: argparse.Namespace) -> int:
    """Run a one-shot retrieval against an on-disk corpus.

    Loads the embedder + chunker, walks the corpus, indexes everything
    in a temporary on-disk store (or a persistent one if ``--cache-dir``
    is given), runs the search, prints results in the same shape the
    MCP server would return. No daemon, no MCP transport.
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import Indexer, IndexerConfig, _embedder_sha256
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    corpus = Path(args.corpus_dir).expanduser().resolve()
    if not corpus.is_dir():
        print(f"error: --corpus-dir not a directory: {corpus}",
              file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir \
        else Path("/tmp") / f"longctx-ask-{corpus.name}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    if args.verbose:
        print(f"[longctx ask] corpus={corpus}", file=sys.stderr)
        print(f"[longctx ask] cache={cache_dir}", file=sys.stderr)
        print(f"[longctx ask] embedder={args.embedder} on {device}",
              file=sys.stderr)

    # ---- Set up stores
    chunk_store = SqliteChunkStore(cache_dir / "index.db")
    embedder = SentenceTransformer(args.embedder, device=device)
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
        on_mismatch="warn",   # ``ask`` is exploratory; don't crash on swaps
    )

    chunker = Chunker(
        tokens_per_chunk=args.chunk_tokens,
        respect_sentences=True,
    )

    # ---- Index (incremental — second run with same cache_dir is fast)
    indexer = Indexer(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        chunker=chunker,
        config=IndexerConfig(
            embed_batch_size=args.batch_size,
            embedder_max_seq_length=args.max_seq_length,
        ),
    )
    project = corpus.name
    indexer.add_project(name=project, root_path=corpus)
    if args.verbose:
        print(f"[longctx ask] scanning {corpus} …", file=sys.stderr)
    scan = indexer.full_scan(project)
    if args.verbose:
        print(
            f"[longctx ask] {scan.n_files:,} files, "
            f"{scan.n_chunks_total:,} chunks "
            f"({scan.n_chunks_new:,} new, {scan.n_chunks_reused:,} reused) "
            f"in {scan.wall_secs:.1f}s",
            file=sys.stderr,
        )

    # ---- Search
    searcher = Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(),
    )
    result = searcher.search(
        query=args.question,
        cwd=str(corpus),
        max_tokens=args.max_tokens,
        max_results=args.top_k,
    )

    # ---- Output
    if args.json:
        out = {
            "question": args.question,
            "corpus_dir": str(corpus),
            "scope_decision": asdict(result.scope_decision),
            "latency_ms": asdict(result.latency_ms),
            "freshness": asdict(result.freshness),
            "chunks": [
                {
                    "project": c.citation.project,
                    "file_path": c.citation.file_path,
                    "start_line": c.citation.start_line,
                    "end_line": c.citation.end_line,
                    "relevance_score": c.relevance_score,
                    "token_count": c.token_count,
                    "text": c.text,
                }
                for c in result.chunks
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        _pretty_print_result(args.question, result)

    chunk_store.close()
    embed_store.close()
    return 0


def _pretty_print_result(question: str, result) -> None:
    """Human-readable terminal output. Mirrors the demo style: scope
    banner → top-K chunks with file:line citations → latency footer."""
    bar = "─" * 72
    print(bar)
    print(f"question: {question}")
    print(bar)
    print()
    print(
        f"  scope: {result.scope_decision.primary_project or '(no primary)'} "
        f"(source: {result.scope_decision.primary_source})"
    )
    if result.scope_decision.fanout_projects:
        print(f"  fan-out: {', '.join(result.scope_decision.fanout_projects)}")
    print(
        f"  freshness: "
        f"{'fresh' if result.freshness.is_fully_fresh else 'partial'}, "
        f"pending={result.freshness.pending_updates}"
    )
    print()
    if not result.chunks:
        print("  (no results)")
        print()
        return
    for rank, chunk in enumerate(result.chunks, start=1):
        cite = (f"{chunk.citation.file_path}:"
                f"{chunk.citation.start_line}-{chunk.citation.end_line}")
        snippet = textwrap.shorten(
            " ".join(chunk.text.split()), width=140, placeholder="…",
        )
        print(f"  #{rank}  score={chunk.relevance_score:.4f}  {cite}")
        print(f"      › {snippet}")
        print()
    print(bar)
    print(
        f"  total {result.latency_ms.total:.1f} ms "
        f"(embed {result.latency_ms.embed_query:.0f} ms, "
        f"bm25 {result.latency_ms.bm25_score:.0f} ms, "
        f"dense {result.latency_ms.dense_score:.0f} ms, "
        f"rrf {result.latency_ms.rrf_fuse:.0f} ms, "
        f"fetch {result.latency_ms.fetch_chunks:.0f} ms)"
    )
    print(bar)


# ------------------------------------------------------------- serve command

def _cmd_serve(args: argparse.Namespace) -> int:
    """Foreground stdio MCP server over a single corpus directory.

    Phase 2.0 ships single-project + foreground only. Daemon mode +
    SSE/streamable-http transports + multi-root land in 2.1.
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.indexer import Indexer, IndexerConfig, _embedder_sha256
    from longctx_daemon.logging import setup_logging
    from longctx_daemon.mcp_server import MCPServer
    from longctx_daemon.replay_log import ReplayLog
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    corpus = Path(args.corpus_dir).expanduser().resolve()
    if not corpus.is_dir():
        print(f"error: --corpus-dir not a directory: {corpus}",
              file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir \
        else Path.home() / ".cache" / "longctx" / corpus.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level=args.log_level,
        file_path=cache_dir / "longctx.log",
        redact_query_text=args.redact_queries,
        redact_cwd=args.redact_cwd,
    )

    device = _resolve_device(args.device)
    chunk_store = SqliteChunkStore(cache_dir / "index.db")
    embedder = SentenceTransformer(args.embedder, device=device)
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
    )
    chunker = Chunker(tokens_per_chunk=args.chunk_tokens, respect_sentences=True)

    indexer = Indexer(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        chunker=chunker,
        config=IndexerConfig(
            embed_batch_size=args.batch_size,
            embedder_max_seq_length=args.max_seq_length,
        ),
    )
    project = corpus.name
    indexer.add_project(name=project, root_path=corpus)
    print(f"[longctx serve] scanning {corpus} …", file=sys.stderr)
    scan = indexer.full_scan(project)
    print(
        f"[longctx serve] indexed {scan.n_files:,} files, "
        f"{scan.n_chunks_total:,} chunks in {scan.wall_secs:.1f}s",
        file=sys.stderr,
    )

    searcher = Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(),
    )

    replay_log = ReplayLog(cache_dir / "interactions.jsonl") \
        if args.replay_log else None

    server = MCPServer(
        searcher=searcher,
        indexer=indexer,
        chunk_store=chunk_store,
        replay_log=replay_log,
    )

    print(f"[longctx serve] mcp-stdio ready (cache: {cache_dir})", file=sys.stderr)
    asyncio.run(server.run_stdio())
    chunk_store.close()
    embed_store.close()
    if replay_log is not None:
        replay_log.close()
    return 0


# ------------------------------------------------------------ version command

def _cmd_version(args: argparse.Namespace) -> int:
    print(f"longctx-daemon {__version__}")
    return 0


# ------------------------------------------------------------ argument parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="longctx",
        description=(
            "longctx — local codebase Q&A daemon. "
            "Phase 2.0 ships ``ask`` (one-shot) and ``serve`` (stdio MCP)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- ask
    ask = sub.add_parser(
        "ask",
        help="Zero-config one-shot retrieval over a local corpus.",
        description=(
            "Indexes a directory and returns top-K chunks for a question. "
            "No daemon, no MCP, no service install — for first-time users "
            "to validate the pipeline before wiring it into an agent."
        ),
    )
    ask.add_argument("question", help="Natural-language question.")
    ask.add_argument(
        "--corpus-dir", required=True,
        help="Directory to index + search (e.g. ~/dev/myapp).",
    )
    ask.add_argument(
        "--cache-dir",
        help="Persistent index location. Defaults to /tmp; pass an explicit "
             "path under ~/.cache for incremental re-runs.",
    )
    ask.add_argument(
        "--embedder", default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformer model id.",
    )
    ask.add_argument("--device", default="auto",
                     help='"cuda" / "mps" / "cpu" / "auto" (default).')
    ask.add_argument("--top-k", type=int, default=5,
                     help="Max chunks to return (default 5).")
    ask.add_argument("--max-tokens", type=int, default=4096,
                     help="Token budget across returned chunks (default 4096).")
    ask.add_argument("--chunk-tokens", type=int, default=2048,
                     help="Tokens per chunk during indexing (default 2048).")
    ask.add_argument("--batch-size", type=int, default=64,
                     help="Embedding batch size (drop for heavy embedders).")
    ask.add_argument(
        "--max-seq-length", type=int, default=None,
        help="Cap embedder seq length (e.g. 512 for bge-m3 on real prose).",
    )
    ask.add_argument("--json", action="store_true",
                     help="Emit JSON instead of pretty-printed terminal output.")
    ask.add_argument("--verbose", "-v", action="store_true",
                     help="Print per-stage progress to stderr.")
    ask.set_defaults(func=_cmd_ask)

    # --- serve
    serve = sub.add_parser(
        "serve",
        help="Foreground stdio MCP server.",
        description=(
            "Indexes a directory, then runs an MCP server over stdio "
            "exposing search_codebase / list_projects / index_status. "
            "Daemon mode + SSE / streamable-http transports + multi-root "
            "land in 2.1."
        ),
    )
    serve.add_argument(
        "--corpus-dir", required=True,
        help="Directory to index + serve.",
    )
    serve.add_argument("--cache-dir",
                       help="Persistent index location "
                            "(default ~/.cache/longctx/<corpus_name>).")
    serve.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--chunk-tokens", type=int, default=2048)
    serve.add_argument("--batch-size", type=int, default=64)
    serve.add_argument("--max-seq-length", type=int, default=None)
    serve.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARN", "ERROR"])
    serve.add_argument("--redact-queries", action="store_true",
                       help="Redact query text from operational logs.")
    serve.add_argument("--redact-cwd", action="store_true",
                       help="Redact cwd from operational logs.")
    serve.add_argument("--no-replay-log", dest="replay_log",
                       action="store_false", default=True,
                       help="Disable interactions.jsonl replay capture.")
    serve.set_defaults(func=_cmd_serve)

    # --- version
    ver = sub.add_parser("version", help="Print version.")
    ver.set_defaults(func=_cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
