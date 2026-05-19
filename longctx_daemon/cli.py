"""longctx daemon CLI.

Phase 2.0 subcommands: ``ask`` (zero-config one-shot retrieval),
``serve`` (foreground stdio MCP), ``version``.

Phase 2.1 adds ``init`` (write config, run auto-discovery; supports
both interactive and non-interactive modes).

Future (2.1+): ``service install/start/stop``, ``status``,
``mcp-stdio``, ``port``, ``clean``, ``replay``, ``watch``. Stubs for
those land in the daemon (Agent F) or the wizard polish step (Agent
I); this CLI just hosts ``init`` for now.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from longctx_daemon import __version__


def _load_searcher_config_from_user_config():
    """Build a ``SearcherConfig`` populated from the user's config
    file when present. Used by both ``longctx ask`` and ``longctx
    serve`` so per-project floors set via ``longctx calibrate
    --project NAME --write-config`` actually reach the searcher
    without callers having to wire it up themselves.

    Returns a ``SearcherConfig`` with ``relevance_floor`` and
    ``project_floors`` reflecting the user's persisted choices.
    Falls back silently to the defaults when the config file is
    absent or unreadable — never blocks the search path on config
    errors.
    """
    from longctx_daemon.searcher import SearcherConfig
    try:
        from longctx_daemon.config import (
            config_path as _config_path,
            load_config,
        )
        cfg_path = _config_path()
        if not cfg_path.exists():
            return SearcherConfig()
        cfg = load_config(cfg_path)
        return SearcherConfig(
            relevance_floor=cfg.index.relevance_floor,
            project_floors=dict(cfg.index.project_floors),
        )
    except Exception:  # noqa: BLE001 — config errors fall back to defaults
        return SearcherConfig()


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
        config=_load_searcher_config_from_user_config(),
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
    """Foreground or detached MCP server over a single corpus directory.

    Phase 2.1 adds:
      * ``--port``       preferred MCP port (walks forward on collision)
      * ``--host``       bind host (default 127.0.0.1)
      * ``--transports`` comma list of {sse, streamable-http, stdio}
      * ``--daemon``     detach to background, write server.info + lock

    Default behavior remains foreground stdio when no overrides are
    given, preserving the Phase 2.0 contract for Claude Desktop's
    command-launch integration.
    """
    from sentence_transformers import SentenceTransformer

    from longctx.rag.chunker import Chunker
    from longctx.rag.pipeline import _resolve_device
    from longctx_daemon.daemon import (
        DEFAULT_MCP_PORT,
        DEFAULT_STATUS_PORT,
        Daemon,
        DaemonConfig,
        detach_and_run,
    )
    from longctx_daemon.indexer import Indexer, IndexerConfig, _embedder_sha256
    from longctx_daemon.logging import setup_logging
    from longctx_daemon.mcp_server import MCPServer
    from longctx_daemon.replay_log import ReplayLog
    from longctx_daemon.searcher import Searcher, SearcherConfig
    from longctx_daemon.storage.memmap_store import MemmapEmbedStore
    from longctx_daemon.storage.sqlite_store import SqliteChunkStore

    # ``--corpus-dir`` is repeatable. Expand each + auto-discover git
    # worktrees off each corpus root so a single ``--corpus-dir
    # ~/dev/foo`` extends coverage to every worktree (`/tmp/foo-a`,
    # `~/work/foo-feat-b`, etc.) without the user enumerating them.
    raw_corpora = args.corpus_dir
    if isinstance(raw_corpora, str):
        raw_corpora = [raw_corpora]
    corpora_specs: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for raw in raw_corpora:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            print(f"error: --corpus-dir not a directory: {root}",
                  file=sys.stderr)
            return 2
        if root in seen_paths:
            continue
        seen_paths.add(root)
        corpora_specs.append((root.name, root))
        if getattr(args, "include_worktrees", False):
            # Worktree extension. Best-effort — never fail startup if
            # git is missing or the dir isn't a repo.
            for sub_name, sub_path in _discover_worktrees(root):
                if sub_path in seen_paths:
                    continue
                seen_paths.add(sub_path)
                corpora_specs.append((sub_name, sub_path))
                print(
                    f"[longctx serve] auto-discovered worktree: "
                    f"{sub_name} -> {sub_path}",
                    file=sys.stderr,
                )

    # The cache root keys on the FIRST corpus's basename so multi-root
    # daemons get a stable cache dir tied to the primary corpus.
    primary = corpora_specs[0][1]
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir \
        else Path.home() / ".cache" / "longctx" / primary.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve transport set. Default is stdio (preserves 2.0 contract);
    # daemon mode flips the default to {sse, streamable-http}.
    transports_raw: Optional[str] = getattr(args, "transports", None)
    if transports_raw:
        transports = tuple(
            t.strip() for t in transports_raw.split(",") if t.strip()
        )
    elif args.daemon:
        transports = ("sse", "streamable-http")
    else:
        transports = ("stdio",)

    host = getattr(args, "host", None) or "127.0.0.1"
    mcp_port = getattr(args, "port", None) or DEFAULT_MCP_PORT
    status_port = getattr(args, "status_port", None) or DEFAULT_STATUS_PORT

    def _run_in_process() -> int:
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
        chunker = Chunker(
            tokens_per_chunk=args.chunk_tokens, respect_sentences=True,
        )

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
        # Index every (name, root) spec. Each becomes its own Project so
        # the scope router can fan out + cross-project search works cleanly.
        total_files = 0
        total_chunks = 0
        for project_name, project_root in corpora_specs:
            indexer.add_project(name=project_name, root_path=project_root)
            print(
                f"[longctx serve] scanning {project_name} ({project_root}) …",
                file=sys.stderr,
            )
            scan = indexer.full_scan(project_name)
            print(
                f"[longctx serve] indexed {scan.n_files:,} files, "
                f"{scan.n_chunks_total:,} chunks in {scan.wall_secs:.1f}s "
                f"({project_name})",
                file=sys.stderr,
            )
            total_files += scan.n_files
            total_chunks += scan.n_chunks_total
        if len(corpora_specs) > 1:
            print(
                f"[longctx serve] all corpora ready: "
                f"{total_files:,} files / {total_chunks:,} chunks across "
                f"{len(corpora_specs)} projects",
                file=sys.stderr,
            )

        searcher = Searcher(
            chunk_store=chunk_store,
            embed_store=embed_store,
            embedder=embedder,
            config=_load_searcher_config_from_user_config(),
        )

        replay_log = ReplayLog(cache_dir / "interactions.jsonl") \
            if args.replay_log else None

        server = MCPServer(
            searcher=searcher,
            indexer=indexer,
            chunk_store=chunk_store,
            embed_store=embed_store,
            replay_log=replay_log,
        ) if _mcpserver_accepts_embed_store() else MCPServer(
            searcher=searcher,
            indexer=indexer,
            chunk_store=chunk_store,
            replay_log=replay_log,
        )

        # Stdio-only is the simple path: no daemon orchestrator, no
        # singleton lock — preserves 2.0 behavior verbatim.
        if transports == ("stdio",):
            print(
                f"[longctx serve] mcp-stdio ready (cache: {cache_dir})",
                file=sys.stderr,
            )
            try:
                asyncio.run(server.run_stdio())
            finally:
                chunk_store.close()
                embed_store.close()
                if replay_log is not None:
                    replay_log.close()
            return 0

        # HTTP / multi-transport: hand off to the Daemon orchestrator.
        daemon_cfg = DaemonConfig(
            host=host,
            mcp_port=mcp_port,
            status_port=status_port,
            transports=transports,
        )
        daemon = Daemon(
            config=daemon_cfg,
            mcp_server=server,
            chunk_store=chunk_store,
            embed_store=embed_store,
            indexer=indexer,
            searcher=searcher,
        )
        print(
            f"[longctx serve] daemon starting on {host}:{mcp_port} "
            f"transports={','.join(transports)}",
            file=sys.stderr,
        )
        try:
            asyncio.run(daemon.run())
        finally:
            chunk_store.close()
            embed_store.close()
            if replay_log is not None:
                replay_log.close()
        return 0

    if args.daemon:
        return detach_and_run(
            _run_in_process, log_dir=cache_dir / "logs",
        )
    return _run_in_process()


def _discover_worktrees(root: Path) -> list[tuple[str, Path]]:
    """Enumerate git worktrees rooted at ``root`` (best-effort).

    Returns a list of (project_name, worktree_path) for every worktree
    EXCEPT the one at ``root`` itself (caller already includes that).
    Each worktree gets a name of the form ``<root.name>@<suffix>`` where
    suffix is the worktree's branch name when available, else the
    worktree path's basename. The leading ``refs/heads/`` prefix on
    branch names is stripped.

    Returns ``[]`` on any failure — `git` missing, not a repo, parse
    error, missing worktree paths. Worktree discovery is a nice-to-have;
    a corpus that's not a git repo (or whose git binary is missing)
    still gets indexed as just its root path.

    Format reference: ``git worktree list --porcelain`` emits blank-
    line-separated records, each with lines like::

        worktree /path
        HEAD <sha>
        branch refs/heads/feat/x
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []

    out: list[tuple[str, Path]] = []
    current_path: Optional[Path] = None
    current_branch: Optional[str] = None
    for line in proc.stdout.splitlines() + [""]:
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree "):]).resolve()
            current_branch = None
        elif line.startswith("branch "):
            ref = line[len("branch "):].strip()
            current_branch = ref.removeprefix("refs/heads/")
        elif not line.strip():
            # Record terminator.
            if current_path is not None and current_path != root:
                if current_path.is_dir():
                    suffix = (
                        current_branch.replace("/", "-")
                        if current_branch else current_path.name
                    )
                    name = f"{root.name}@{suffix}"
                    out.append((name, current_path))
            current_path = None
            current_branch = None
    return out


def _mcpserver_accepts_embed_store() -> bool:
    """The MCPServer signature was extended in 2.1 (Agent H wired in
    embed_store for find_related). We try the new signature first and
    fall back so this CLI keeps working against either version of the
    module."""
    import inspect
    from longctx_daemon.mcp_server import MCPServer
    try:
        sig = inspect.signature(MCPServer.__init__)
        return "embed_store" in sig.parameters
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------- calibrate command

def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Run a per-corpus relevance-floor sweep and print the table.

    Sweeps a range of dense-cosine thresholds against the abstention
    suite (30 off-corpus + 12 in-corpus queries) and prints the
    suppression / FNR trade-off plus the recommended floor (highest
    floor whose FNR is at most ``--max-fnr``, default 0.05).

    Default behavior is observation-only: print the table + exit. To
    also persist the recommendation to the user's config, pass
    ``--write-config`` — that path explicitly opts in and rewrites
    ``[index].relevance_floor``.
    """
    from longctx_daemon.eval.abstention import (
        render_sweep_table,
        sweep_floor,
    )

    corpus = Path(args.corpus_dir).expanduser().resolve()
    if not corpus.is_dir():
        print(
            f"error: --corpus-dir not a directory: {corpus}",
            file=sys.stderr,
        )
        return 2

    # Build the floor list. We construct the sequence once here so the
    # CLI surface stays simple (min/max/step) while ``sweep_floor``
    # accepts any sequence.
    floors: list[float] = []
    f = args.floor_min
    while f <= args.floor_max + 1e-9:
        floors.append(round(f, 6))
        f += args.floor_step

    if args.verbose:
        print(
            f"[longctx calibrate] sweeping {len(floors)} floors "
            f"({floors[0]:.2f}..{floors[-1]:.2f}) on {corpus}",
            file=sys.stderr,
        )

    report = sweep_floor(
        corpus,
        floors=floors,
        embedder_id=args.embedder,
        device=args.device,
        max_fnr=args.max_fnr,
    )

    if args.json:
        print(json.dumps({
            "corpus_dir": str(corpus),
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

    if args.write_config:
        return _persist_floor_to_config(
            report.recommended_floor,
            config_path_override=(
                Path(args.config).expanduser() if args.config else None
            ),
            project=getattr(args, "project", None),
            verbose=args.verbose,
        )
    return 0


def _persist_floor_to_config(
    floor: float,
    *,
    config_path_override: Optional[Path],
    project: Optional[str] = None,
    verbose: bool,
) -> int:
    """Update either ``[index].relevance_floor`` (global) or
    ``[index].project_floors[project]`` (per-project) in the user's
    config file.

    When ``project`` is provided, the floor lands in the per-project
    map — the searcher consults that map first when a search resolves
    to a matching primary project, falling back to the global floor.
    This is the Phase 3 dogfood-audit fix for "0.50 doesn't generalize
    across corpus types."

    Loads the existing config (or default if missing), updates the
    field, validates, and writes back. Idempotent. Returns the
    standard exit code (0 ok, 2 invalid, 3 IO error).
    """
    from dataclasses import replace

    from longctx_daemon.config import (
        ConfigError,
        ServiceConfig,
        config_path as _default_config_path,
        default_config,
        load_config,
        write_config,
    )

    cfg_path = config_path_override or _default_config_path()
    try:
        if cfg_path.exists():
            cfg = load_config(cfg_path)
        else:
            cfg = default_config()
    except ConfigError as exc:
        print(f"error: failed to read config: {exc}", file=sys.stderr)
        return 2

    if project:
        # Per-project: merge into the existing project_floors map.
        new_floors = dict(cfg.index.project_floors)
        new_floors[project] = float(floor)
        new_index = replace(cfg.index, project_floors=new_floors)
    else:
        # Global default.
        new_index = replace(cfg.index, relevance_floor=float(floor))
    new_cfg = ServiceConfig(
        corpus=cfg.corpus, server=cfg.server, index=new_index,
        watcher=cfg.watcher, cleanup=cfg.cleanup, search=cfg.search,
        logging=cfg.logging,
    )
    try:
        write_config(cfg_path, new_cfg)
    except OSError as exc:
        print(f"error: failed to write config: {exc}", file=sys.stderr)
        return 3

    if project:
        print(
            f"updated {cfg_path}: "
            f"[index].project_floors[{project!r}] = {floor:.2f}"
        )
    else:
        print(f"updated {cfg_path}: [index].relevance_floor = {floor:.2f}")
    if verbose:
        print(
            "  reload the daemon (`longctx reload`) to pick up the new "
            "floor.",
            file=sys.stderr,
        )
    return 0


# ------------------------------------------------------------ version command

def _cmd_version(args: argparse.Namespace) -> int:
    print(f"longctx-daemon {__version__}")
    return 0


# ------------------------------------------------------------ init command

def _interactive_pick_projects(discovered):
    """Interactive per-project picker for ``longctx init``.

    Numbered checkbox-style UI. All projects default to selected
    (except those with ``.longctxignore``). User can:

      <Enter> / 'y' / 'yes'   accept defaults
      'n' / 'none'            deselect all
      'all'                   select all (overrides .longctxignore)
      '1 3 5' or '1,3,5'      toggle specific projects by number
      'q' / 'quit'            cancel — returns None

    Returns the selected list of DiscoveredProjects, or None if the
    user bailed.
    """
    if not discovered:
        return []

    # Default selection: include all EXCEPT those marked .longctxignore
    selected: dict[int, bool] = {
        i: not p.has_longctxignore
        for i, p in enumerate(discovered, start=1)
    }

    while True:
        print()
        print(f"Discovered {len(discovered)} project(s):")
        for i, p in enumerate(discovered, start=1):
            mark = "[✓]" if selected[i] else "[ ]"
            note = " (.longctxignore)" if p.has_longctxignore else ""
            print(f"  {mark} {i}. {p.name}{note}  —  {p.sentinel}")
        print()
        try:
            ans = input(
                "[Enter] accept · 'all' / 'none' · '1 3 5' to toggle · "
                "'q' to cancel: "
            ).strip().lower()
        except EOFError:
            ans = ""

        if ans in ("", "y", "yes"):
            return [p for i, p in enumerate(discovered, start=1) if selected[i]]
        if ans in ("q", "quit", "exit"):
            return None
        if ans in ("all",):
            for i in selected:
                selected[i] = True
            continue
        if ans in ("n", "none"):
            for i in selected:
                selected[i] = False
            continue

        # Parse toggle list
        tokens = [t for t in ans.replace(",", " ").split() if t]
        any_valid = False
        for t in tokens:
            if not t.isdigit():
                continue
            n = int(t)
            if 1 <= n <= len(discovered):
                selected[n] = not selected[n]
                any_valid = True
        if not any_valid:
            print("(no valid toggles parsed; try again)", file=sys.stderr)


def _cmd_init(args: argparse.Namespace) -> int:
    """Run ``longctx init`` — set up config + auto-discover projects.

    Two flavors:
      * ``--non-interactive``: write a default config (with overrides
        from --parent-dir / --include / --no-auto-discover / env vars),
        report the discovered project count, exit. No prompts. PRD §11.1.
      * Interactive (default): walk the user through parent_dir +
        per-project pick. The full Charmbracelet-style picker UI is
        the integration step; this minimal version prompts for
        parent_dir then accepts/declines per project.

    Both paths:
      * Refuse to overwrite an existing config unless ``--force``.
      * Write to ``--config`` (override) or ``config_path()`` (default).
      * Skip service install + daemon start + initial indexing — those
        are separate explicit commands per PRD §11.1.
    """
    from longctx_daemon.config import (
        ConfigError,
        CorpusConfig,
        ServiceConfig,
        config_path as _default_config_path,
        default_config,
        load_config,
        write_config,
    )
    from longctx_daemon.discovery import discover_projects

    # Resolve config path
    if args.config:
        cfg_path = Path(args.config).expanduser()
    else:
        cfg_path = _default_config_path()

    if cfg_path.exists() and not args.force:
        print(
            f"error: config already exists at {cfg_path}. "
            f"Use --force to overwrite.",
            file=sys.stderr,
        )
        return 2

    # Build the config
    base = default_config()

    # Resolve parent_dir from explicit flag or default
    if args.parent_dir:
        parent_dir = Path(args.parent_dir).expanduser()
    else:
        parent_dir = base.corpus.parent_dir

    # auto_discover semantics:
    #   --no-auto-discover → False; require --include
    #   --include AND --no-auto-discover → use list as allow-list
    #   --include alone (auto_discover=true) → treat list as deny-list
    auto_discover = not args.no_auto_discover
    include_list: tuple[str, ...] = ()
    if args.include:
        include_list = tuple(p.strip() for p in args.include.split(",") if p.strip())

    if not auto_discover and not include_list:
        print(
            "error: --no-auto-discover requires --include <comma-separated-projects>",
            file=sys.stderr,
        )
        return 2

    corpus = CorpusConfig(
        parent_dir=parent_dir,
        auto_discover=auto_discover,
        sentinels=base.corpus.sentinels,
        include=include_list,
        exclude=base.corpus.exclude,
        exclude_patterns=base.corpus.exclude_patterns,
        secret_patterns=base.corpus.secret_patterns,
        allow_secret_patterns=base.corpus.allow_secret_patterns,
        forbidden_dirs=base.corpus.forbidden_dirs,
        respect_gitignore=base.corpus.respect_gitignore,
        max_file_size_kb=base.corpus.max_file_size_kb,
        include_extensions=base.corpus.include_extensions,
    )
    cfg = ServiceConfig(
        corpus=corpus,
        server=base.server,
        index=base.index,
        watcher=base.watcher,
        cleanup=base.cleanup,
        search=base.search,
        logging=base.logging,
    )

    # Discover (informational; we still write the config even if
    # parent_dir is missing, so the user can run init before mkdir'ing).
    discovered = []
    if parent_dir.is_dir():
        discovered = discover_projects(
            parent_dir,
            sentinels=corpus.sentinels,
            exclude=corpus.exclude,
            forbidden_dirs=corpus.forbidden_dirs,
        )

    # Apply auto_discover semantics for reporting
    if auto_discover:
        # include = deny-list
        deny = set(corpus.include)
        will_index = [p for p in discovered if p.name not in deny]
    else:
        allow = set(corpus.include)
        will_index = [p for p in discovered if p.name in allow]

    # Honor .longctxignore
    will_index = [p for p in will_index if not p.has_longctxignore]

    # Interactive picker (numbered checkbox-style)
    if not args.non_interactive:
        if not parent_dir.is_dir():
            print(
                f"warning: parent_dir does not exist yet: {parent_dir}",
                file=sys.stderr,
            )
        elif discovered:
            selected = _interactive_pick_projects(discovered)
            if selected is None:
                print(
                    "skipping config write; rerun with "
                    "--non-interactive or --include to customize.",
                    file=sys.stderr,
                )
                return 1
            # Translate selection back to the deny/allow list semantics
            selected_names = {p.name for p in selected}
            if auto_discover:
                # auto_discover=True: include[] is the deny-list
                deny_list = tuple(
                    p.name for p in discovered if p.name not in selected_names
                )
                if deny_list != include_list:
                    include_list = deny_list
                    corpus = CorpusConfig(
                        parent_dir=parent_dir,
                        auto_discover=True,
                        sentinels=corpus.sentinels,
                        include=include_list,
                        exclude=corpus.exclude,
                        exclude_patterns=corpus.exclude_patterns,
                        secret_patterns=corpus.secret_patterns,
                        allow_secret_patterns=corpus.allow_secret_patterns,
                        forbidden_dirs=corpus.forbidden_dirs,
                        respect_gitignore=corpus.respect_gitignore,
                        max_file_size_kb=corpus.max_file_size_kb,
                        include_extensions=corpus.include_extensions,
                    )
                    cfg = ServiceConfig(
                        corpus=corpus, server=base.server,
                        index=base.index, watcher=base.watcher,
                        cleanup=base.cleanup, search=base.search,
                        logging=base.logging,
                    )
            will_index = list(selected)

    # Validate before writing — load_config does this on read, but the
    # write path goes through tomli_w directly so we re-run validation
    # here.
    try:
        from longctx_daemon.config import _validate
        _validate(cfg)
    except ConfigError as exc:
        print(f"error: invalid config: {exc}", file=sys.stderr)
        return 2

    # Write
    write_config(cfg_path, cfg)

    # Confirm by re-loading (catches accidental round-trip drift)
    try:
        load_config(cfg_path)
    except ConfigError as exc:
        print(
            f"error: round-trip validation failed: {exc}", file=sys.stderr,
        )
        return 2

    # Report
    print(f"Configuration written to {cfg_path}")
    print(f"Parent directory: {parent_dir}")
    print(f"Auto-discover: {auto_discover}")
    if include_list:
        label = "Allow-list" if not auto_discover else "Deny-list"
        print(f"{label}: {', '.join(include_list)}")
    if parent_dir.is_dir():
        print(
            f"Discovered {len(discovered)} sentinel-marked project(s); "
            f"{len(will_index)} will be indexed."
        )
        if discovered and args.verbose:
            for p in discovered:
                marker = " [opted out via .longctxignore]" \
                    if p.has_longctxignore else ""
                will = " (will index)" if p in will_index else " (skipped)"
                print(f"  - {p.name} ({p.sentinel}){will}{marker}")
    else:
        print(
            f"Note: {parent_dir} does not exist yet; create it before "
            f"running `longctx serve`."
        )
    print()
    print("Next steps:")
    print("  longctx service install    # register launchd / systemd unit")
    print("  longctx service start      # start the daemon")
    print("  longctx status             # confirm it's running")
    return 0


# ============================================================ daemon-aware
# Phase 2.1 commands that talk to a running daemon via server.info.

def _read_running_info(args: argparse.Namespace):
    """Resolve + read ``server.info`` from CLI args.

    Returns the ``ServerInfo`` or prints a diagnostic + returns None.
    The caller decides what exit code maps to "no daemon".
    """
    from longctx_daemon.server_info import (
        DEFAULT_SERVER_INFO,
        read_server_info,
    )
    info_path = (
        Path(args.server_info).expanduser() if getattr(args, "server_info", None)
        else DEFAULT_SERVER_INFO
    )
    info = read_server_info(info_path)
    return info, info_path


def _cmd_port(args: argparse.Namespace) -> int:
    """Print the MCP or status port from ``server.info``.

    Used by curl/Hermes-style clients that want to discover the live
    daemon's port without parsing the JSON themselves (PRD §8.4).
    """
    info, info_path = _read_running_info(args)
    if info is None:
        print(
            f"error: no running longctx daemon "
            f"(missing or stale {info_path}). "
            f"Start with `longctx serve --daemon` or `longctx service start`.",
            file=sys.stderr,
        )
        return 3
    which = (args.which or "mcp").lower()
    if which == "mcp":
        print(info.mcp_port)
    elif which == "status":
        print(info.status_port)
    else:
        print(f"error: unknown port name {which!r}", file=sys.stderr)
        return 2
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Print a rich status report based on ``server.info`` + /health.

    PRD §10.2: read server.info (Agent F) for PID + ports, then GET
    ``/health`` (Agent D's StatusServer) and pretty-print the
    combined info. Falls back to the leaner server.info-only banner
    if /health is unreachable so we still tell the user *something*
    when the HTTP endpoint flakes.
    """
    info, info_path = _read_running_info(args)
    if info is None:
        print("longctx daemon: not running", file=sys.stdout)
        print(f"  (no live {info_path})", file=sys.stdout)
        print("Start with: longctx serve --daemon", file=sys.stdout)
        return 1

    # Try /health for the rich §10.2 banner.
    payload = _fetch_health(args.host or "127.0.0.1", info.status_port)
    if args.json:
        out = {
            "server_info": info.to_dict(),
            "health": payload,
        }
        print(json.dumps(out, indent=2))
        return 0

    if payload is None:
        # Status endpoint unreachable — fall back to the server.info
        # summary so the user still sees PID / ports / version.
        print(f"longctx daemon: running (PID {info.pid})")
        print(f"  started_at:   {info.started_at}")
        print(
            f"  mcp:          "
            f"{','.join(info.mcp_transports)} on 127.0.0.1:{info.mcp_port}"
        )
        print(f"  status:       http on 127.0.0.1:{info.status_port} "
              f"(unreachable)")
        print(f"  version:      {info.version}")
        print(f"  server.info:  {info_path}")
        return 0

    _pretty_print_status(info, payload)
    return 0


def _fetch_health(host: str, port: int) -> dict | None:
    """GET ``/health`` against the running daemon. stdlib only — keep
    the CLI cold-start cheap and don't drag aiohttp into every shell
    completion.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _format_uptime(seconds: int) -> str:
    """1h 4m / 12m / 47s / 2d 3h — the §10.2 banner format."""
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{secs}s"


def _format_count(n: int) -> str:
    """Format chunk-count: 1234 → '1,234'."""
    return f"{n:,}"


def _format_tokens(n: int) -> str:
    """Format token-count: 1543000 → '~1.5M tokens'."""
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f}M tokens"
    if n >= 1_000:
        return f"~{n / 1_000:.0f}K tokens"
    return f"~{n} tokens"


def _format_age(value) -> str:
    """Compact "3h ago" rendering. Accepts ISO-8601 string OR unix
    timestamp (int). Returns "never" for None / 0."""
    if value is None or value == 0:
        return "never"
    if isinstance(value, (int, float)):
        ts = int(value)
    else:
        try:
            from datetime import datetime
            ts = int(datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            ).timestamp())
        except (ValueError, TypeError):
            return str(value)
    import time as _time
    delta = max(0, int(_time.time()) - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _pretty_print_status(info, payload: dict) -> None:
    """Render the §10.2 six-line summary."""
    pid = info.pid
    uptime = _format_uptime(int(payload.get("uptime_seconds", 0)))
    print(f"longctx daemon: running (PID {pid}, uptime {uptime})")

    projects = payload.get("projects", [])
    if projects:
        print("Indexed projects:")
        max_name = max(len(str(p.get("name", "?"))) for p in projects)
        for proj in projects:
            name = str(proj.get("name", "?")).ljust(max_name)
            chunks = _format_count(int(proj.get("chunks", 0)))
            tokens = _format_tokens(int(proj.get("tokens_approx", 0)))
            age = _format_age(proj.get("last_updated"))
            print(
                f"  {name}  {chunks:>10} chunks   {tokens:<14} updated {age}"
            )
        total_chunks = sum(int(p.get("chunks", 0)) for p in projects)
        total_tokens = sum(int(p.get("tokens_approx", 0)) for p in projects)
        print(
            f"Total: {_format_count(total_chunks)} chunks, "
            f"{_format_tokens(total_tokens)}"
        )
    else:
        print("Indexed projects: (none)")

    watcher = payload.get("watcher", {})
    print(
        f"Watcher: {watcher.get('queued', 0)} queued, "
        f"{_format_count(int(watcher.get('events_processed_total', 0)))} "
        f"events processed"
    )

    transports = ",".join(info.mcp_transports) or "stdio"
    print(
        f"MCP server: {transports} on 127.0.0.1:{info.mcp_port}"
    )

    embedder = payload.get("embedder", {})
    sha = embedder.get("sha256", "")
    sha_short = (sha[:4] + "…") if sha else "unknown"
    print(
        f"Embedder: {embedder.get('model', 'unknown')} "
        f"(sha256: {sha_short})"
    )

    rss = payload.get("memory_rss_mb", 0)
    if rss >= 1024:
        print(f"Memory: {rss / 1024:.1f} GB RSS")
    else:
        print(f"Memory: {rss} MB RSS")


def _cmd_reload(args: argparse.Namespace) -> int:
    """Send SIGHUP to the running daemon."""
    info, info_path = _read_running_info(args)
    if info is None:
        print(
            f"error: no running longctx daemon (missing or stale "
            f"{info_path}).",
            file=sys.stderr,
        )
        return 3
    sig = getattr(signal, "SIGHUP", None)
    if sig is None:
        print(
            "error: SIGHUP not available on this platform; "
            "reload not supported.", file=sys.stderr,
        )
        return 2
    try:
        os.kill(info.pid, sig)
    except ProcessLookupError:
        print(
            f"error: pid {info.pid} from server.info is gone", file=sys.stderr,
        )
        return 3
    except PermissionError:
        print(
            f"error: cannot signal pid {info.pid} (different user?)",
            file=sys.stderr,
        )
        return 3
    print(f"sent SIGHUP to longctx daemon (pid {info.pid})")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    """Send SIGTERM to the running daemon."""
    info, info_path = _read_running_info(args)
    if info is None:
        print(
            f"error: no running longctx daemon (missing or stale "
            f"{info_path}).",
            file=sys.stderr,
        )
        return 3
    try:
        os.kill(info.pid, signal.SIGTERM)
    except ProcessLookupError:
        print(
            f"error: pid {info.pid} from server.info is gone", file=sys.stderr,
        )
        return 3
    except PermissionError:
        print(
            f"error: cannot signal pid {info.pid} (different user?)",
            file=sys.stderr,
        )
        return 3
    print(f"sent SIGTERM to longctx daemon (pid {info.pid})")
    return 0


def _cmd_mcp_stdio(args: argparse.Namespace) -> int:
    """Bridge stdio↔SSE for the calling MCP client.

    PRD §8.4: clients configure ``longctx mcp-stdio`` and we read
    server.info, open an SSE connection to the running daemon, and
    forward MCP messages bidirectionally over our stdio. Decouples
    client config from runtime ports.

    Implementation note: we use ``mcp.client.sse.sse_client`` to talk
    to the daemon and ``mcp.server.stdio.stdio_server`` to talk to the
    calling client; messages are forwarded raw. Either side
    disconnecting closes the bridge.

    Test path: when ``--check`` is passed, we just resolve server.info
    and exit 0 / 3. The full bridge runs an event loop until either
    side disconnects, which we don't drive in unit tests.
    """
    info, info_path = _read_running_info(args)
    if info is None:
        print(
            f"error: no running longctx daemon (missing or stale "
            f"{info_path}). Start with `longctx serve --daemon` or "
            f"`longctx service start`.",
            file=sys.stderr,
        )
        return 3

    if args.check:
        print(
            f"longctx daemon ready: pid={info.pid} "
            f"mcp=http://127.0.0.1:{info.mcp_port}/sse",
            file=sys.stderr,
        )
        return 0

    transports = set(info.mcp_transports)
    if "sse" not in transports:
        print(
            f"error: running daemon does not expose SSE "
            f"(transports={sorted(transports)}); cannot bridge.",
            file=sys.stderr,
        )
        return 4

    sse_url = f"http://127.0.0.1:{info.mcp_port}/sse"
    return asyncio.run(_run_mcp_stdio_bridge(sse_url))


async def _run_mcp_stdio_bridge(sse_url: str) -> int:
    """Drive the stdio↔SSE bridge until either side disconnects.

    The MCP SDK ships both the client SSE transport and the server
    stdio transport; we just wire their streams together. No payload
    introspection — every message goes through unmodified, including
    initialize/notifications/tool calls.
    """
    from mcp.client.sse import sse_client
    from mcp.server.stdio import stdio_server

    async with sse_client(sse_url) as (sse_read, sse_write):
        async with stdio_server() as (stdio_read, stdio_write):
            async def _client_to_daemon() -> None:
                async for msg in stdio_read:
                    await sse_write.send(msg)

            async def _daemon_to_client() -> None:
                async for msg in sse_read:
                    await stdio_write.send(msg)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_client_to_daemon(), name="cli→daemon")
                    tg.create_task(_daemon_to_client(), name="daemon→cli")
            except* Exception:  # noqa: BLE001
                # Either side closed; drain and exit cleanly.
                pass
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
        "--corpus-dir", required=True, action="append",
        help="Directory to index + serve. Repeat the flag to cover multiple "
             "roots (e.g. main repo + a sibling git worktree + an unrelated "
             "project). Each root becomes its own Project so cross-project "
             "scope routing + fanout works.",
    )
    serve.add_argument(
        "--include-worktrees", action="store_true", default=False,
        help="For each corpus root that IS a git repo, also auto-index every "
             "worktree of that repo as its own Project named "
             "``<corpus>@<branch>``. Off by default because worktrees of "
             "the same repo share most files and indexing all of them "
             "duplicates storage. Useful when each branch has meaningfully "
             "diverged code you want retrievable.",
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
    # ---- Phase 2.1 transport / daemon flags
    serve.add_argument(
        "--host", default=None,
        help="Bind host for SSE / streamable-http (default 127.0.0.1).",
    )
    serve.add_argument(
        "--port", type=int, default=None,
        help="Preferred MCP port (walks forward on collision; default 8765).",
    )
    serve.add_argument(
        "--status-port", type=int, default=None,
        help="Preferred status HTTP port (default 8766).",
    )
    serve.add_argument(
        "--transports", default=None,
        help="Comma-separated subset of {stdio, sse, streamable-http}. "
             "Default: stdio if --daemon not given, else 'sse,streamable-http'.",
    )
    serve.add_argument(
        "--daemon", action="store_true",
        help="Detach to background (writes server.info + holds lock).",
    )
    serve.set_defaults(func=_cmd_serve)

    # --- calibrate (Phase 2.0.1+)
    cal = sub.add_parser(
        "calibrate",
        help="Sweep relevance-floor thresholds on a real corpus + recommend.",
        description=(
            "Index a corpus and run the abstention suite (30 off-corpus + "
            "12 in-corpus queries) at a range of dense-cosine thresholds. "
            "Print suppression rate / false-negative rate per floor + the "
            "recommended floor (highest floor with FNR ≤ --max-fnr). "
            "Default behavior is observation-only — pass --write-config to "
            "persist the recommendation to the user's config file."
        ),
    )
    cal.add_argument(
        "--corpus-dir", required=True,
        help="Directory to index + evaluate.",
    )
    cal.add_argument("--floor-min", type=float, default=0.30)
    cal.add_argument("--floor-max", type=float, default=0.70)
    cal.add_argument("--floor-step", type=float, default=0.05)
    cal.add_argument(
        "--max-fnr", type=float, default=0.05,
        help="Max acceptable false-negative rate when picking the "
             "recommended floor (default 0.05).",
    )
    cal.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    cal.add_argument("--device", default="auto")
    cal.add_argument("--json", action="store_true",
                     help="Emit JSON instead of pretty-printed table.")
    cal.add_argument(
        "--write-config", action="store_true",
        help="Persist the recommended floor to the user's config file "
             "(default ~/.config/longctx/config.toml). Without --project, "
             "writes to [index].relevance_floor (global). With --project, "
             "writes to [index].project_floors[NAME] (per-project, takes "
             "precedence over global when scope resolves to that project).",
    )
    cal.add_argument(
        "--project", default=None,
        help="When set with --write-config, persist as a per-project "
             "floor under [index].project_floors[NAME] instead of the "
             "global [index].relevance_floor. Recommended for any non-"
             "code corpus (mixed-content needs ~0.60, prose-heavy ~0.65).",
    )
    cal.add_argument(
        "--config",
        help="Override config-file path for --write-config "
             "(default: platformdirs).",
    )
    cal.add_argument("--verbose", "-v", action="store_true")
    cal.set_defaults(func=_cmd_calibrate)

    # --- version
    ver = sub.add_parser("version", help="Print version.")
    ver.set_defaults(func=_cmd_version)

    # --- init (Phase 2.1)
    init = sub.add_parser(
        "init",
        help="Auto-discover projects + write the longctx config.",
        description=(
            "Create or replace the longctx config file. Walks "
            "<parent-dir> for sentinel-marked subdirectories and writes "
            "the result. Does NOT install the service or start the "
            "daemon — run `longctx service install` and `longctx service "
            "start` separately."
        ),
    )
    init.add_argument(
        "--non-interactive", action="store_true",
        help="Use defaults without prompting (PRD §11.1).",
    )
    init.add_argument(
        "--parent-dir",
        help="Override the parent directory to scan "
             "(default ~/dev or LONGCTX_PARENT_DIR).",
    )
    init.add_argument(
        "--include",
        help="Comma-separated project names. With --no-auto-discover "
             "this is the explicit allow-list; without, it is the "
             "deny-list against discovered projects.",
    )
    init.add_argument(
        "--no-auto-discover", action="store_true",
        help="Disable sentinel-based auto-discovery; require --include.",
    )
    init.add_argument(
        "--config",
        help="Override config-file path (default: platformdirs).",
    )
    init.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing config file.",
    )
    init.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-project discovery results.",
    )
    init.set_defaults(func=_cmd_init)

    # ============================================================ daemon-aware
    # Common shared arg: ``--server-info`` overrides the default
    # ``~/.cache/longctx/server.info`` path. Tests use this; users
    # don't need to.

    # --- port
    port = sub.add_parser(
        "port",
        help="Print the MCP or status port from server.info.",
        description=(
            "Read ``~/.cache/longctx/server.info`` and print just the "
            "port the running daemon is listening on. Suitable for "
            "shell scripts that drive curl against the daemon."
        ),
    )
    port.add_argument(
        "which", nargs="?", default="mcp", choices=["mcp", "status"],
        help="Which port to print (default: mcp).",
    )
    port.add_argument(
        "--server-info", default=None,
        help="Override server.info path (test hook).",
    )
    port.set_defaults(func=_cmd_port)

    # --- status
    status = sub.add_parser(
        "status",
        help="Show daemon liveness + transports + ports.",
        description=(
            "Print a human-readable summary of the running daemon. "
            "Reads ``~/.cache/longctx/server.info``; reports 'not "
            "running' if the file is missing or the recorded PID is "
            "gone. Future versions will also fetch the status HTTP "
            "endpoint for chunk/token totals."
        ),
    )
    status.add_argument(
        "--server-info", default=None,
        help="Override server.info path (test hook).",
    )
    status.add_argument(
        "--host", default="127.0.0.1",
        help="Host to query for /health (default 127.0.0.1).",
    )
    status.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON (server.info + /health) instead of pretty-printing.",
    )
    status.set_defaults(func=_cmd_status)

    # --- reload
    reload_p = sub.add_parser(
        "reload",
        help="Send SIGHUP to the running daemon (reconcile config).",
    )
    reload_p.add_argument(
        "--server-info", default=None,
        help="Override server.info path (test hook).",
    )
    reload_p.set_defaults(func=_cmd_reload)

    # --- stop
    stop_p = sub.add_parser(
        "stop",
        help="Send SIGTERM to the running daemon (graceful shutdown).",
    )
    stop_p.add_argument(
        "--server-info", default=None,
        help="Override server.info path (test hook).",
    )
    stop_p.set_defaults(func=_cmd_stop)

    # --- mcp-stdio (bridge subcommand)
    mcp_stdio = sub.add_parser(
        "mcp-stdio",
        help="Bridge stdio↔SSE for an MCP client; daemon must be running.",
        description=(
            "Reads ``~/.cache/longctx/server.info`` and bridges this "
            "process's stdio with the running daemon's SSE transport. "
            "Suitable for Claude Desktop's command-launch model — it "
            "starts ``longctx mcp-stdio`` per session, never starts a "
            "daemon itself. The shared daemon is started separately "
            "via `longctx serve --daemon` or `longctx service start`."
        ),
    )
    mcp_stdio.add_argument(
        "--check", action="store_true",
        help="Just verify a daemon is reachable; don't bridge.",
    )
    mcp_stdio.add_argument(
        "--server-info", default=None,
        help="Override server.info path (test hook).",
    )
    mcp_stdio.set_defaults(func=_cmd_mcp_stdio)

    # --- clean (manual sweeps; auto-cleanup is the watcher's job)
    clean_p = sub.add_parser(
        "clean",
        help="Manual cleanup: drop idle/orphan/missing projects + replay shards.",
        description=(
            "Manual side of PRD §12.4.4. Auto-cleanup tiers (Tier 1 "
            "missing-root, Tier 2 cold-pause) are the watcher's job. "
            "This command is for user-driven sweeps."
        ),
    )
    clean_p.add_argument("--idle", default=None,
                         help="Drop projects not queried in this duration "
                              "(e.g. 90d, 2w, 12h, 30m).")
    clean_p.add_argument("--orphans", action="store_true",
                         help="Drop session-bound projects whose creating "
                              "session is no longer alive.")
    clean_p.add_argument("--missing", action="store_true",
                         help="Drop projects whose root_path no longer "
                              "exists, no grace period.")
    clean_p.add_argument("--all", action="store_true",
                         help="Equivalent to --idle 90d --orphans --missing "
                              "--replay-older-than 30d.")
    clean_p.add_argument("--replay-older-than", default=None,
                         help="Delete interactions.jsonl shards older than "
                              "this duration.")
    clean_p.add_argument("--disk-budget", type=float, default=None,
                         metavar="GB",
                         help="EXPERIMENTAL: enforce a soft cap on total "
                              "cache footprint (Tier 3 disk-budget LRU "
                              "eviction). Pass a value in GB; oldest-queried "
                              "projects evict first until under budget.")
    clean_p.add_argument("--cache-dir", default=None,
                         help="Override cache dir (default ~/.cache/longctx).")
    clean_p.add_argument("--yes", "-y", action="store_true",
                         help="Skip the confirmation prompt; actually clean.")
    def _cmd_clean(args):
        from longctx_daemon.clean import cmd_clean
        return cmd_clean(args)
    clean_p.set_defaults(func=_cmd_clean)

    # --- replay (regression-test against captured interactions.jsonl)
    replay_p = sub.add_parser(
        "replay",
        help="Replay captured MCP traffic against the current index.",
        description=(
            "Reads interactions.jsonl (or .gz shards) and re-runs each "
            "captured search_codebase against the current index. Prints "
            "per-call rank/score deltas — useful for asking 'did this "
            "embedder/chunker/RRF tweak help or hurt what real agents "
            "actually asked yesterday?' more meaningful than synthetic NIAH."
        ),
    )
    replay_p.add_argument("log_path",
                          help="Path to interactions.jsonl or .gz shard.")
    replay_p.add_argument("--cache-dir", default=None)
    replay_p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    replay_p.add_argument("--device", default="auto")
    replay_p.add_argument("--json", action="store_true",
                          help="Emit per-call deltas as JSONL on stdout.")
    def _cmd_replay(args):
        from longctx_daemon.replay import cmd_replay
        return cmd_replay(args)
    replay_p.set_defaults(func=_cmd_replay)

    # --- watch (live MCP-traffic tail)
    watch_p = sub.add_parser(
        "watch",
        help="Pretty-printed real-time tail of MCP activity.",
        description=(
            "Tails the structured operational log and renders one row per "
            "MCP call with color-coded harness identification, citations, "
            "freshness flags. The 'debug TV' for what agents are doing "
            "right now — keep open in one terminal while running real "
            "agents in others."
        ),
    )
    watch_p.add_argument("--cache-dir", default=None)
    watch_p.add_argument("--log-file", default=None,
                         help="Override path; default <cache_dir>/longctx.log.")
    watch_p.add_argument("--filter", default=None,
                         help="Comma-separated filter spec, e.g. "
                              "client=opencode,tool=search_codebase,min_ms=50.")
    watch_p.add_argument("--since", default=None,
                         help="Backfill records from the last N "
                              "(e.g. 1h, 30m, 5d).")
    watch_p.add_argument("--no-follow", action="store_true",
                         help="Print existing records and exit; don't tail.")
    watch_p.add_argument("--verbose", "-v", action="store_true",
                         help="Show trace IDs + per-stage latency breakdown.")
    def _cmd_watch(args):
        from longctx_daemon.watch_cmd import cmd_watch
        return cmd_watch(args)
    watch_p.set_defaults(func=_cmd_watch)

    # --- service install/start/stop/status/uninstall (macOS launchd in 2.3)
    service = sub.add_parser(
        "service",
        help="Install/start/stop/status the platform service "
             "(launchd / systemd / Windows Service).",
        description=(
            "Phase 2.3 ships macOS launchd. Linux systemd + Windows "
            "Service ship in 2.4 (NotImplementedError on those "
            "platforms for now). All sub-actions are idempotent."
        ),
    )
    service_sub = service.add_subparsers(dest="service_action", required=True)

    s_install = service_sub.add_parser("install",
                                       help="Generate + register the service definition.")
    s_install.add_argument("--corpus-dir", default=None,
                           help="Initial corpus to serve (multi-root via "
                                "config-driven discovery in 2.4).")
    s_install.add_argument("--executable", default=None,
                           help="Path to the longctx binary (default: argv[0] resolved).")

    service_sub.add_parser("start", help="Start the installed service.")
    service_sub.add_parser("stop", help="Stop the installed service.")
    service_sub.add_parser("status", help="Print service liveness.")
    service_sub.add_parser("uninstall", help="Stop + remove service definition.")

    def _cmd_service(args):
        from longctx_daemon.service import current_installer
        installer = current_installer()
        action = args.service_action
        try:
            if action == "install":
                exe = (
                    Path(args.executable).expanduser()
                    if args.executable
                    else Path(sys.argv[0]).resolve()
                )
                corpus = (
                    Path(args.corpus_dir).expanduser()
                    if args.corpus_dir
                    else None
                )
                p = installer.install(executable=exe, corpus_dir=corpus)
                print(f"installed: {p}")
                return 0
            if action == "start":
                installer.start()
                print("started.")
                return 0
            if action == "stop":
                installer.stop()
                print("stopped.")
                return 0
            if action == "uninstall":
                installer.uninstall()
                print("uninstalled.")
                return 0
            if action == "status":
                state = installer.status()
                if not state.installed:
                    print(f"{state.label}: not installed")
                    return 0
                pid = state.pid if state.pid else "(not running)"
                running = "running" if state.running else "stopped"
                print(f"{state.label}: {running}, pid={pid}")
                if state.plist_path is not None:
                    print(f"  definition: {state.plist_path}")
                return 0
        except NotImplementedError as e:
            print(f"not implemented on this platform: {e}", file=sys.stderr)
            return 1
        return 1
    service.set_defaults(func=_cmd_service)

    # --- pack: registry for pre-built (SqliteChunkStore, MemmapEmbedStore)
    # pairs that the daemon can mount as additional projects. See
    # ``longctx_daemon/packs.py`` for the on-disk format. v1 is manual
    # add/rm/list/info; the Mini-LLM controller will add auto-pick later.
    pack = sub.add_parser(
        "pack",
        help="Manage pre-built corpus packs.",
        description=(
            "Register, list, and inspect pre-built corpus packs (the "
            "drop-in artifacts produced by offline ingest, e.g. "
            "corpora/swezero-12m/ingest.py). v1 is manual selection — "
            "the Mini-LLM controller will auto-pick in a later phase."
        ),
    )
    pack_sub = pack.add_subparsers(dest="pack_command", required=True)

    pack_list = pack_sub.add_parser(
        "list", help="List registered packs.",
    )
    pack_list.add_argument(
        "--json", action="store_true",
        help="Emit JSON for scripting; default is a pretty table.",
    )
    pack_list.set_defaults(func=_cmd_pack_list)

    pack_add = pack_sub.add_parser(
        "add", help="Register a pack from a directory.",
    )
    pack_add.add_argument(
        "path",
        help="Pack directory (must contain chunks.sqlite + embeds/*.idx).",
    )
    pack_add.add_argument(
        "--name", default=None,
        help="Pack name. Defaults to <corpus>-<lang> for paths under "
             "corpora/<corpus>/<lang>, else the directory basename.",
    )
    pack_add.add_argument(
        "--dedup", default="auto", choices=("auto", "on", "off"),
        help="Apply dedup_by_doc_root at search time. 'auto' = on for "
             "packs whose name starts with swezero-, else off.",
    )
    pack_add.set_defaults(func=_cmd_pack_add)

    pack_rm = pack_sub.add_parser(
        "rm", help="Unregister a pack (on-disk files NOT deleted).",
    )
    pack_rm.add_argument("name", help="Pack name (see ``longctx pack list``).")
    pack_rm.set_defaults(func=_cmd_pack_rm)

    pack_info = pack_sub.add_parser(
        "info", help="Show full pack details + embedder identity.",
    )
    pack_info.add_argument("name", help="Pack name.")
    pack_info.add_argument(
        "--json", action="store_true",
        help="Emit JSON for scripting.",
    )
    pack_info.set_defaults(func=_cmd_pack_info)

    pack_scan = pack_sub.add_parser(
        "scan",
        help="Auto-register every valid pack found under a directory.",
        description=(
            "Walks the given root (default ~/.cache/longctx/corpora) up "
            "to 4 levels deep and registers any directory that looks "
            "like a pack (contains chunks.sqlite + embeds/*.idx). "
            "Idempotent: already-registered packs get their stats "
            "refreshed in place."
        ),
    )
    pack_scan.add_argument(
        "path", nargs="?", default=None,
        help="Root directory to scan (default ~/.cache/longctx/corpora).",
    )
    pack_scan.add_argument(
        "--prune", action="store_true",
        help="Also unregister packs whose path no longer exists on disk.",
    )
    pack_scan.add_argument(
        "--json", action="store_true",
        help="Emit JSON for scripting.",
    )
    pack_scan.set_defaults(func=_cmd_pack_scan)

    return p


def _human_bytes(n: int) -> str:
    """Format ``n`` bytes as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n = n / 1024
    return f"{n:.1f} PB"


def _cmd_pack_list(args: argparse.Namespace) -> int:
    """``longctx pack list``."""
    from longctx_daemon.packs import PackRegistry
    reg = PackRegistry()
    packs = reg.list()

    if args.json:
        print(json.dumps(
            {"packs": [asdict(p) for p in packs]}, indent=2,
        ))
        return 0

    if not packs:
        print("no packs registered. add one with: longctx pack add <path>")
        return 0

    print(f"{'name':<28} {'chunks':>12} {'size':>10} dedup  embedder")
    print("-" * 80)
    for p in packs:
        print(
            f"{p.name:<28} {p.chunk_count:>12,} "
            f"{_human_bytes(p.size_bytes):>10} "
            f"{'on' if p.dedup_by_doc_root else 'off':<6} "
            f"{p.embedder_model}"
        )
    return 0


def _cmd_pack_add(args: argparse.Namespace) -> int:
    """``longctx pack add <path>``."""
    from longctx_daemon.packs import PackRegistry
    dedup_arg = {
        "auto": None,
        "on": True,
        "off": False,
    }[args.dedup]
    try:
        pack = PackRegistry().register(
            Path(args.path),
            name=args.name,
            dedup_by_doc_root=dedup_arg,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"registered: {pack.name}\n"
        f"  path: {pack.path}\n"
        f"  chunks: {pack.chunk_count:,}\n"
        f"  size: {_human_bytes(pack.size_bytes)}\n"
        f"  embedder: {pack.embedder_model} ({pack.embed_dim}d)\n"
        f"  dedup_by_doc_root: {'on' if pack.dedup_by_doc_root else 'off'}"
    )
    return 0


def _cmd_pack_rm(args: argparse.Namespace) -> int:
    """``longctx pack rm <name>``."""
    from longctx_daemon.packs import PackRegistry
    reg = PackRegistry()
    if not reg.unregister(args.name):
        print(f"error: no pack named {args.name!r}", file=sys.stderr)
        return 1
    print(f"unregistered: {args.name} (on-disk files preserved)")
    return 0


def _cmd_pack_scan(args: argparse.Namespace) -> int:
    """``longctx pack scan [<path>]``."""
    from longctx_daemon.packs import PackRegistry, DEFAULT_PACK_ROOT
    root = Path(args.path) if args.path else DEFAULT_PACK_ROOT
    reg = PackRegistry()
    added, refreshed, pruned = reg.scan(root, prune_missing=args.prune)

    if args.json:
        print(json.dumps({
            "root": str(root.expanduser().resolve()),
            "added": [p.name for p in added],
            "refreshed": [p.name for p in refreshed],
            "pruned": list(pruned),
        }, indent=2))
        return 0

    print(f"scanned: {root.expanduser().resolve()}")
    print(f"  added:     {len(added)}")
    for p in added:
        print(f"    + {p.name}  ({p.chunk_count:,} chunks, "
              f"{_human_bytes(p.size_bytes)}, dedup="
              f"{'on' if p.dedup_by_doc_root else 'off'})")
    print(f"  refreshed: {len(refreshed)}")
    for p in refreshed:
        print(f"    ~ {p.name}  ({p.chunk_count:,} chunks, "
              f"{_human_bytes(p.size_bytes)})")
    if args.prune:
        print(f"  pruned:    {len(pruned)}")
        for n in pruned:
            print(f"    - {n}")
    if not added and not refreshed and not pruned:
        print("  (nothing to do)")
    return 0


def _cmd_pack_info(args: argparse.Namespace) -> int:
    """``longctx pack info <name>``."""
    from longctx_daemon.packs import PackRegistry
    reg = PackRegistry()
    pack = reg.get(args.name)
    if pack is None:
        print(f"error: no pack named {args.name!r}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(asdict(pack), indent=2))
        return 0
    import time as _t
    reg_iso = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(pack.registered_at))
    used_iso = (
        _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(pack.last_used_at))
        if pack.last_used_at > 0 else "never"
    )
    print(
        f"name: {pack.name}\n"
        f"path: {pack.path}\n"
        f"chunks: {pack.chunk_count:,}\n"
        f"size: {_human_bytes(pack.size_bytes)}\n"
        f"embed_dim: {pack.embed_dim}\n"
        f"embedder_model: {pack.embedder_model}\n"
        f"embedder_sha256: {pack.embedder_sha256}\n"
        f"dedup_by_doc_root: {'on' if pack.dedup_by_doc_root else 'off'}\n"
        f"registered_at: {reg_iso}\n"
        f"last_used_at: {used_iso}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
