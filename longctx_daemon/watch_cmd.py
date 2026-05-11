"""longctx watch — live MCP-traffic tail for harness debugging.

Per spec §14.7: pretty-printed real-time stream of MCP activity. The
"debug TV" for what agents are doing right now.

Reads the structured operational log (loguru JSON-lines sink) and
renders one row per MCP call with color-coded harness identification,
inline citations, and freshness flags.

Phase 2.0: tails the log file; supports ``--filter client=opencode``
and ``--since 1h``. Verbose mode shows trace IDs + per-stage latency.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------- ANSI colors

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# 4-bit ANSI palette (works on every terminal). Picks a stable color
# per harness name via hashing — first-seen client gets cyan, second
# magenta, etc. so the side-by-side recording reads cleanly even with
# 5+ concurrent harnesses.
_PALETTE = [36, 35, 33, 32, 34, 31]   # cyan, magenta, yellow, green, blue, red


def _color(idx: int) -> str:
    return f"\033[{_PALETTE[idx % len(_PALETTE)]}m"


# ------------------------------------------------------------ duration

_DURATION_RE = re.compile(r"^(\d+)\s*([smhd])$")


def parse_since_seconds(s: str) -> int:
    """``--since 1h`` → 3600. Defaults to seconds if no suffix."""
    s = s.strip().lower()
    m = _DURATION_RE.match(s)
    if m:
        n = int(m.group(1))
        return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"unrecognized --since value {s!r}") from e


# ---------------------------------------------------------- filter spec

@dataclass(frozen=True)
class WatchFilter:
    """Caller-provided filters on the live log stream."""
    client_name: Optional[str] = None
    tool: Optional[str] = None
    min_total_ms: Optional[float] = None
    only_errors: bool = False

    def matches(self, rec: dict) -> bool:
        if self.client_name:
            client = (rec.get("client") or {}).get("name", "")
            if client != self.client_name:
                return False
        if self.tool and rec.get("tool") != self.tool:
            return False
        if self.min_total_ms is not None:
            t = (rec.get("latency_ms") or {}).get("total", 0.0)
            if t < self.min_total_ms:
                return False
        if self.only_errors and rec.get("level") not in ("ERROR", "CRITICAL"):
            return False
        return True


def parse_filters(spec: str | None) -> WatchFilter:
    """``--filter client=opencode,tool=search_codebase`` → WatchFilter."""
    if not spec:
        return WatchFilter()
    kw = {}
    for piece in spec.split(","):
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        kw[k.strip()] = v.strip()
    return WatchFilter(
        client_name=kw.get("client"),
        tool=kw.get("tool"),
        min_total_ms=float(kw["min_ms"]) if "min_ms" in kw else None,
        only_errors=kw.get("only_errors", "").lower() in ("1", "true", "yes"),
    )


# ----------------------------------------------------------- pretty print

class _ClientPalette:
    """Allocates a stable palette index per first-seen client name."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def color_for(self, name: str) -> str:
        if name not in self._seen:
            self._seen[name] = len(self._seen)
        return _color(self._seen[name])


def render_record(rec: dict, palette: _ClientPalette,
                  *, verbose: bool = False) -> str:
    """One log record → one (multi-line) terminal block."""
    ts = (rec.get("ts") or "")[11:19]   # HH:MM:SS portion of ISO
    client = rec.get("client") or {}
    cname = client.get("name", "?")
    cver = client.get("version", "?")
    tool = rec.get("tool", "?")
    args = rec.get("args") or {}
    scope = rec.get("scope") or {}
    result = rec.get("result") or {}
    lat = rec.get("latency_ms") or {}
    color = palette.color_for(cname)

    head = (
        f"{_DIM}{ts}{_RESET}  "
        f"{color}{cname}/{cver}{_RESET}  "
        f"{_BOLD}{tool}{_RESET}"
    )
    if tool == "search_codebase":
        q = args.get("query")
        head += f"  \"{_truncate(q, 80)}\""

    lines = [head]

    if tool == "search_codebase":
        primary = scope.get("primary_project") or "(none)"
        src = scope.get("primary_source") or "?"
        n_chunks = result.get("chunk_count", 0)
        total_ms = lat.get("total", 0.0)
        fresh = "fresh" if result.get("is_fully_fresh") else (
            f"{result.get('pending_updates', 0)} pending"
        )
        lines.append(
            f"          scope={primary} ({src}) → {n_chunks} chunks "
            f"({total_ms:.0f}ms, {fresh})"
        )
        for f in (result.get("files") or [])[:5]:
            lines.append(f"          ↳ {_DIM}{f}{_RESET}")
    elif tool in ("list_projects", "index_status"):
        ms = lat.get("total", 0.0)
        lines.append(f"          → {ms:.0f}ms")
    else:
        lines.append(f"          {json.dumps(result, default=str)[:200]}")

    if verbose:
        trace = rec.get("trace_id", "?")
        lines.append(f"          {_DIM}trace={trace}{_RESET}")
        if tool == "search_codebase" and lat:
            lines.append(
                f"          {_DIM}embed={lat.get('embed_query', 0):.0f} "
                f"bm25={lat.get('bm25_score', 0):.0f} "
                f"dense={lat.get('dense_score', 0):.0f} "
                f"rrf={lat.get('rrf_fuse', 0):.0f} "
                f"fetch={lat.get('fetch_chunks', 0):.0f}{_RESET}"
            )

    return "\n".join(lines)


def _truncate(s: Optional[str], n: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


# -------------------------------------------------------- log tailing

def iter_log_records(
    log_path: Path,
    *,
    follow: bool = True,
    since_seconds: int = 0,
) -> Iterable[dict]:
    """Yield records from the JSON-lines log. ``follow=True`` blocks
    waiting for new lines (like ``tail -f``)."""
    cutoff = time.time() - since_seconds if since_seconds > 0 else 0
    if not log_path.exists():
        if not follow:
            return
        # Wait up to a few seconds for the daemon to create it
        for _ in range(40):
            if log_path.exists():
                break
            time.sleep(0.25)
        else:
            return

    with log_path.open(encoding="utf-8", errors="ignore") as f:
        # Backfill phase: scan from start, filter by since.
        for line in f:
            rec = _parse_line(line)
            if rec is None:
                continue
            if cutoff and _record_ts(rec) < cutoff:
                continue
            yield rec
        # Follow phase
        if not follow:
            return
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            rec = _parse_line(line)
            if rec is not None:
                yield rec


def _parse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _record_ts(rec: dict) -> float:
    """ISO 8601 → epoch seconds. Returns 0 on parse failure."""
    s = rec.get("ts")
    if not s:
        return 0
    try:
        # Tolerate a trailing 'Z' (loguru emits it).
        from datetime import datetime
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return 0


# ----------------------------------------------------------------- CLI

def cmd_watch(args: argparse.Namespace) -> int:
    """``longctx watch`` entry point."""
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".cache" / "longctx"
    )
    log_path = (
        Path(args.log_file).expanduser()
        if args.log_file
        else cache_dir / "longctx.log"
    )
    flt = parse_filters(args.filter)
    since = parse_since_seconds(args.since) if args.since else 0
    palette = _ClientPalette()

    print(f"# tailing {log_path} (Ctrl-C to stop)", file=sys.stderr)
    if flt.client_name or flt.tool:
        print(f"# filter: client={flt.client_name} tool={flt.tool}",
              file=sys.stderr)
    try:
        for rec in iter_log_records(
            log_path, follow=not args.no_follow, since_seconds=since,
        ):
            if not flt.matches(rec):
                continue
            if rec.get("component") != "mcp":
                # Only render MCP-call records by default; daemon
                # internals are noise here.
                if args.verbose:
                    pass   # verbose passes through everything
                else:
                    continue
            print(render_record(rec, palette, verbose=args.verbose))
    except KeyboardInterrupt:
        return 0
    return 0
