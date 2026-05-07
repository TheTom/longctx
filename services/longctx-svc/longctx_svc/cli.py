"""longctx-svc CLI entry point."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="longctx-svc",
        description="Local retrieval companion service for inference servers.",
    )
    ap.add_argument(
        "command", nargs="?", default="serve",
        choices=["serve", "version", "clean"],
    )
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind host (default 127.0.0.1; localhost only)")
    ap.add_argument("--port", type=int, default=8765,
                    help="bind port (default 8765)")
    ap.add_argument("--reload", action="store_true",
                    help="reload on code changes (dev only)")
    ap.add_argument("--older-than", type=int, default=30,
                    help="for `clean`: drop disk caches older than N days")
    ap.add_argument(
        "--upstream", default=None,
        help=("optional OpenAI-compat backend URL. When set, longctx-svc "
              "exposes /v1/chat/completions and /v1/completions as a "
              "passthrough proxy that splices retrieved chunks into the "
              "request before forwarding. Works with llama.cpp's llama-"
              "server, vLLM (CUDA/ROCm/AMD), vllm-swift, etc."),
    )
    args = ap.parse_args(argv)

    if args.command == "version":
        from longctx_svc import __version__
        print(__version__)
        return 0

    if args.command == "clean":
        from longctx_svc.cache.disk import (
            cache_root_size_bytes, clean_older_than, list_cached,
        )
        before = len(list_cached())
        before_bytes = cache_root_size_bytes()
        removed = clean_older_than(args.older_than)
        print(f"clean older-than={args.older_than}d: "
              f"removed {removed} of {before} scopes "
              f"({before_bytes / 1e6:.1f} MB → "
              f"{cache_root_size_bytes() / 1e6:.1f} MB)")
        return 0

    # serve
    if args.upstream:
        os.environ["LONGCTX_UPSTREAM"] = args.upstream
    import uvicorn
    proxy_note = (f" → proxy upstream: {args.upstream}"
                  if args.upstream else "")
    print(f"longctx-svc serving on http://{args.host}:{args.port} "
          f"(local-only){proxy_note}")
    uvicorn.run(
        "longctx_svc.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
