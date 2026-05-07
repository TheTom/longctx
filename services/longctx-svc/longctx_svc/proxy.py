"""OpenAI-compat passthrough proxy.

Run longctx-svc with `--upstream URL` (or LONGCTX_UPSTREAM env). Any
client that already speaks OpenAI Chat Completions can point at the
service instead of the engine — longctx-svc detects scope from the
messages, runs /retrieve, splices chunks into the system message, and
forwards the rewritten request to the upstream engine. Streaming pass-
through preserved.

Engine-agnostic by design: works against llama.cpp's `llama-server`,
vLLM (CUDA / ROCm / AMD), vllm-swift, or any OpenAI-compat backend.
The longctx layer is *optional* — if you don't run the proxy, your
engine is unchanged.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from longctx_svc.indexer.builder import build_index
from longctx_svc.scope.detect import detect_scope
from longctx_svc.scope.walk import walk_hot_scope, walk_package_scope
from longctx_svc.session.manager import extract_session_id
from longctx_svc.state import get_state


router = APIRouter()


def _upstream() -> str | None:
    return os.environ.get("LONGCTX_UPSTREAM")


def _debug_dump_dir() -> Path | None:
    """When set, every rewritten upstream request body is dumped here as
    <timestamp>-<route>.json so you can verify what the engine actually
    receives."""
    d = os.environ.get("LONGCTX_DEBUG_DUMP")
    if not d:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump(label: str, body: dict) -> None:
    d = _debug_dump_dir()
    if d is None:
        return
    import json as _json
    ts = f"{int(time.time() * 1000):013d}"
    path = d / f"{ts}-{label}.json"
    try:
        path.write_text(_json.dumps(body, indent=2, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass


def _last_user_query(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        return c.get("text", "")
    return ""


def _flatten_messages_to_prefill(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _splice_into_messages(messages: list[dict[str, Any]],
                          chunk_block: str) -> list[dict[str, Any]]:
    """Prepend chunk_block to the system message (or insert a system msg)."""
    out = [dict(m) for m in messages]
    for i, m in enumerate(out):
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                m["content"] = [
                    {"type": "text", "text": chunk_block}
                ] + content
            else:
                m["content"] = chunk_block + str(content)
            return out
    return [{"role": "system", "content": chunk_block.rstrip()}] + out


def _format_chunks_block(chunks: list[Any]) -> str:
    """Render retrieved chunks into a fenced markdown system block.

    Caps total bytes to `splice_max_chars` (default ~16KB ≈ 4K tokens).
    Without the cap, 8 chunks × 50-line windows balloon to ~12K tokens
    of splice every turn, blowing out small-context Hermes sessions.
    Earliest (highest-rank) chunks fully kept; later ones truncated or
    dropped.
    """
    from longctx_svc.config import get_config
    budget = get_config().limits.splice_max_chars
    parts = ["## Retrieved code context", ""]
    used = sum(len(p) for p in parts)
    for c in chunks:
        fp = c.file_path if hasattr(c, "file_path") else c["file_path"]
        sl = c.start_line if hasattr(c, "start_line") else c["start_line"]
        el = c.end_line if hasattr(c, "end_line") else c["end_line"]
        text = c.text if hasattr(c, "text") else c["text"]
        text = text.rstrip()
        header = f"// {fp}:{sl}-{el}"
        # Reserve ~30 chars for fence + newlines.
        chunk_overhead = len(header) + 12
        remaining = budget - used - chunk_overhead
        if remaining <= 200:  # not enough room for anything useful
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n... [truncated]"
        block = f"{header}\n```\n{text}\n```\n\n"
        parts.append(block)
        used += len(block)
    return "".join(parts).rstrip() + "\n\n"


async def _retrieve_for_messages(
    messages: list[dict[str, Any]], request: Request,
    top_k: int = 8,
) -> tuple[str | None, list, dict[str, str]]:
    """Run scope detect + retrieve. Returns (scope_path, chunks, hdrs)."""
    state = get_state()
    sid, detected_via = extract_session_id(
        dict(request.headers), body_metadata=None,
    )
    prefill = _flatten_messages_to_prefill(messages)
    query = _last_user_query(messages)

    scope = detect_scope(prefill)
    hdrs: dict[str, str] = {
        "x-longctx-session": sid or "ephemeral",
        "x-longctx-scope": "",
        "x-longctx-chunks-used": "0",
        "x-longctx-scope-status": "no-scope",
    }
    if scope is None or not query:
        return None, [], hdrs

    if sid is not None:
        state.sessions.bind(sid, scope.scope_hash, detected_via or "")

    entry = state.register_scope(
        scope.root, scope.scope_hash, scope.sentinel,
    )
    with entry.lock:
        if entry.index is None or entry.index.embeddings is None:
            entry.status = "indexing"
            try:
                files = walk_hot_scope(
                    scope.root,
                    mentioned_paths=list(scope.mentioned_paths),
                )
                if len(files) < 50:
                    files = walk_package_scope(scope.root)[:50_000]
                entry.index = build_index(
                    scope.root, files, scope.scope_hash,
                )
                entry.status = "ready" \
                    if entry.index.embeddings is not None else "empty"
            except Exception as e:  # noqa: BLE001
                entry.status = "error"
                entry.error = str(e)
                return str(scope.root), [], {
                    **hdrs,
                    "x-longctx-scope": str(scope.root),
                    "x-longctx-scope-status": "error",
                }

    if entry.index is None or entry.index.embeddings is None:
        return str(scope.root), [], {
            **hdrs,
            "x-longctx-scope": str(scope.root),
            "x-longctx-scope-status": entry.status,
        }

    result = state.pipeline.retrieve(
        query=query, index=entry.index, top_k=top_k,
    )
    hdrs.update({
        "x-longctx-scope": str(scope.root),
        "x-longctx-chunks-used": str(len(result.chunks)),
        "x-longctx-scope-status": entry.status,
    })
    return str(scope.root), result.chunks, hdrs


@router.get("/v1/models")
async def proxy_list_models(request: Request) -> Response:
    """Pass-through to upstream so clients can discover served models."""
    upstream = _upstream()
    if not upstream:
        raise HTTPException(
            status_code=503,
            detail="proxy disabled: set LONGCTX_UPSTREAM or pass --upstream",
        )
    forward_headers = {}
    auth = request.headers.get("authorization")
    if auth:
        forward_headers["authorization"] = auth
    target = f"{upstream.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=30.0) as ac:
        r = await ac.get(target, headers=forward_headers)
    return Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    upstream = _upstream()
    if not upstream:
        raise HTTPException(
            status_code=503,
            detail="proxy disabled: set LONGCTX_UPSTREAM or pass --upstream",
        )
    body = await request.json()
    messages = body.get("messages") or []
    top_k = int(body.get("longctx_top_k", 8))
    _, chunks, hdrs = await _retrieve_for_messages(
        messages, request, top_k=top_k,
    )
    if chunks:
        block = _format_chunks_block(chunks)
        body["messages"] = _splice_into_messages(messages, block)

    _dump("chat-completions", body)

    forward_headers = {"content-type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        forward_headers["authorization"] = auth

    is_stream = bool(body.get("stream", False))
    target = f"{upstream.rstrip('/')}/v1/chat/completions"

    if is_stream:
        async def gen():
            async with httpx.AsyncClient(timeout=None) as ac:
                async with ac.stream(
                    "POST", target, json=body, headers=forward_headers,
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
        sr = StreamingResponse(gen(), media_type="text/event-stream")
        for k, v in hdrs.items():
            sr.headers[k] = v
        return sr

    async with httpx.AsyncClient(timeout=120.0) as ac:
        r = await ac.post(target, json=body, headers=forward_headers)
    out = Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )
    for k, v in hdrs.items():
        out.headers[k] = v
    return out


@router.post("/v1/completions")
async def proxy_completions(request: Request) -> Response:
    upstream = _upstream()
    if not upstream:
        raise HTTPException(
            status_code=503,
            detail="proxy disabled: set LONGCTX_UPSTREAM or pass --upstream",
        )
    body = await request.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    fake_messages = [{"role": "user", "content": str(prompt)}]
    top_k = int(body.get("longctx_top_k", 8))
    _, chunks, hdrs = await _retrieve_for_messages(
        fake_messages, request, top_k=top_k,
    )
    if chunks:
        block = _format_chunks_block(chunks)
        body["prompt"] = block + str(prompt)

    _dump("completions", body)

    forward_headers = {"content-type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        forward_headers["authorization"] = auth

    target = f"{upstream.rstrip('/')}/v1/completions"
    async with httpx.AsyncClient(timeout=120.0) as ac:
        r = await ac.post(target, json=body, headers=forward_headers)
    out = Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )
    for k, v in hdrs.items():
        out.headers[k] = v
    return out
