"""FastAPI app exposing POST /retrieve and GET /longctx/status. PRD §4."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from longctx_svc import __version__
from longctx_svc.eviction_store import EvictedChunk, get_eviction_store
from longctx_svc.indexer.builder import build_index
from longctx_svc.scope.detect import detect_scope
from longctx_svc.scope.walk import walk_hot_scope, walk_package_scope
from longctx_svc.session.manager import extract_session_id
from longctx_svc.state import get_state


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start the idle-eviction janitor when serving (PRD §5.7).

    Disabled in test contexts where LONGCTX_NO_JANITOR=1, so unit tests
    don't have to wait on a background thread shutdown.
    """
    task: asyncio.Task | None = None
    if not os.environ.get("LONGCTX_NO_JANITOR"):
        async def _janitor() -> None:
            while True:
                try:
                    await asyncio.sleep(60.0)
                    state = get_state()
                    state.evict_idle_indexes()
                    state.sessions.evict_idle()
                    # V3 evict-to-vector store (Tier 2): drop sessions
                    # that haven't been touched in 30 min. Keeps the
                    # store from leaking memory across long server
                    # uptimes when V3 sessions never explicitly close.
                    get_eviction_store().evict_idle()
                except asyncio.CancelledError:
                    return
                except Exception:  # noqa: BLE001
                    # Janitor must never crash the loop.
                    pass
        task = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


app = FastAPI(
    title="longctx-svc",
    version=__version__,
    description=(
        "Local retrieval companion for inference servers. "
        "POST /retrieve to get top-K chunks for a session+prefill+query."
    ),
    lifespan=_lifespan,
)

from longctx_svc.proxy import router as _proxy_router  # noqa: E402
app.include_router(_proxy_router)


class RetrieveRequest(BaseModel):
    """PRD §4: {session_id, prefill_text, query, top_k}."""
    session_id: str | None = Field(
        default=None,
        description="Session id; if None, falls back to header detection",
    )
    prefill_text: str = Field(
        ...,
        description="Combined system + user messages from the LLM prompt",
    )
    query: str = Field(..., description="The user's actual question")
    top_k: int = Field(default=8, ge=1, le=64)
    explicit_scope: str | None = Field(
        default=None,
        description="Override scope detection with an explicit project root",
    )
    default_scope: str | None = Field(
        default=None,
        description=(
            "Fallback scope when no path is detected in the prefill. "
            "Used by engine adapters to auto-bind a scope (e.g., the "
            "engine's working directory) so tool-using agents that "
            "don't put absolute paths in messages still get retrieval."
        ),
    )


class ChunkOut(BaseModel):
    text: str
    file_path: str
    start_line: int
    end_line: int
    file_type: str
    score: float


class RetrieveResponse(BaseModel):
    chunks: list[ChunkOut]
    scope_path: str | None
    scope_hash: str | None
    scope_status: str
    scope_sentinel: str | None
    session_id: str | None
    used_rerank: bool
    paraphrases_count: int


class StatusResponse(BaseModel):
    version: str
    mode: str
    sessions_active: int
    scopes_loaded: int
    scopes_total: int
    memory_mb: float
    scopes: list[dict]


# ---------------------------------------------------------------------------
# V3 evict-to-vector — Tier 2 + Tier 3 of the rescue stack.
# Design doc: TriAttention V3 — 3-Tier Eviction Rescue Architecture (obsidian).
# ---------------------------------------------------------------------------


class EvictWriteChunk(BaseModel):
    """One evicted span as posted by V3.

    `text` is decoded by the V3 caller (vLLM has the tokenizer); the store
    deliberately does not ship a tokenizer to keep the surface narrow.
    """
    text: str = Field(..., description="Decoded text of the evicted span")
    token_range: tuple[int, int] = Field(
        ..., description="(start, end) in the original prompt token-id sequence"
    )
    layer: int = Field(..., description="V3 attention layer index that evicted this")
    score: float = Field(..., description="V3 eviction score at write time")


class EvictWriteRequest(BaseModel):
    session_id: str = Field(..., description="V3 session id; per-session isolated")
    chunks: list[EvictWriteChunk] = Field(default_factory=list)


class EvictWriteResponse(BaseModel):
    accepted: int
    session_total: int


class EvictRetrieveRequest(BaseModel):
    session_id: str
    query: str = Field(..., description="The user's question for this turn")
    top_k: int = Field(default=8, ge=1, le=64)
    score_floor: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum (post-fusion) score to surface a chunk (0 = no floor)",
    )
    # v0.3.1: hybrid scoring + cross-encoder rerank. Both default to
    # backwards-compatible behavior: hybrid_alpha=None reads the env
    # default (0.5), use_rerank=True only kicks in when the session has
    # ≥100 chunks (cheap CPU guard). Existing callers (vllm-turboquant
    # V3 prefill hook) keep working without changes.
    hybrid_alpha: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Cosine/BM25 fusion weight in [0, 1]. 1.0 = pure cosine "
            "(legacy v0.3.0 behavior), 0.0 = pure BM25, 0.5 = balanced. "
            "None reads VLLM_TRIATT_RESCUE_HYBRID_ALPHA (default 0.5)."
        ),
    )
    use_rerank: bool = Field(
        default=True,
        description=(
            "Apply cross-encoder rerank when the session has enough "
            "chunks to justify the CPU cost (≥100). No-op below that."
        ),
    )
    prefilter: int = Field(
        default=32, ge=1, le=512,
        description=(
            "Candidate pool size for cross-encoder rerank. Must be "
            "≥ top_k. Ignored when rerank is skipped."
        ),
    )


class EvictRetrieveChunk(BaseModel):
    text: str
    token_range: tuple[int, int]
    layer: int
    score: float


class EvictRetrieveResponse(BaseModel):
    chunks: list[EvictRetrieveChunk]
    session_total: int  # how many evicted chunks total in this session


class EvictDumpResponse(BaseModel):
    """Coverage-instrumentation dump: full chunk listing for a session.

    Harness use only — lets the 10M PRD scorer measure what fraction of
    planted-fact token positions fell inside any captured eviction span.
    Emits token_range + layer + score; text omitted by default (large).
    """
    session_id: str
    session_total: int
    token_ranges: list[tuple[int, int]]
    layers: list[int]
    scores: list[float]


@app.post("/evict/write", response_model=EvictWriteResponse)
async def evict_write(req: EvictWriteRequest) -> EvictWriteResponse:
    """V3 calls this after each eviction round. Persists evicted-span text
    into a per-session vector index for later rehydration."""
    if not req.chunks:
        return EvictWriteResponse(accepted=0, session_total=0)
    store = get_eviction_store()
    chunks = [
        EvictedChunk(
            text=c.text,
            token_range=c.token_range,
            layer=int(c.layer),
            score=float(c.score),
        )
        for c in req.chunks
    ]
    total = store.write(req.session_id, chunks)
    return EvictWriteResponse(accepted=len(chunks), session_total=total)


@app.post("/evict/retrieve", response_model=EvictRetrieveResponse)
async def evict_retrieve(req: EvictRetrieveRequest) -> EvictRetrieveResponse:
    """vLLM prefill hook calls this on each new turn. Returns top-K
    evicted chunks for the session, ranked by cosine similarity to the
    current query. Strictly session-isolated.

    Auto-resumes from cold storage if the session was hibernated since
    last access. Resume is transparent to the caller — same response
    shape, same isolation guarantees, just an extra disk read.
    """
    store = get_eviction_store()
    # Tier 4: auto-resume if hibernated. No-op when the session is
    # already in RAM or has no snapshot.
    try:
        store.maybe_resume(req.session_id)
    except Exception:  # noqa: BLE001
        # Snapshot fingerprint mismatch / schema mismatch / disk error
        # — fall through to empty retrieve rather than 500-ing.
        pass
    chunks = store.retrieve(
        req.session_id, req.query, top_k=req.top_k,
        score_floor=req.score_floor,
        hybrid_alpha=req.hybrid_alpha,
        use_rerank=req.use_rerank,
        prefilter=req.prefilter,
    )
    total = store.stats()["per_session"].get(req.session_id, 0)
    return EvictRetrieveResponse(
        chunks=[
            EvictRetrieveChunk(
                text=c.text,
                token_range=c.token_range,
                layer=c.layer,
                score=c.score,
            )
            for c in chunks
        ],
        session_total=total,
    )


@app.get("/evict/dump", response_model=EvictDumpResponse)
async def evict_dump(session_id: str) -> EvictDumpResponse:
    """Coverage instrumentation: dump every captured chunk's token_range
    for a session so the harness can compute fact-span coverage. No
    semantic ranking, no query — just a flat enumeration."""
    store = get_eviction_store()
    with store._lock:  # noqa: SLF001
        idx = store._sessions.get(session_id)  # noqa: SLF001
        if idx is None:
            return EvictDumpResponse(
                session_id=session_id, session_total=0,
                token_ranges=[], layers=[], scores=[],
            )
        ranges = [tuple(c.token_range) for c in idx.chunks]
        layers = [int(c.layer) for c in idx.chunks]
        scores = [float(c.score) for c in idx.chunks]
    return EvictDumpResponse(
        session_id=session_id, session_total=len(ranges),
        token_ranges=ranges, layers=layers, scores=scores,
    )


# ---------------------------------------------------------------------------
# Tier 4 — session hibernation (cold storage). Explicit endpoints so clients
# can hibernate/resume on their own lifecycle. The janitor auto-hibernates
# on idle; /evict/retrieve auto-resumes from snapshot.
# ---------------------------------------------------------------------------


class HibernateResponse(BaseModel):
    session_id: str
    hibernated: bool
    n_chunks: int


class ResumeResponse(BaseModel):
    session_id: str
    resumed: bool
    n_chunks: int
    error: str | None = None


class HibernationStatusResponse(BaseModel):
    session_id: str
    in_ram: bool
    on_disk: bool
    n_chunks_in_ram: int
    n_chunks_on_disk: int
    embedder_fingerprint: str | None
    hibernated_at: float | None


@app.post(
    "/sessions/{session_id}/hibernate", response_model=HibernateResponse,
)
async def session_hibernate(session_id: str) -> HibernateResponse:
    """Explicitly hibernate a session — serialise its evict-store to
    disk + drop from RAM. Returns ``hibernated=False`` when the session
    has no chunks worth persisting (no-op success)."""
    store = get_eviction_store()
    # Read chunk count BEFORE hibernate (it'll be dropped after).
    with store._lock:  # noqa: SLF001
        idx = store._sessions.get(session_id)  # noqa: SLF001
        n_chunks = len(idx.chunks) if idx is not None else 0
    ok = store.hibernate(session_id)
    return HibernateResponse(
        session_id=session_id, hibernated=ok, n_chunks=n_chunks,
    )


@app.post(
    "/sessions/{session_id}/resume", response_model=ResumeResponse,
)
async def session_resume(session_id: str) -> ResumeResponse:
    """Explicitly resume a hibernated session into RAM."""
    store = get_eviction_store()
    try:
        n = store.resume(session_id)
        return ResumeResponse(
            session_id=session_id, resumed=n > 0, n_chunks=n,
        )
    except Exception as e:  # noqa: BLE001
        return ResumeResponse(
            session_id=session_id, resumed=False, n_chunks=0,
            error=f"{type(e).__name__}: {e}",
        )


@app.get(
    "/sessions/{session_id}/hibernation-status",
    response_model=HibernationStatusResponse,
)
async def session_hibernation_status(
    session_id: str,
) -> HibernationStatusResponse:
    """Inspect the session's storage tiers. Used by operators + tests."""
    store = get_eviction_store()
    with store._lock:  # noqa: SLF001
        idx = store._sessions.get(session_id)  # noqa: SLF001
        in_ram = idx is not None
        n_chunks_in_ram = len(idx.chunks) if idx is not None else 0
    on_disk = store.is_hibernated(session_id)
    n_chunks_on_disk = 0
    fingerprint: str | None = None
    hibernated_at: float | None = None
    if on_disk:
        try:
            import json as _json
            with (store._session_dir(session_id) / "metadata.json").open() as f:  # noqa: SLF001
                meta = _json.load(f)
            n_chunks_on_disk = int(meta.get("n_chunks", 0))
            fingerprint = meta.get("embedder_fingerprint")
            hibernated_at = float(meta.get("hibernated_at", 0.0)) or None
        except Exception:  # noqa: BLE001
            pass
    return HibernationStatusResponse(
        session_id=session_id, in_ram=in_ram, on_disk=on_disk,
        n_chunks_in_ram=n_chunks_in_ram, n_chunks_on_disk=n_chunks_on_disk,
        embedder_fingerprint=fingerprint, hibernated_at=hibernated_at,
    )


def _build_scope_index(state, scope, entry) -> None:
    """Synchronous index build. Caller holds entry.lock."""
    if entry.index is not None and entry.index.embeddings is not None:
        return
    entry.status = "indexing"
    try:
        files = walk_hot_scope(
            scope.root, mentioned_paths=list(scope.mentioned_paths),
        )
        mode = "hot"
        if len(files) < 50:
            files = walk_package_scope(scope.root)[: 50_000]
            mode = "package"
        entry.index = build_index(
            scope.root, files, scope.scope_hash,
            embedder=getattr(state.pipeline, "_embedder", None),
        )
        entry.indexed_files = frozenset(str(f) for f in files)
        entry.mode = mode
        entry.status = "ready" if entry.index.embeddings is not None \
            else "empty"
        if entry.status == "ready":
            try:
                from longctx_svc.cache.disk import save_index
                save_index(entry.index, scope.sentinel)
            except Exception:  # noqa: BLE001
                pass
            state.attach_watcher(scope.scope_hash)
    except Exception as e:
        entry.status = "error"
        entry.error = str(e)
        raise


def _ensure_scope(state, scope, async_kickoff: bool = False) -> Optional["object"]:
    """Ensure the scope has an index.

    Disk-reload happens inside `register_scope`; this only kicks off a
    build on a fresh scope or one that was evicted from RAM.

    When `async_kickoff` is True (request opt-in via `x-longctx-async`),
    return immediately with the entry in `indexing` state and a worker
    thread doing the build in the background (PRD §5.2).
    """
    entry = state.register_scope(
        scope.root, scope.scope_hash, scope.sentinel,
    )
    if entry.index is not None and entry.index.embeddings is not None:
        return entry
    if async_kickoff:
        if entry.status == "indexing":
            return entry
        entry.status = "indexing"
        import threading

        def _worker() -> None:
            with entry.lock:
                try:
                    _build_scope_index(state, scope, entry)
                except Exception:  # noqa: BLE001
                    pass

        t = threading.Thread(
            target=_worker, daemon=True,
            name=f"longctx-build-{scope.scope_hash[:8]}",
        )
        t.start()
        return entry
    with entry.lock:
        _build_scope_index(state, scope, entry)
    return entry


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, request: Request,
                   response: Response) -> RetrieveResponse:
    state = get_state()

    # Session resolution (PRD §5.5)
    sid = req.session_id
    detected_via = "explicit-body" if sid else None
    if sid is None:
        sid, detected_via = extract_session_id(
            dict(request.headers), body_metadata=None,
        )

    # PRD §6.3 / v0.3.3 — workspace mode: explicit_scope == "ws:" means
    # "search across every scope this session has touched".
    if req.explicit_scope == "ws:":
        return await _retrieve_workspace(req, request, response, sid, state)

    explicit_root = Path(req.explicit_scope) if req.explicit_scope else None
    # Pass default_scope so detect_scope can resolve relative path mentions
    # against the caller's repo root (2026-05-11 fix — Aider and other
    # cwd-relative agents previously fell through to the last-resort
    # default_scope branch below, which wedged the scope to the
    # vllm-swift cwd instead of the actual repo).
    scope = detect_scope(req.prefill_text,
                         explicit_root=explicit_root,
                         default_scope=req.default_scope)
    # UX fallback (no PRD section — testers asked for it 2026-05-07):
    # if path detection found nothing AND a default_scope is provided,
    # treat the default as the scope. Lets tool-using agents (Hermes,
    # OpenCode agentic mode) get retrieval without forcing the user to
    # paste absolute paths in every message.
    if scope is None and req.default_scope:
        scope = detect_scope(
            "", explicit_root=Path(req.default_scope),
        )
        if scope is not None:
            scope = type(scope)(
                root=scope.root, sentinel="default-scope",
                scope_hash=scope.scope_hash,
                mentioned_paths=scope.mentioned_paths,
            )
    # PRD §6.3 / v0.3.3 — bind every additional sentinel root the prefill
    # mentions so subsequent `ws:` queries can fan out across them.
    if sid and scope is not None and not explicit_root:
        from longctx_svc.scope.detect import detect_scopes
        for s in detect_scopes(req.prefill_text,
                               default_scope=req.default_scope):
            if s.scope_hash != scope.scope_hash:
                state.sessions.bind(sid, s.scope_hash, "multi-mention")
                # Build the secondary scope's index synchronously so a
                # follow-up workspace query can actually search it.
                try:
                    _ensure_scope(state, s, async_kickoff=False)
                except Exception:  # noqa: BLE001
                    pass
        # Re-bind the primary so its detected_via wins
        state.sessions.bind(sid, scope.scope_hash, "header:x-session-affinity")

    if scope is None:
        # PRD §5.1: no path → no scope → no retrieval
        response.headers["x-longctx-session"] = sid or "ephemeral"
        response.headers["x-longctx-scope"] = ""
        response.headers["x-longctx-chunks-used"] = "0"
        response.headers["x-longctx-scope-status"] = "no-scope"
        return RetrieveResponse(
            chunks=[], scope_path=None, scope_hash=None,
            scope_status="no-scope", scope_sentinel=None,
            session_id=sid, used_rerank=False, paraphrases_count=0,
        )

    # Bind session to scope
    if sid is not None:
        state.sessions.bind(sid, scope.scope_hash, detected_via or "")

    # Build / fetch the index. Honor x-longctx-async for background kickoff.
    async_kickoff = (request.headers.get("x-longctx-async") or "").lower() \
        in ("1", "true", "yes")
    try:
        _ensure_scope(state, scope, async_kickoff=async_kickoff)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"index build failed: {exc!s}",
        )

    entry = state.get_scope(scope.scope_hash)
    if entry is None:
        raise HTTPException(status_code=500, detail="scope build vanished")

    # Async path: index may not be ready yet — return empty chunks with
    # scope_status=indexing so the client knows to retry.
    if entry.index is None or entry.index.embeddings is None:
        response.headers["x-longctx-session"] = sid or "ephemeral"
        response.headers["x-longctx-scope"] = str(scope.root)
        response.headers["x-longctx-chunks-used"] = "0"
        response.headers["x-longctx-scope-status"] = entry.status
        return RetrieveResponse(
            chunks=[], scope_path=str(scope.root),
            scope_hash=scope.scope_hash,
            scope_status=entry.status,
            scope_sentinel=scope.sentinel,
            session_id=sid,
            used_rerank=False, paraphrases_count=0,
        )

    result = state.pipeline.retrieve(
        query=req.query, index=entry.index, top_k=req.top_k,
        scope_root=scope.root,
    )
    entry.last_query_at = time.time()
    # PRD §6.1 / v0.3.1: path-based promotion
    try:
        mentioned = [str(p) for p in scope.mentioned_paths]
        state.maybe_promote(scope.scope_hash, mentioned)
    except Exception:  # noqa: BLE001
        pass
    # PRD §6.2 / v0.3.2: confidence-driven promotion
    top_score = float(result.scores[0]) if result.scores else 0.0
    response.headers["x-longctx-confidence"] = f"{top_score:.3f}"
    try:
        if sid:
            state.maybe_promote_on_confidence(
                scope.scope_hash, sid, top_score,
            )
    except Exception:  # noqa: BLE001
        pass

    response.headers["x-longctx-session"] = sid or "ephemeral"
    response.headers["x-longctx-scope"] = str(scope.root)
    response.headers["x-longctx-chunks-used"] = str(len(result.chunks))
    response.headers["x-longctx-scope-status"] = entry.status

    return RetrieveResponse(
        chunks=[ChunkOut(
            text=c.text, file_path=c.file_path,
            start_line=c.start_line, end_line=c.end_line,
            file_type=c.file_type, score=s,
        ) for c, s in zip(result.chunks, result.scores)],
        scope_path=str(scope.root),
        scope_hash=scope.scope_hash,
        scope_status=entry.status,
        scope_sentinel=scope.sentinel,
        session_id=sid,
        used_rerank=result.used_rerank,
        paraphrases_count=len(result.paraphrases),
    )


async def _retrieve_workspace(req: "RetrieveRequest", request: Request,
                                response: Response, sid: str | None,
                                state) -> "RetrieveResponse":
    """PRD §6.3 / v0.3.3 — workspace query across all scopes the
    session has bound. Returns top-K merged across loaded indexes.
    """
    scope_hashes: list[str] = []
    if sid:
        sess = state.sessions.get(sid)
        if sess is not None:
            scope_hashes = list(sess.scope_hashes)
    entries = state.get_scopes(scope_hashes)
    if not entries:
        response.headers["x-longctx-session"] = sid or "ephemeral"
        response.headers["x-longctx-scope"] = "ws:"
        response.headers["x-longctx-chunks-used"] = "0"
        response.headers["x-longctx-scope-status"] = "no-scope"
        return RetrieveResponse(
            chunks=[], scope_path="ws:", scope_hash=None,
            scope_status="no-scope", scope_sentinel="workspace",
            session_id=sid, used_rerank=False, paraphrases_count=0,
        )
    indexes = [e.index for e in entries if e.index is not None]
    result = state.pipeline.retrieve_multi(
        query=req.query, indexes=indexes, top_k=req.top_k,
    )
    for e in entries:
        e.last_query_at = time.time()

    response.headers["x-longctx-session"] = sid or "ephemeral"
    response.headers["x-longctx-scope"] = "ws:"
    response.headers["x-longctx-chunks-used"] = str(len(result.chunks))
    response.headers["x-longctx-scope-status"] = "ready"
    top_score = float(result.scores[0]) if result.scores else 0.0
    response.headers["x-longctx-confidence"] = f"{top_score:.3f}"
    return RetrieveResponse(
        chunks=[ChunkOut(
            text=c.text, file_path=c.file_path,
            start_line=c.start_line, end_line=c.end_line,
            file_type=c.file_type, score=s,
        ) for c, s in zip(result.chunks, result.scores)],
        scope_path="ws:",
        scope_hash=None,
        scope_status="ready",
        scope_sentinel="workspace",
        session_id=sid,
        used_rerank=result.used_rerank,
        paraphrases_count=len(result.paraphrases),
    )


def _format_status_text(state, scopes_meta: list[dict]) -> str:
    """Render the Sarah-visible status block per PRD §5.9."""
    lines = ["[longctx] mode: local-only"]
    sessions = state.sessions.all_sessions()
    if sessions:
        most_recent = max(sessions, key=lambda e: e.last_seen_at)
        lines.append(
            f"[longctx] session: {most_recent.session_id} "
            f"(via: {most_recent.detected_via})"
        )
    if scopes_meta:
        most_recent_scope = scopes_meta[-1]
        lines.append(
            f"[longctx] scope: {most_recent_scope['scope_path']} "
            f"(sentinel: {most_recent_scope['sentinel']})"
        )
        lines.append(
            f"[longctx] indexed: {most_recent_scope['files']} files, "
            f"{most_recent_scope['chunks']} chunks, "
            f"status {most_recent_scope['status']}"
        )
    lines.append(
        f"[longctx] memory: {state.loaded_scope_count()} scopes loaded, "
        f"{state.total_memory_bytes() / 1e6:.1f} MB total"
    )
    try:
        from longctx_svc.cache.disk import (
            cache_root_size_bytes, list_cached,
        )
        from longctx_svc.config import get_config
        n_disk = len(list_cached())
        bytes_disk = cache_root_size_bytes()
        cfg_dir = get_config().cache_dir
        lines.append(
            f"[longctx] disk cache: {cfg_dir}, {n_disk} scopes, "
            f"{bytes_disk / 1e6:.1f} MB"
        )
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


@app.get("/longctx/status")
async def status(request: Request) -> Response:
    """Status endpoint. JSON by default; Sarah-style plain text on
    Accept: text/plain (PRD §5.9)."""
    state = get_state()
    scopes_meta = []
    for e in state.all_scopes():
        scopes_meta.append({
            "scope_path": str(e.scope_root),
            "scope_hash": e.scope_hash,
            "sentinel": e.sentinel,
            "status": e.status,
            "chunks": e.index.chunk_count if e.index else 0,
            "files": e.index.file_count if e.index else 0,
            "memory_mb": (e.index.memory_bytes / 1e6) if e.index else 0.0,
        })

    accept = (request.headers.get("accept") or "").lower()
    if "text/plain" in accept:
        text = _format_status_text(state, scopes_meta)
        return Response(content=text, media_type="text/plain")

    from fastapi.responses import JSONResponse
    payload = StatusResponse(
        version=__version__,
        mode="local-only",
        sessions_active=len(state.sessions),
        scopes_loaded=state.loaded_scope_count(),
        scopes_total=len(scopes_meta),
        memory_mb=state.total_memory_bytes() / 1e6,
        scopes=scopes_meta,
    )
    return JSONResponse(payload.model_dump())


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "version": __version__}
