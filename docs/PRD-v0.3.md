# longctx v0.3 — PRD

**Status:** Draft
**Owner:** TheTom
**Targets:** v0.3.0, v0.3.1, v0.3.2, v0.3.3
**Last updated:** 2026-05-06

---

## 1. Problem

Developers running local inference servers (vllm-swift, llama.cpp, vLLM) get no retrieval out of the box. To get codebase-aware answers from a local LLM, they currently:

- Paste large chunks of code into prompts (slow, expensive on cloud, OOM-prone locally)
- Use closed tools (Cursor, Aider) that bundle proprietary retrieval plumbing
- Build their own retrieval glue (boilerplate, no shared standard)

The longctx v0.2 library proves the retrieval pattern works on long-context evals. v0.3 turns that pattern into an **automatic, scoped, session-aware retrieval companion** that any developer can run alongside a local inference server with zero per-project setup.

---

## 2. Sarah, the user

Sarah is building a Next.js + FastAPI + Docker app. Her codebase is 50K LOC across 200 files. She runs `vllm-swift` locally serving Qwen2.5-14B-Instruct-1M, and chats with it through OpenCode (or Hermes).

### Sarah's day with v0.2

She has to manually export her codebase, embed it with a script, store the faiss index, and call `LongCtxClient` from a custom Python wrapper. When she edits a file, the index is stale until she re-runs the embed step. Switching between her billing app and her auth library means swapping config files. She does it once, hates it, falls back to pasting code into prompts.

### Sarah's day with v0.3

She launches `vllm-swift` and opens OpenCode in `~/dev/myapp`. First message: *"why is auth failing in the docker build?"*

- vllm-swift sees the prefill, parses paths from OpenCode's system context, detects `~/dev/myapp` as the project root (sentinel: `package.json`)
- longctx-svc starts indexing in the background. By the second turn, retrieval is warm.
- Subsequent answers cite real files at real line numbers. Sub-second latency, $0/query.
- She edits `auth.middleware.ts`. The watcher debounces, re-embeds the touched file. Next query reflects the change.
- She switches to a side terminal in `~/dev/auth-lib` with another OpenCode session. vllm-swift detects the new session (different `x-session-affinity`), detects the new scope, indexes it independently. The two sessions don't cross-contaminate.

She types `/longctx status` and sees:

```
[longctx] session: opencode:abc123 (header: x-session-affinity)
[longctx] scope: ~/dev/myapp (sentinel: package.json)
[longctx] indexed: 842 files, 3,918 chunks, updated 12s ago
[longctx] memory: 2 scopes loaded, 1.4GB total
```

When something feels off, she can see exactly what's indexed and how. Invisible — but not mysterious.

---

## 3. Goals

- **Zero-config retrieval** for local-inference-server users on a project
- **Scoped indexing** that doesn't blow up on monorepos or massive directories
- **Session-isolated** indexes so concurrent harnesses don't cross-contaminate
- **Debug-visible** state for when retrieval feels wrong
- **Cross-engine portable** — same retrieval companion works with vllm-swift, TheTom/llama-cpp-turboquant, TheTom/vllm

## Non-goals (v0.3)

- Full Cursor/Aider replacement (no agentic loops, no edit-and-apply)
- Tool-use retrieval (RAG-on-tool-output is a different problem)
- Cloud-hosted retrieval (local-only)
- Multi-user / LAN-exposed deployments (single-user-on-localhost only)
- Fine-tuned rerankers (off-the-shelf bi-encoder + caps)
- Workspace-wide indexing by default (manual override only until v0.3.3)

---

## 4. Architecture

```
┌──────────────────┐     OpenAI HTTP      ┌──────────────────┐
│ Hermes / OpenCode│ ───────────────────▶ │  vllm-swift      │
└──────────────────┘                      │  (or fork)       │
                                          │                  │
                                          │ --retrieval-     │
                                          │   endpoint       │
                                          │   ↓              │
                                          └────┬─────────────┘
                                               │
                                               ▼
                                          ┌──────────────────┐
                                          │  longctx-svc     │
                                          │  (FastAPI)       │
                                          │  - scope detect  │
                                          │  - index cache   │
                                          │  - retrieve      │
                                          │  - file watcher  │
                                          └──────────────────┘
```

### Components

**longctx-svc** (new in v0.3): FastAPI service, single endpoint `POST /retrieve`. Accepts `{session_id, prefill_text, query, top_k}`, returns `{chunks, scope_path, scope_status}`.

**Inference-server hook** (new in each fork): `--retrieval-endpoint URL` flag. When set, server calls longctx-svc before each `/v1/chat/completions`. Augments the system message with retrieved chunks. Forward to inference normally.

**Persistent cache**: `~/.longctx/<scope-hash>/` containing `index.faiss`, `chunks.jsonl`, `metadata.json`. Survives restarts.

---

## 5. Functional requirements

### 5.1 Scope detection (v0.3.0)

**Trigger**: first request of a session, or any request whose session has no detected scope.

**Detection sequence** (synchronous, <100ms):
1. Parse prefill (system + user messages) for absolute paths matching `^/(?:Users|home)/[^/]+/[^\s]+`
2. From mentioned paths, walk up to nearest project sentinel
3. Sentinels (in priority order): `.git`, `pyproject.toml`, `package.json`, `pnpm-workspace.yaml`, `Cargo.toml`, `go.mod`, `WORKSPACE`, `BUILD.bazel`, `pom.xml`, `Gemfile`
4. If multiple mentioned files share a sub-package inside a monorepo, prefer the package, not the workspace root
5. If no path detected: no scope, no retrieval, fall through to plain inference

**Scope path is canonicalized** before hashing: `realpath()`, lowercase on case-insensitive volumes, strip trailing slashes.

### 5.2 Hot scope (v0.3.0)

When a project root is detected, **Hot scope** = the indexable subset to embed first:
- Mentioned/open files
- Their containing directories (non-recursive, immediate siblings only)
- The nearest package's `src/` directory (recursive, capped)
- **Hard cap: 1000 files**

If Hot scope yields <500 files, expand to **Package scope** = the entire detected project root, recursive, capped at 10K-50K files.

### 5.3 Caps and ignores (v0.3.0)

| Limit | Default |
|---|---|
| Max single file size | 5 MB |
| Max files (Hot) | 1000 |
| Max files (Package) | 50,000 |
| Soft index time budget | 60s |
| Hard index time budget | 5 min |
| Max in-memory indexes | 4 (LRU eviction) |

**Always skipped**: `.git`, `node_modules`, `.venv`, `__pycache__`, `dist`, `build`, `target`, `.next`, `.cache`, lockfiles (`package-lock.json`, `yarn.lock`, `Cargo.lock`, `go.sum`), binary files (detected by extension + magic bytes).

**Always respected**: `.gitignore` rules at every directory level.

### 5.4 Chunking strategy (v0.3.0)

| File type | Strategy |
|---|---|
| Code (`.py`, `.ts`, `.go`, `.rs`, `.swift`, etc.) | Top-level definitions where parseable; fallback to N-line windows (default 50 lines) |
| Prose (`.md`, `.txt`, `.rst`) | Paragraph-aware, with overlap |
| Config (`.json`, `.yaml`, `.toml`) | Whole-file as one chunk if under cap; else N-line windows |
| Other | N-line windows |

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (CPU, default). User-configurable.

### 5.5 Session isolation (v0.3.0)

**Session identification, in priority order:**
1. `x-session-affinity` header (OpenCode default)
2. `x-session-id` header
3. `metadata.session_id` in OpenAI body
4. **Stateless fallback**: no session header → ephemeral request, scope detected per-request, no caching

**No synthetic fingerprint.** If no session header, treat each request independently. (See Risk #2.)

**Indexes are keyed by canonical scope path, not by session.** Sessions point to scopes:

```
sessions: hermes:abc → scope_hash_foo
          opencode:def → scope_hash_bar
          opencode:ghi → scope_hash_foo  (shares index)
scopes: scope_hash_foo → loaded index, watcher, status
        scope_hash_bar → loaded index, watcher, status
```

### 5.6 Concurrency (v0.3.0)

**RW-lock per scope.** Reads (queries) acquire shared lock. Writes (re-embed on file change) acquire exclusive lock. Brief read latency during writes is acceptable for v0.3.0; copy-on-write swap is a v0.4 consideration.

### 5.7 Eviction (v0.3.0)

- **Sessions**: 2h idle → drop session entry. Underlying scope-index unchanged.
- **In-memory indexes**: 30-60min idle → evict from RAM, retain on disk. Re-load on next request to that scope (~ms from disk).
- **Disk cache**: never auto-evicted. Manual cleanup via `longctx clean --older-than 30d`.

### 5.8 Manual override (v0.3.0)

```bash
# Server-side flag
vllm-swift serve --retrieval-endpoint http://localhost:8000 --longctx-scope ./apps/billing

# Future: in-client command (v0.3.1+)
/longctx scope ./apps/billing
```

### 5.9 Debug visibility (v0.3.0)

**Response header on every request** that triggered retrieval:

```
x-longctx-session: opencode:abc123
x-longctx-scope: /Users/tom/dev/myapp
x-longctx-chunks-used: 8
x-longctx-scope-status: ready
```

**`/longctx status` endpoint on longctx-svc**:

```
[longctx] mode: local-only
[longctx] session: opencode:abc123 (header: x-session-affinity)
[longctx] scope: /Users/tom/dev/myapp (sentinel: package.json)
[longctx] indexed: 842 files, 3,918 chunks, updated 12s ago
[longctx] memory: 2 scopes loaded, 1.4GB total
[longctx] disk cache: ~/.longctx/, 8 scopes, 4.2GB
```

**Server log line** on scope detection:
```
[longctx] detected project: /Users/tom/dev/myapp (sentinel: package.json) for session opencode:abc123
```

### 5.10 Privacy (v0.3.0)

- All embeddings, indexes, and content stay on local disk
- No network calls outside `localhost`
- Stated in README, `--help`, and `[longctx] mode: local-only` in debug output
- Disk cache location is user-configurable via `LONGCTX_CACHE_DIR`

---

## 6. Release plan

### v0.3.0 — "Scoped autofill"

**Ships:**
- longctx-svc as separate package with `POST /retrieve` and `GET /longctx/status`
- Scope detection from prefill (no auto-promotion)
- Hot + Package scopes, capped
- Persistent per-scope disk cache
- File watcher with debounce (default 1s)
- Session isolation with header-based identification
- Manual override (`--longctx-scope` CLI flag)
- Debug visibility (response headers, status endpoint, server logs)
- `--retrieval-endpoint` flag in vllm-swift

**Acceptance criteria:**
- `pip install longctx-svc && longctx-svc serve` runs without further config
- 5 of 5 scenarios in §7 pass
- Average overhead vs plain inference: <100ms on warm cache
- Documented compatibility with at least 2 harnesses (OpenCode + Hermes)

### v0.3.1 — "Cross-scope promotion"

- Auto-promote Hot → Package when user references a file outside Hot
- `/longctx scope` slash command (in clients that support custom commands)
- Improved chunking heuristics (tree-sitter-based code splitting where available)

### v0.3.2 — "Confidence-driven promotion"

- Auto-promote on retrieval-confidence dip (top-K cosine score below threshold for 2+ consecutive turns)
- Per-turn classifier for multi-scope routing (when conversation spans projects)
- Telemetry on `/longctx scope` usage to inform threshold tuning

### v0.3.3 — "Workspace and multi-scope"

- Explicit workspace-wide scope via `/longctx scope ws:` or `--longctx-scope-workspace`
- Multi-scope routing in conversation: maintain N indexes, route queries by detected per-turn context
- Cross-engine parity: ship `--retrieval-endpoint` in TheTom/llama-cpp-turboquant + TheTom/vllm

---

## 7. Smoke-test scenarios (must pass before v0.3.0 ship)

Run against OpenCode + Hermes + at least one third harness.

1. **Single-project root**
   Open harness in `~/dev/myapp` (Next.js, 50K LOC). First message asks about `auth.middleware.ts`. Verify scope detected as `~/dev/myapp` (sentinel: `package.json`), Hot scope indexed within 60s, retrieval reflects the auth file by turn 2.

2. **Monorepo, single sub-package**
   Open harness at `~/dev/monorepo` (Turborepo, 5 packages). First message references `apps/billing/src/Charge.tsx`. Verify scope = `apps/billing`, NOT the monorepo root. `apps/payments` not indexed.

3. **Monorepo, ambiguous root**
   Open harness at `~/dev/monorepo`. First message has no path mention. Verify no scope detected, no indexing, plain inference.

4. **File change reflected**
   In an active session with indexed scope, edit a file. Within 5s, query that file's content. Verify retrieval reflects the change.

5. **Concurrent sessions, separate projects**
   Run OpenCode session A in `~/dev/foo` and Hermes session B in `~/dev/bar` against the same vllm-swift. Verify queries in A retrieve from foo, queries in B retrieve from bar. No cross-contamination.

6. **Concurrent sessions, same project**
   Two OpenCode sessions in `~/dev/foo`. Verify they share the same on-disk index and in-memory loaded state.

7. **`.gitignore` respected**
   Project with `node_modules`, `.venv`, `dist`, `build`. Verify none indexed.

8. **Large file skipped**
   Project containing a 20MB file. Verify it's skipped without crashing.

9. **No-header request**
   Send `/v1/chat/completions` without any session header. Verify ephemeral handling: scope detected, retrieval served, no cache entry persisted.

10. **Cache reload**
    Run scenario 1, kill longctx-svc, restart. Verify second-session-on-same-project loads from disk cache in <500ms.

---

## 8. Risks

### R1: Harness prefill format variance
**Risk**: OpenCode and Hermes prefill paths differently. Detection regex misses one or both.
**Mitigation**: Per-harness adapter modules in `longctx_svc/parsers/`. Smoke tests against both. Fallback to no-scope when detection fails.

### R2: Synthetic session fingerprints collide
**Risk**: If we synthesize session IDs from IP+UA+model+message-hash, two browser tabs collide or first-message-hash drift causes session fragmentation.
**Mitigation**: Don't synthesize. Header-or-ephemeral. Documented in §5.5.

### R3: Memory pressure from concurrent monorepos
**Risk**: 4+ concurrent large monorepo scopes blow RAM.
**Mitigation**: Hard cap of 4 in-memory indexes (configurable). LRU eviction to disk-only. Disk cache survives.

### R4: Stale cache after external git operations
**Risk**: User runs `git checkout other-branch`, file watcher catches per-file changes but loses the bigger context. Indexed embeddings are now mismatched.
**Mitigation (v0.3.0)**: Watcher catches per-file changes; large-scale changes (`git checkout`, `git pull`) trigger a full re-embed of touched files. Documented limitation.
**Mitigation (v0.4)**: Detect `.git/HEAD` changes, prompt user via `[longctx] branch changed: re-indexing` log line; consider full reindex.

### R5: Re-embed during query causes latency spike
**Risk**: RW-lock blocks reads during writes; user sees a stutter.
**Mitigation (v0.3.0)**: Document. Watcher debounces (default 1s) so changes batch.
**Mitigation (v0.4)**: Copy-on-write index swap.

### R6: Path normalization edge cases
**Risk**: Symlinks, Docker mounts, case-insensitive vs case-sensitive volumes produce different cache keys for the same content.
**Mitigation**: `realpath()` + case normalization on macOS-default volumes + strip trailing slashes. Documented behavior.

---

## 9. Open questions

- **Q1**: Should longctx-svc auto-launch as a subprocess of vllm-swift, or expect the user to start it separately?
  *Lean: separate process for clean lifecycle. Document `brew services start longctx-svc` or similar.*

- **Q2**: Default `top_k`?
  *Lean: 8, matching the v0.2 selector benchmark. User-overridable via header `x-longctx-top-k`.*

- **Q3**: Tree-sitter for code chunking — v0.3.0 or v0.3.1?
  *Lean: v0.3.1. v0.3.0 ships line-window fallback. Tree-sitter adds dependency weight.*

- **Q4**: Cache portability across machines (e.g., shared NFS dev volume)?
  *Out of scope for v0.3. Cache is per-machine.*

- **Q5**: Telemetry / opt-in metrics?
  *Out of scope for v0.3. Local-only stance demands no telemetry by default.*

---

## 10. Distribution / packaging

- **Package**: `longctx-svc` published to PyPI alongside existing `longctx`
- **Repo**: Lives in same `TheTom/longctx` repo under `services/longctx-svc/`
- **Docker**: Optional `Dockerfile` for users who prefer containerized deployment
- **Docs**:
  - `README.md` quickstart updated
  - `docs/v03-quickstart.md` new
  - `docs/PRD-v0.3.md` (this doc)
  - `docs/architecture.md` for the selector + retrieval companion pattern

---

## 11. Out of scope (deferred or rejected)

- Agentic loops with apply-edit functionality
- Code-aware semantic chunking via LSP (defer to v0.4 if demand exists)
- Authentication / multi-user namespacing (single-user local only)
- Cloud retrieval backends (local-only)
- Fine-tuned rerankers (off-the-shelf bi-encoder + caps)
- IDE plugin (downstream community work)

---

## 12. Success metrics (post-launch)

- v0.3.0 reproduces longctx v0.2 MRCR scores via the new svc + hook plumbing (regression check)
- ≥3 third-party reports of working setups against OpenCode or Hermes within 30 days of release
- Median retrieval-augmented response latency overhead: <150ms (p95: <500ms)
- Issue tracker: <5 open bugs related to scope detection within 30 days
- v0.3.1 cuts manual `/longctx scope` invocations by ≥50% (telemetry-light, opt-in)
