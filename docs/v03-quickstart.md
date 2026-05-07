# longctx v0.3 quickstart

Local retrieval for any inference engine in 60 seconds. Tool is optional —
if you don't run it, your engine is unchanged.

## Install

```bash
pip install -e services/longctx-svc
```

## Run alongside your engine

### Mode A — proxy (zero engine changes, recommended)

```bash
# 1. Run your engine on its usual port
llama-server -m model.gguf --port 8080 &
# (works with vLLM CUDA / vLLM AMD / vllm-swift / any OpenAI-compat
# server)

# 2. Run longctx-svc in front of it
longctx-svc serve --upstream http://localhost:8080

# 3. Point your OpenAI-compatible client at longctx-svc
export OPENAI_BASE_URL=http://localhost:8765/v1
```

That's it. longctx-svc detects the project root from the messages,
retrieves top-K chunks, splices them into the system message, and
forwards to the upstream. Responses (including SSE streams) pass
through untouched.

### Mode B — embedded (engine calls /retrieve)

Engine code calls longctx-svc directly:

```python
from longctx_svc.client import LongctxClient

cli = LongctxClient.from_env()      # honors LONGCTX_ENDPOINT
if cli is not None:                 # tool optional
    res = cli.retrieve(
        prefill_text=full_prompt,
        query=user_message,
        session_id=session_id,
        top_k=8,
    )
    full_prompt = cli.splice(full_prompt, res)
```

Network failure → empty result → engine takes the no-retrieval path.

## What you get for free

- **Project sentinel detection** (.git, package.json, pyproject.toml,
  Cargo.toml, go.mod, ...). Walks up from any path mentioned in the
  prompt.
- **Monorepo sub-package preference** — multiple files in
  `apps/billing/` resolve to that package, not the workspace root.
- **Hot scope first**: 1000-file cap of files near the mentioned ones.
  Falls through to a 50K-file Package scope when Hot is too small.
- **Persistent disk cache** at `~/.longctx/<scope-hash>/`. Reloads in
  <500ms on next process start.
- **File watcher** with 1s debounce. Edited files re-embed
  incrementally — no full reindex.
- **Session isolation** via `x-session-affinity` /  `x-session-id` /
  `metadata.session_id`. Concurrent harnesses don't cross-contaminate.
  No header → ephemeral request, no caching.
- **LRU + idle eviction**: 4 indexes max in RAM, 30 min idle drops
  to disk-only.
- **Async index kickoff**: send `x-longctx-async: 1` on the first
  request and the server returns immediately with
  `scope_status=indexing`.
- **Sarah-visible status** at `GET /longctx/status` — `Accept:
  text/plain` for the human-readable block.
- **Debug headers** on every retrieve / proxy response:
  - `x-longctx-session: <id|ephemeral>`
  - `x-longctx-scope: <project-root>`
  - `x-longctx-chunks-used: <n>`
  - `x-longctx-scope-status: ready|empty|indexing|error|no-scope`

## Configuration via env

| Var | Purpose | Default |
|-----|---------|---------|
| `LONGCTX_CACHE_DIR` | disk cache location | `~/.longctx` |
| `LONGCTX_EMBEDDER` | sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `LONGCTX_RERANKER` | optional CrossEncoder | `BAAI/bge-reranker-v2-m3` |
| `LONGCTX_MULTIQUERY` | template paraphrases on/off | `1` |
| `LONGCTX_UPSTREAM` | proxy upstream URL | unset (proxy disabled) |
| `LONGCTX_MAX_FILE_BYTES` | per-file size cap | 5 MB |
| `LONGCTX_MAX_HOT` | Hot scope file cap | 1000 |
| `LONGCTX_MAX_PACKAGE` | Package scope file cap | 50000 |
| `LONGCTX_NO_JANITOR` | disable background eviction sweep (tests) | unset |

## Cleanup

```bash
longctx-svc clean --older-than 30
# clean older-than=30d: removed 4 of 12 scopes (812.4 MB → 234.1 MB)
```

## Privacy

Local-only. No network calls outside `localhost`. No telemetry. Cache
lives under `LONGCTX_CACHE_DIR`. The string `[longctx] mode: local-only`
is rendered on every status response so you can verify.

## Compatibility matrix (v0.3.0)

| Engine | Mode A (proxy) | Mode B (embedded) |
|--------|:-:|:-:|
| `vllm-swift` | ✅ | wired in `feature/longctx-endpoint` |
| `TheTom/llama-cpp-turboquant` (`llama-server`) | ✅ | TBD |
| `TheTom/vllm` (`feature/turboquant-amd-noautotune`) | ✅ | TBD |
| vLLM CUDA upstream | ✅ | n/a |
| anything OpenAI-compat | ✅ | n/a |

Mode A works today on all of the above. Mode B is the tighter path
where engines call `/retrieve` directly — work in progress on the
respective forks.

## Smoke scenarios (PRD §7)

The `services/longctx-svc/tests/` suite covers all 10 v0.3.0 smoke
scenarios (single-project root, monorepo sub-package, ambiguous root,
file-change reflection, concurrent sessions on separate / same projects,
.gitignore respect, large-file skip, no-header ephemeral, cache reload).
109 tests, all green.

```bash
cd services/longctx-svc
pytest tests/ --no-cov
```
