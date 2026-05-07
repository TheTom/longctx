# longctx-svc

Local retrieval companion for inference servers. Scoped, session-aware,
file-watching. Tool is **optional** — if you don't run it, your engine
behaves exactly as before.

WIP. Apache-2.0.

## Engine-agnostic by design

longctx-svc speaks plain HTTP/JSON. It works with any engine that accepts
a prompt — no engine forks required for the basic path.

| Engine | Mode | Wiring |
|--------|------|--------|
| `vllm-swift` | embedded | optional `--retrieval-endpoint URL` flag (engine-side) |
| `TheTom/llama-cpp-turboquant` (`llama-server`) | proxy | point client at longctx-svc; longctx-svc forwards to llama-server |
| `TheTom/vllm` (`feature/turboquant-amd-noautotune`) | proxy or embedded | OpenAI-compat passthrough; or call `LongctxClient` from a custom hook |
| vLLM (CUDA) | proxy | OpenAI-compat passthrough |
| anything OpenAI-compat | proxy | OpenAI-compat passthrough |

### Mode A — proxy (zero engine changes)

```bash
# 1. Run your engine as usual
llama-server -m model.gguf --port 8080 &
# (or vLLM AMD, vLLM CUDA, vllm-swift, ...)

# 2. Run longctx-svc in front of it
longctx-svc serve --upstream http://localhost:8080

# 3. Point your OpenAI client at longctx-svc instead of the engine
export OPENAI_BASE_URL=http://localhost:8765/v1
```

`longctx-svc` detects the project from the messages, retrieves top-K
chunks, splices them into the system message, and forwards the request
to the upstream. Response (including SSE stream) is passed straight
back. If no path is mentioned in the messages, the request is forwarded
unmodified.

### Mode B — embedded (engine calls /retrieve)

For tighter integration (e.g. so the engine can reuse retrieved chunks
across KV cache boundaries), engines import `LongctxClient`:

```python
from longctx_svc.client import LongctxClient

cli = LongctxClient.from_env()        # honors LONGCTX_ENDPOINT
if cli is not None:                   # tool is optional
    res = cli.retrieve(
        prefill_text=full_prompt,
        query=user_message,
        session_id=session_id,
        top_k=8,
    )
    full_prompt = cli.splice(full_prompt, res)
```

Network failure → empty result → engine falls back to the no-retrieval
path. Optional tool stays optional.

## HTTP surface

| Endpoint | Purpose |
|----------|---------|
| `POST /retrieve` | engine-side retrieval (Mode B) |
| `POST /v1/chat/completions` | OpenAI-compat passthrough (Mode A) |
| `POST /v1/completions` | legacy OpenAI-compat passthrough (Mode A) |
| `GET /longctx/status` | JSON status; `Accept: text/plain` for the Sarah-visible block |
| `GET /healthz` | liveness probe |

## Headers

Every retrieve / proxy response sets:

- `x-longctx-session: <session-id|ephemeral>`
- `x-longctx-scope: <project-root|"">`
- `x-longctx-chunks-used: <n>`
- `x-longctx-scope-status: ready|empty|error|no-scope`

Session affinity is sent on the request side via:

1. `x-session-affinity: <id>` (preferred)
2. `x-session-id: <id>`
3. `metadata.session_id` in the JSON body

No header → ephemeral request, no caching.

## Install (alpha)

```bash
pip install -e services/longctx-svc
longctx-svc serve              # http://127.0.0.1:8765
```

## Tests

```bash
cd services/longctx-svc
pytest tests/ --no-cov
```

85 tests cover: scope detection, walk + .gitignore, chunker, indexer,
session manager, the Sarah-journey end-to-end, and the engine-agnostic
client + OpenAI-compat proxy.
