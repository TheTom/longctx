# longctx

> **v0.3.1.** APIs are stable for v0.3.x; numbers and framing may still
> tighten. Issues + PRs welcome. Apache-2.0.

**Open long-context retrieval for evaluations and live coding sessions.** One
repo, three entry points:

- **CLI** — ``longctx ask`` against a directory, no infra required.
- **Daemon + MCP** — long-lived service, exposes ``search_codebase`` to
  Claude Code / OpenCode / Hermes.
- **Inference-side service** (``longctx-svc``) — drops in front of an
  OpenAI-compatible engine and splices retrieved chunks into the prompt
  automatically. **Primary target: [vllm-swift](https://github.com/TheTom/vllm-swift)
  on Apple Silicon.** Upstream vLLM and llama.cpp work via the generic
  proxy path; first-class ``--enable-longctx`` integration for them is
  future work.

It also doubles as the **rescue layer** for TriAttention V3 — KV-cache
eviction without losing the evicted context, because longctx catches the
evicted spans, indexes them, and serves them back on the next turn.

---

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │             your client              │
                          │  CLI  │  MCP agent  │  curl  │  ...  │
                          └────┬───────────┬───────────────┬─────┘
                               │           │               │
                ┌──────────────▼──┐   ┌────▼────┐   ┌──────▼──────────┐
                │   longctx CLI   │   │  MCP    │   │  OpenAI HTTP    │
                │  (`longctx ask`)│   │ stdio   │   │  /v1/chat/...   │
                └──────────┬──────┘   └────┬────┘   └──────┬──────────┘
                           │               │               │
                           │               │               ▼
                           │               │     ┌──────────────────────┐
                           │               │     │  inference engine    │
                           │               │     │  vllm-swift  ◀ main  │
                           │               │     │  vLLM / llama.cpp    │
                           │               │     │   (via proxy mode)   │
                           │               │     └──────┬───────────────┘
                           │               │            │ --enable-longctx
                           │               │            ▼
                           │               │     ┌──────────────────────┐
                           │               │     │     longctx-svc      │
                           │               │     │   (FastAPI sidecar)  │
                           │               │     │ /retrieve            │
                           │               │     │ /evict/{write,retrieve}
                           │               │     └──────┬───────────────┘
                           │               │            │
                           ▼               ▼            ▼
                  ┌─────────────────────────────────────────────┐
                  │              longctx_daemon                 │
                  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                  │   │ Searcher │  │ Indexer  │  │ Watcher  │  │
                  │   └────┬─────┘  └────┬─────┘  └────┬─────┘  │
                  │        │             │             │        │
                  │   ┌────▼─────────────▼─────────────▼─────┐  │
                  │   │ SqliteChunkStore  +  MemmapEmbedStore│  │
                  │   └──────────────────────────────────────┘  │
                  └──────────────────────┬──────────────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │     longctx (library)        │
                          │  rag/coarse_filter           │
                          │  rag/chunker                 │
                          │  rag/symbol_augment          │
                          │  rag/pipeline                │
                          └──────────────────────────────┘
```

Three retrieval shapes share the same library and storage layer:

- ``longctx ask`` and the MCP daemon hit the **daemon's Searcher** directly.
- ``longctx-svc`` is an **HTTP companion** for inference engines — it owns
  its own scope/index/watcher stack and the V3 evict-rehydrate endpoints,
  but pulls retrieval primitives from the same ``longctx.rag`` package.
- The **inference engine** takes one CLI flag and the rest is transparent:
  completions get a ``## Retrieved code context`` block prepended at the
  system level. ``vllm-swift`` has first-class ``--enable-longctx`` wiring.
  vLLM and llama.cpp work via the OpenAI proxy mode; native CLI-flag
  integration for them is on the roadmap.

---

## Install

```bash
pip install longctx                 # eval library + daemon (v0.3.0)
pip install longctx-svc             # local retrieval service (v0.3.0)
```

For local vLLM:
```bash
pip install longctx[serve]
```

---

## Quick start

```bash
# Ask one question, no daemon needed
longctx ask --project ./my-repo \
            --question "Where do we validate the JWT signature?" \
            --model gpt-4o-mini
```

First call indexes the repo (cached at ``~/.longctx/``). Subsequent calls
re-embed only the chunks whose ``content_hash`` changed.

---

## Pick your use case

### 1. Library + CLI

For one-off questions, evals, and scripts. No daemon, no service.

```bash
# Ask a question
longctx ask --project ./my-repo --question "..." --model gpt-4o-mini

# Or import the library directly
python -c "from longctx_daemon.searcher import Searcher; ..."

# Run a coarse-filter sweep over a million-LOC corpus
python -m longctx.eval.bench_coarse_filter_real \
    --corpus-dir ~/dev/your-monorepo \
    --extensions .py,.swift,.md \
    --top-k 1000
```

Cached indices live under ``~/.longctx/<scope-hash>/``. Move it with
``LONGCTX_CACHE_DIR``.

### 2. Daemon + MCP for coding agents

For Claude Code, OpenCode, Hermes, or any MCP-aware client.

```bash
longctx daemon install          # macOS launchd / Linux systemd
longctx daemon status
```

MCP client config (Claude Code, etc.):
```json
{
  "mcpServers": {
    "longctx": { "command": "longctx", "args": ["mcp"] }
  }
}
```

The daemon exposes two MCP tools:

- ``search_codebase(query, top_k=8, ...)`` — BM25 + dense + RRF over your
  indexed projects.
- ``set_active_project(name)`` — sticks subsequent queries to one project
  in a multi-project setup.

It watches indexed projects with ``watchfiles`` and re-embeds only the
changed chunks. Searches always reflect the working-tree state.

### 3. Service behind an inference engine

For local LLMs. ``longctx-svc`` sits next to the engine and splices
retrieved chunks into every chat completion. The model just sees a normal
prompt with a ``## Retrieved code context`` system block at the top — no
agent loop required.

#### vllm-swift — primary target (Apple Silicon)

[``vllm-swift``](https://github.com/TheTom/vllm-swift) has first-class
``--enable-longctx`` wiring. The engine auto-spawns longctx-svc as a
sidecar; the rest is transparent.

```bash
vllm-swift serve ~/models/Qwen3-4B-4bit --enable-longctx
```

510/510 vllm-swift tests still green with the flag wired. Flag absent =
bit-for-bit unchanged engine behavior.

#### vLLM / llama.cpp — proxy mode (any OpenAI-compatible engine)

Native ``--enable-longctx`` integration for upstream vLLM and llama.cpp is
**future work**. Until then, run longctx-svc as a transparent OpenAI proxy
in front of the engine:

```bash
# Engine on :8080 (unchanged)
llama-server -m model.gguf --port 8080 &

# longctx-svc proxy on :8765 — rewrites incoming requests, forwards to engine
longctx-svc serve --upstream http://localhost:8080

# Point your client at the proxy
export OPENAI_BASE_URL=http://localhost:8765/v1
```

This works with any OpenAI-compatible server (upstream vLLM, upstream
llama.cpp, ollama, LM Studio, anything) — the proxy doesn't care what's
upstream. Tradeoff vs the sidecar path: one extra HTTP hop per request and
no engine-side ergonomics (no ``--enable-longctx`` flag).

A proper integration would push the splice into the engine's prompt-build
path so the engine owns scope detection + retrieval lifecycle. Open issue
welcomed — see ``services/longctx-svc/integration/`` for the vllm-swift
reference.

#### Fine-grained: hit ``/retrieve`` directly

```bash
curl -s http://127.0.0.1:8080/retrieve -H 'content-type: application/json' \
  -d '{"prefill_text": "fix the JWT validation in src/auth/jwt.py",
       "query": "JWT signature verification",
       "default_scope": "/path/to/repo",
       "top_k": 8}'
```

### 4. TriAttention V3 rescue mode (advanced)

For unbounded effective context. longctx catches tokens that V3 evicts from
the KV cache, indexes them by salience, and serves them back when the next
user turn needs them.

```bash
VLLM_TRIATT_ENABLED=1 \
LONGCTX_ENDPOINT=http://127.0.0.1:5054 \
vllm-swift serve <model> --enable-longctx
```

**End-to-end receipt: 256K NIAH on Qwen3.5-2B-4bit (M5 Max)**

```
ctx     arm           v3-overhead  recall   total
32K     baseline-tq8v4 0.00%        ✓HIT      5.6s
32K     v3-only        3.72%        ✗miss     6.9s
32K     v3+longctx     3.72%        ✓HIT      8.3s
128K    baseline-tq8v4 0.00%        ✓HIT     76.3s
128K    v3-only        1.42%        ✗miss    67.6s
128K    v3+longctx     1.42%        ✓HIT     70.9s
256K    baseline-tq8v4 0.00%        ✓HIT    186.7s
256K    v3-only        1.32%        ✗miss   221.9s
256K    v3+longctx     1.32%        ✓HIT    229.3s
```

V3+longctx ✓HIT every rung 32K → 256K. V3-only ✗miss every rung. The pair
gets you unbounded effective context with NIAH-passing recall. Design
write-up: [`triattention-v3.md`](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/triattention-v3.md).

**How the wiring works:**

1. Engine boots with ``VLLM_TRIATT_ENABLED=1`` + ``LONGCTX_ENDPOINT=...``.
2. V3 fires per-token eviction during prefill. Each round: decoded token
   IDs → ``POST /evict/write`` on longctx-svc.
3. longctx-svc embeds the chunks (MiniLM by default) and indexes them in a
   per-session faiss store.
4. Next user turn: ``ChatSession``'s auto-Tier-3 hook fires
   ``rescue.rehydratePrompt(query: <user_msg>)`` → ``POST /evict/retrieve``
   → top-K relevant chunks → prepended as a system message.

The rescue path **only auto-fires through ``ChatSession``**. Bare
``container.generate()`` will not rescue.

---

## How longctx helps tool-use

longctx helped tool-use in two distinct ways during the overnight
2026-05-11 → 2026-05-12 Hermes-on-vllm-swift bench, one expected and
one surprising.

### 1. Expected — relevant code is injected without burning context

The standard story: retriever finds top-K chunks for the current turn's
query, splices them into the system message. The model then has the
right code in front of it when deciding which tool to call. Saves the
agent from having to ``read_file`` every file just to see what's there.

### 2. Surprising — longctx is load-bearing for Hermes' agentic flow itself

This came out of overnight Test 4 (``--max-model-len 65536`` but
``--enable-longctx`` **off**):

- Hermes started, sent ONE prompt to vllm-swift.
- Model emitted a friendly text response: *"Let me read the prompt file
  first."*
- Zero tool calls in that response.
- Hermes' agent loop saw 0 tool calls → terminated cleanly after 24
  seconds.
- Test directory stayed empty.

With longctx ON (Tests 1, 2, 3 — same prompt, same model, same
everything else), the model immediately emitted tool calls and started
building files.

The mechanism appears to be: longctx's chunk splice changes the shape
of the system context the model sees. Without that splice, the system
prompt is shorter and the model defaults to conversational mode —
*"I'll read it then tell you what I find."* With splice, the system
message contains code-block fenced chunks, which primes the model to
operate in *"I'm in a codebase, I should use tools"* mode and emit a
``read_file`` or ``write_file`` tool call instead.

So longctx isn't just retrieval — for at least Hermes on
Qwen3.6-27B-ConfigI, it's a **behavioral primer** that flips the model
from chat-mode into agent-mode. Without it, the tool-call rate drops
dramatically and the agentic loop terminates.

This was the night's biggest unexpected finding. Practical implication:
for the stack we ship, benchmarking the model + harness without
longctx measures a different system than what end users experience.

---

## Tuning knobs

All knobs are env vars (so the engine sidecar can inherit them without code
changes). Per-call overrides exist on ``Searcher.search`` for the daemon
path.

| Env var | Default | What it does |
|---|---|---|
| ``LONGCTX_SYMBOL_AUGMENT`` | ``1`` | Symbol-aware augment — grep ``class X`` / ``def X`` for identifiers in the query, boost ``.py`` over docs when the query has a code signal. Set ``0`` to disable. |
| ``LONGCTX_COARSE_FILTER`` | ``0`` | BM25 + dense RRF fusion. Engages at corpora ≥ ``coarse_filter_min_chunks``. |
| ``LONGCTX_COARSE_FILTER_MIN_CHUNKS`` | ``5000`` | Threshold for the coarse-filter lane. |
| ``LONGCTX_MULTIQUERY`` | ``1`` | Paraphrase-fusion retrieval. |
| ``LONGCTX_EMBEDDER`` | ``MiniLM-L6-v2`` | Embedding model. ``BAAI/bge-m3`` recommended at ≥32K context. |
| ``LONGCTX_RERANKER`` | ``bge-reranker-v2-m3`` | Cross-encoder rerank. Set empty to disable. |
| ``LONGCTX_TS`` | ``0`` | Tree-sitter chunker (Python / TS / JS / Go / Rust). Off by default — line-window chunking is the production path. |
| ``LONGCTX_CACHE_DIR`` | ``~/.longctx`` | Where indices live. |
| ``LONGCTX_ENDPOINT`` | unset | V3 rescue mode — point engines at a running longctx-svc. |
| ``LONGCTX_AUTO_LEARN`` | ``1`` | Ambient learning — daemon auto-registers repos when the agent's ``cwd`` arg points to one. Set ``0`` to disable. See [Ambient Learning](#ambient-learning) below. |

---

## Ambient Learning

Starting with v0.3.4 (2026-05-21), the daemon **learns** which repos to index by
watching the ``cwd`` arg on every ``search_codebase`` MCP call. No upfront
``--corpus-dir`` setup, no filesystem walk on startup, no per-user config of
"where my code lives." The daemon converges to the user's real working set as
the agent uses it.

**How it works:**

1. Agent calls ``search_codebase(query="...", cwd="/Users/tom/dev/myrepo")``.
2. Daemon resolves ``cwd`` to a repo root (walks up looking for ``.git``; uses
   ``cwd`` itself if no git ancestor).
3. If that root isn't already a registered project AND isn't under a forbidden
   parent (see below), the daemon registers it as a session-bound project and
   kicks off a background ``full_scan``.
4. The response carries a ``learning_signal`` field telling the caller the new
   project is being indexed.
5. The next search call against the same repo returns real chunks.

**Default-on.** Disable with ``LONGCTX_AUTO_LEARN=0``.

**Forbidden parents** — the daemon will NEVER auto-index paths under any of
these, even if the agent's ``cwd`` points there. Explicit ``--corpus-dir`` and
the ``add_project`` MCP tool keep working for any directory the user chooses.

```
~/.ssh
~/.aws
~/.gnupg
~/Library
~/Downloads
~/.Trash
~/Desktop
/private/var
/tmp
/private/tmp
```

Plus bare ``/`` and ``~`` if the agent literally passes one of those.

**Coexists with explicit ``--corpus-dir``.** Both signal channels add to the
same registry. A repo registered by ambient learning lives alongside any
explicitly-added corpora.

**Session scope.** Ambient-registered projects default to ``persist=False``
(session-bound) in v0.3.4. Phase 2 of the [Ambient Indexing PRD] adds the
cross-session tier model where actively-touched repos persist across daemon
restarts.

[Ambient Indexing PRD]: https://github.com/TheTom/longctx/blob/main/docs/PRD-ambient-indexing.md

---

## Recommended models

Plumbing is identical across all models; answer quality is the model's
job. Cross-model bake-off:

**Apple Silicon (vllm-swift / llama.cpp):**
- First try: **Qwen3-4B-4bit** via vllm-swift — small, fast, good code recall.
- Best small coder: **Qwen3-Coder-30B-A3B-MLX-6bit** (Mac mini sized).
- Long context: any Qwen3-1M / Llama-4-1M / Gemma-4-128k variant.

**CUDA / AMD:**
- **Qwen2.5-32B-Instruct** (verified on MI300X) — solid baseline.
- DeepSeek-Coder-V2 / Codestral 22B / Qwen2.5-Coder 32B for code-heavy work.

Full bake-off harness: [`integration/cross_model_bakeoff.py`](services/longctx-svc/integration/cross_model_bakeoff.py).

---

## Numbers

**MRCR v2 8-needle, MI300X, Qwen2.5-32B-Instruct (2026-05-06/07)**

| bin | recipe | n | longctx | SubQ |
|---|---|---:|---:|---:|
| 8K | plain RAG | 30 | **0.822** | — |
| 32K | plain RAG | 30 | 0.697 | — |
| 64K | chunked (cs=2000) | 30 | 0.670 | — |
| 1M | Selector + bge-rerank + det copy (single-query) | 60 | 0.601 *(mass-val)* | 0.659 |
| 1M | **MultiQ** Selector + bge-rerank + det copy | 30 | **0.688** *(directional)* | 0.659 |

**MRCR v2 8-needle, M5 Max, Qwen3-32B + bge-m3 (2026-05-08)**

| bin | longctx |
|---|---:|
| 32K | **0.784** |
| 64K | **0.748** |
| 1M (hierarchical) | 0.553 |

**13.4M-token real-corpus NIAH** (4 of my own repos: mlx-swift-lm,
llama.cpp, vllm-swift, the obsidian vault, plus longctx itself —
3,396 files / 53.6M chars / 7,423 chunks):

| | min | median | p90 | p95 | max | misses |
|---|---:|---:|---:|---:|---:|---:|
| single-query | 1 | 9.5 | 25 | 47 | 177 | 0/20 |
| multi-query | 1 | **4** | 17 | **41** | 108 | 0/20 |

**longctx-svc latency** (target: <100 ms warm):
- Cold build (20-file project): 12.7 s
- Warm ``/retrieve`` mean: **63.8 ms** ✅
- Warm p95: 63.2 ms
- Cache reload from disk: 8.9 s

**Test coverage:**
- ``longctx-svc``: **221 tests, all green** — scope detection, walk +
  .gitignore, chunker (line + tree-sitter), indexer, session manager,
  async kickoff, idle eviction, disk cache, file watcher, OpenAI-compat
  proxy, sidecar spawn + port-collision, V3 evict/rehydrate roundtrip.
- ``longctx`` library + daemon: see ``tests/`` and ``tests/daemon/``.
- ``vllm-swift``: 510 tests, full suite green.

Full curves + receipts in [`docs/results.md`](docs/results.md),
[`benchmark/mrcr_e2e/RESULTS.md`](benchmark/mrcr_e2e/RESULTS.md),
[`benchmark/coarse_filter/RESULTS.md`](benchmark/coarse_filter/RESULTS.md).

---

## Features (v0.3.0–v0.4.0, all in)

| Feature | Status |
|---|:-:|
| **Ambient learning** — daemon auto-registers repos from agent's `cwd` (v0.4.0) | ✅ |
| **Iterative retrieval API** — `suppress_ids` + `prior_context` on `search_codebase` (v0.4.0) | ✅ |
| **`suggested_followup`** response field — server-side kwarg hints (v0.4.0) | ✅ |
| **Live FS watcher** wired into daemon (v0.4.0) | ✅ |
| **Multi-corpus daemon** — repeatable `--corpus-dir`, opt-in `--include-worktrees` (v0.4.0) | ✅ |
| **MCP tool descriptions** — Anthropic-aligned USE WHEN / DO NOT / per-arg / response-action sections (v0.4.0) | ✅ |
| Scope detection from prefill paths (absolute + relative) | ✅ |
| Hot scope (1K files) → Package scope (50K) | ✅ |
| Caps + .gitignore + always-skip dirs | ✅ |
| Line-window chunker | ✅ |
| Tree-sitter chunker (Python/TS/JS/Go/Rust, opt-in `LONGCTX_TS=1`) | ✅ |
| Header-based session isolation (`x-session-affinity` / etc) | ✅ |
| RW-lock per scope, file watcher (1s debounce, incremental re-embed) | ✅ |
| LRU + idle eviction (sessions 2h, indexes 30m) | ✅ |
| Manual scope override (`explicit_scope` body field) | ✅ |
| Debug headers + `/longctx/status` | ✅ |
| Local-only privacy stance | ✅ |
| OpenAI-compat passthrough proxy + sidecar spawn | ✅ |
| Disk cache `~/.longctx/<scope-hash>/` | ✅ |
| Auto Hot→Package promotion when out-of-Hot path mentioned | ✅ |
| Confidence-driven promotion (top-K cosine across N turns) | ✅ |
| Workspace `ws:` mode (multi-scope query merge) | ✅ |
| First-class `--enable-longctx` wiring (vllm-swift) | ✅ |
| Generic OpenAI proxy mode (vLLM / llama.cpp / any compat) | ✅ |
| Symbol-aware retrieval (sym-grep + file-type prior) | ✅ |
| Auto-policy router (context-size + query-shape adaptive) | ✅ |
| Per-corpus relevance floor + ``longctx calibrate`` | ✅ |
| Native CLI-flag integration for vLLM / llama.cpp | 🛣️ future |

---

## Repo layout

```
longctx/
├── longctx/                   # eval library (RAG primitives, MRCR scoring,
│   │                          # coarse filter, symbol-aware augment)
│   └── rag/
│       ├── coarse_filter.py   # BM25 + dense RRF fusion
│       ├── chunker.py         # token-aware chunking
│       ├── pipeline.py        # retrieve_chunked
│       └── symbol_augment.py  # symbol-grep + file-type prior
├── longctx_daemon/            # long-lived daemon (MCP, CLI, watcher)
│   ├── searcher.py            # BM25 + dense + RRF over persistent storage
│   ├── storage/               # SqliteChunkStore + MemmapEmbedStore
│   ├── mcp_server.py          # MCP transport
│   ├── policy.py              # auto-policy router
│   └── eval/                  # MRCR e2e + Recall@K + NIAH rigs
├── docs/
│   ├── v03-quickstart.md
│   └── results.md
├── benchmark/                 # bench outputs (mrcr_e2e, coarse_filter, ...)
└── services/
    └── longctx-svc/           # local retrieval companion (v0.3)
        ├── longctx_svc/       # FastAPI app + scope/indexer/retrieve/cache/watcher/proxy
        ├── tests/
        ├── integration/       # cross-fork harness + bake-off
        ├── benchmarks/        # latency.py
        └── scripts/           # llama-server-longctx wrapper
```

---

## What's next

Out-of-scope for v0.3, tracked separately:
- **First-class ``--enable-longctx`` integration for upstream vLLM and
  llama.cpp** — pushes scope detection + retrieval into the engine's
  prompt-build path so users get the same one-flag UX as vllm-swift.
  Until then, proxy mode covers the gap.
- Agentic loops with apply-edit
- Tree-sitter for more languages (currently 5)
- Multi-user / LAN deployments
- Cloud retrieval backends
- Fine-tuned rerankers (off-the-shelf bi-encoder + cross-encoder still wins by margin)

Alpha-tester gate: drop me an issue, post in the OpenCode / Hermes Discords,
or hit me up on X with results.
