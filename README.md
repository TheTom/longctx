# longctx

> **🚧 Pre-alpha.** APIs, numbers, and framing will change. Pin versions if you build on this. Issues + PRs welcome. Apache-2.0.

Open long-context retrieval — for evaluations *and* live coding sessions. One repo, two shipping pieces:

- **`longctx`** — the library/eval framework. Per-bin MRCR v2 8-needle on a single MI300X with Qwen2.5-32B-Instruct: 8K **0.822** (n=30); 1M mass-val (single-query Selector + det copy) **0.601** (n=60, below SubQ's 0.659); 1M directional best (MultiQ Selector + bge-rerank + det copy) **0.688** (n=30, above SubQ; mass-val pending). Full per-bin curve in [`docs/results.md`](docs/results.md).
- **`longctx-svc`** — the local retrieval companion service for inference engines (vllm-swift, llama.cpp, vLLM). Adds RAG-on-codebase to any OpenAI-compatible engine via a single CLI flag. **173 tests, all green.**

This README covers both. v0.2 (library) is for evaluators. v0.3 (service) is for developers running local LLMs.

---

## How it works

```
┌────────────┐  OpenAI HTTP   ┌──────────────┐
│ OpenCode / │ ─────────────▶ │  inference   │
│ Hermes /   │                │  engine      │
│ curl / ... │ ◀───────────── │              │
└────────────┘                └──────┬───────┘
                                     │  --enable-longctx
                                     ▼
                              ┌──────────────┐
                              │  longctx-svc │
                              │  (FastAPI)   │
                              │              │
                              │ • scope      │
                              │ • index      │
                              │ • retrieve   │
                              │ • watch      │
                              └──────────────┘
```

Every chat completion's prompt is parsed for absolute file paths. The first path's project root is detected via sentinel (`package.json`, `.git`, `pyproject.toml`, `Cargo.toml`, …). longctx-svc indexes that scope (Hot first → Package on demand), retrieves top-K chunks for the user's query, splices them into a system message, and forwards the request. The model just sees a normal chat completion with a `## Retrieved code context` block at the top.

---

## Works with your forks (and any OpenAI-compatible engine)

Tested end-to-end this session:

| Engine | Mode A (proxy) | Mode B (`--enable-longctx`) | Smoke status |
|---|:-:|:-:|---|
| **TheTom/vllm-swift** | ✅ | ✅ | local Mac + Mac mini |
| **TheTom/llama-cpp-turboquant** | ✅ | ✅ via `llama-server-longctx` wrapper | local Mac |
| **TheTom/vllm-turboquant** (AMD MI300X) | ✅ | ✅ wired (CLI flag in serve.py) | droplet via SSH tunnel |
| upstream vLLM (CUDA), llama.cpp, anything OpenAI-compat | ✅ | n/a | – |

**Mode A — proxy.** Engine unchanged. longctx-svc sits in front, rewrites the request, forwards to upstream:

```bash
llama-server -m model.gguf --port 8080 &
longctx-svc serve --upstream http://localhost:8080
export OPENAI_BASE_URL=http://localhost:8765/v1
```

**Mode B — single flag.** Engine auto-spawns longctx-svc as a sidecar:

```bash
vllm-swift serve ~/models/Qwen3-4B-4bit --enable-longctx
# or
vllm serve <model> --enable-longctx                          # TheTom/vllm-turboquant
# or
llama-server-longctx -m model.gguf --port 8000 --enable-longctx
```

Tool is **optional everywhere**. Flag absent + env unset = bit-for-bit unchanged engine behavior. 487/487 vllm-swift tests still green after wiring.

---

## Models — what to recommend testers run

Plumbing (chunk retrieval) is identical across all models. Answer quality depends on the model. Cross-model bake-off run this session ([`integration/bakeoff_results.json`](services/longctx-svc/integration/bakeoff_results.json)):

| Model | Family / Size | Engine | Recall (cited the spliced secret) |
|---|---|---|:-:|
| Mistral-Small-24B-Q4 | Mistral 24B | llama.cpp | ✅ cleanest natural answer |
| Qwen2.5-1.5B-Q4 | Qwen 1.5B | llama.cpp | ✅ |
| Qwen3-4B-4bit | Qwen 4B | vllm-swift | ✅ |
| Llama-3.2-1B | Llama 1B | vllm-swift | ✅ |
| Gemma-4-E2B-Q4 | Gemma 2B | llama.cpp | ✗ retrieved chunks but emitted 0 chars |
| Gemma-3-4B-4bit | Gemma 4B | vllm-swift | – boot timeout |

**Tester recommendations (Apple Silicon):**
- **First try**: `Qwen3-4B-4bit` via `vllm-swift` — small, fast, good code recall
- **Best small coder**: `Qwen3-Coder-30B-A3B-MLX-6bit` if you have it (Mac mini does)
- **Long context**: any Qwen3-1M / Llama-4-1M / Gemma-4-128k variant
- **Avoid for now**: Gemma-4-E2B-Q4 (silent on retrieved code in our test)

**Tester recommendations (CUDA/AMD):**
- Qwen2.5-32B-Instruct (verified on MI300X droplet) — solid baseline
- DeepSeek-Coder-V2 / Codestral 22B / Qwen2.5-Coder 32B for code-heavy work

---

## Metrics so far

**MRCR v2 8-needle, single MI300X, Qwen2.5-32B-Instruct via vLLM (2026-05-06/07)**

| bin | recipe | n | longctx | SubQ |
|---|---|---:|---:|---:|
| 8K | plain RAG | 30 | **0.822** | — |
| 32K | plain RAG | 30 | 0.697 | — |
| 64K | plain RAG | 30 | 0.641 | — |
| 64K | chunked (cs=2000) | 30 | 0.670 | — |
| 1M | plain RAG (baseline) | 30 | 0.440 | — |
| 1M | Selector + bge-rerank + det copy (single-query) | **60** | **0.601** *(mass-val)* | 0.659 |
| 1M | **MultiQ** Selector + bge-rerank + det copy | 30 | **0.688** *(directional)* | 0.659 |

- The **n=60 selector at 0.601** is the mass-validated AMD result; below SubQ's 0.659
- The **n=30 MultiQ selector at 0.688** is the directional best, above SubQ — n=80 priority run OOM'd before completing, mass-val rerun pending
- Full curve, raw logs, and recipe details in [`docs/results.md`](docs/results.md) and on the droplet at `/root/results/longctx_1m_*.log`

**longctx-svc latency (PRD §6 acceptance: <100ms warm)**
- Cold build (20-file project): 12.7 s
- Warm `/retrieve` mean: **63.8 ms** ✅
- Warm p95: 63.2 ms
- Cache reload from disk: 8.9 s (mostly embedder cold-load; the disk read itself is <500 ms)
- See [`benchmarks/latency.py`](services/longctx-svc/benchmarks/latency.py) and `latency_results.json`.

**Test coverage**
- `longctx-svc`: **173 tests, all green** — covers scope detection, walk + .gitignore, chunker (line-window + tree-sitter), indexer, session manager, async kickoff, idle eviction, disk cache, file watcher, OpenAI-compat proxy, sidecar spawn + port-collision, cross-fork integration, Sarah's full PRD §7 journey, auto-promotion (path-based + confidence-driven), workspace `ws:` mode.
- `vllm-swift`: 487/487 (after `--retrieval-endpoint` and `--enable-longctx` landed)

---

## Features (v0.3.0–v0.3.3, all in)

| PRD ref | Feature | Status |
|---|---|:-:|
| §5.1 | Scope detection from prefill paths | ✅ |
| §5.2 | Hot scope (1K files) → Package scope (50K) | ✅ |
| §5.3 | Caps + .gitignore + always-skip dirs | ✅ |
| §5.4 | Line-window chunker | ✅ |
| §5.4 | Tree-sitter chunker (Python/TS/JS/Go/Rust, opt-in `LONGCTX_TS=1`) | ✅ v0.3.1 |
| §5.5 | Header-based session isolation (`x-session-affinity` / etc) | ✅ |
| §5.6 | RW-lock per scope, file watcher (1s debounce, incremental re-embed) | ✅ |
| §5.7 | LRU + idle eviction (sessions 2h, indexes 30m) | ✅ |
| §5.8 | Manual scope override (`explicit_scope` body field) | ✅ |
| §5.9 | Debug headers + `/longctx/status` (JSON + Sarah-visible text) | ✅ |
| §5.10 | Local-only privacy stance | ✅ |
| §6.0 | `--retrieval-endpoint URL` flag in vllm-swift | ✅ |
| §6.0 | OpenAI-compat passthrough proxy | ✅ |
| §6.0 | Disk cache `~/.longctx/<scope-hash>/` (smoke §7.10 reload) | ✅ |
| §6.1 | Auto Hot→Package promotion when out-of-Hot path mentioned | ✅ v0.3.1 |
| §6.2 | Confidence-driven promotion (top-K cosine across N turns) | ✅ v0.3.2 |
| §6.3 | Workspace `ws:` mode (multi-scope query merge) | ✅ v0.3.3 |
| §6.3 | Multi-scope routing (one turn → multiple bound scopes) | ✅ v0.3.3 |
| §6.3 | Cross-engine parity (`--enable-longctx` on all 3 forks) | ✅ v0.3.3 |
| §7 | All 10 smoke scenarios | ✅ test-covered |

---

## Use cases

**Coding agent (OpenCode, Hermes, custom):**
"Why does `authMiddleware` fail in the Docker build?" → longctx finds the auth file + the Dockerfile, splices both into context, model answers grounded in real code.

**Repo Q&A (any model, any engine):**
Point a curl / OpenAI client at longctx-svc, ask about anything in your codebase, get cited line ranges back.

**Multi-project conversation (`ws:`):**
You start in `~/dev/auth-lib` then ask about something in `~/dev/myapp`. longctx auto-binds both scopes; later workspace queries fan out across them.

**Local-only / air-gapped:**
Everything stays on `localhost`. No telemetry. No cloud calls. Status endpoint announces `[longctx] mode: local-only` on every check.

**Drop-in for testers:** one command, no config:
```bash
vllm-swift serve ~/models/Qwen3-4B-4bit --enable-longctx
```

---

## Install

```bash
pip install longctx                 # the eval library (v0.2.x)
pip install -e services/longctx-svc # the v0.3 service (alpha; not yet on PyPI)
```

For local vLLM:
```bash
pip install longctx[serve]
```

---

## Repo layout

```
longctx/
├── longctx/                   # the eval library (MRCR retrieval, scoring)
├── docs/
│   ├── PRD-v0.3.md           # the v0.3 spec we're shipping against
│   ├── v03-quickstart.md     # 60-second setup
│   └── results.md            # MRCR runs + curves
└── services/
    └── longctx-svc/          # the v0.3 service
        ├── longctx_svc/      # FastAPI app + scope/index/retrieve/cache/watcher/proxy
        ├── tests/            # 173 tests
        ├── integration/      # cross-fork harness + bake-off
        ├── benchmarks/       # latency.py
        └── scripts/          # llama-server-longctx wrapper
```

---

## What's next

Out-of-scope for v0.3 (per PRD §11), tracked separately:
- Agentic loops with apply-edit
- Tree-sitter for more languages (currently 5)
- Multi-user / LAN deployments
- Cloud retrieval backends
- Fine-tuned rerankers (off-the-shelf bi-encoder + cross-encoder still wins by margin)

Alpha-tester gate: drop me an issue, post in the OpenCode / Hermes Discords, or hit me up on X with results.
