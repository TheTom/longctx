# Changelog

## 0.3.2 — 2026-05-13

Patch release. No API changes; coverage + test hardening only.

* **Coverage gates green.** `longctx` 81% → 97%; `longctx-svc` 62% →
  87%. Adds `[tool.coverage.run]` `omit` for CLI bench entrypoints
  (`bench_coarse_filter*.py`) and the tree-sitter optional path so
  gates measure library code, not script orchestration. New 85%
  fail-under gate on `longctx-svc`'s `pyproject.toml`.

* **+80 new unit tests across both packages:**
  - `longctx`: 5 branch tests on `symbol_augment` (dotted-symbol
    decomposition, ripgrep error-swallow, neutral-weight fallback).
  - `longctx-svc/client.py` (77% → 100%): env construction,
    `healthz`, sync + async `retrieve` error degradation, splice
    helper, full payload parsing.
  - `longctx-svc/proxy.py` (73% → 92%): message helpers, dump,
    503-disabled guards, `/v1/models` passthrough.
  - `longctx-svc/eviction_store.py` (77% → 90%): BM25 tokenize edge
    cases, env_float parsing, `rank_bm25` ImportError fallback,
    empty corpus guard, reranker lazy-load cache.
  - `longctx-svc/state.py` (81% → 87%): pipeline lazy init,
    `evict_idle` skip branches, watcher attach/detach edge cases,
    `maybe_promote` early returns, `reset_state` cleanup.

* Total: longctx 1172 → 1177 passing tests; longctx-svc 216 → 291.

## 0.3.1 — 2026-05-11

Patch release: CI gate + audit-driven cleanups on top of v0.3.0.

* **CI coverage gate** — added `[tool.coverage.run]` ``omit`` for CLI
  bench drivers (``bench_coarse_filter_real.py``, ``eval/cli.py``) so
  the 90% library-coverage gate doesn't trip on script orchestration
  code. Total covered: 93% (up from 81%).
* **Version strings synced.** ``longctx``, ``longctx_daemon``, and
  ``longctx_svc`` ``__version__`` strings all bumped to 0.3.1 in lock-
  step with their pyproject.toml.
* **README cleanup.** ``longctx-svc`` test count corrected (210 →
  221). Dangling ref to ``integration/bakeoff_results.json`` (file
  doesn't exist) replaced with link to the runnable
  ``cross_model_bakeoff.py`` harness.
* **CHANGELOG.** Promoted the "unreleased — 12M coarse filter"
  section to v0.3.0 and added this v0.3.1 entry.

## 0.3.0 — 2026-05-11

First real release. 58 commits collapsed into 13 clean ones off main.

Headline features (see commit log + README for full surface):

* **longctx_daemon** — long-lived service with persistent
  `SqliteChunkStore` + `MemmapEmbedStore`, MCP transport
  (`search_codebase`, `set_active_project`), file watcher with
  incremental re-embed, macOS launchd + Linux systemd installers.
* **longctx-svc v0.3.0** — local retrieval companion for inference
  engines. First-class ``--enable-longctx`` on vllm-swift, generic
  OpenAI proxy mode for vLLM / llama.cpp / any-compat.
  Drops the alpha suffix; classifier moves to Beta.
* **Symbol-aware retrieval augment** — grep ``class X`` / ``def X``
  for identifiers in the query, boost ``.py`` over docs when the
  query has a code signal. Recovered 5/10 SWE-bench retrieval_miss
  cases.
* **Auto-policy router** — context-size + query-shape adaptive
  retrieval (BM25/dense weights + embedder hint).
* **Per-corpus relevance floor** + ``longctx calibrate`` CLI.
* **TriAttention V3 rescue mode** — V3 evicts KV cells, longctx
  catches the spans + serves them back on the next turn. End-to-end
  256K NIAH receipt: V3+longctx ✓HIT every rung 32K → 256K.

## 0.2.x — 12M coarse filter (hierarchical-selector pipeline)

Originally landed on `feature/coarse-filter-12m` pre-v0.3.0.

### Why

`RetrievalPipeline.retrieve_chunked` works well up to ~100K tokens.
Above that, the dense embed pass and the optional cross-encoder
rerank stage are both unbounded — at 12M tokens the rerank stage
sees thousands of chunks. The fix is a cheap **coarse filter** in
front of the existing rerank+pick pipeline so downstream stages
always see a bounded top-K regardless of input size.

### Added

- **`longctx.rag.coarse_filter.CoarseFilter`** — BM25 + dense embedding
  hybrid with Reciprocal Rank Fusion (k=60). Default embedder
  `BAAI/bge-small-en-v1.5`. `bm25_weight=0` or `dense_weight=0`
  selects pure-dense or pure-BM25 mode.
- **`CoarseFilter.filter_multi_query(chunks, queries)`** — RRF-fuses
  rankings across N paraphrase queries. One BM25 build + one dense
  embed pass; per-query cost is just an argsort. ~150× rank
  improvement on the 12M hard mode worst case (rank 451 → 1) at the
  same wall time.
- **`longctx.rag.chunker.Chunker`** — token-aware chunker with stable
  Chunk ids (position-prefixed, content-hashed). Char-proxy mode
  (default) or HF tokenizer mode via `offset_mapping`.
- **`RetrievalPipeline.retrieve_chunked` integration** — new args
  `coarse_filter_threshold_chars` (default ~400K chars / 100K tokens)
  and `coarse_filter_top_n` (default 1000). Below threshold the path
  is byte-identical to today's behavior so MRCR v2 numbers stay
  pinned. Above threshold the prefilter trims chunks before the
  dense rerank stage.
- **`longctx-svc.RetrievePipeline` fusion lane** — optional BM25 +
  dense RRF fusion for huge scopes. Enable with
  `LONGCTX_COARSE_FILTER=1` or `use_coarse_filter=True`. Fires above
  `coarse_filter_min_chunks` (default 5000). Per-index BM25 cache,
  invalidates on chunk-count change. Surfaces `used_coarse_filter`
  in `RetrieveResult`.
- **`longctx.eval.bench_coarse_filter`** — synthetic NIAH bench
  (`longctx-coarse-bench` CLI). `--hard` mode swaps in topically-
  overlapping filler. `--query-index` selects from four paraphrase
  queries. `--multi-query` exercises filter_multi_query.
- **`longctx.eval.bench_coarse_filter_real`** — real-corpus NIAH
  bench. Walks a directory, plants a needle, runs the pipeline.
  Privacy-preserving: only summary metrics saved.
- **`benchmark/coarse_filter/RESULTS.md`** — sweep tables + caveats.

### Numbers (M5 Max / MPS / bge-small / synthetic NIAH)

| target | top-K | n_chunks → kept | recall@top-K | total |
|---:|---:|---:|---:|---:|
| 100K  |  100 |   55 →  55 | 100% | <1s |
| 1M    | 1000 |  545 → 545 | 100% | ~3s |
| 4M    | 1000 | 2177 →1000 | 100% | ~9s |
| 12M   | 1000 | 6531 →1000 | 100% | ~30s |

Hard-mode (topically-overlapping filler) recall@top-1000 = 24/24
across {100K, 1M, 4M, 12M} × 4 paraphrase queries. Borderline case:
12M + literal query + hard filler hits rank 451 inside top-1000;
multi-query fusion drops that to rank 1.

### Real-corpus data point (obsidian vault, ~1.5M tokens)

| top-K | mode | needle rank |
|---:|---|---:|
| 1000 | single literal     | 32 |
| 1000 | multi-query (4)    | 3 |
|   10 | multi-query (4)    | 3 |

### Embedder ablation (multi-query, top-1000)

| corpus | MiniLM-L6 | bge-small | bge-m3 |
|---|---:|---:|---:|
| synth 1M hard | 26 | **8** | 11 |
| obsidian vault | **1** | 3 | OOM |

bge-small remains the right default. bge-m3 OOMs on real vault
(146 GiB request). MiniLM-L6 is a viable fallback.

### Test coverage

- 16 unit tests on `CoarseFilter` (includes multi-query)
- 15 unit tests on `Chunker`
- 3 unit tests on `retrieve_chunked` coarse-filter integration
- 6 unit tests on the bench harness
- 6 unit tests on the longctx-svc fusion lane
- Full suite: **310/310** passing (94 longctx + 216 longctx-svc)

### Compat

Existing `retrieve_chunked` callers see no behavior change unless
total candidate text is ≥ ~400K chars AND chunk count exceeds
`coarse_filter_top_n` (default 1000). Pass
`coarse_filter_threshold_chars=None` to disable the prefilter
unconditionally. The longctx-svc fusion lane is **off by default** —
enable explicitly via `LONGCTX_COARSE_FILTER=1`.

---

## longctx-svc v0.3.0a3 (2026-05-07) — splice budget cap + sidecar leak guards

### Why

Tom's Hermes alpha session at the Qwen3.6-35B-A3B-4bit + vllm-swift +
turbo4v2 stack hit two real problems in 0.3.0a2:

1. The default 50-line line-window chunker at `top_k=8` produced a
   ~12K-token splice block per turn. On a 32K-context Hermes session
   this filled the prompt budget after a few turns and triggered
   Hermes' compress / retry cascade.
2. Hermes' aux-provider fan-out (summarizer, vision, tools, mcp) plus
   client-side timeout retries left the engine queue full of zombie
   gens that never received cancellation. Memory pressure built up
   across long sessions.

### Added

- **Splice budget cap.** `Limits.splice_max_chars` (default 16384 chars
  ≈ 4K tokens). `_format_chunks_block` now truncates the `## Retrieved
  code context` system block to fit the budget — highest-rank chunks
  fully kept, later ones truncated or dropped. Same cap logic ships in
  vllm-swift's `response_rewriter` so embedded-mode consumers get the
  bound for free.
- **Cancel-on-disconnect propagation.** When the rewriter sees a
  `ClientConnectionResetError` mid-stream, it now calls
  `upstream_resp.close()` so vLLM's `is_disconnected()` check fires
  and aborts the in-flight generation instead of letting it finish
  into a void and pin a decode slot.
- **Shared upstream `aiohttp.ClientSession`.** The rewriter previously
  spawned a fresh ClientSession per request, leaking connection pools
  / FDs under fan-out. Now a single bounded `TCPConnector`
  (`limit=64, enable_cleanup_closed=True`) is reused across requests.
- **`engine_dirty` diagnostic.** Per-request `[longctx]` stderr line
  also reports `vmmap --summary` DIRTY size on the EngineCore pid —
  the macOS private-memory metric. `ps rss` undercounts mmap-shared
  pages on big inference processes, which is how a 4–5 GB/turn leak
  hid for hours during the Hermes session that surfaced this work.

### Compat

Everything ships behind the existing flags; defaults preserve 0.3.0a2
behavior unless tripped. No API changes. 173 longctx-svc tests +
487 vllm-swift tests still green.

### Pairs with vllm-swift 0.5.1

`vllm-swift[longctx]` extra pinned to `longctx-svc>=0.3.0a3`, so
`pip install vllm-swift==0.5.1` always pulls a `longctx-svc` that
includes the splice cap + cancel-on-disconnect.

---

## v0.2.0 (2026-05-06) — embedding cache + GPU autodetect

### **Milestone: 1M-context inference reached on open stack**

On 2026-05-06 longctx ran the MRCR v2 1M bin (135 samples available, 30-sample mass-val) end-to-end on a single AMD MI300X droplet against vLLM-served Qwen2.5-32B-Instruct (vanilla, no fine-tuning). Each sample is a 2-5M character haystack with ~1200 assistant-message candidates. 100% prefix-pass, no crashes, ~5-7s wall-clock per sample.

Score: 0.440 plain / 0.409 chunked at top-K=8. The figure is below SubQ Inc.'s published 0.659, but the rig WORKS — open weights, standard attention, no custom architecture, fully reproducible with `pip install longctx`. Score-narrowing experiments (BM25 prefilter, top-K sweeps, oracle gen-ceiling) are in flight; the milestone of "1M context end-to-end on commodity open stack" stands regardless of where the score lands.



### Added

- **Disk-backed per-chunk embedding cache.** `RetrievalPipeline(cache_dir="default")` (the new default) caches each chunk's embedding at `~/.cache/longctx/<embedder>/<sha256>.npy`. Subsequent calls with identical chunks skip embedding entirely. Override the cache root with `LONGCTX_CACHE_DIR`. Pass `cache_dir=None` to disable.
- **GPU autodetect.** `RetrievalPipeline(device="auto")` (the new default) walks CUDA → MPS (Apple Silicon) → CPU. On Apple M-series this is a 10-20× embedding speedup over the previous CPU default; on NVIDIA it's 30-50×. Pass an explicit `device="cpu"` to force CPU.
- New module `longctx.rag.embed_cache` with the `EmbedCache` class. Cache failures (corrupt files, disk full, read-only home) degrade silently to a fresh embed — never fatal.

### Why

Embed time scales linearly with haystack size (`O(n)`). For short contexts (≤30K chars) the cost is imperceptible. For long contexts (~1M chars on CPU MiniLM-L6 = ~7s) the cost is visible. The cache makes the first-call cost a one-time tax for any pattern that re-queries the same haystack (chat-with-PDF, codebase chat, persistent KBs); GPU autodetect makes the first-call cost ~0.5s on MPS / ~0.2s on CUDA. Together they cover ~95% of real usage patterns.


### Test coverage

55 tests, 98% line coverage. New tests cover device resolution, cache enable/disable, cache hit/miss, partial-hit batches, cache-disabled pass-through, end-to-end pipeline cache integration.

## v0.1.1 (2026-05-06) — mass-validated headline + 99% test coverage

Pre-public-push hardening. No API changes.

### Mass validation (n=80+ per bin, AMD MI300X, Qwen2.5-14B-Instruct-1M)

Single-run numbers from v0.1.0 had ±0.05 swing variance between adjacent runs of the same config. Replaced with mass-validated table:

| MRCR v2 8-needle bin | pipeline | n | avg_score |
| -------------------- | -------- | -- | --------- |
| 8K  | RAG          | 82 | 0.822 |
| 32K | RAG          | 98 | 0.697 |
| 64K | RAG          | 95 | 0.641 |
| 64K | chunked-RAG  | 95 | 0.670 |

Three of three bins clear SubQ Inc.'s published 0.659 headline with the right pipeline.

### Test coverage gate

- 39 tests, 99% line coverage (`pytest --cov=longctx`)
- All tests run in 2.4s with no real model load (sentence-transformers + CrossEncoder mocked)
- `pytest-cov>=5.0` added to dev deps
- `RetrievalPipeline.retrieve_chunked` infinite-loop bug fixed (`chunk_overlap >= chunk_size` no longer hangs)

## v0.1.0 (2026-05-06) — initial private release

The library that ships the open-stack-vs-SubQ comparison from the
2026-05-06 X thread. Pre-alpha. Local-only for now; not yet pushed
to public GitHub.

### Headline result

`LongCtxClient` defaults running Qwen2.5-32B-Instruct (vanilla, no
long-context retrieval training) on AMD MI300X scored **~0.80 ± 0.05
on MRCR v2 8K bin (3-run mean)**. Matches Anthropic's Opus 4.6
(0.783, per SubQ Inc.'s published comparison table) within sample
noise. Clears SubQ Inc.'s claimed 0.659 by 0.14 absolute.

Validated end-to-end via `longctx.eval.MRCRRunner`: library
reproduction matches inline reference within ±0.01 once the system
prompt was tightened.

### Components

- `RetrievalPipeline`: bi-encoder retrieve + optional chunked retrieval
  + optional cross-encoder reranker
- `LongCtxClient`: end-to-end retrieve + generate against any
  OpenAI-compatible chat completions endpoint
- `MRCRRunner`: 8-needle MRCR v2 eval runner with verbatim-prefix
  scoring
- `longctx-eval` CLI for single-bin runs
- `longctx-bench` CLI for the canonical multi-pipeline-multi-bin
  comparison
- Per-family chat templates (Qwen2.5, Qwen3 with `/no_think`,
  Mistral with prefix-first emphasis)

### Commits

- `e5be0dc` longctx v0.1.0 initial structure
- `8a164dc` README with validated MRCR 8K results
- `bed08f0` client prompt fix (drop conditional prefix language)
- `a97f344` `RetrievalPipeline.retrieve_chunked()` hierarchical chunking
- `4069cb8` `longctx-bench` command + chunked retrieval tests

### Validated negative results (documented to save the next user time)

These do **not** improve MRCR-style retrieval at 64K bin in our
2026-05-06 testing on AMD MI300X with Qwen2.5-1M generators:

- ms-marco-MiniLM-L-6-v2 cross-encoder reranker: degrades
- bge-reranker-base cross-encoder reranker: degrades
- bge-large-en-v1.5 embedder (vs MiniLM-L6): degrades
- bge-large-en-v1.5 with bge-recommended instruction prefix: degrades
- Naively reducing top-K to 4 or raising to 16: both degrade

Off-the-shelf upgrades on top of our default pipeline did not help.
A retrieval-component fine-tuned on retrieval-style training data
is the actual improvement vector. On the roadmap.

### Not yet shipped

- Public GitHub push (deliberately deferred)
- Multi-seed averaging for the canonical bench output
- Fine-tuned reranker checkpoint
- RULER / NIAH cross-evals
