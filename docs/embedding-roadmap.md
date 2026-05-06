# longctx — embedding-time roadmap

Embed time is the single visible cost of retrieval-augmented inference. Below is the full menu of optimizations the codebase will work through, organized by which usage pattern each one unlocks.

## How embed time scales (non-obvious bonus)

`embed_time = num_chunks × per_chunk_cost`. Both factors are bounded:

- `num_chunks` grows **linearly** with haystack size (`haystack_chars / chunk_size`).
- `per_chunk_cost` is roughly constant (one transformer forward pass per chunk).
- Therefore total embed time scales **`O(n)`** in haystack size.

### Concrete numbers (CPU MiniLM-L6)

| haystack | num chunks | embed time |
| -------- | ---------- | ---------- |
| 4K chars | 2 | ~50ms |
| 16K chars (8K bin) | 8 | ~150ms |
| 64K chars (32K bin) | 32 | ~600ms |
| 128K chars (64K bin) | 64 | ~1.2s |
| 1M chars | 500 | ~7s |
| 5M chars (1M bin worst) | 2500 | ~35s |

**Short contexts are basically free.** A 10-page PDF is ~30K chars = 0.3s embed on CPU, imperceptible. A 50-message chat history is ~10K chars = 0.1s. A code file is 0.2s. The embed tax only becomes visible when the haystack is genuinely huge — entire codebases, multi-million-char benchmarks, big knowledge bases. And those scenarios are exactly where caching pays off most because users don't re-upload a codebase per query.

### The non-obvious bonus

Retrieval's advantage *grows* over full-context inference at long context. Compare growth rates:

| | embed time (RAG) | prefill time (full-context attention) |
| --- | --- | --- |
| complexity | `O(n)` | `O(n²)` |
| 100× context | 100× slower | 10,000× slower |

Self-attention prefill is quadratic; embedding is linear. So as haystack size climbs, RAG keeps pulling further ahead of full-context inference even though the absolute embed cost is climbing.

Translated to wall-clock: at 100K context, RAG's edge is ~10× over full-context. At 1M context, ~100× edge. At 10M context, ~1000× edge.

**Practical implication:** the embed cost looks scary as a flat number, but as a percentage of "the alternative (full-context prefill)," it shrinks the longer the haystack gets. RAG is *more* useful at longer context, not less, even with the embed tax included.

This is the framing used throughout this roadmap: every optimization below either reduces the linear constant or removes the linear term entirely. None of them change the fundamental `O(n)` shape because they don't have to — `O(n)` already wins against `O(n²)`.

---

## Usage patterns

| pattern | example | embed cost is felt | priority |
| ------- | ------- | ------------------ | -------- |
| A. fresh haystack every query | one-shot PDF Q&A | every call | high (first-time UX) |
| B. multi-Q on same doc | chat-with-PDF, ask-my-contract | first call only | very high (most apps) |
| C. persistent corpus | codebase chat, KB, support ticket lookup | offline indexing | very high (production) |
| D. growing context | long agent loops, chat histories | per new chunk | high (agents) |
| E. real-time streaming | live transcript Q&A | overlaps with input | medium |

Each tier below targets a class of patterns. Tier 1 + 2 ship in v0.2; the rest are sequenced behind real demand.

---

## Tier 1 — Persistence (caching)

The win: skip embedding work that's already been done. Disk-backed.

### 1.1 Per-chunk content-hash cache *(SHIPPING in v0.2)*

Hash each chunk's text with sha256, store its embedding at `~/.cache/longctx/<embedder>/<hash>.npy`. On the next call, only embed chunks whose hash isn't in cache. Survives process restarts.

- **Unlocks:** patterns B, C, D, and any pattern A where chunks repeat across queries.
- **Effort:** ~80 LOC.
- **Limits:** doesn't help genuine pattern A (truly novel haystack every query).

### 1.2 LRU eviction

Disk cache grows without bound. Evict by least-recently-used past a configurable cap (default 5 GB). One scan-and-delete cron path.

- **Unlocks:** running longctx in long-lived servers / agent processes.
- **Effort:** ~100 LOC.
- **Status:** v0.3.

### 1.3 Distributed cache backend (Redis / S3)

Multi-node deployments share embeddings via a cloud KV store. Configurable `cache_backend="redis://..."`.

- **Unlocks:** multi-replica production servers.
- **Effort:** ~200 LOC + deps.
- **Status:** v0.5+ (gated on enterprise demand).

### 1.4 Persistent vector index format

For pattern C at scale: instead of re-loading per-chunk numpy files, ship a single `.lance`/`.parquet`/`.faiss` artifact. Loaded once at startup, queried in milliseconds.

- **Unlocks:** "ship a longctx index alongside your docs" pattern.
- **Effort:** ~300 LOC.
- **Status:** v1.0 hand-off to lancedb / chromadb-style storage.

---

## Tier 2 — Hardware (faster per chunk)

The win: do the same number of forward passes faster.

### 2.1 GPU autodetect *(SHIPPING in v0.2)*

Today longctx defaults to CPU. Auto-detect MPS (Apple Silicon), CUDA, ROCm; fall back to CPU. One-line change to constructor default.

- **Unlocks:** 10-50× speedup for every pattern.
- **Throughput rough order:** CPU 100K chars/s → MPS 2M → CUDA 5M → ANE 10M.
- **Effort:** ~30 LOC.
- **Limits:** still bound by encoder size on the chosen device.

### 2.2 ANE (Apple Neural Engine) via Core ML conversion

MiniLM converted to `.mlmodel` runs on the iPhone/Mac Neural Engine at ~10× MPS speed for free.

- **Unlocks:** consumer iOS / Mac apps with imperceptible embed latency even on big haystacks.
- **Effort:** ~150 LOC + tooling for the conversion script.
- **Status:** v0.4 (ties into Pal iOS app).

### 2.3 ONNX Runtime / TensorRT

Compile MiniLM to ONNX. CPU inference 2-3× faster; CUDA inference 1.5-2× faster than vanilla PyTorch.

- **Unlocks:** maximum CPU performance for users without a GPU.
- **Effort:** ~100 LOC + a compile-time conversion step.
- **Status:** v0.5.

### 2.4 INT8 / FP8 quantized embedder

A quantized MiniLM-L6 runs at ~2× FP16 throughput on CPU and most GPUs with negligible quality loss.

- **Unlocks:** another 2× on top of GPU autodetect.
- **Effort:** ~50 LOC + a model variant.
- **Status:** v0.5.

---

## Tier 3 — Algorithmic (do less work)

The win: don't embed everything; embed only what's likely to be relevant.

### 3.1 BM25 prefilter + bi-encoder rerank

Stage 1: BM25 (keyword) on all candidates → top-100. Stage 2: bi-encoder embed only that top-100 → top-K.

- **Speedup:** 10-50× on big haystacks, minimal recall loss for typical queries.
- **Effort:** ~150 LOC (rank_bm25 dep).
- **Status:** v0.4.
- **Caveat:** loses on queries where keyword signal is poor (paraphrase-heavy, semantic).

### 3.2 Hierarchical chunking

Embed at coarse granularity first (chapter → section → paragraph). Drill into matching coarse chunks only.

- **Speedup:** 5-10× for structured docs (legal, books, code), ~0 for flat text.
- **Effort:** ~250 LOC.
- **Status:** v0.4.

### 3.3 Adaptive chunk sizing

Don't use a fixed chunk size. Size by content boundaries: paragraphs, function definitions, message turns. Reduces total chunk count for cohesive content.

- **Speedup:** 1.5-3× when content has natural seams.
- **Effort:** ~100 LOC.
- **Status:** v0.3.

### 3.4 MinHash / SimHash prefilter

For huge corpora, use locality-sensitive hashing to drop obviously-irrelevant chunks before any embedding pass.

- **Speedup:** 5-20× on noisy haystacks (logs, web scrapes).
- **Effort:** ~200 LOC + datasketch dep.
- **Status:** v0.6 (niche).

---

## Tier 4 — Skip embedding entirely (extreme cases)

The win: query-time embed cost is exactly zero.

### 4.1 Static index artifact

Ship `<doc>.longctx-index` alongside the document. Query loads the artifact, runs faiss, returns top-K. No embed work at runtime ever.

- **Unlocks:** distributing pre-indexed datasets, RAG-as-an-asset.
- **Effort:** ~150 LOC + format spec.
- **Status:** v1.0.

### 4.2 ColBERT-style late interaction

Embed each token (not each chunk), store on disk. Retrieval is a matrix multiplication, not a forward pass. ~10× memory cost for instant query.

- **Unlocks:** production search engines competing with Vespa / Elasticsearch.
- **Effort:** ~800 LOC + storage refactor.
- **Status:** v1.5+.

### 4.3 Pure BM25 retrieval

For corpora dominated by keyword signal (code, technical docs, transcripts), skip embeddings entirely. Pure BM25 + the LLM does the heavy lifting.

- **Unlocks:** zero-dependency long-context Q&A.
- **Effort:** ~50 LOC.
- **Status:** v0.4 (offered as alternate `retrieve_bm25()` method).

---

## Tier 5 — Async / pipelining (UX-only)

The win: the user doesn't wait for embedding because it happens during their typing.

### 5.1 Async embedding API

`async def aretrieve(...)` for non-blocking calls in async agent frameworks (LangChain, LlamaIndex, OpenAI Agents).

- **Unlocks:** no UX-perceived latency in modern async stacks.
- **Effort:** ~60 LOC.
- **Status:** v0.3.

### 5.2 Background pre-indexing

A `LongCtxClient.warm_index(candidates)` call returns immediately and embeds in a background thread. Subsequent retrieve calls block only if indexing is still running.

- **Unlocks:** chat UIs that index while the user reads the first sentence.
- **Effort:** ~80 LOC.
- **Status:** v0.3.

### 5.3 Streaming chunk-by-chunk embed

For corpora that arrive incrementally (live transcripts, chat history), embed each new chunk as it arrives. Constant per-turn cost.

- **Unlocks:** real-time agent loops, live transcription Q&A.
- **Effort:** ~120 LOC.
- **Status:** v0.4.

---

## Sequencing — what ships when

| version | features | total LOC | unlocks |
| ------- | -------- | --------- | ------- |
| **v0.2.0** *(now)* | per-chunk content-hash cache + GPU autodetect | ~110 | patterns B, C, D drop to 0; pattern A drops 10-50× |
| **v0.3.0** | LRU eviction, async API, background pre-indexing, adaptive chunking | ~340 | long-running servers, async stacks, chat UIs |
| **v0.4.0** | BM25 prefilter, hierarchical chunking, BM25-only mode, streaming chunks, ANE Core ML | ~620 | huge-haystack pattern A, mobile, real-time |
| **v0.5.0** | ONNX runtime, INT8 embedder, distributed cache | ~350 | extreme low-latency, multi-node servers |
| **v1.0** | static index artifact, persistent vector format | ~450 | RAG-as-shippable-asset, production search |
| **v1.5** | ColBERT late interaction | ~800 | search-engine-grade retrieval |

Total: ~2700 LOC over 6 releases. Every release stands on its own with new use cases unlocked.

---

## Strategic frame

The single thread through this menu: **make the embed cost match the workload, not impose a fixed tax.**

- Tier 1 makes repeats free.
- Tier 2 makes one-shot fast.
- Tier 3 makes huge corpora tractable.
- Tier 4 makes some workloads embed-cost-zero.
- Tier 5 hides the cost entirely in async stacks.

A user paying 7 seconds today on first query of a 1M-character haystack will be paying 0.5 seconds after v0.2 (GPU), 0.1 seconds after v0.4 (BM25 prefilter), and 0 seconds after v0.5 (cached + ONNX) for the same query.

Each release is its own X-postable improvement. The roadmap doubles as a launch calendar.
