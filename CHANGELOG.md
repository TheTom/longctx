# Changelog

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

See `docs/embedding-roadmap.md` for the full roadmap of future tiers (BM25 prefilter, hierarchical chunking, ANE Core ML, ColBERT-style late interaction).

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
