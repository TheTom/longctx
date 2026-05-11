# PRD — 12M-token coarse filter (hierarchical-selector pipeline)

**Status:** Phase 1–4 implemented (2026-05-08). Branch: `feature/coarse-filter-12m`.
**Owner:** TheTom
**Scope:** longctx (Python). Paired vllm-swift work TBD.

---

## 1. Problem

`RetrievalPipeline.retrieve_chunked` works well up to ~100K tokens of
candidate text. Above that, the dense embed pass dominates wall time
and the optional cross-encoder rerank stage scales linearly with the
chunk count it sees — so on the 12M-token target both stages become
unbounded. The fix is to put a cheap **coarse filter** in front of the
existing rerank+pick pipeline so the downstream stages always see at
most ~1000 candidate chunks regardless of input size.

Concretely: trim O(N) candidate chunks down to a bounded top-K cheaply,
so the existing BGE cross-encoder rerank stage stays bounded
regardless of input size. Designed for inputs up to 12M tokens
(~6,000 chunks at 2K each) on commodity hardware.

## 2. Pipeline shape

```
chunks (any size)
    │
    ▼
Stage 0 — Chunker (token-aware, char-proxy by default)
    │
    ▼
Stage 1 — Coarse filter (BM25 ⊕ dense-embedding via RRF)   ← NEW
    │
    ▼
Stage 2 — Mid rerank (existing dense scoring per chunk)
    │
    ▼
Stage 3 — Pick top-K parents (existing)
    │
    ▼
Stage 4 — Splice into the model context (downstream consumers)
```

Below `coarse_filter_threshold_chars` (default ≈ 400K chars / 100K
tokens) the coarse filter is skipped — `RetrievalPipeline` runs as it
does today, preserving the existing MRCR v2 numbers exactly.

## 3. Why hybrid (BM25 + dense)

- BM25 is **lexical** — catches exact keyword matches, milliseconds,
  no model. Wins on rare-term queries (codes, identifiers, names).
- Dense embedding is **semantic** — catches paraphrases, seconds,
  needs a model. Wins on natural-language queries with no shared
  surface tokens.
- Each fails what the other catches. **Reciprocal Rank Fusion**
  (Cormack, Clarke, Buettcher 2009; `k = 60`) combines them
  deterministically without learnable weights:
  ```
  score(d) = sum over rankers r of  1 / (k + rank_r(d))
  ```

Tier 3.1 of `docs/embedding-roadmap.md` originally targeted v0.4 for
this; the 12M push pulls it forward.

## 4. Components

### 4.1 `longctx.rag.coarse_filter.Chunk`
Plain dataclass with:
- `id` — stable string (position-prefixed, content-hashed) so
  re-running the chunker on identical input produces the same id and
  the embed cache stays warm.
- `text`, `start_offset`, `end_offset`, `token_count`, `metadata`.

### 4.2 `longctx.rag.coarse_filter.CoarseFilter`
Hybrid prefilter. Default embedder `BAAI/bge-small-en-v1.5`
(33M params, 384-dim) — chosen for quality-per-dollar at the coarse
stage; bge-large is overkill here.
- `filter(chunks, query, top_k=1000, bm25_weight=1.0, dense_weight=1.0)`
- Reuses `EmbedCache` and `_resolve_device` from the existing
  `RetrievalPipeline` so cold/warm-cache costs are matched.
- `bm25_weight=0` or `dense_weight=0` skips that stage entirely
  (pure-dense or pure-BM25 modes — tier 4.3 in embedding-roadmap.md).

### 4.3 `longctx.rag.chunker.Chunker`
Token-aware splitter producing `Chunk` objects.
- **char proxy mode** (default): 4 chars/token, sentence-end backoff
  inside a slack window. Same heuristic the existing
  `retrieve_chunked` path uses, so chunk shapes agree when the two
  paths are mixed.
- **HF tokenizer mode**: pass any HF fast tokenizer; uses
  `offset_mapping` for true token boundaries. Slow tokenizers (no
  offset_mapping support) silently drop to char-proxy with a
  documented note.

### 4.4 `RetrievalPipeline.retrieve_chunked` integration
New args: `coarse_filter_threshold_chars` and `coarse_filter_top_n`.
Below threshold: behavior is byte-identical to the pre-coarse-filter
path. Above threshold AND chunk count > top_n: BM25 + dense + RRF
trims chunks before the dense rerank stage. The coarse filter is
built with the pipeline's already-loaded embedder + cache so we don't
pay a second model load.

### 4.5 `longctx.eval.bench_coarse_filter`
Self-contained NIAH bench — synthetic haystack, planted needle,
per-stage timing, recall@top-K. Validates the **retrieval** stage in
isolation; pair with a generator for end-task accuracy.

## 5. Acceptance criteria

| target tokens | top-K | recall@top-K | latency target (M5 Max / MPS) |
|---:|---:|---:|---:|
| 100,000     |  100 | ≥90% | <1s   |
| 1,000,000   | 1000 | ≥90% | <5s   |
| 4,000,000   | 1000 | ≥85% | <15s  |
| 12,000,000  | 1000 | ≥80% | <30s  |

Status as of 2026-05-08: **all rungs hit on synthetic NIAH** —
recall 100% across seeds × rungs, 12M cold-cache 30s end-to-end. See
`benchmark/coarse_filter/RESULTS.md` for the per-seed table and
caveats.

## 6. Out of scope (this PRD)

- Real-corpus NIAH (TODO — synthetic-only for now).
- End-task accuracy with a generator (lives in `longctx-bench` /
  the v0.3 svc work).
- Embedder ablation (bge-base / bge-large / MiniLM-L6).
- vllm-swift internal changes — the longctx side calls vllm-swift
  through the existing serving interface; no engine work needed for
  Phase 1–4.
- Production deployment (`longctx-svc` integration is a follow-up).

## 7. Risks / known caveats

- **Synthetic haystack is easy.** Filler is repetitive and the needle
  uses a high-IDF rare phrase. A real corpus with topical overlap
  will produce harder cases.
- **Single embedder.** All current numbers use `bge-small-en-v1.5`.
- **MPS-only timings.** CPU and CUDA timings expected to differ.
- **RRF rank-only.** BM25 score magnitudes are discarded inside the
  fusion, by design — keeps fusion robust to score-scale mismatch
  between BM25 and dense cosine.

## 8. Test coverage

- 13 unit tests on `CoarseFilter` (BM25, dense, RRF fusion, edge cases,
  weight tilting, synthetic-haystack needle find).
- 15 unit tests on `Chunker` (char proxy, HF mode, sentence backoff,
  stable ids, slow-tokenizer fallback, stats).
- 3 unit tests on the `retrieve_chunked` coarse-filter integration
  (below-threshold byte-identical, above-threshold prefilter fires,
  needle still found).
- 6 unit tests on the bench harness + result struct (mocked).
- Full longctx suite: 91/91 passing.

## 9. Sequencing

- Phase 1 — `CoarseFilter` module + tests. ✓
- Phase 2 — pipeline integration + threshold gate. ✓
- Phase 3 — token-aware `Chunker`. ✓
- Phase 4 — 12M NIAH bench + sweep results. ✓
- Phase 5 (follow-up) — real-corpus NIAH (e.g. wikipedia or a code
  monorepo) + embedder sweep + longctx-svc wiring.
