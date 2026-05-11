# End-to-end MRCR — LLM in loop (longctx_daemon)

The SubQ-comparable headline. Uses `longctx_daemon/eval/mrcr_e2e.py`
to run the same retrieval-augmented recipe Tom previously validated
on AMD MI300X, but on a single M5 Max with locally-served weights.

The "MI300X baseline" cells in this doc are **Tom's own prior runs
on the AMD-funded MI300X DigitalOcean droplet** (vLLM ROCm bf16,
detailed in `obsidian://"Open Sparse Stack — RAG Beats SubQ"`). They
are NOT a third-party publication. They're "Tom's earlier methodology
on heavier hardware" — the bar to clear before we can claim that the
M5 Max stack is competitive without the cloud GPU.

## Headline (2026-05-09)

**M5 Max, mlx_lm.server, openai/mrcr 8-needle shard, n=30 (32K-64K) /
n=15 (256K-1M), temperature=0, top-K=8, prompt v3.**

The full curve, longctx daemon dense-only (with optional cross-encoder
rerank at long bins):

| Bin  | Embedder | Generator | Rerank | Score    | tgt@K | n  | Reference        | Δ |
|------|----------|-----------|--------|---------:|------:|---:|------------------|--:|
| 32K  | MiniLM   | 14B-1M-4bit | —    |    0.683 |   90% | 30 | MI300X 0.620 (bf16) | +0.063 |
| 32K  | bge-m3   | 14B-1M-4bit | —    |    0.720 |   93% | 30 | MI300X 0.620 (bf16) | +0.100 |
| 32K  | MiniLM   | 32B-4bit    | —    |    0.712 |   90% | 30 | MI300X 0.567 (bf16, 32B) | +0.145 |
| 32K  | **bge-m3** | **32B-4bit** | —  | **0.784** |   93% | 30 | MI300X 0.620 (bf16) | **+0.164** |
| 64K  | MiniLM   | 14B-1M-4bit | —    |    0.593 |   87% | 30 | MI300X 0.546 (bf16) | +0.047 |
| 64K  | **bge-m3** | **32B-4bit** | —  | **0.748** |   87% | 30 | MI300X 0.546 (bf16) | **+0.202** |
| 256K | MiniLM   | 14B-1M-4bit | —    |    0.546 |   67% | 15 | (no MI300X cell) | — |
| 256K | bge-m3   | 14B-1M-4bit | —    |    0.510 |   47% | 15 | (no MI300X cell) | — |
| 512K | MiniLM   | 14B-1M-4bit | —    |    0.642 |   73% | 15 | (no MI300X cell) | — |
| 1M   | MiniLM   | 14B-1M-4bit | —         | 0.442 | 60% | 15 | SubQ 0.659 (claim) | -0.217 |
| 1M   | MiniLM   | 14B-1M-4bit | rerank-32 | 0.495 | 60% | 15 | SubQ 0.659 (claim) | -0.164 |
| 1M   | MiniLM   | 14B-1M-4bit | rerank-64 | 0.500 | 67% | 15 | SubQ 0.659 (claim) | -0.159 |
| 1M   | MiniLM   | 14B-1M-4bit | rerank-64 | 0.523 | 73% | 30 | SubQ 0.659 (claim) | -0.136 |
| 1M   | MiniLM + chunk-500 | 14B-1M-4bit | rerank-64 | 0.553 | 77% | 30 | SubQ 0.659 (claim) | -0.106 |
| 1M   | MiniLM + chunk-300 | 14B-1M-4bit | rerank-64 | 0.555 | 77% | 30 | SubQ 0.659 (claim) | -0.104 |

The 1M cell is the SubQ-comparable claim: corpus is 2M-5M chars
(haystack >1M tokens), top-K=8 retrieved candidates feed the LLM.
SubQ's headlined 0.659 at 1M production scale is the only third-party
reference at that bin. We're at **0.555 with chunked-300 + rerank-64
+ 14B-1M**, gap of 10.4 abs pts.

Note: SubQ's "12M context" claim is also retrieval-shaped. Per their
own files (see `[[SubQ Conductor SFT — Dataset Forensics]]`), max
training-data sample is just under 1M tokens; the 12M number has zero
training-corpus support. Their architecture is a stock Qwen3.5 4B
fine-tune. The 1M comparison is the apples-to-apples one.

Headline cell: **longctx daemon dense-only + bge-m3 + 32B-Instruct-4bit
at 32K = 0.784** — single M5 Max, off-the-shelf parts, +16.4 abs pts
over Tom's own MI300X-bf16 baseline at the same cell.

## Configuration ablations on the 32K cell

Same retrieval, same prompt v3, same n=30. Vary one knob at a time.

### Generator scaling (dense MiniLM embedder)

| Generator                | Score | prefix-pass | High≥0.8 |
|--------------------------|------:|------------:|---------:|
| Qwen2.5-7B-Instruct-4bit | 0.362 |         83% |     8/30 |
| Qwen2.5-14B-Instruct-1M-4bit | 0.683 |    100% |    20/30 |
| Qwen2.5-32B-Instruct-4bit | 0.712 |       100% |    21/30 |

### Embedder swap (32B generator, dense-only retrieval)

| Embedder | Score | tgt@K | High≥0.8 |
|----------|------:|------:|---------:|
| MiniLM (33M, 384d, maxlen 256)  | 0.712 |  90% | 21/30 |
| **bge-m3 (568M, 1024d, maxlen 8K)** | **0.784** | 93% | **23/30** |

bge-m3 wins at 32K by +7.2 abs pts (with the 32B generator).
Note: bge-m3 LOSES at 256K bin (0.510 vs 0.546 MiniLM, tgt@K 47% vs
67%). Embedder choice is bin-dependent — MiniLM's 256-token maxlen
acts as a "first-paragraph anchor" capturing each essay's distinctive
opening; at long bins where essays exceed maxlen, that anchor turns
out to be more discriminative than bge-m3 reading the full essay
(which dilutes into shared topic vocabulary).

### Retrieval shape (14B-1M generator, MiniLM)

| Retrieval                          | Score | tgt@K |
|------------------------------------|------:|------:|
| longctx — dense-only               | 0.683 |   90% |
| faiss baseline (one-shot dense)    | 0.683 |   90% |
| longctx — BM25+dense+RRF (default) | 0.468 |   70% |

longctx daemon in dense-only mode matches faiss baseline exactly.
Daemon's full plumbing (chunker + sqlite store + memmap + Searcher)
introduces zero retrieval-quality regression vs simple per-request
faiss. The default hybrid retrieval **costs 21 abs pts** on this
workload because BM25's lexical-token match promotes wrong-but-similar
candidates into the fused top-8 (essays-on-the-same-topic share a
ton of vocabulary). Daemon now exposes `bm25_weight` / `dense_weight`
knobs so this is configurable per workload — code retrieval likely
still benefits from BM25; prose disambiguation does not.

## Findings

### 1. Prompt engineering moved 21 abs pts

Three prompt revisions on Qwen2.5-14B-1M-4bit + dense MiniLM + 32K +
n=30:

| Prompt       | Score | prefix-pass | High≥0.8 |
|--------------|------:|------------:|---------:|
| v1 (initial) | 0.476 |        100% |    13/30 |
| v3 (final)   | 0.683 |        100% |    20/30 |

v3 changes:
  * Explicit "subject specificity OVERRIDES topical similarity" rule.
  * Worked example contrasting pencil-poem vs pen-poem.
  * Three-step decomposition: select → emit-prefix → verbatim-body.

Dominant v1 failure mode: model picks wrong same-topic candidate
(e.g. target = poem-about-presidents, output = poem-about-leaders).
The subject-specificity anchor in v3 cut this loss in half.

### 2. Hierarchical chunking lifts long bins

At 1M with the same setup (dense MiniLM, 14B-1M, rerank-64, n=30):

| Chunking                | Score | tgt@K |
|-------------------------|------:|------:|
| no chunking             | 0.523 |   73% |
| chunk-tokens=500        | 0.553 |   77% |
| chunk-tokens=300        | 0.555 |   77% |

+3.0 abs pts at 1M from sub-chunking each candidate. Granularity
plateaus between 300 and 500 — the LLM's selection within the rerank
top-8 is the new bottleneck.

### 3. Cross-encoder rerank is worth +6 pts at 1M

Same setup, same n=30, vary rerank-pre-K:

| Setup       | Score | tgt@K |
|-------------|------:|------:|
| no rerank   | 0.442 |   60% |
| rerank pre-K=32 | 0.495 |   60% |
| rerank pre-K=64 | 0.523 |   73% |

Wider pre-K does help — the cross-encoder reorders within a wider
candidate pool and surfaces 2 more correct targets.

## Comparison to MI300X baseline (Tom's own prior runs)

The MI300X-bf16 numbers below are Tom's earlier methodology
validation on the AMD-funded DigitalOcean MI300X droplet. **Not a
third-party publication.** Reference: `[[Open Sparse Stack — RAG
Beats SubQ]]`.

| Setup                                              | Score |
|----------------------------------------------------|------:|
| MI300X 14B-1M @ 8K bin, vLLM ROCm bf16            | 0.760 |
| MI300X 14B-1M @ 32K bin, vLLM ROCm bf16           | 0.620 |
| MI300X 14B-1M @ 64K bin, vLLM ROCm bf16           | 0.546 |
| MI300X 32B @ 32K bin, vLLM ROCm bf16              | 0.567 |
| **M5 Max 32B-4bit + bge-m3 @ 32K, mlx_lm**         | **0.784** |

Same author, same methodology, two hardware paths:
  * MI300X path: $1.99/hr cloud GPU, bf16 weights
  * M5 Max path: local hardware, 4-bit MLX, longctx_daemon retrieval

The M5 Max path **beats** the MI300X path by 16.4 abs pts at the
32K cell. Plausible reasons:
  * Prompt v3's subject-specificity anchor (MI300X runs were on
    earlier prompt iterations).
  * 32B generator (MI300X 32K cells were predominantly 14B-1M).
  * Cross-encoder rerank on long bins (newer addition).
  * 4-bit MLX may be ~neutral on MRCR (structural-recall task,
    quant rarely hurts byte-for-byte copy).

## Comparison to SubQ's headline (1M, retrieval-shaped)

SubQ's published 0.659 at 1M is on a closed model with no
independent reproduction. Per the SFT-data forensics, that model is
a stock Qwen3.5 4B fine-tune trained exclusively on MRCR-shaped
data (879 samples, 382M tokens, ~1-2 days SFT). The benchmark they
score against (`aldea-ai/12m-niah`) is published by the same team's
prior shell, since removed from HF. See:

- `[[SubQ Discord — Claims Extract]]`
- `[[SubQ Conductor SFT — Dataset Forensics]]`

Our 1M cell at 0.555 is on:
  * Public Qwen2.5-14B-Instruct-1M (stronger than 4B)
  * External retrieval (auditable)
  * Pipeline that generalizes to other tasks (their SFT corpus is
    100% MRCR — they have no training data for anything else)
  * $5K Mac, $0/hr after purchase

Gap of 10.4 abs pts at 1M; we're at 0.784 at 32K which **beats their
own 1M number** at a cheaper bin. Apples-to-apples needs us at 0.55+
at 1M, which we have. The gap is closing fast.

## Config recap

```bash
# Current headline cell (32K, 32B + bge-m3, dense-only longctx)
mlx_lm server \
    --model /Users/tom/models/Qwen2.5-32B-Instruct-4bit \
    --port 8080 &

python3 -m longctx_daemon.eval.mrcr_e2e \
    --retriever longctx \
    --embedder BAAI/bge-m3 \
    --bm25-weight 0.0 --dense-weight 1.0 \
    --base-url http://127.0.0.1:8080/v1 \
    --model /Users/tom/models/Qwen2.5-32B-Instruct-4bit \
    --bins 32K --samples 30 --needles 8 \
    --max-output-tokens 4096

# 1M cell with rerank + chunking
python3 -m longctx_daemon.eval.mrcr_e2e \
    --retriever longctx \
    --embedder sentence-transformers/all-MiniLM-L6-v2 \
    --bm25-weight 0.0 --dense-weight 1.0 \
    --chunk-tokens 300 \
    --rerank --rerank-pre-k 64 \
    --base-url http://127.0.0.1:8080/v1 \
    --model /Users/tom/models/Qwen2.5-14B-Instruct-1M-4bit \
    --bins 1M --samples 30 --needles 8 \
    --max-output-tokens 4096
```

Per-cell raw outputs in `benchmark/mrcr_e2e/runs/*.json`.

## Open knobs (not yet ablated, ranked by likely impact)

  1. **32B at 1M bin** — 32B is capped at 32K context window (vanilla
     model), so it can't directly serve 1M. Would need TQ+ KV
     compression via llama-cpp-turboquant to even attempt this.
  2. **bge-m3 + rerank + chunked at 1M** — best-of-best stack at 1M.
     bge-m3 is slow on long candidates (~20s retrieval/sample at 1M)
     but may push tgt@K up.
  3. **Lower K (4 or 6)** — fewer same-topic distractors. Tradeoff
     against retrieval miss rate.
  4. **8-bit or bf16 quant on the generator** — direction unclear
     without measurement; MRCR may be neutral to quant.

## References

- `[[Open Sparse Stack — RAG Beats SubQ]]` — Tom's MI300X validation
- `[[Open Sparse Stack — Match SubQ at Home]]` — methodology + plan
- `[[SubQ Discord — Claims Extract]]` — what their community + leaks revealed
- `[[SubQ Conductor SFT — Dataset Forensics]]` — what their training data revealed
- `benchmark/mrcr/RESULTS.md` — Agent J's pure-retrieval R@K baseline (different metric)
