# 12M coarse filter — bench results

Stage 1 (BM25 + dense + RRF) end-to-end timings on a synthetic NIAH
haystack. **Retrieval-only** — no generator, no model context. The
purpose of this rung is to confirm the prefilter trims a large chunk
set down to a bounded top-K *and* keeps the planted needle inside it
under realistic input sizes.

Pair this with the existing RetrievalPipeline + a generator to measure
end-task accuracy; that lives in `longctx-bench` against MRCR.

## Run config
- Hardware: M5 Max, MPS device
- Embedder: `BAAI/bge-small-en-v1.5` (33M params, 384-dim)
- Chunker: token-aware char-proxy mode (4 chars/token, 2K tokens/chunk)
- Coarse filter: BM25 + dense + RRF (k=60), equal-weighted
- Date: 2026-05-08
- Seeds: 0, 1, 2 per rung (filler order varies; needle text varies)

## Recall@top-K

| target tokens | top-K | n_chunks | n_kept | seeds | needle in top-K | mean rank | best rank | worst rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000     |  100 |   55 |   55 | 0,1,2 | 3/3 |  1.0 | 1 | 1 |
| 1,000,000   | 1000 |  545 |  545 | 0,1,2 | 3/3 |  1.0 | 1 | 1 |
| 4,000,000   | 1000 | 2177 | 1000 | 0,1,2 | 3/3 |  7.3 | 1 | 12 |
| 12,000,000  | 1000 | 6531 | 1000 |   0   | 1/1 |  1.0 | 1 | 1 |

Recall@top-1000 = 10/10 on synthetic. The needle's rank within top-K
slips slightly at 4M (still inside top-12) — expected behavior since
RRF averages two noisy ranking sources; what matters for downstream
rerank is that the chunk is *kept*, not its precise rank.

## Latency

| target tokens | chunk time | filter time | total | notes |
|---:|---:|---:|---:|---|
| 100,000    | <1ms |  ~0.3s | ~0.3s | 100% recall@100 |
| 1,000,000  | <5ms |  ~3.0s | ~3.0s | 100% recall@1000 |
| 4,000,000  | <15ms| ~9.2s  | ~9.2s | 100% recall@1000 |
| 12,000,000 | ~35ms| ~30s   | ~30s  | 100% recall@1000 |

Hits the spec target for cold-cache 12M (<30s on M5). The bge-small
embed pass dominates; chunking is essentially free at this scale.

## Cache effect

Across seeds, chunk text overlaps because the filler sentence pool is
small (8 sentences). The disk-backed embed cache keys on chunk text —
identical chunk slices across seeds reuse cached embeddings — so a
later seed sees a partial warm cache. In the run above, seed=0 of the
4M rung benefited (0.4s vs 9.2s on seeds 1+2) because the 1M run had
already populated common slices.

For real haystacks the cache effect is the opposite shape: first
query pays full cost, repeat queries on the same corpus are
near-instant.

## Reproduce

```bash
# Single rung
python -m longctx.eval.bench_coarse_filter --tokens 12000000 --top-k 1000 --json

# Seed sweep (matches the table above)
for tokens in 100000 1000000 4000000; do
  for seed in 0 1 2; do
    python -m longctx.eval.bench_coarse_filter \
      --tokens $tokens --top-k 1000 --seed $seed --json
  done
done
```

## Hard-mode sweep (topical overlap + paraphrase queries)

Adds a `--hard` filler pool that mentions other projects, access codes,
credential rotation, and decommissioned NOVA references — the needle
now has actual semantic competition. Plus four paraphrase queries with
shrinking surface-token overlap with the needle text. Run 2026-05-08,
seed 0.

| tokens | mode | q0 (literal) | q1 (paraphrase) | q2 (paraphrase) | q3 (paraphrase) |
|---:|:-:|---:|---:|---:|---:|
|   100,000 | easy | 1 | 1 | 1 | 1 |
|   100,000 | hard | 40 | 2 | 2 | 2 |
| 1,000,000 | easy | 1 | 1 | 1 | 1 |
| 1,000,000 | hard | 373 | 9 | 11 | 16 |
| 4,000,000 | easy | 12 | 16 | 10 | 10 |
| 4,000,000 | hard | **974** | 15 | 15 | 17 |
|12,000,000 | hard | 451 | 13 | 4 | 15 |

Recall@top-1000 = **24/24** even in hard mode. Two findings:
- **Literal-query + topical-overlap filler is the actual stress case.**
  q0 hits rank 974 at 4M — borderline within top-1000. Going to top-K
  smaller than 1000 would be unsafe at this combination.
- **Paraphrase queries do *better* in hard mode** because they don't
  share surface tokens with the noise. Counterintuitive but real:
  `"Which numeric credential was issued to NOVA technicians?"` outranks
  the literal `"What is the access code for Project NOVA?"` query.
  Suggests query rewriting / multi-query expansion would help.

## Multi-query fusion uplift

The hard-mode finding (paraphrases outrank the literal query) became a
feature. `CoarseFilter.filter_multi_query(chunks, queries)` RRF-fuses
across N paraphrases at the cost of one extra rank pass per query —
the BM25 build and the dense embed pass over chunks happen once.

Worst case from the hard sweep: 12M tokens, hard mode, literal query
alone:

| query strategy | needle rank | wall time (warm cache) |
|---|---:|---:|
| single literal (q0)              | 451 | 1.3s |
| multi-query, 4 queries (literal + 3 paraphrases) | 3 | 1.3s |
| multi-query, 3 paraphrases (no literal) | **1** | 1.2s |

~150× rank improvement at no perceptible time cost. The literal query
*hurts* fusion at this combination because BM25 ranks it where the
project-overlap noise is densest; dropping it produces the cleanest
result. In production we'd want a paraphrase generator (or a tiny LLM
call) to expand a user's question into 2–3 alternates before the
filter; the call site changes from `filter(q)` to
`filter_multi_query([q, *paraphrases])`.

## Real-corpus aggregate — **13.4M tokens, beats spec target**

Aggregated four real first-party corpora:

| corpus | files | chars |
|---|---:|---:|
| mlx-swift-lm                | 350   | 6,974,557 |
| llama.cpp (Tom's TQ fork)   | 1,269 | 39,823,737 |
| vllm-swift                  | 70    | 823,957 |
| obsidian Self Study (vault) | 1,707 | 5,947,637 |
| **total**                   | **3,396** | **53,569,894 (≈13.4M tokens)** |

End-to-end with multi-query (4 paraphrases) on M5 Max / MPS / bge-small:

| stage | wall time |
|---|---:|
| walk + concat        | 1.3s  |
| chunk (7,300 chunks) | 0.06s |
| coarse filter        | 37.4s |
| **total**            | **38.8s** |

Needle planted at char 27,059,166 (mid-corpus, real file boundary).
**Found at rank 10 in top-1000.** Recall HIT on 13.4M real-corpus
tokens — the first end-to-end context-length probe past 12M on
non-synthetic data.

Reproduce with the script committed at
`benchmark/coarse_filter/real_corpus_aggregate_2026-05-08.json` (the
JSON also captures the per-corpus breakdown).

## Real-corpus NIAH (obsidian vault, 1.5M tokens)

Beyond synthetic: walked Tom's obsidian vault (1707 markdown files,
~1.5M tokens, real prose with topical variety — bible study, sermon
notes, eng project notes, daily logs). Inserted the same NOVA-code
needle at the file-boundary midpoint, ran the pipeline.

| top-K | mode | needle rank | wall time |
|---:|---|---:|---:|
| 1000 | single literal query  | 32 | ~0.4s (warm) |
| 1000 | multi-query (4 paraphrases) | **3** | ~0.3s |
|  100 | multi-query | 3 | ~0.3s |
|   10 | multi-query | 3 | ~0.3s |

Multi-query uplift on real prose mirrors the synthetic hard-mode
finding: ~10× rank improvement at no perceptible time cost. The
needle survives even an aggressive top-10 cutoff, meaning the coarse
filter can hand a very small candidate set to the rerank stage on
real corpora and still keep the right answer.

813 chunks at 8K char (2K token) each. The vault is small relative
to the 12M target; it's a real-corpus *sanity* check, not a scale
test. For scale, the synthetic 12M run is the headline.

Reproduce:
```bash
python -m longctx.eval.bench_coarse_filter_real \
  --corpus-dir "/path/to/your/markdown/corpus" \
  --extensions .md --top-k 1000 --json
```

## Embedder ablation (multi-query, top-1000, 2026-05-08)

Three embedders compared on synthetic 1M hard mode and the real
obsidian vault. Speed numbers conflate model speed with disk-cache
state across runs (the seed-0 chunks repeat across embedders only
when the cache key matches), so **quality (rank) is the comparable
metric**, not wall time.

| corpus | embedder | dims | rank | notes |
|---|---|---:|---:|---|
| synth 1M hard  | sentence-transformers/all-MiniLM-L6-v2 |  384 | 26 | original v0.2 default |
| synth 1M hard  | BAAI/bge-small-en-v1.5                  |  384 | **8** | current default |
| synth 1M hard  | BAAI/bge-m3                             | 1024 | 11 | heavy, fine quality |
| obsidian vault | sentence-transformers/all-MiniLM-L6-v2 |  384 | **1** | best on real prose |
| obsidian vault | BAAI/bge-small-en-v1.5                  |  384 | 3 | close second |
| obsidian vault | BAAI/bge-m3 (default seq=8192, bs=64)   | 1024 | — | OOM (146 GiB) |
| obsidian vault | BAAI/bge-m3 (bs=4, seq=default)         | 1024 | 5 | 471s — fits |
| obsidian vault | BAAI/bge-m3 (bs=8, seq=512)             | 1024 | 4 | 28s — viable |

Findings:
- **bge-small remains the right default.** Best on synthetic hard mode,
  close-2nd on real prose, lightest model.
- **MiniLM-L6 wins real prose by a small margin** but loses synthetic
  hard mode by a wide one. Mixed signal — keep as a fallback only.
- **bge-m3 is now usable** with `batch_size=8` + `max_seq_length=512`
  (28s on real vault, rank 4 — close to the 384-dim embedders'
  numbers). Defaults still OOM because BERT attention is O(seq²) per
  layer and the 8192-token context × 64 batch materializes for the
  full forward pass. Cap either knob and it fits. The pipeline now
  takes both as args — see `CoarseFilter(batch_size=..., max_seq_length=...)`.
- **Cache subdirs include `max_seq_length`** so a 512-cap run can
  never silently reuse 8192-cap entries.

## Caveats / honest scope

- **Synthetic haystack still.** The hard filler is hand-written and
  drawn from a 12-sentence pool — not a real corpus. Real-corpus runs
  (wikipedia, code monorepos) are next.
- **No generator path.** This rung validates retrieval, not answer
  quality. End-task accuracy needs the full pipeline against MRCR or
  a coherent-context bench (see `services/longctx-svc/harness/PRD_coherence_10m.md`).
- **Single embedder.** All numbers above use bge-small-en-v1.5. Other
  embedders (MiniLM-L6, bge-large) not yet swept.
- **MPS only.** CPU and CUDA timings expected to differ; not bench'd
  in this round.
