# NIAH advanced — bench results

Three NIAH variants run through the **daemon's full pipeline** (real
`Indexer` + `Searcher`, SqliteChunkStore + MemmapEmbedStore). Confirms
the persistent-storage path agrees with Phase 1's in-memory
`CoarseFilter` numbers and surfaces failure modes that single-needle
benches don't expose: partial recall, semantic ambiguity, positional
bias.

Pair this with the existing `bench_coarse_filter` for the in-memory
side and with the messy-query rig for the relevance-floor / query-type
calibration; this rig's job is the **multi-fact and ambiguity edge**.

## Run config
- Hardware: M5 Max, MPS device
- Embedder: `BAAI/bge-small-en-v1.5` (33M params, 384-dim)
- Chunker: char-proxy mode, 256 tokens/chunk (smaller than the Phase 1
  benchmark's 2K — needed so adjacent multi-needles land in distinct
  chunks at small corpus sizes)
- Searcher: BM25 + dense + RRF (k=60), `relevance_floor=0.0` (we want
  raw retrieval behavior in the synthetic regime; the production floor
  is calibrated against the messy-query rig, not synthetic noise)
- Date: 2026-05-09
- Seed: 0 (per-sample seeds derived deterministically)

## 1. Multi-needle (3 facts, ALL required in top-K)

| n_facts | corpus_tokens | samples | R@50 (per-fact) | All-of-N@50 | mean wall (s) |
|---:|---:|---:|---:|---:|---:|
| 3 | 100,000 | 5 | 1.00 | 1.00 | 3.7 |

Per-fact recall and All-of-N both 1.00 at 100K with bge-small. Each
fact uses a distinct project codename (ATLAS / ORION / TITAN) +
unique numeric code; the dense path nails them with codes-in-query
boost. Larger corpora and more facts will compress the All-of-N
metric — this is the expected regime where a single weak fact tanks
the whole sample. Expand the matrix when wiring the daemon to 1M+
corpora.

## 2. Needle-near-needle (semantic distractor)

| corpus_tokens | samples | real recall | real_beats_distractor | mean wall (s) |
|---:|---:|---:|---:|---:|
| 100,000 | 5 | 1.00 | 0.60 | 3.6 |

Both needles always retrieved (real_recall=1.00) — the dense path has
no trouble finding either. The interesting number is
`real_beats_distractor=0.60`: bge-small only ranks the real needle
above the lexically-distinct distractor 60% of the time, even though
the distractor explicitly says "deprecated" and "legacy".

This is a known weakness of small bi-encoders on ambiguity. Production
mitigation paths: (a) cross-encoder rerank for top-K, (b) raise the
chunk size so contextual cues like "deprecated" carry more weight in
the embedding, (c) add a query-side rewrite that emphasizes "current"
or "active". For Phase 2.0.1 we surface the metric so agents know
ambiguity is a real failure mode; agents should NOT assume the top-1
is the canonical fact when multiple project-NOVA-shaped chunks exist.

## 3. Depth sweep (positional bias)

| depth | 100,000 |
|---:|---:|
| 10% | 1.00 |
| 30% | 1.00 |
| 50% | 1.00 |
| 70% | 1.00 |
| 90% | 1.00 |

Flat at 100K — no positional bias detectable with 3 samples per cell.
This is the healthy outcome: the chunker handles tail / head /
midpoint identically, and the embed-store search ordering doesn't
favor low row indices. Re-run at 1M and 12M (with `--include-12m`)
once the daemon's storage is large enough to make positional artefacts
visible.

## Reproduce

```bash
python3 -m longctx_daemon.eval.niah_advanced multi-needle \
    --n-facts 3 --tokens 100000 --samples 5 --save
python3 -m longctx_daemon.eval.niah_advanced near-needle \
    --tokens 100000 --samples 5 --save
python3 -m longctx_daemon.eval.niah_advanced depth \
    --tokens 100000 --samples 3 --save
# All-in-one smoke (5 samples per variant):
python3 -m longctx_daemon.eval.niah_advanced all
```

JSON snapshots land at `benchmark/niah_advanced/<variant>_<date>.json`.

## Open questions

- 1M-token cell on multi-needle and near-needle: how does All-of-N
  scale? Hypothesis: drops faster than per-fact recall as N grows;
  TODO confirm.
- 12M depth-sweep with `--include-12m`: opt-in only because each
  cell is ~30s; budget ~12 min to fill the row.
- Cross-encoder rerank as a fix for the near-needle 0.60 number —
  not part of Phase 2.0.1 but the metric here is the gate for whether
  it's worth the latency.
