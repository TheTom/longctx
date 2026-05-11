# Real-embedder messy-query observation (2026-05-09)

## Setup

- Embedder: `BAAI/bge-small-en-v1.5` on MPS
- Corpus: 5-file synthetic project (`auth.py`, `billing.py`, `inventory.py`, `notes.md`, `README.md`) with one planted needle in `docs/notes.md`
- Pipeline: full `Indexer` → `SqliteChunkStore` + `MemmapEmbedStore` → `Searcher` (BM25 + dense + RRF, no multi-query)

The point is to surface **dense cosine + BM25 raw** scores per case, not just the fused RRF — RRF is rank-driven and useless as an absolute relevance signal.

## Results

| case                    | rank | RRF top-1 | dense cos | BM25 raw | top-1 file |
|-------------------------|-----:|----------:|----------:|---------:|------------|
| clean (control)         |    1 |    0.0323 |  **0.748** |     3.49 | `docs/notes.md` ✓ |
| multi-question          |    1 |    0.0325 |    0.616 |     4.64 | `docs/notes.md` ✓ |
| run-on no-punc          |    2 |    0.0325 |    0.409 |     1.40 | `README.md` ✗ |
| blob+question           |    1 |    0.0323 |    0.653 |     5.38 | `docs/notes.md` ✓ |
| blob only (find similar)|    3 |    0.0323 |    0.504 |     5.86 | `billing.py` |
| fragments               |    2 |    0.0325 |    0.483 |     3.29 | `README.md` ✗ |
| typo                    |    1 |    0.0323 |    0.645 |     2.09 | `docs/notes.md` ✓ |
| off-corpus              |    1 |    0.0323 |  **0.439** |     1.15 | `docs/notes.md` ✗ false positive |

## Headline findings

1. **Dense cosine separates real from noise.** Clean control hits **0.748**; off-corpus ("capital of france") hits **0.439**. There's a real ~0.30 gap.

2. **RRF score is useless for thresholding.** Every top-1 sits at ~0.0323-0.0325 regardless of match quality. Confirms what the synthetic suite suggested: any future `no_relevant_results` flag has to use the **dense cosine**, not the fused RRF.

3. **`relevance_floor ≈ 0.50` is a defensible starting threshold.** Above: trustworthy. Below: probably noise. Sits in the gap between "honest matches" (≥0.6 in this corpus) and "false positives" (≤0.45).

4. **Multi-question still finds the needle** (rank 1, dense 0.616). bge-small handles dilution surprisingly well at this corpus size. May not scale to 12M-token corpora — re-run there before any conclusion.

5. **Off-corpus IS a confident false positive today.** "what is the capital of france" returns `notes.md` at rank 1 with 0.439 cosine. This is the killer case for `no_relevant_results`: the agent has no signal to know the answer is garbage.

6. **BM25 raw is anchored too** — clean=3.49, off-corpus=1.15. Could be a secondary signal but the dense cosine is the cleaner anchor.

## Implications for Phase 2.0.1 messy-query design

When we wire the messy-query handling, **anchor everything in the dense cosine**:

- `no_relevant_results: true` when top-1 dense cosine < `relevance_floor` (default ~0.50)
- `topic_diversity` heuristic: Jaccard over BM25 top-N vs dense top-N (we have both rankings cheaply)
- Multi-question handling: `query: str | list[str]` API; per-query result groups

## Reproduce

```bash
python3 -m longctx_daemon.eval.messy_queries_real -v
python3 -m longctx_daemon.eval.messy_queries_real --json > out.json
```

JSON output is committed at `real_embedder_2026-05-09.json` for diffing across embedder/chunker/RRF tweaks.
