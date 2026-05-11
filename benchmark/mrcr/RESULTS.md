# Clean-MRCR Retrieval-Only Recall@K (2026-05-09)

## Setup

- Embedder: `BAAI/bge-small-en-v1.5` on MPS
- Pipeline: `longctx_daemon.indexer.Indexer` → `SqliteChunkStore` + `MemmapEmbedStore` → `Searcher` (BM25 + dense + RRF, no multi-query, single literal query, default `relevance_floor=0.50`)
- Chunker: `longctx.rag.chunker.Chunker(tokens_per_chunk=256, respect_sentences=False)` — tuned so most assistant messages stay in 1-2 chunks (typical MRCR assistant message ~1700 chars / ~425 tokens at 4 chars/token)
- Dataset: `openai/mrcr` 8-needle shard, samples streamed and binned by `n_chars`
- Sample budget: 30 per bin
- Reachable bins (where the 8-needle shard has rows): **32K, 64K, 256K, 512K**. The 8-needle shard skips 8K/16K/128K/1M by construction; runner falls back to a synthetic fixture when a bin is empty (not done here)

## Results — Retrieval-only Recall@K, 8-needle, 30 samples per bin

| bin   | n  | R@1   | R@3   | R@5   | R@8   |
|-------|----|-------|-------|-------|-------|
| 32K   | 30 | 0.267 | 0.500 | 0.567 | 0.667 |
| 64K   | 30 | 0.133 | 0.467 | 0.467 | 0.633 |
| 256K  | 30 | 0.033 | 0.200 | 0.267 | 0.433 |
| 512K  | 30 | 0.200 | 0.400 | 0.433 | 0.733 |

Average top-1 dense cosine sits in the 0.68-0.70 range across all bins — well above the 0.50 floor — so `no_relevant_results` never fires (`floor_tripped_count = 0` in every bin). The miss cases (rank=None) are always **legitimately missed** (the right chunk fell off the top-32 ranking entirely), not floor-suppressed.

## What this measures vs what SubQ measured

This is **retrieval-only** — no LLM is invoked. For each sample we:

1. Treat each prior assistant message as one document in a per-sample synthetic corpus
2. Index that corpus through the production daemon stack (chunker → SQLite + memmap)
3. Run the user's final question through `Searcher.search` (BM25 + dense + RRF)
4. Score: did the chunk containing the gold needle text rank in top-K?

SubQ's published **0.659** and Tom's prior LongCtx end-to-end MRCR numbers (**0.601** mass-validation + **0.688** directional) are **end-to-end RAG → generator** scores. They factor in:

- The retriever (this score)
- The generator's instruction-following (prepend the random string verbatim)
- The generator's verbatim-copy fidelity on a 1.7K-character target

A high R@8 here is therefore a **necessary upper bound** for end-to-end MRCR performance, not a 1:1 comparison. If R@8 < 0.7, no generator on earth can break 0.7 end-to-end. The numbers here say:

- The current pipeline already exposes the right candidate to the generator about **63-73% of the time** at depth 8 in the small-context bins
- The 256K bin is the visible bottleneck — R@8 drops to 0.43, half of 32K. Worth investigating whether 256K-specific chunk-count growth is dispersing the answer signal

## Headline observations

1. **R@1 is brutally low (3-27%)** — the right chunk almost never lands top-1. BM25 + dense cosine without query rewriting / multi-query expansion struggle to disambiguate "the 6th essay about X" from the other essays-about-X.

2. **The relevance floor never fires.** Average top-1 cosine is 0.68-0.70, meaning even when retrieval is **wrong** the top-1 chunk is still semantically close to the query. This is the SubQ failure mode: confident wrong answer. Telling the agent "no results" wouldn't help here — the agent legitimately needs disambiguation, not a no-go signal.

3. **R@8 grows slowly with depth** — going from K=5 to K=8 adds ~10 percentage points across bins. The right chunk is genuinely scattered through the ranked list (we see ranks 12, 18, 23, 26 in the per-sample records).

4. **256K is the worst bin (R@8=0.43).** Hypothesis: chunk count grows roughly linearly with `n_chars`; at 256K-512K char haystacks the per-chunk distractor mass passes a tipping point where pure BM25 + dense fusion can't surface the right one. 512K rebounds (R@8=0.73) — possibly because the 512K samples in this shard have more lexically-distinctive needles, or the embedder hits a sweet spot. Worth replication.

5. **No `query_type=find_similar` triggers.** All MRCR queries are imperative natural-language ("Prepend X to the Nth Y…") — the heuristic correctly classifies them as `natural_language`.

## Implications

This is a **realistic baseline** for what longctx's coarse filter can do on the MRCR retrieval-only metric without paraphrase rewriting, query expansion, or multi-query fusion. Levers to pull next:

- **Query expansion**: synthesize 2-3 paraphrases of the user's question and pass via the existing `paraphrases` parameter on `Searcher.search`. The multi-query RRF path is already wired; just needs the paraphrase generator.
- **Position-aware retrieval**: MRCR queries explicitly mention ordinal position ("the 6th essay"). Adding a position-aware reranker (or just embedding the asst index in the chunk text) should sharply improve R@1.
- **Hybrid scoring tuned for "Nth"**: the BM25 / dense weight ratio is currently 1:1. For ordinal queries BM25 is probably stronger (the random prefix helps).
- **Investigate 256K dropout**: per-bin chunk count and cosine distribution suggests something specific to that chunk-count regime.

## Reproduce

```bash
python3 -m longctx_daemon.eval.clean_mrcr_retrieval \
    --needles 8 --samples 30 --bins 32K,64K,256K,512K --verbose

# Smoke (5 samples, 32K bin)
python3 -m longctx_daemon.eval.clean_mrcr_retrieval \
    --needles 8 --samples 5 --bins 32K --json

# Slower bins via per-bin override
python3 -m longctx_daemon.eval.clean_mrcr_retrieval \
    --needles 8 --samples 30 --bins 32K,512K \
    --samples-per-bin '{"512K": 10}'
```

Raw per-sample records (rank, top-1 cosine, query/answer heads) live at `clean_recall_2026-05-09.json` for diffing across pipeline tweaks.

## Caveats

- The 8-needle shard's natural distribution doesn't fill 8K/16K/128K/1M bins. If those need real numbers, sample from 2-needle / 4-needle shards or split essays manually. The 8K bin in the original spec table will require either smaller MRCR samples or a synthetic fallback.
- 30 samples per bin is enough to see the shape; not enough for tight CIs. Run 100+ before any conclusion.
- `Searcher._scope_to_chunk_ids` is bypassed in this eval (set to `lambda _: None`) to work around a pre-existing protocol-vs-implementation indexing offset (`chunk_ids_in_scope` is documented as row indices but production passes chunk IDs). Doesn't change the result for retrieval — the corpus has only one project — but flags a real bug worth fixing in the searcher itself.
