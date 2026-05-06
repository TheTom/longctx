# Changelog

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
