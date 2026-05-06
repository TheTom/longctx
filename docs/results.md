# longctx — measured results

Live results from mass-validation runs. Rows update as new data lands.

## MRCR v2 8-needle, Qwen2.5-32B-Instruct, single AMD MI300X

| bin | char range | n | pipeline | top-K | chunk size | avg score | prefix pass | run date |
| --- | ---------- | -- | -------- | ----- | ---------- | --------- | ----------- | -------- |
| 8K | 16K–32K | 82 | RAG | 8 | message-level | **0.822** | 100% | 2026-05-06 |
| 32K | 64K–128K | 98 | RAG | 8 | message-level | 0.697 | 97% | 2026-05-06 |
| 64K | 128K–256K | 95 | RAG | 8 | message-level | 0.641 | 98% | 2026-05-06 |
| 64K | 128K–256K | 95 | chunked | 8 | 500 | 0.670 | 98% | 2026-05-06 |
| 1M | 2M–5M | 30 | RAG | 8 | message-level | 0.440 | 100% | 2026-05-06 |
| 1M | 2M–5M | 30 | chunked | 8 | 500 | 0.409 | 97% | 2026-05-06 |

The 1M bin scores reflect an in-progress score-narrowing campaign. Several retrieval levers are being characterized (top-K sweeps, BM25 hybrid, position-aware retrieval, oracle gen-ceiling). Numbers will be updated as runs complete.

## Reproducing

```bash
pip install longctx[eval]
longctx-eval --bin 8k --n 30 --model qwen25-32b \
    --server http://localhost:5050/v1/chat/completions \
    --data-dir /path/to/mrcr/v2
```

Or for the full multi-bin curve:

```bash
longctx-bench --data-dir /path/to/mrcr/v2 --model qwen25-32b \
    --bins 8k 32k 64k --n 80 --include-chunked
```

## Generator notes

- **Qwen2.5-32B-Instruct** (vanilla, 32K native context window, served via vLLM `--max-model-len 32768`): the headline numbers above. longctx feeds the model only the retrieved top-K, so the 32K window is sufficient regardless of haystack size.
- **Qwen2.5-14B-Instruct-1M** (1M native context): also tested. Scores in the same ballpark at small bins; the 1M-context model is not required for longctx since the model never sees the full haystack.
- Other generators (Mistral-7B-Instruct-v0.3, Qwen3-8B) need the bundled `longctx.templates.MISTRAL_VERBATIM_TEMPLATE` / `QWEN3_NO_THINK_TEMPLATE` to produce verbatim-prefix outputs.

## Methodology

Mass-validation runs use n ≥ 80 for the headline cells (8K, 32K, 64K). Single-run scores at n ≤ 30 have ±0.05 swing across adjacent runs; trust the n ≥ 80 numbers for any cross-cell comparison. The 1M bin is currently characterized at n=30; mass-val will land once the score-narrowing campaign converges on a recipe.
