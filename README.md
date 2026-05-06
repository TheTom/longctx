# longctx

> **🚧 Work in progress.** Pre-alpha. APIs, numbers, and framing will change as the score-narrowing experiments and roadmap items land. Pin a version (`pip install longctx==0.2.0`) if you depend on it. Issues and PRs welcome.

Open long-context inference stack. Retrieval + open weights, no closed parts.

A small library that bundles the components needed to reach frontier-class long-context retrieval performance on a single accessible GPU using only open weights.

## What it is

`longctx` is a thin wrapper over standard tools:

- **Retrieval**: sentence-transformers (bi-encoder) + faiss
- **Generation**: any OpenAI-compatible LLM endpoint (vLLM, SGLang, llama.cpp server)
- **Defaults tuned for**: Qwen2.5-14B-Instruct-1M, but works with any instruction-following open model

## Why

A stack of `longctx` defaults running Qwen2.5-14B-Instruct-1M on a single MI300X scored **0.822 on MRCR v2 8K bin** (n=82, mass-validated 2026-05-06). The same stack reaches the **1M context bin end-to-end** on commodity open weights — no fine-tuning, no custom architecture, no closed parts. Cross-bin score curve is in [`docs/results.md`](docs/results.md) and continues to be characterized.

This library exists so the rest of the open ecosystem can reproduce that result with one `pip install`.

## Install

```bash
pip install longctx
```

For local vLLM serving:

```bash
pip install longctx[serve]
```

## Quickstart

```python
from longctx import LongCtxClient

# Defaults: sentence-transformers/all-MiniLM-L6-v2 + local vLLM at port 5050
client = LongCtxClient()

# Pass your candidate chunks and a query
result = client.ask(
    query="What was the third response about regulatory compliance?",
    candidates=[
        "Response 1: brief on regulatory compliance...",
        "Response 2: legal analysis of...",
        "Response 3: detailed compliance walkthrough...",
        # ... up to thousands of candidates
    ],
    top_k=8,
)

print(result.content)
print(f"Retrieved indices: {result.retrieved_indices}")
print(f"Prompt tokens: {result.prompt_tokens}")
```

## Custom embedder

```python
from longctx import LongCtxClient, RetrievalPipeline

# Default uses MiniLM-L6 (23M params, CPU-friendly).
# For higher quality at the cost of compute:
pipeline = RetrievalPipeline(embedder_model="BAAI/bge-large-en-v1.5")
client = LongCtxClient(pipeline=pipeline)
```

## Notes on rerankers

`longctx` does not enable cross-encoder reranking by default. Off-the-shelf rerankers (ms-marco-MiniLM, bge-reranker-base) **degraded** retrieval quality on MRCR-style tasks in our 2026-05-06 testing. They are trained for web-search relevance, which doesn't transfer to "find the Nth message of type X" task semantics.

A retrieval-style reranker fine-tuned on appropriate data is on the roadmap. Until then, pure bi-encoder retrieval is the default.

## Status

Pre-alpha v0.2.0. APIs may change.

### Headline numbers (mass-validated)

End-to-end validation 2026-05-06 on AMD MI300X with vLLM-served Qwen2.5-32B-Instruct, default `LongCtxClient` config (sentence-transformers MiniLM-L6 + faiss top-K=8). Score-curve characterization is ongoing — see [`docs/results.md`](docs/results.md) for live numbers.

| MRCR v2 8-needle bin | pipeline | n | avg_score | prefix_pass |
| -------------------- | -------- | -- | --------- | ----------- |
| 8K  (16K-32K char)   | RAG          | 82 | **0.822** | 100% |
| 32K (64K-128K char)  | RAG          | 98 | **0.697** |  97% |
| 64K (128K-256K char) | RAG          | 95 | 0.641 |  98% |
| 64K (128K-256K char) | chunked-RAG  | 95 | **0.670** |  98% |
| 1M (2M-5M char)      | RAG          | 30 | 0.440 | 100% |
| 1M (2M-5M char)      | chunked-RAG  | 30 | 0.409 |  97% |

The 1M bin scores are an in-progress characterization; the score-narrowing campaign (top-K sweeps, layered retrieval, position-aware ordering) is ongoing.

### Other tested generators (single-run, n=30, not mass-validated)

- Qwen2.5-7B-Instruct + RAG: 0.567 (2.4× faster, fits 16GB GPU)
- Qwen2.5-32B-Instruct + RAG: 0.237 (vanilla 32K context window, training-data fit limits the result)
- Qwen3-Next-80B-A3B + RAG: 0.281 (linear-attention hybrid, MoE)

Single-run scores at n=30 have substantial variance (we observed ±0.05 swings between adjacent runs of the same config). Trust the mass-validated numbers above for headline claims.

Mistral-7B-Instruct-v0.3 and Qwen3-8B failed with the default Qwen2.5-style template (prefix-first instruction). Templates are provided for both: `longctx.templates.MISTRAL_VERBATIM_TEMPLATE` and `longctx.templates.QWEN3_NO_THINK_TEMPLATE`. Validation against MRCR for these templates is on the roadmap.

### Reproduce

```bash
longctx-bench --data-dir /path/to/mrcr/v2 --model qwen2.5-14b-instruct-1m \
    --bins 8k 32k 64k --n 80 --include-chunked
```

## License

Apache 2.0.
