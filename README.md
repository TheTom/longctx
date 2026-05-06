# longctx

Open long-context inference stack. Retrieval + open weights, no closed parts.

A small library that bundles the components needed to reach Anthropic-class long-context retrieval performance on a single accessible GPU using only open weights.

## What it is

`longctx` is a thin wrapper over standard tools:

- **Retrieval**: sentence-transformers (bi-encoder) + faiss
- **Generation**: any OpenAI-compatible LLM endpoint (vLLM, SGLang, llama.cpp server)
- **Defaults tuned for**: Qwen2.5-14B-Instruct-1M, but works with any instruction-following open model

## Why

A stack of `longctx` defaults running Qwen2.5-14B-Instruct-1M on a single MI300X scored **0.760 on MRCR v2 8K bin**, beating the headline number a $29M-funded closed-weight startup published with their custom subquadratic architecture. The architectural moat narrative wasn't load-bearing for the workload. Retrieval + open weights solve it.

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

Pre-alpha. APIs may change. Tested against:

- Qwen2.5-14B-Instruct-1M (best results)
- Qwen2.5-7B-Instruct (good results, 2.4× faster)
- Qwen2.5-32B-Instruct (works)
- Qwen3-Next-80B-A3B-Instruct (works)

Models that fail the default prompt format (Mistral, Qwen3) need a different chat template; fix planned.

## License

Apache 2.0.
