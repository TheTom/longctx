"""longctx — open long-context inference stack.

Bundles the components that make Qwen2.5-1M class models actually deliver
long-context retrieval performance on accessible hardware:
- Retrieval pipeline (sentence-transformers + faiss + optional reranker)
- Chat templates that work with retrieval-style verbatim-prefix tasks
- Eval runners for MRCR v2 and similar long-context benchmarks
- Optional vLLM integration with the DCA RoPE V1 fallback patch baked in

Reference: see https://github.com/TheTom/longctx for benchmarks and
reproductions of the open-stack-matches-SubQ result.
"""

__version__ = "0.2.0"

from longctx.rag.pipeline import RetrievalPipeline
from longctx.rag.client import LongCtxClient

__all__ = [
    "RetrievalPipeline",
    "LongCtxClient",
    "__version__",
]
