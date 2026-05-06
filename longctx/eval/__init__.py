"""Eval runners for long-context retrieval benchmarks.

Currently bundled:
- MRCR v2 (8-needle): OpenAI's published long-context coreference benchmark
"""

from longctx.eval.mrcr import MRCRRunner

__all__ = ["MRCRRunner"]
