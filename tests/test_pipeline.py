"""Smoke tests for the retrieval pipeline.

These don't hit a network LLM endpoint; they verify the retrieval
component in isolation.
"""
import pytest


def test_imports():
    """Public API surface is importable."""
    from longctx import LongCtxClient, RetrievalPipeline

    assert RetrievalPipeline is not None
    assert LongCtxClient is not None


def test_retrieval_basic():
    """Pure bi-encoder retrieval finds the obvious match."""
    pytest.importorskip("sentence_transformers")
    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline()
    candidates = [
        "The capital of France is Paris.",
        "Bananas grow in tropical climates.",
        "Quantum computing uses qubits.",
        "Mount Everest is the tallest mountain.",
    ]

    result = pipeline.retrieve(
        query="What is the capital of France?",
        candidates=candidates,
        top_k=1,
    )

    assert len(result.candidates) == 1
    assert "France" in result.candidates[0] or "Paris" in result.candidates[0]
    assert result.indices[0] == 0


def test_retrieval_preserves_order():
    """preserve_order=True returns candidates in input order."""
    pytest.importorskip("sentence_transformers")
    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline()
    candidates = [
        "The third paragraph discussed regulatory matters.",
        "The first paragraph introduced the topic.",
        "The second paragraph elaborated on context.",
    ]

    result = pipeline.retrieve(
        query="paragraph",
        candidates=candidates,
        top_k=3,
        preserve_order=True,
    )

    # Indices should be returned in original input order
    assert result.indices == sorted(result.indices)


def test_templates_present():
    """All declared templates are non-empty strings."""
    from longctx.templates import TEMPLATES

    for name, template in TEMPLATES.items():
        assert isinstance(template, str), name
        assert len(template) > 50, name
