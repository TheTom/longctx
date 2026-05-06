"""Shared test fixtures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest


@dataclass
class FakeRetrievalResult:
    indices: list[int]
    candidates: list[str]
    scores: list[float]


class FakePipeline:
    """Drop-in RetrievalPipeline that returns whatever you tell it to.

    Use to test LongCtxClient and MRCRRunner without needing
    sentence-transformers installed.
    """

    def __init__(self, fixed_indices=None):
        self.fixed_indices = fixed_indices or [0, 1]

    def retrieve(self, query, candidates, top_k=8, retrieve_k=None,
                 preserve_order=True):
        idx = list(self.fixed_indices)[:top_k]
        idx = [i for i in idx if i < len(candidates)]
        return FakeRetrievalResult(
            indices=idx,
            candidates=[candidates[i] for i in idx],
            scores=[1.0 - i * 0.1 for i in range(len(idx))],
        )

    def retrieve_chunked(self, query, candidates, top_k=8, chunk_size=500,
                         chunk_overlap=50, preserve_order=True):
        return self.retrieve(query, candidates, top_k=top_k,
                             preserve_order=preserve_order)


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def mock_post_factory():
    """Returns a function that builds a fake `requests.post` response."""

    def _make(content="hello", prompt_tokens=10, completion_tokens=2,
              status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        })
        return response

    return _make


@pytest.fixture
def fake_8needle_df():
    """A tiny fake MRCR-style DataFrame with 3 samples in the 8K bin range."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame([
        {
            "n_chars": 20_000,
            "prompt": '[{"role":"user","content":"start"},'
                      '{"role":"assistant","content":"first response"},'
                      '{"role":"user","content":"more"},'
                      '{"role":"assistant","content":"second response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"third response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"fourth response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"fifth response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"sixth response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"seventh response"},'
                      '{"role":"user","content":"again"},'
                      '{"role":"assistant","content":"eighth response"},'
                      '{"role":"user","content":"final"}]',
            "answer": "PFXfirst response",
            "random_string_to_prepend": "PFX",
        },
        {
            "n_chars": 25_000,
            "prompt": '[{"role":"user","content":"start"},'
                      '{"role":"assistant","content":"alpha"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"beta"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"gamma"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"delta"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"epsilon"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"zeta"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"eta"},'
                      '{"role":"user","content":"x"},'
                      '{"role":"assistant","content":"theta"},'
                      '{"role":"user","content":"final"}]',
            "answer": "Q1Qbeta",
            "random_string_to_prepend": "Q1Q",
        },
        {
            "n_chars": 1_000,  # below 8K bin (16-32K char), should be filtered
            "prompt": '[]',
            "answer": "Z",
            "random_string_to_prepend": "Z",
        },
    ])
