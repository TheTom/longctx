"""Tests for chunked retrieval."""
import pytest


def test_chunked_basic():
    """Chunked retrieval finds the right parent on a simple case."""
    pytest.importorskip("sentence_transformers")
    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline()
    candidates = [
        # A long-ish passage — chunking should find the relevant part
        "Paragraph 1: introduces the topic of supply chain logistics. "
        "It covers definitions, scope, and historical context. "
        "Mostly background information for the reader.",
        "Paragraph 2: discusses inventory turnover ratios in detail. "
        "Average ratios for retail are between 4 and 8 turns per year. "
        "Manufacturing tends to be lower at 6-12 turns annually.",
        "Paragraph 3: regulatory environment for cross-border shipments. "
        "Customs declarations, tariff codes, harmonized system numbering. "
        "Compliance failure rates are highest in the apparel sector.",
    ]
    result = pipeline.retrieve_chunked(
        query="What is a typical retail inventory turnover ratio?",
        candidates=candidates,
        top_k=1,
        chunk_size=20,
    )
    # Paragraph 2 is the answer; chunked retrieval should find it
    assert result.indices[0] == 1, (
        f"expected paragraph 2 (index 1), got {result.indices}"
    )


def test_chunked_returns_unique_parents():
    """top_k chunks may come from same parent; result should dedupe."""
    pytest.importorskip("sentence_transformers")
    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline()
    candidates = [
        "completely irrelevant text about cooking pasta with garlic",
        "this very long paragraph mentions bananas multiple times. "
        "bananas are great fruit. people love bananas. bananas are yellow. "
        "we sell many bananas at the store. bananas grow in the tropics. "
        "did i mention bananas? yes, bananas are everywhere here.",
        "another irrelevant paragraph about car repair manuals",
    ]
    result = pipeline.retrieve_chunked(
        query="bananas",
        candidates=candidates,
        top_k=2,
        chunk_size=15,
    )
    # Should return 2 UNIQUE parents, not 2 chunks of the bananas paragraph
    assert len(set(result.indices)) == len(result.indices)
    assert len(result.indices) == 2
    # The bananas paragraph should be #1
    assert 1 in result.indices


def test_chunked_preserves_order():
    """preserve_order=True returns parents in original input order."""
    pytest.importorskip("sentence_transformers")
    from longctx import RetrievalPipeline

    pipeline = RetrievalPipeline()
    candidates = [
        "third item, mentions baseball games and World Series",
        "first item, talks about gardening tips for tomatoes",
        "second item, baseball strategy and pitching mechanics",
    ]
    result = pipeline.retrieve_chunked(
        query="baseball",
        candidates=candidates,
        top_k=2,
        chunk_size=10,
        preserve_order=True,
    )
    # Indices should be returned in original input order
    assert result.indices == sorted(result.indices)
