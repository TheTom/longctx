"""Token-aware chunker tests — char proxy mode + HF tokenizer mode.

Avoids downloading any real tokenizer; HF mode is exercised with a
fake tokenizer that mimics the BatchEncoding interface enough to
drive the chunker.
"""
from __future__ import annotations

import pytest

from longctx.rag.chunker import Chunker, _stable_id, _walk_to_sentence_end
from longctx.rag.coarse_filter import Chunk


# --------------------------------------------------------------- helpers

class _FakeTokenizer:
    """Whitespace-split fake tokenizer that returns offset_mapping the way
    HF fast tokenizers do. One token per whitespace-delimited word."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        offsets: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            offsets.append((start, i))
        out = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class _SlowTokenizer:
    """Mimics a HF slow tokenizer that doesn't accept offset_mapping."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        if return_offsets_mapping:
            raise TypeError("slow tokenizer doesn't support offset_mapping")
        return {"input_ids": text.split()}


# ------------------------------------------------------------- stable IDs

def test_stable_id_is_deterministic():
    a = _stable_id(0, "hello world")
    b = _stable_id(0, "hello world")
    assert a == b


def test_stable_id_changes_with_position():
    a = _stable_id(0, "hello world")
    b = _stable_id(1, "hello world")
    assert a != b


def test_stable_id_changes_with_text():
    a = _stable_id(0, "hello world")
    b = _stable_id(0, "hello earth")
    assert a != b


# ----------------------------------------------------- sentence backoff

def test_walk_to_sentence_end_finds_period():
    text = "Sentence one. Sentence two is longer than first. Sentence three."
    end = len("Sentence one. Sentence two is longer ")  # mid-word
    result = _walk_to_sentence_end(text, end, slack=80)
    # Should pull back to right after "one. "
    assert text[result - 2:result] in (". ", "! ", "? ")


def test_walk_to_sentence_end_falls_back_when_none_in_window():
    text = "no_sentence_terminators_here_at_all_a_long_string"
    end = 30
    result = _walk_to_sentence_end(text, end, slack=10)
    assert result == 30


# --------------------------------------------------------- char proxy

def test_char_proxy_basic_split():
    text = "abc " * 1000  # 4000 chars
    chunker = Chunker(tokens_per_chunk=100, token_overlap=10,
                      respect_sentences=False)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Every chunk's offsets must be inside the input
    for c in chunks:
        assert 0 <= c.start_offset < c.end_offset <= len(text)
        assert text[c.start_offset:c.end_offset] == c.text


def test_char_proxy_respects_overlap():
    text = "abc " * 1000
    chunker = Chunker(tokens_per_chunk=100, token_overlap=20,
                      respect_sentences=False)
    chunks = chunker.chunk(text)
    # Adjacent chunks must overlap when overlap > 0
    if len(chunks) >= 2:
        a, b = chunks[0], chunks[1]
        assert b.start_offset < a.end_offset


def test_char_proxy_short_text_one_chunk():
    chunker = Chunker(tokens_per_chunk=2048)
    chunks = chunker.chunk("short text here")
    assert len(chunks) == 1
    assert chunks[0].text == "short text here"


def test_char_proxy_empty_returns_empty():
    chunker = Chunker(tokens_per_chunk=10)
    assert chunker.chunk("") == []


def test_chunk_many_assigns_monotonic_positions():
    chunker = Chunker(tokens_per_chunk=2048)
    chunks = chunker.chunk_many(["doc one text", "doc two text", "doc three"])
    assert len(chunks) == 3
    ids = [c.id for c in chunks]
    assert len(set(ids)) == 3  # all unique


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        Chunker(tokens_per_chunk=0)
    with pytest.raises(ValueError):
        Chunker(tokens_per_chunk=10, token_overlap=10)
    with pytest.raises(ValueError):
        Chunker(tokens_per_chunk=10, token_overlap=20)


# --------------------------------------------------------- HF mode

def test_hf_mode_uses_token_boundaries():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunker = Chunker(tokens_per_chunk=3, token_overlap=1,
                      tokenizer=_FakeTokenizer())
    chunks = chunker.chunk(text)
    assert all(c.token_count <= 3 for c in chunks)
    # First chunk starts at 0, ends near token 3
    assert chunks[0].start_offset == 0
    assert chunks[0].text.startswith("alpha")


def test_hf_mode_slow_tokenizer_falls_back():
    """A slow tokenizer that rejects offset_mapping should silently
    drop to char-proxy mode, not crash."""
    chunker = Chunker(tokens_per_chunk=10, tokenizer=_SlowTokenizer())
    chunks = chunker.chunk("alpha beta gamma " * 50)
    assert len(chunks) >= 1


# ------------------------------------------------------------- stats

def test_stats_aggregates():
    chunker = Chunker(tokens_per_chunk=50, token_overlap=5,
                      respect_sentences=False)
    chunks = chunker.chunk("word " * 500)
    s = chunker.stats(chunks)
    assert s.n_chunks == len(chunks)
    assert s.total_chars > 0
    assert s.total_tokens_est > 0
    assert s.avg_chunk_tokens > 0


def test_chunks_are_compatible_with_coarse_filter():
    """Smoke: chunks produced by Chunker can feed straight into
    CoarseFilter (no shape mismatch). We don't actually run the filter
    here — just confirm types line up."""
    chunker = Chunker(tokens_per_chunk=20, respect_sentences=False)
    chunks = chunker.chunk("alpha " * 200)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.id and c.text for c in chunks)
