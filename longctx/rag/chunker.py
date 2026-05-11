"""Token-aware chunker for the 12M coarse-filter pipeline.

Stage 0 of the hierarchical-selector pipeline (see docs/PRD-12m-coarse-filter.md).

Splits raw text into ``Chunk`` objects with stable identity. Two modes:

  - **char proxy** (default): 4 chars / token heuristic. Dependency-free,
    deterministic, fast — fine for scoring. Used everywhere unless
    you opt in.
  - **HF tokenizer**: pass ``tokenizer=AutoTokenizer.from_pretrained(...)``
    for exact token boundaries. Slower (one tokenize pass) but lines up
    with the model's real tokenizer when that matters.

Stable Chunk ids = ``f"{i}_{sha1(text)[:10]}"`` so the embed cache keys
stay aligned across re-runs of the chunker on identical input.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from longctx.rag.coarse_filter import Chunk

# Conservative English-text approximation. Real tokenizers compress
# punctuation + common words harder, but for the *boundary* decision
# (where to slice) a 4-char average is the same heuristic the existing
# RetrievalPipeline.retrieve_chunked uses, so the two paths agree on
# chunk shape when the user mixes them.
DEFAULT_CHARS_PER_TOKEN = 4

# Sentence boundary heuristic. Walks back from the proposed split point
# to the nearest sentence terminator so chunks don't slice through a
# sentence mid-word.
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _stable_id(position: int, text: str) -> str:
    """``{position}_{first-10-of-sha1(text)}`` — deterministic, short, unique."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{position:06d}_{h}"


def _walk_to_sentence_end(text: str, end: int, slack: int) -> int:
    """Pull the proposed split point back to the nearest sentence-end
    within ``slack`` chars. Falls back to the proposed split when no
    sentence-end is reachable (long URL, code block, dense punctuation).
    """
    lo = max(end - slack, 0)
    window = text[lo:end]
    matches = list(_SENTENCE_END_RE.finditer(window))
    if not matches:
        return end
    last = matches[-1]
    return lo + last.end()


@dataclass
class ChunkerStats:
    """Diagnostic info per chunker run — for benchmarks + CLI demo."""
    n_chunks: int
    total_chars: int
    total_tokens_est: int
    avg_chunk_tokens: float


class Chunker:
    """Token-aware chunker with stable Chunk ids.

    Usage (char proxy, no deps beyond stdlib):
        chunker = Chunker(tokens_per_chunk=2048)
        chunks = chunker.chunk(big_text)

    Usage (true tokenizer):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        chunker = Chunker(tokens_per_chunk=2048, tokenizer=tok)
        chunks = chunker.chunk(big_text)

    Args:
        tokens_per_chunk: target chunk size in tokens. Spec default 2K.
        token_overlap: overlap between adjacent chunks, in tokens.
            Default ~10% of tokens_per_chunk.
        tokenizer: optional HF tokenizer for exact boundaries. ``None``
            means use the char-proxy heuristic.
        chars_per_token: only used in char-proxy mode.
        respect_sentences: in char-proxy mode, walk the split point back
            to the nearest sentence end within ``sentence_slack`` chars.
        sentence_slack: how far back the chunker is willing to walk to
            find a sentence end. Default ~5% of chunk char size.
    """

    def __init__(
        self,
        tokens_per_chunk: int = 2048,
        token_overlap: int | None = None,
        tokenizer=None,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        respect_sentences: bool = True,
        sentence_slack: int | None = None,
    ) -> None:
        if tokens_per_chunk < 1:
            raise ValueError("tokens_per_chunk must be >= 1")
        self.tokens_per_chunk = tokens_per_chunk
        self.token_overlap = (
            token_overlap if token_overlap is not None
            else max(tokens_per_chunk // 10, 1)
        )
        if self.token_overlap >= self.tokens_per_chunk:
            raise ValueError("token_overlap must be < tokens_per_chunk")
        self.tokenizer = tokenizer
        self.chars_per_token = chars_per_token
        self.respect_sentences = respect_sentences
        self.sentence_slack = (
            sentence_slack if sentence_slack is not None
            else max(int(tokens_per_chunk * chars_per_token * 0.05), 32)
        )

    # ------------------------------------------------------------------ public

    def chunk(self, text: str, *, base_position: int = 0) -> list[Chunk]:
        """Chunk one document. ``base_position`` lets callers offset chunk
        positions when concatenating multiple documents into one stream."""
        if not text:
            return []
        if self.tokenizer is not None:
            return self._chunk_with_tokenizer(text, base_position)
        return self._chunk_char_proxy(text, base_position)

    def chunk_many(self, texts: Iterable[str]) -> list[Chunk]:
        """Chunk a sequence of documents into a single flat list of Chunks
        with monotonic positions."""
        out: list[Chunk] = []
        for t in texts:
            out.extend(self.chunk(t, base_position=len(out)))
        return out

    def stats(self, chunks: Sequence[Chunk]) -> ChunkerStats:
        n = len(chunks)
        chars = sum(len(c.text) for c in chunks)
        tokens = sum(c.token_count for c in chunks)
        return ChunkerStats(
            n_chunks=n,
            total_chars=chars,
            total_tokens_est=tokens,
            avg_chunk_tokens=(tokens / n) if n else 0.0,
        )

    # ------------------------------------------------------------------ internals

    def _chunk_char_proxy(self, text: str, base_position: int) -> list[Chunk]:
        char_size = self.tokens_per_chunk * self.chars_per_token
        char_overlap = self.token_overlap * self.chars_per_token
        stride = max(char_size - char_overlap, 1)

        chunks: list[Chunk] = []
        n = len(text)
        start = 0
        position = base_position
        while start < n:
            end = min(start + char_size, n)
            if self.respect_sentences and end < n:
                end = _walk_to_sentence_end(text, end, self.sentence_slack)
            piece = text[start:end]
            chunks.append(Chunk(
                id=_stable_id(position, piece),
                text=piece,
                start_offset=start,
                end_offset=end,
                token_count=max(len(piece) // self.chars_per_token, 1),
            ))
            if end == n:
                break
            start = max(end - char_overlap, start + 1)
            position += 1
        return chunks

    def _chunk_with_tokenizer(self, text: str, base_position: int) -> list[Chunk]:
        # One tokenize pass, then slice token IDs into windows and decode
        # each window back to a contiguous text span. We keep
        # offset_mapping so chunk start/end are real char positions in
        # the source — not the decoded text, which can differ by
        # whitespace / special-token noise.
        try:
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        except TypeError:
            # Slow tokenizers don't support offset_mapping; fall back to
            # char-proxy boundaries with the (still useful) real token
            # count. Documented limitation.
            return self._chunk_char_proxy(text, base_position)

        offsets = enc["offset_mapping"]
        n_tok = len(offsets)
        if n_tok == 0:
            return []

        chunks: list[Chunk] = []
        start_tok = 0
        position = base_position
        while start_tok < n_tok:
            end_tok = min(start_tok + self.tokens_per_chunk, n_tok)
            char_start = offsets[start_tok][0]
            char_end = offsets[end_tok - 1][1]
            piece = text[char_start:char_end]
            chunks.append(Chunk(
                id=_stable_id(position, piece),
                text=piece,
                start_offset=char_start,
                end_offset=char_end,
                token_count=end_tok - start_tok,
            ))
            if end_tok == n_tok:
                break
            start_tok = max(end_tok - self.token_overlap, start_tok + 1)
            position += 1
        return chunks
