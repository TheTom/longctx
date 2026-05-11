"""End-to-end MRCR with LLM in loop — port of the AMD MI300X recipe
that scored 0.760 on the 8K bin (vs SubQ's claimed 0.659 at 1M).

Pipeline (per sample)::

    parse 8-needle MRCR conversation → embed each assistant message +
    final user query → top-K=8 cosine → re-sort top-K by original
    conversation position → pack into a compact system+user prompt →
    run through generator LLM → score = difflib.SequenceMatcher(
        model_output, target).ratio(), zeroed if prefix-pass fails.

Two retrieval backends compared on the same task:

    * "faiss"   — one-shot per-request brute-force cosine over the
                  candidates (the AMD baseline; faiss is overkill for
                  K=8 candidates so we use plain numpy).
    * "longctx" — longctx_daemon's persistent BM25+dense+RRF Searcher,
                  building an ephemeral chunk + embed store per sample.

The cell that matters: longctx + same generator + same scoring vs
the faiss baseline. If longctx ≥ faiss, the daemon's hybrid retrieval
isn't a regression vs the simple per-request recipe.

Why this is NOT what ``clean_mrcr_retrieval`` measures: that runner
reports retrieval Recall@K with no LLM in loop. It's a useful upper
bound (if retrieval misses, no generator recovers) but doesn't
produce a number directly comparable to SubQ's published 0.659. The
SubQ-comparable metric is end-to-end SequenceMatcher.ratio. That's
this module.

Usage::

    # Smoke (synthetic samples, stub generator — no LLM needed)
    python3 -m longctx_daemon.eval.mrcr_e2e --smoke

    # Real MRCR + faiss baseline + vllm-swift-served Qwen2.5-14B-1M
    python3 -m longctx_daemon.eval.mrcr_e2e \\
        --retriever faiss \\
        --base-url http://127.0.0.1:8080/v1 \\
        --model Qwen2.5-14B-Instruct-1M \\
        --bins 8K --samples 30

    # Same again with longctx-daemon retrieval for comparison
    python3 -m longctx_daemon.eval.mrcr_e2e \\
        --retriever longctx \\
        --base-url http://127.0.0.1:8080/v1 \\
        --model Qwen2.5-14B-Instruct-1M \\
        --bins 8K --samples 30

References:
    * obsidian://"Open Sparse Stack — RAG Beats SubQ"
    * obsidian://"Open Sparse Stack — Match SubQ at Home"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

# Bin definitions match clean_mrcr_retrieval.BINS_BY_CHAR so retrieval-
# only and end-to-end share a vocabulary.
BINS_BY_CHAR: dict[str, tuple[int, int]] = {
    "8K":   (16_000,    32_000),
    "16K":  (32_000,    64_000),
    "32K":  (64_000,   128_000),
    "64K":  (128_000,  256_000),
    "128K": (256_000,  512_000),
    "256K": (512_000, 1_000_000),
    "512K": (1_000_000, 2_000_000),
    "1M":   (2_000_000, 5_000_000),
}

DATASET_REPO = "openai/mrcr"
NEEDLE_SHARDS: dict[int, str] = {
    2: "2needle/2needle_0.parquet",
    4: "4needle/4needle_0.parquet",
    8: "8needle/8needle_0.parquet",
}

DEFAULT_TOP_K = 8
DEFAULT_BINS = ("8K", "32K", "64K")
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def _apply_auto_policy(args, sample: Optional["MRCRSample"] = None):
    """Apply the auto-policy router to ``args`` IN PLACE.

    Looks up a ``RetrievalPolicy`` for (corpus-size, query-shape) and
    overrides the relevant CLI args (``bm25_weight``, ``dense_weight``,
    ``chunk_tokens``, ``rerank``, ``rerank_pre_k``). Embedder /
    generator hints surface as warnings but are NOT applied — the CLI
    can't swap models on the fly without restart.

    Use ``sample`` (per-sample dispatch) when each MRCR sample has a
    different ``n_chars``. Without ``sample`` we default to a typical
    32K conversation (sane fallback for smoke runs).
    """
    from longctx_daemon.policy import (
        QueryShape, detect_query_shape, select_policy,
    )
    if sample is not None:
        size = sample.n_chars
        shape = detect_query_shape(sample.query)
    else:
        size = 64_000  # sane fallback ≈ 32K bin midpoint
        shape = QueryShape.UNKNOWN
    policy = select_policy(corpus_size_chars=size, query_shape=shape)
    args.bm25_weight = policy.bm25_weight
    args.dense_weight = policy.dense_weight
    if policy.chunk_tokens is not None:
        args.chunk_tokens = policy.chunk_tokens
    if policy.rerank_pre_k is not None:
        args.rerank = True
        args.rerank_pre_k = policy.rerank_pre_k
    return policy

# AMD-recipe defaults — see "Open Sparse Stack — RAG Beats SubQ".
DEFAULT_BASELINE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LONGCTX_EMBEDDER = "BAAI/bge-small-en-v1.5"

# Prompt template — small, explicit candidate labels, prefix-first
# instruction. Mirrors the AMD recipe; the verbatim prompt that scored
# 0.760 is documented inline.
_SYSTEM_PROMPT = (
    "You are a strict verbatim copier of one specific candidate.\n\n"
    "INPUTS: numbered candidate responses + a question. The question "
    "names the EXACT response wanted (often by its subject matter, "
    "format, or both — e.g. 'the poem about pencils', 'the email "
    "about Q3 sales', 'the limerick about cats').\n\n"
    "STEP 1 — SELECT. Find the ONE candidate whose body matches the "
    "question's EXACT subject AND format. If the question asks for "
    "a poem about presidents, do NOT pick a poem about leaders, "
    "monarchs, or rulers — only presidents. If the question asks "
    "for an email about pencils, do NOT pick a poem about pencils. "
    "Subject specificity OVERRIDES topical similarity.\n\n"
    "STEP 2 — EMIT. The question may instruct you to begin with a "
    "specific random alphanumeric prefix; if so, that prefix is the "
    "FIRST characters of your output, then the chosen candidate's "
    "full body verbatim. Otherwise, output begins with the chosen "
    "candidate's first character.\n\n"
    "STEP 3 — VERBATIM. Reproduce the chosen candidate's body byte-"
    "for-byte. Do not summarize, paraphrase, shorten, or 'clean up' "
    "formatting. No commentary, no candidate number, no quotes, no "
    "trailing notes.\n\n"
    "Worked example (illustrative):\n"
    "  Question: 'Reproduce the limerick about apples, beginning "
    "with XyZ7AbCdEf.'\n"
    "  Candidate 3 body: 'There once was an apple from Maine, ...'\n"
    "  Correct output: 'XyZ7AbCdEfThere once was an apple from "
    "Maine, ...'\n"
    "Note: prefix from the question + body from the candidate, "
    "concatenated. No spaces, no labels."
)

_USER_TEMPLATE = (
    "{candidates}\n\n"
    "Question: {query}\n\n"
    "Pick the candidate whose subject AND format EXACTLY match the "
    "question. Emit the prefix the question asks for (if any) "
    "concatenated with the chosen candidate's body verbatim. No "
    "commentary, no candidate label, no quotes."
)


# ─────────────────────────────────────────────────────────── parsing


@dataclass(frozen=True)
class MRCRSample:
    """A single parsed 8-needle MRCR conversation sample.

    ``candidates`` are the assistant message texts in the order they
    appeared in the original conversation. The retriever returns
    indices INTO this tuple. Re-sorting by original position before
    prompting the LLM means index ordering = conversation ordering,
    which matters for prefix-following.
    """
    bin: str
    n_chars: int
    candidates: tuple[str, ...]
    query: str
    target: str
    random_prefix: str = ""


def _parse_messages(raw_prompt: Any) -> tuple[list[dict[str, str]], str]:
    """Decode the MRCR ``prompt`` field (may be JSON string or list of
    dicts) into a flat ``[{"role", "content"}, ...]`` list. Returns
    ``(messages, raw_repr)`` where ``raw_repr`` is the original string
    (for debugging when parsing fails).
    """
    if isinstance(raw_prompt, str):
        try:
            messages = json.loads(raw_prompt)
        except json.JSONDecodeError as e:
            raise ValueError(f"prompt is not valid JSON: {e}") from e
    else:
        messages = raw_prompt
    if not isinstance(messages, list):
        raise ValueError(f"expected list of messages, got {type(messages)}")
    return messages, str(raw_prompt)


def _bin_for_chars(n_chars: int) -> Optional[str]:
    for label, (lo, hi) in BINS_BY_CHAR.items():
        if lo <= n_chars < hi:
            return label
    return None


def parse_sample(
    raw_sample: dict[str, Any], *, default_bin: str = "?",
) -> MRCRSample:
    """Convert a raw HF ``openai/mrcr`` row into a ``MRCRSample``.

    Each row has shape::

        {
          "prompt": <JSON-encoded message list — last message is the user query>,
          "answer": <expected verbatim assistant response>,
          "random_string_to_prepend": <random prefix the model must echo first>,
          "n_chars": <int>,
        }
    """
    messages, _ = _parse_messages(raw_sample["prompt"])
    if not messages:
        raise ValueError("empty prompt")
    # Final message is the user query.
    if messages[-1].get("role") != "user":
        raise ValueError(
            "expected final message to be user query; got "
            f"{messages[-1].get('role')}"
        )
    query = str(messages[-1].get("content", ""))
    candidates = tuple(
        str(m.get("content", ""))
        for m in messages[:-1]
        if m.get("role") == "assistant"
    )
    if not candidates:
        raise ValueError("no assistant messages in conversation")
    n_chars = int(raw_sample.get("n_chars", sum(len(c) for c in candidates)))
    bin_label = _bin_for_chars(n_chars) or default_bin
    return MRCRSample(
        bin=bin_label,
        n_chars=n_chars,
        candidates=candidates,
        query=query,
        target=str(raw_sample.get("answer", "")),
        random_prefix=str(raw_sample.get("random_string_to_prepend", "")),
    )


def synthetic_sample(
    *, n_candidates: int = 8, target_idx: int = 3, bin: str = "8K",
    random_prefix: str = "REPLY:",
) -> MRCRSample:
    """Build a synthetic MRCR-shaped sample for tests. Each candidate
    is an "essay about <topic_i>"; the target is the candidate at
    ``target_idx``; the query asks for that topic specifically.
    """
    topics = [
        "marbles", "tigers", "harpsichords", "asteroids",
        "ferns", "cinnamon", "binary trees", "mariachi",
        "weather balloons", "petrichor",
    ]
    if n_candidates > len(topics):
        topics = topics * (n_candidates // len(topics) + 1)
    topics = topics[:n_candidates]
    candidates = tuple(
        f"{random_prefix}\nThis is essay number {i + 1} about {topic}. "
        f"It explores the {topic} from a unique perspective and "
        f"contains identifying detail-{i:03d}-{topic} for retrieval."
        for i, topic in enumerate(topics)
    )
    target = candidates[target_idx]
    query = (
        f"Reproduce the essay about {topics[target_idx]}. "
        f"Begin with {random_prefix}"
    )
    return MRCRSample(
        bin=bin, n_chars=sum(len(c) for c in candidates),
        candidates=candidates, query=query, target=target,
        random_prefix=random_prefix,
    )


# ─────────────────────────────────────────────────────────── scoring


@dataclass(frozen=True)
class SampleResult:
    """End-to-end result for one sample."""
    bin: str
    n_chars: int
    score: float
    prefix_pass: bool
    retrieved_indices: tuple[int, ...]
    target_in_topk: bool
    target_rank: Optional[int]
    retrieval_ms: float
    generation_ms: float
    model_output: str
    target: str


def score_output(
    model_output: str, target: str, random_prefix: str = "",
) -> tuple[float, bool]:
    """SequenceMatcher ratio between model output and the expected
    answer. If a non-empty ``random_prefix`` is set, prefix-pass is
    required: the model's output MUST start with the prefix exactly,
    otherwise the score is zeroed (matches MRCR v2 protocol — fails
    fast on models that wander off-format).
    """
    prefix_pass = (not random_prefix) or model_output.startswith(random_prefix)
    if not prefix_pass:
        return 0.0, False
    return float(SequenceMatcher(None, model_output, target).ratio()), True


def _target_index(sample: MRCRSample) -> Optional[int]:
    """Locate the target candidate's index in ``sample.candidates``.

    The MRCR dataset stores ``target`` as ``random_prefix + body``,
    while the ``candidates`` (extracted from prior assistant messages
    in the conversation) only contain ``body`` — the prefix is what
    the user *asks* the model to prepend at output time, not what was
    in the original response.

    Strategy:
      1. Strip the random prefix from ``target`` to get the body.
      2. First-200-char body match — exact, robust to short tail
         differences. The body is unique per sample (by construction)
         so the first 200 chars are a near-perfect anchor.
      3. Verbatim full-string fallback for the no-prefix case.

    Returns None if no candidate looks like the target.
    """
    target = sample.target
    if not target:
        return None
    prefix = sample.random_prefix
    body = target[len(prefix):] if prefix and target.startswith(prefix) else target
    if not body:
        return None
    head = body[:200]
    head_short = body[:80]
    for i, c in enumerate(sample.candidates):
        if c.startswith(head) or head_short in c[:300]:
            return i
    # Last-resort: full verbatim match on the (no-prefix) raw target.
    for i, c in enumerate(sample.candidates):
        if c == target or c == body:
            return i
    return None


# ─────────────────────────────────────────────────────────── retrievers


class Retriever(Protocol):
    """Adapter interface for the two retrieval backends.

    Both consume an ``MRCRSample`` and return at most ``k`` indices
    into ``sample.candidates``, ranked by relevance. The pipeline
    re-sorts the result by original-position before prompting.
    """
    def retrieve(self, sample: MRCRSample, k: int) -> tuple[int, ...]: ...

    def close(self) -> None: ...


class FaissRetriever:
    """One-shot per-request retrieval — the AMD-recipe baseline.

    The name is historical: at K=8 over <100 candidates plain numpy
    cosine is faster than building a faiss index, so we skip faiss.
    But the *behavior* is the AMD recipe verbatim — embed each
    candidate, embed the query, top-K cosine, return positional
    indices.
    """

    def __init__(
        self,
        *,
        embedder_model: str = DEFAULT_BASELINE_EMBEDDER,
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self._model_name = embedder_model
        self._encoder = SentenceTransformer(embedder_model, device=device)

    @property
    def embedder_model(self) -> str:
        return self._model_name

    def retrieve(self, sample: MRCRSample, k: int) -> tuple[int, ...]:
        import numpy as np
        embs = self._encoder.encode(
            list(sample.candidates) + [sample.query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cand_embs = embs[:-1]
        query_emb = embs[-1]
        sims = cand_embs @ query_emb
        order = list(np.argsort(-sims)[: max(k, 1)])
        return tuple(int(i) for i in order)

    def close(self) -> None:
        self._encoder = None  # type: ignore[assignment]


class LongctxRetriever:
    """longctx_daemon Searcher backed by a per-sample SqliteChunkStore +
    MemmapEmbedStore. Each candidate is written as a separate "file"
    so the chunker may or may not split it — chunking behavior is the
    knob that mirrors the AMD experiments' "hierarchical chunking"
    win at 64K.

    To compare backends fairly, the result is mapped back to candidate
    indices: top-K chunks → unique parent message indices, ranked by
    first-appearance in the chunk ranking, capped at K.
    """

    def __init__(
        self,
        *,
        embedder_model: str = DEFAULT_LONGCTX_EMBEDDER,
        chunk_tokens: Optional[int] = None,
        device: Optional[str] = None,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self._model_name = embedder_model
        self._encoder = SentenceTransformer(embedder_model, device=device)
        # 0 / None → effectively-unbounded chunk size = one chunk per
        # candidate (mirrors faiss baseline). Otherwise the chunker
        # splits candidates into N-token sub-chunks.
        self._chunk_tokens = chunk_tokens or 1_000_000
        # Weights for the daemon's RRF fusion. (1.0, 1.0) = production
        # default. (0.0, 1.0) = dense-only — kills the BM25 channel,
        # used to confirm BM25 is hurting MRCR-style essay
        # disambiguation (lexical overlap between same-topic essays
        # promotes wrong candidates).
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    @property
    def embedder_model(self) -> str:
        return self._model_name

    def retrieve(self, sample: MRCRSample, k: int) -> tuple[int, ...]:
        # Lazy import — keep the test surface small for callers that
        # only need the FaissRetriever or scoring helpers.
        from longctx.rag.chunker import Chunker
        from longctx_daemon.indexer import Indexer, IndexerConfig
        from longctx_daemon.searcher import Searcher, SearcherConfig
        from longctx_daemon.storage.memmap_store import MemmapEmbedStore
        from longctx_daemon.storage.sqlite_store import SqliteChunkStore

        with tempfile.TemporaryDirectory(prefix="lctx-mrcr-e2e-") as tmp:
            tmp_path = Path(tmp)
            corpus = tmp_path / "corpus"
            corpus.mkdir()
            (corpus / ".git").mkdir()
            # Write each candidate as message_NNN.txt — flat layout,
            # path-stable index recovery.
            for i, text in enumerate(sample.candidates):
                (corpus / f"message_{i:04d}.txt").write_text(text)

            # NOTE: the SentenceTransformer the retriever already loaded
            # is reused as the embedder for the ephemeral indexer.
            sha = _stable_sha(self._model_name)
            chunk_store = SqliteChunkStore(tmp_path / "index.db")
            embed_store = MemmapEmbedStore(
                tmp_path / "embeddings",
                model_name=self._model_name,
                model_sha256=sha,
                dim=self._encoder.get_sentence_embedding_dimension(),
            )
            try:
                wrapped = _SentenceTransformerEmbedder(
                    self._encoder, model_name=self._model_name,
                    sha=sha,
                )
                chunker = Chunker(
                    tokens_per_chunk=self._chunk_tokens,
                    respect_sentences=False,
                )
                indexer = Indexer(
                    chunk_store=chunk_store, embed_store=embed_store,
                    embedder=wrapped, chunker=chunker,
                    config=IndexerConfig(),
                )
                indexer.add_project("mrcr-sample", corpus)
                indexer.full_scan("mrcr-sample")
                # relevance_floor=0 because MRCR samples are by
                # construction "in-topic" — disabling the floor lets
                # the recipe match the faiss baseline's behavior.
                searcher = Searcher(
                    chunk_store=chunk_store, embed_store=embed_store,
                    embedder=wrapped,
                    config=SearcherConfig(
                        relevance_floor=0.0,
                        bm25_weight=self._bm25_weight,
                        dense_weight=self._dense_weight,
                    ),
                )
                # Pull a generous max_results; we'll dedupe to K
                # message indices below.
                result = searcher.search(
                    query=sample.query, cwd=str(corpus),
                    max_results=max(k * 4, 16),
                    max_tokens=1_000_000,  # budget unconstrained
                )
            finally:
                chunk_store.close()
                embed_store.close()

        # chunks are returned ranked; map each chunk's file_path back
        # to its candidate index. Dedupe preserves first-appearance.
        seen: list[int] = []
        for chunk in result.chunks:
            idx = _candidate_index_from_path(chunk.citation.file_path)
            if idx is None or idx in seen:
                continue
            seen.append(idx)
            if len(seen) >= k:
                break
        return tuple(seen)

    def close(self) -> None:
        self._encoder = None  # type: ignore[assignment]


_PATH_RE = re.compile(r"message_(\d+)\.txt$")


def _rerank_indices(
    *, sample: MRCRSample, wide_indices: tuple[int, ...], k: int,
    reranker: Any,
) -> tuple[int, ...]:
    """Cross-encoder rerank stage. Scores ``(query, candidate body)``
    pairs with the reranker model; sorts wide_indices by score descending;
    returns the top-k. Empty input → empty output."""
    if not wide_indices:
        return ()
    pairs = [
        [sample.query, sample.candidates[i][:4000]]
        for i in wide_indices
    ]
    scores = reranker.predict(pairs, show_progress_bar=False)
    # Sort wide_indices by score desc, keep top-k.
    ranked = sorted(
        zip(wide_indices, scores), key=lambda p: -float(p[1]),
    )
    return tuple(int(i) for i, _ in ranked[:k])


def _build_reranker(model_name: str) -> Any:
    """Lazy-load a cross-encoder reranker (sentence_transformers
    CrossEncoder). Cached if already loaded for the same model name."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _candidate_index_from_path(path: str) -> Optional[int]:
    m = _PATH_RE.search(path)
    if m is None:
        return None
    return int(m.group(1))


def _stable_sha(model_name: str) -> str:
    """Produce a deterministic 64-char hex string from the model name.
    We don't need a real SHA — the chunk store just wants ANY stable
    identifier so embed-store version-stamping works. The real SHA
    would be the safetensors digest, which is overkill here."""
    import hashlib
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()


class _SentenceTransformerEmbedder:
    """Adapter so an already-loaded SentenceTransformer satisfies the
    longctx_daemon Embedder duck-type. Caches dim + name + sha so the
    indexer doesn't reload.
    """

    def __init__(self, model: Any, *, model_name: str, sha: str) -> None:
        self._model = model
        self.model_name = model_name
        self.fake_sha = sha
        self.model_sha256 = sha
        try:
            dim = int(model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001 — best-effort
            dim = int(model.encode(["x"], normalize_embeddings=True).shape[-1])
        self.dim = dim
        self._max_seq_length: Optional[int] = getattr(
            model, "max_seq_length", None,
        )

    @property
    def max_seq_length(self) -> Optional[int]:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self._max_seq_length = value
        try:
            self._model.max_seq_length = value
        except Exception:  # noqa: BLE001 — best-effort
            pass

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self, texts, *, convert_to_numpy: bool = True,
        normalize_embeddings: bool = True, **kw,
    ):
        # Drop show_progress_bar from kw if a caller forwarded one —
        # we always silence progress for eval runs.
        kw.pop("show_progress_bar", None)
        return self._model.encode(
            list(texts),
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            **kw,
        )


# ─────────────────────────────────────────────────────────── generators


class Generator(Protocol):
    """Adapter interface for chat-completion endpoints."""
    def complete(
        self, *, system: str, user: str,
        max_tokens: int, temperature: float,
    ) -> str: ...


class StubGenerator:
    """Returns the target verbatim. For tests + plumbing smoke runs.

    Optionally simulates failure modes by configuration: ``broken_prefix``
    drops the random_prefix to exercise prefix-pass=False zeroing.
    """

    def __init__(self, *, broken_prefix: bool = False) -> None:
        self._broken = broken_prefix

    def complete(
        self, *, system: str, user: str,  # noqa: ARG002 — Protocol shape
        max_tokens: int, temperature: float,  # noqa: ARG002
    ) -> str:
        # The pipeline packs the target as one of the candidates and
        # asks the model to reproduce it; for a stub we just look at
        # the first candidate label that matches.
        # Stub must remain decoupled from the real prompt structure —
        # honest tests provide a known target via SampleHarness.
        return _STUB_OUTPUT_OVERRIDE.get(id(self), "")


_STUB_OUTPUT_OVERRIDE: dict[int, str] = {}


def _stub_set_output(stub: StubGenerator, output: str) -> None:
    """Test helper: program a stub to return ``output`` for its next
    completion call. Lives at module level so tests don't have to
    monkey-patch through layers."""
    _STUB_OUTPUT_OVERRIDE[id(stub)] = output


class OpenAICompatGenerator:
    """HTTP client for any OpenAI-compatible chat-completions endpoint.

    Points at vllm-swift on M5 Max by default (port 8080). Override
    via ``base_url`` to target a remote vLLM, Anthropic Messages
    bridge, or any other compatible endpoint.

    Stdlib-only — no ``requests`` dependency. Deliberately minimal so
    the eval doesn't drag a heavy HTTP stack into the daemon."""

    def __init__(
        self, *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen2.5-14B-Instruct-1M",
        api_key: str = "EMPTY",
        timeout_s: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_s

    def complete(
        self, *, system: str, user: str,
        max_tokens: int, temperature: float,
    ) -> str:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        # Reasoning-mode-on-by-default models (Qwen3.x family) eat the
        # token budget on `<think>...</think>` blocks and emit empty
        # ``content``. Detect and disable thinking via the chat-
        # template kwargs the OpenAI-compat layer threads through.
        # Pattern catches `Qwen3` / `Qwen3.5` / `Qwen3.6` etc but NOT
        # `Qwen2.5`, `Qwen2`. Case-insensitive.
        is_thinking_default_on = bool(
            re.search(r"\bqwen3(\.\d+)?\b", self._model, re.IGNORECASE)
        )

        def _post(messages: list[dict]) -> str:
            body = {
                "model": self._model, "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": False,
            }
            if is_thinking_default_on:
                # mlx_lm.server passes ``chat_template_kwargs`` through
                # to the tokenizer's chat-template render call. Setting
                # ``enable_thinking=False`` skips the
                # ``<think>...</think>`` phase entirely, so the model
                # emits the answer directly into ``content``. This is
                # the official Qwen3 mechanism for non-reasoning use.
                body["chat_template_kwargs"] = {"enable_thinking": False}
            req = Request(
                f"{self._base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            with urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError(
                    f"generator returned no choices: {payload!r}"
                )
            msg = choices[0].get("message", {}) or {}
            # Reasoning models (Qwen3.x, DeepSeek-R1) emit the answer
            # in ``reasoning_content`` and leave ``content`` null or
            # absent. Try content first, fall back so we don't lose
            # the answer.
            content = msg.get("content")
            if content is None or content == "":
                content = msg.get("reasoning_content")
            return str(content or "")

        # First try: standard system+user. Most models accept this.
        try:
            return _post([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except HTTPError as e:
            err_body = e.read().decode("utf-8", "replace")
            # Mistral / Gemma / certain Llama checkpoints reject the
            # system role or strict-alternation. Fold system → first
            # user message and retry once. Pattern catches the most
            # common errors:
            #   * "Conversation roles must alternate user/assistant…" (Mistral)
            #   * "System role not supported" (some Gemma chat templates)
            needs_fold = (
                "roles must alternate" in err_body
                or "system role" in err_body.lower()
                or "system" in err_body.lower() and "support" in err_body.lower()
            )
            if needs_fold:
                merged = f"{system}\n\n---\n\n{user}"
                try:
                    return _post([{"role": "user", "content": merged}])
                except HTTPError as e2:
                    raise RuntimeError(
                        f"generator HTTP {e2.code} (after system-fold "
                        f"retry): {e2.read().decode('utf-8', 'replace')}"
                    ) from e2
            raise RuntimeError(
                f"generator HTTP {e.code}: {err_body}"
            ) from e
        except URLError as e:
            raise RuntimeError(f"generator unreachable: {e.reason}") from e


# ─────────────────────────────────────────────────────────── pipeline


def build_prompt(
    sample: MRCRSample, retrieved_indices: Sequence[int],
) -> tuple[str, str]:
    """Pack the system + user prompt for ``sample``, restricted to the
    retrieved candidates re-sorted by original conversation position.
    Returns ``(system, user)``. Mirrors the AMD recipe explicitly.
    """
    ordered = sorted(set(retrieved_indices))
    candidates_text = "\n\n".join(
        f"Candidate {i + 1} (originally message {orig + 1}):\n{sample.candidates[orig]}"
        for i, orig in enumerate(ordered)
    )
    return _SYSTEM_PROMPT, _USER_TEMPLATE.format(
        candidates=candidates_text, query=sample.query,
    )


def _evaluate_sample(
    sample: MRCRSample, *,
    retriever: Retriever,
    generator: Generator,
    top_k: int,
    temperature: float,
    max_output_tokens: int,
    reranker: Optional[Any] = None,
    rerank_pre_k: int = 32,
) -> SampleResult:
    t0 = time.perf_counter()
    if reranker is None:
        indices = retriever.retrieve(sample, top_k)
    else:
        # Two-stage: pull a wider candidate pool from retrieval,
        # cross-encoder rerank to the final top_k. This recovers the
        # retrieval-recall losses we see at very long bins (1M+) where
        # 100+ candidates compete and the right one falls out of a
        # one-shot top-K=8 cosine cut.
        pre_k = max(rerank_pre_k, top_k)
        wide = retriever.retrieve(sample, pre_k)
        indices = _rerank_indices(
            sample=sample, wide_indices=wide, k=top_k, reranker=reranker,
        )
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    target_idx = _target_index(sample)
    target_rank: Optional[int] = None
    if target_idx is not None and target_idx in indices:
        target_rank = list(indices).index(target_idx) + 1
    target_in_topk = target_idx is not None and target_idx in indices

    system, user = build_prompt(sample, indices)
    t0 = time.perf_counter()
    output = generator.complete(
        system=system, user=user,
        max_tokens=max_output_tokens, temperature=temperature,
    )
    generation_ms = (time.perf_counter() - t0) * 1000.0

    score, prefix_pass = score_output(
        output, sample.target, sample.random_prefix,
    )
    return SampleResult(
        bin=sample.bin, n_chars=sample.n_chars,
        score=score, prefix_pass=prefix_pass,
        retrieved_indices=tuple(indices),
        target_in_topk=target_in_topk, target_rank=target_rank,
        retrieval_ms=retrieval_ms, generation_ms=generation_ms,
        model_output=output, target=sample.target,
    )


@dataclass
class BinSummary:
    bin: str
    n_samples: int
    avg_score: float
    prefix_pass_rate: float
    target_in_topk_rate: float
    avg_retrieval_ms: float
    avg_generation_ms: float


@dataclass
class E2EReport:
    config: dict[str, Any]
    summaries: dict[str, BinSummary]
    samples: list[SampleResult] = field(default_factory=list)


def _summarize(samples: Sequence[SampleResult]) -> dict[str, BinSummary]:
    by_bin: dict[str, list[SampleResult]] = {}
    for s in samples:
        by_bin.setdefault(s.bin, []).append(s)
    out: dict[str, BinSummary] = {}
    for bin_label, rows in by_bin.items():
        n = len(rows)
        out[bin_label] = BinSummary(
            bin=bin_label, n_samples=n,
            avg_score=sum(r.score for r in rows) / n if n else 0.0,
            prefix_pass_rate=(
                sum(1 for r in rows if r.prefix_pass) / n if n else 0.0
            ),
            target_in_topk_rate=(
                sum(1 for r in rows if r.target_in_topk) / n if n else 0.0
            ),
            avg_retrieval_ms=(
                sum(r.retrieval_ms for r in rows) / n if n else 0.0
            ),
            avg_generation_ms=(
                sum(r.generation_ms for r in rows) / n if n else 0.0
            ),
        )
    return out


def _samples_per_bin_normalized(
    raw: Any, bins: Sequence[str],
) -> dict[str, int]:
    if isinstance(raw, dict):
        return {b: int(raw.get(b, 0)) for b in bins}
    return {b: int(raw) for b in bins}


def run(
    *,
    retriever: Retriever,
    generator: Generator,
    bins: Sequence[str] = DEFAULT_BINS,
    samples_per_bin: Any = 30,
    needles: int = 8,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
    samples_iter: Optional[Sequence[MRCRSample]] = None,
    reranker: Optional[Any] = None,
    rerank_pre_k: int = 32,
) -> E2EReport:
    """Run the e2e MRCR pipeline.

    ``samples_iter`` overrides MRCR loading — pass synthetic samples
    for plumbing tests without hitting HF or the real dataset.

    For real runs leave ``samples_iter=None`` and the runner pulls
    ``openai/mrcr`` from HF, filters to ``bins`` and ``samples_per_bin``,
    feeds each through retriever + generator + scorer.
    """
    sp_bin = _samples_per_bin_normalized(samples_per_bin, bins)
    if samples_iter is None:
        samples = _load_mrcr_samples(
            bins=bins, samples_per_bin=sp_bin, needles=needles,
        )
    else:
        samples = list(samples_iter)

    rows: list[SampleResult] = []
    for s in samples:
        rows.append(_evaluate_sample(
            s, retriever=retriever, generator=generator,
            top_k=top_k, temperature=temperature,
            max_output_tokens=max_output_tokens,
            reranker=reranker, rerank_pre_k=rerank_pre_k,
        ))

    return E2EReport(
        config={
            "bins": list(bins),
            "samples_per_bin": sp_bin,
            "needles": needles,
            "top_k": top_k,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "retriever": type(retriever).__name__,
            "retriever_embedder": getattr(retriever, "embedder_model", None),
            "generator": type(generator).__name__,
            "generator_model": getattr(generator, "_model", None),
        },
        summaries=_summarize(rows),
        samples=rows,
    )


def _load_mrcr_samples(
    *, bins: Sequence[str], samples_per_bin: dict[str, int],
    needles: int,
) -> list[MRCRSample]:
    """Stream the openai/mrcr shard and pick samples up to the bin
    quotas. Returns a list of parsed samples.
    """
    if needles not in NEEDLE_SHARDS:
        raise ValueError(
            f"needles must be one of {sorted(NEEDLE_SHARDS)}; got {needles}"
        )
    from datasets import load_dataset
    shard = NEEDLE_SHARDS[needles]
    ds = load_dataset(
        DATASET_REPO,
        data_files={"train": shard},
        split="train",
        streaming=True,
    )
    bin_set = set(bins)
    quota = {b: int(samples_per_bin.get(b, 0)) for b in bins}
    out: list[MRCRSample] = []
    for raw in ds:
        if all(quota[b] <= 0 for b in bins):
            break
        try:
            sample = parse_sample(raw)
        except Exception:  # noqa: BLE001 — skip malformed
            continue
        if sample.bin not in bin_set:
            continue
        if quota.get(sample.bin, 0) <= 0:
            continue
        out.append(sample)
        quota[sample.bin] -= 1
    return out


# ─────────────────────────────────────────────────────────── CLI


def render_table(report: E2EReport) -> str:
    head = (
        "| bin   |   n | score   | prefix | tgt@topK | retr_ms | gen_ms |\n"
        "|-------|----:|--------:|-------:|---------:|--------:|-------:|\n"
    )
    rows = [
        f"| {s.bin:<5s} | {s.n_samples:>3d} | {s.avg_score:>7.4f} | "
        f"{s.prefix_pass_rate:>6.0%} | {s.target_in_topk_rate:>8.0%} | "
        f"{s.avg_retrieval_ms:>7.1f} | {s.avg_generation_ms:>6.1f} |"
        for s in sorted(report.summaries.values(), key=lambda b: b.bin)
    ]
    return head + "\n".join(rows)


def _build_retriever(
    name: str, *, embedder: Optional[str], chunk_tokens: Optional[int],
    bm25_weight: float = 1.0, dense_weight: float = 1.0,
) -> Retriever:
    if name == "faiss":
        return FaissRetriever(
            embedder_model=embedder or DEFAULT_BASELINE_EMBEDDER,
        )
    if name == "longctx":
        return LongctxRetriever(
            embedder_model=embedder or DEFAULT_LONGCTX_EMBEDDER,
            chunk_tokens=chunk_tokens,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
    raise ValueError(f"unknown retriever: {name!r}")


def _build_generator(
    *, smoke: bool, base_url: str, model: str, api_key: str,
) -> Generator:
    if smoke:
        return StubGenerator()
    return OpenAICompatGenerator(
        base_url=base_url, model=model, api_key=api_key,
    )


def _smoke_samples(n: int = 4) -> list[MRCRSample]:
    """Tiny synthetic suite used by --smoke."""
    out: list[MRCRSample] = []
    for i in range(n):
        out.append(synthetic_sample(target_idx=i % 8, bin="8K"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrcr_e2e",
        description="End-to-end MRCR with LLM in loop (AMD recipe port)",
    )
    parser.add_argument("--retriever", choices=("faiss", "longctx"),
                        default="faiss")
    parser.add_argument("--embedder", default=None,
                        help="Override embedder. Defaults: faiss → "
                             f"{DEFAULT_BASELINE_EMBEDDER}, longctx → "
                             f"{DEFAULT_LONGCTX_EMBEDDER}")
    parser.add_argument("--chunk-tokens", type=int, default=None,
                        help="longctx-only: tokens per chunk; default = "
                             "unbounded (one chunk per candidate)")
    parser.add_argument("--bm25-weight", type=float, default=1.0,
                        help="longctx-only: RRF BM25 weight (0 = disable)")
    parser.add_argument("--dense-weight", type=float, default=1.0,
                        help="longctx-only: RRF dense weight (0 = disable)")
    parser.add_argument("--auto-policy", action="store_true",
                        help="Per-sample, look up the (corpus-size, "
                             "query-shape) policy in longctx_daemon.policy "
                             "and override --bm25-weight, --dense-weight, "
                             "--chunk-tokens, --rerank, and --rerank-pre-k. "
                             "Embedder/generator hints surface as warnings "
                             "but are NOT swapped (would require restart).")
    parser.add_argument("--bins", default=",".join(DEFAULT_BINS),
                        help=f"comma-separated bin labels from "
                             f"{sorted(BINS_BY_CHAR)}")
    parser.add_argument("--samples", type=int, default=30,
                        help="samples per bin (uniform). For per-bin "
                             "control pass --samples-json.")
    parser.add_argument("--samples-json", default=None,
                        help="JSON dict mapping bin label → sample count")
    parser.add_argument("--needles", type=int, default=8,
                        choices=sorted(NEEDLE_SHARDS))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank", action="store_true",
                        help="Two-stage retrieval: pull rerank_pre_k "
                             "candidates, cross-encoder rerank to top-K. "
                             "Recovers retrieval recall at long bins (1M+).")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--rerank-pre-k", type=int, default=32,
                        help="Candidate pool size before cross-encoder "
                             "rerank (must be >= top_k)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8080/v1",
        help="OpenAI-compat chat-completions URL (vllm-swift default)",
    )
    parser.add_argument("--model", default="Qwen2.5-14B-Instruct-1M")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--smoke", action="store_true",
                        help="Synthetic samples + stub generator. No "
                             "network, no model load — pure plumbing test.")
    parser.add_argument("--json", action="store_true",
                        help="Print full report JSON instead of summary table")
    args = parser.parse_args(argv)

    bins = tuple(b.strip() for b in args.bins.split(",") if b.strip())
    if args.samples_json:
        samples_per_bin: Any = json.loads(args.samples_json)
    else:
        samples_per_bin = args.samples

    if args.auto_policy:
        # Pick a representative size by taking the midpoint of the
        # smallest bin in --bins (conservative — biases toward
        # symbolic-friendly hybrid retrieval if multiple bins span
        # short and long).
        from longctx_daemon.policy import (
            QueryShape, select_policy,
        )
        smallest_bin = min(
            (b for b in bins if b in BINS_BY_CHAR),
            key=lambda b: BINS_BY_CHAR[b][0],
            default="32K",
        )
        lo, hi = BINS_BY_CHAR[smallest_bin]
        rep_size = (lo + hi) // 2
        # MRCR is prose-disambiguation by construction.
        pol = select_policy(
            corpus_size_chars=rep_size, query_shape=QueryShape.PROSE,
        )
        print(
            f"[auto-policy] bin={smallest_bin} (chars≈{rep_size}) "
            f"shape=PROSE → bm25={pol.bm25_weight} "
            f"dense={pol.dense_weight} chunk_tokens={pol.chunk_tokens} "
            f"rerank_pre_k={pol.rerank_pre_k}",
            file=sys.stderr,
        )
        if pol.embedder_hint:
            print(
                f"[auto-policy] embedder_hint: {pol.embedder_hint} "
                f"(NOT swapped automatically — set --embedder to apply)",
                file=sys.stderr,
            )
        if pol.rationale:
            print(f"[auto-policy] rationale: {pol.rationale}",
                  file=sys.stderr)
        args.bm25_weight = pol.bm25_weight
        args.dense_weight = pol.dense_weight
        if pol.chunk_tokens is not None:
            args.chunk_tokens = pol.chunk_tokens
        if pol.rerank_pre_k is not None:
            args.rerank = True
            args.rerank_pre_k = pol.rerank_pre_k

    retriever = _build_retriever(
        args.retriever, embedder=args.embedder, chunk_tokens=args.chunk_tokens,
        bm25_weight=args.bm25_weight, dense_weight=args.dense_weight,
    )
    reranker = _build_reranker(args.reranker_model) if args.rerank else None
    generator = _build_generator(
        smoke=args.smoke, base_url=args.base_url,
        model=args.model, api_key=args.api_key,
    )

    if args.smoke:
        # Smoke mode: ignore --samples / --bins; use the tiny synthetic
        # suite. The stub generator is programmed per-sample to echo
        # the target so plumbing returns score=1.0 when wired correctly.
        smoke_samples = _smoke_samples()
        # Program the stub: for each sample we evaluate, set the next
        # output to the sample's target. Done inside the loop via a
        # one-shot generator wrapper.
        rows: list[SampleResult] = []
        for s in smoke_samples:
            _stub_set_output(generator, s.target)  # type: ignore[arg-type]
            rows.append(_evaluate_sample(
                s, retriever=retriever, generator=generator,
                top_k=args.top_k, temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            ))
        report = E2EReport(
            config={"smoke": True, "retriever": type(retriever).__name__,
                    "samples": len(smoke_samples)},
            summaries=_summarize(rows),
            samples=rows,
        )
    else:
        report = run(
            retriever=retriever, generator=generator,
            bins=bins, samples_per_bin=samples_per_bin,
            needles=args.needles, top_k=args.top_k,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reranker=reranker, rerank_pre_k=args.rerank_pre_k,
        )

    retriever.close()

    if args.json:
        print(json.dumps(
            {
                "config": report.config,
                "summaries": {k: asdict(v) for k, v in report.summaries.items()},
                "samples": [asdict(s) for s in report.samples],
            },
            indent=2,
        ))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
