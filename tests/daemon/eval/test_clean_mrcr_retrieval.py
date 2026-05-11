"""Tests for ``longctx_daemon.eval.clean_mrcr_retrieval``.

The retrieval-only MRCR runner has three things worth covering with
unit tests:

1. **Per-sample mechanics** — corpus materialization writes one file
   per assistant message, the needle file is detectable from the
   bare answer, and the embedder + indexer + searcher pipeline
   actually produces a usable rank for the planted needle.

2. **Aggregation** — sample records roll up correctly into
   per-bin Recall@K via the ``_metrics`` helpers.

3. **CLI / report shape** — bin parsing, JSON serialization, and
   the markdown table render are pure transforms; check them
   against expected output.

We deliberately do NOT load ``openai/mrcr`` in unit tests. The
synthetic-fixture path inside ``_synthetic_fixture`` mirrors the real
schema so we can exercise the whole pipeline offline. The smoke test
that DOES talk to HF is in ``tests/daemon/eval/test_clean_mrcr_smoke.py``
(skipped unless ``LONGCTX_RUN_MRCR_SMOKE=1``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from longctx_daemon.eval import clean_mrcr_retrieval as cmr


# ────────────────────────────────────────────── deterministic embedder

class _NeedleEmbedder:
    """Tiny test embedder.

    Encodes "needle-keyworded" text into a 4-d vector with a 1.0 in
    the first axis when the text contains the bare needle marker
    (``NEEDLE_MARKER``), and a 1.0 in the second axis otherwise.
    Cosine between query (axis-0) and the needle chunk is high; cosine
    between query and noise chunks is low. Mirrors the
    ``_DeterministicEmbedder`` pattern in test_messy_queries.py.
    """

    fake_sha = "n" * 64
    model_name = "test-needle-embedder"
    dim = 4

    @property
    def max_seq_length(self) -> int:
        return 256

    @max_seq_length.setter
    def max_seq_length(self, _v: int) -> None:
        pass

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self, texts, convert_to_numpy=True,
        normalize_embeddings=True, **_,
    ):
        out = []
        for t in texts:
            tl = (t or "").lower()
            v = np.array([
                1.0 if NEEDLE_MARKER in tl else 0.0,
                1.0 if NEEDLE_MARKER not in tl else 0.0,
                0.5,
                0.0,
            ], dtype=np.float32)
            n = float(np.linalg.norm(v)) + 1e-9
            out.append(v / n)
        return np.stack(out)


NEEDLE_MARKER = "secret-mrcr-needle"


def _build_synthetic_sample(
    *,
    bin_label: str,
    needle_idx: int = 3,
    n_msgs: int = 8,
) -> dict:
    """Hand-built MRCR-shaped sample with a needle at ``needle_idx``."""
    rsp = "PFXabc"
    bare_answer = (
        f"This message contains the {NEEDLE_MARKER} planted answer text. "
        "Lorem ipsum filler so the chunker has substance to chew on."
    )
    messages: list[dict[str, str]] = []
    for k in range(n_msgs):
        messages.append({
            "role": "user",
            "content": f"User question {k}",
        })
        if k == needle_idx:
            messages.append({"role": "assistant", "content": bare_answer})
        else:
            messages.append({
                "role": "assistant",
                "content": f"Filler assistant reply number {k} unrelated content",
            })
    # Query contains the same marker so the deterministic embedder
    # aligns query↔needle chunk on axis 0. Real MRCR queries don't
    # echo the answer, but bge-small still learns "prepend X to the
    # Nth essay about Y" → essay-Y embeddings align via subword sim.
    # Test rig short-circuits that with a literal token.
    messages.append({
        "role": "user",
        "content": f"Prepend {rsp} to the {needle_idx + 1}th about {NEEDLE_MARKER}.",
    })
    n_chars = sum(len(m["content"]) for m in messages)
    return {
        "messages": messages,
        "answer": rsp + bare_answer,
        "random_string_to_prepend": rsp,
        "n_chars": n_chars,
        "n_needles": 8,
    }


# ────────────────────────────────────────────── small helpers

def test_bins_by_char_covers_expected_range():
    # Sanity: bin labels match what the runner exposes.
    for label in ("8K", "32K", "64K", "128K", "256K", "1M"):
        assert label in cmr.BINS_BY_CHAR


def test_default_k_values_includes_recall_at_8():
    # Header table reports R@1/3/5/8; if this drifts, render_table
    # would silently lose a column.
    assert 1 in cmr.DEFAULT_K_VALUES
    assert 8 in cmr.DEFAULT_K_VALUES


def test_parse_bins_handles_commas_and_spaces():
    assert cmr._parse_bins("8K,32K") == ("8K", "32K")
    assert cmr._parse_bins("8k 32k") == ("8K", "32K")
    assert cmr._parse_bins("8K, 32K , 64K") == ("8K", "32K", "64K")


def test_parse_bins_lowercases_to_uppercase():
    assert cmr._parse_bins("1m") == ("1M",)


def test_final_user_query_returns_last_user_message():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second-and-final"},
    ]
    assert cmr._final_user_query(msgs) == "second-and-final"


def test_final_user_query_empty_when_none_present():
    assert cmr._final_user_query([]) == ""
    assert cmr._final_user_query(
        [{"role": "assistant", "content": "x"}]
    ) == ""


def test_bare_answer_strips_prefix():
    assert cmr._bare_answer("PFXfoo bar", "PFX") == "foo bar"
    # Robust to mismatch — return as-is rather than mutating.
    assert cmr._bare_answer("foo", "PFX") == "foo"


# ────────────────────────────────────────────── corpus materialization

def test_materialize_corpus_writes_one_file_per_assistant(tmp_path: Path):
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1 content"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2 content"},
        {"role": "user", "content": "final question"},
    ]
    rels = cmr._materialize_corpus(msgs, tmp_path / "proj")
    assert len(rels) == 2
    assert (tmp_path / "proj" / rels[0]).read_text() == "a1 content"
    assert (tmp_path / "proj" / rels[1]).read_text() == "a2 content"
    # Sentinel for indexer's project-root walk.
    assert (tmp_path / "proj" / ".git").is_dir()


def test_materialize_corpus_skips_non_assistant_messages(tmp_path: Path):
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "system", "content": "s1"},
        {"role": "assistant", "content": "a1"},
    ]
    rels = cmr._materialize_corpus(msgs, tmp_path / "proj")
    assert len(rels) == 1


# ────────────────────────────────────────────── synthetic fixture path

def test_synthetic_fixture_produces_n_samples():
    samples = cmr._synthetic_fixture(needles=8, bin_label="8K", n_samples=3)
    assert len(samples) == 3
    for s in samples:
        assert s["n_needles"] == 8
        # Has the structural keys the runner expects.
        for key in ("messages", "answer", "random_string_to_prepend", "n_chars"):
            assert key in s


def test_synthetic_fixture_answer_starts_with_random_prefix():
    samples = cmr._synthetic_fixture(needles=8, bin_label="8K", n_samples=2)
    for s in samples:
        assert s["answer"].startswith(s["random_string_to_prepend"])


# ────────────────────────────────────────────── end-to-end runner

def test_run_with_embedder_returns_recall_report():
    """Full pipeline against a synthetic loader + deterministic
    embedder. Needle-marker embedding makes recall deterministic."""
    embedder = _NeedleEmbedder()

    def loader(*, needles, bin_label, n_samples):
        # Fixed needle position so we can hand-check the rank.
        return [
            _build_synthetic_sample(bin_label=bin_label, needle_idx=2)
            for _ in range(n_samples)
        ]

    report = cmr._run_with_embedder(
        needles=8,
        n_samples=2,
        bins=("8K",),
        embedder=embedder,
        embedder_id="test-needle-embedder",
        verbose=False,
        samples_per_bin_override=None,
        sample_loader=loader,
    )
    assert report.bins == ("8K",)
    # The embedder pumps the needle chunk to top-1 cleanly.
    assert report.recall_at_1["8K"] == 1.0
    assert report.recall_at_3["8K"] == 1.0
    assert report.recall_at_5["8K"] == 1.0
    assert report.recall_at_8["8K"] == 1.0
    assert len(report.sample_records) == 2


def test_run_with_embedder_records_sample_metadata():
    embedder = _NeedleEmbedder()

    def loader(*, needles, bin_label, n_samples):
        return [_build_synthetic_sample(bin_label=bin_label)]

    report = cmr._run_with_embedder(
        needles=8, n_samples=1, bins=("8K",),
        embedder=embedder, embedder_id="test-needle-embedder",
        verbose=False, samples_per_bin_override=None,
        sample_loader=loader,
    )
    rec = report.sample_records[0]
    assert rec["bin"] == "8K"
    assert rec["n_needles"] == 8
    # The deterministic embedder gives high cosine on the needle chunk
    # because the query contains the marker; floor should not trip.
    assert rec["needle_rank"] == 1
    # Top-1 dense cosine should be high (>= 0.9 in this rig — query
    # and needle chunk both project on axis 0).
    assert rec["top1_dense_cosine"] > 0.5


def test_run_with_embedder_handles_per_bin_overrides():
    embedder = _NeedleEmbedder()
    seen: dict[str, int] = {}

    def loader(*, needles, bin_label, n_samples):
        seen[bin_label] = n_samples
        return [_build_synthetic_sample(bin_label=bin_label) for _ in range(n_samples)]

    report = cmr._run_with_embedder(
        needles=8,
        n_samples=3,
        bins=("8K", "32K"),
        embedder=embedder,
        embedder_id="test-needle-embedder",
        verbose=False,
        samples_per_bin_override={"32K": 1},
        sample_loader=loader,
    )
    assert seen["8K"] == 3
    assert seen["32K"] == 1
    # 8K had 3 samples, 32K had 1 — sample_records has 4 total.
    assert len(report.sample_records) == 4


def test_run_with_embedder_skip_bin_when_override_zero():
    embedder = _NeedleEmbedder()

    def loader(*, needles, bin_label, n_samples):
        return [_build_synthetic_sample(bin_label=bin_label) for _ in range(n_samples)]

    report = cmr._run_with_embedder(
        needles=8,
        n_samples=1,
        bins=("8K", "1M"),
        embedder=embedder,
        embedder_id="test-needle-embedder",
        verbose=False,
        samples_per_bin_override={"1M": 0},  # skip the slow bin
        sample_loader=loader,
    )
    bins_run = {r["bin"] for r in report.sample_records}
    assert bins_run == {"8K"}


def test_run_invalid_bin_label_raises():
    with pytest.raises(ValueError, match="unknown bin labels"):
        # Passing a fake embedder via the public API is hard, but the
        # bin validation lives BEFORE embedder creation so we'd hit
        # the same branch via _run_with_embedder.
        cmr._run_with_embedder(
            needles=8, n_samples=1, bins=("BOGUS",),
            embedder=_NeedleEmbedder(),
            embedder_id="x",
            verbose=False, samples_per_bin_override=None,
            sample_loader=lambda **_: [],
        )


# ────────────────────────────────────────────── report rendering

def test_render_table_includes_header_and_per_bin_row():
    report = cmr.RecallReport(
        bins=("8K", "32K"),
        needles=8,
        n_samples_per_bin=2,
        recall_at_1={"8K": 1.0, "32K": 0.5},
        recall_at_3={"8K": 1.0, "32K": 1.0},
        recall_at_5={"8K": 1.0, "32K": 1.0},
        recall_at_8={"8K": 1.0, "32K": 1.0},
        sample_records=[
            {"bin": "8K"}, {"bin": "8K"},
            {"bin": "32K"}, {"bin": "32K"},
        ],
        embedder_id="x",
    )
    table = cmr.render_table(report)
    assert "| bin   | n  | R@1   | R@3   | R@5   | R@8   |" in table
    assert "| 8K    |  2 | 1.000" in table
    assert "| 32K   |  2 | 0.500" in table


def test_recall_report_serializes_to_json():
    report = cmr.RecallReport(
        bins=("8K",),
        needles=8,
        n_samples_per_bin=1,
        recall_at_1={"8K": 1.0},
        recall_at_3={"8K": 1.0},
        recall_at_5={"8K": 1.0},
        recall_at_8={"8K": 1.0},
        sample_records=[{"bin": "8K", "needle_rank": 1}],
        embedder_id="x",
    )
    from dataclasses import asdict
    payload = json.dumps(asdict(report))
    parsed = json.loads(payload)
    assert parsed["recall_at_1"]["8K"] == 1.0
    assert parsed["sample_records"][0]["needle_rank"] == 1


# ────────────────────────────────────────────── needle-rank probe

def test_find_needle_rank_matches_by_basename():
    class _Cit:
        def __init__(self, fp):
            self.file_path = fp

    class _Chunk:
        def __init__(self, fp, text):
            self.citation = _Cit(fp)
            self.text = text

    chunks = (
        _Chunk("proj/asst_00000.md", "different content"),
        _Chunk("proj/asst_00003.md", "the planted answer"),
        _Chunk("proj/asst_00007.md", "more"),
    )
    rank = cmr._find_needle_rank(
        chunks, bare_answer="the planted answer",
        needle_rel_path="asst_00003.md",
    )
    assert rank == 2


def test_find_needle_rank_falls_back_to_text_match():
    class _Cit:
        def __init__(self, fp):
            self.file_path = fp

    class _Chunk:
        def __init__(self, fp, text):
            self.citation = _Cit(fp)
            self.text = text

    # No needle_rel_path supplied; rely on text-prefix match.
    chunks = (
        _Chunk("proj/asst_00000.md", "blah"),
        _Chunk("proj/asst_00001.md", "this is the planted answer text more"),
    )
    rank = cmr._find_needle_rank(
        chunks, bare_answer="this is the planted answer",
        needle_rel_path=None,
    )
    assert rank == 2


def test_find_needle_rank_returns_none_on_miss():
    class _Cit:
        def __init__(self, fp):
            self.file_path = fp

    class _Chunk:
        def __init__(self, fp, text):
            self.citation = _Cit(fp)
            self.text = text

    chunks = (
        _Chunk("proj/asst_00000.md", "irrelevant"),
        _Chunk("proj/asst_00001.md", "also irrelevant"),
    )
    rank = cmr._find_needle_rank(
        chunks, bare_answer="never appears",
        needle_rel_path="asst_99999.md",
    )
    assert rank is None
