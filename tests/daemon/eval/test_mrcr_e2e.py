"""Unit tests for the e2e MRCR rig.

These cover the bits that don't require a real embedder or LLM:
parsing, scoring, prompt construction, summarization, and a full
plumbing round-trip with a stub retriever + stub generator.

Heavier tests (real SentenceTransformer + ephemeral SqliteChunkStore)
live in test_mrcr_e2e_live.py — skipped unless explicitly opted in,
since they hit disk and may download HF models.
"""
from __future__ import annotations

import json

import pytest

from longctx_daemon.eval import mrcr_e2e as mod


# ───────────────────────────────────────────────────── parsing


def _build_raw_sample(*, candidates, query, answer, n_chars=20000,
                      prefix="REPLY:") -> dict:
    messages = [{"role": "user", "content": "begin"}]
    for c in candidates:
        messages.append({"role": "assistant", "content": c})
        messages.append({"role": "user", "content": "ok"})
    # final user query (REPLACES the last "ok")
    messages[-1] = {"role": "user", "content": query}
    return {
        "prompt": json.dumps(messages),
        "answer": answer,
        "random_string_to_prepend": prefix,
        "n_chars": n_chars,
    }


class TestParseSample:
    def test_extracts_candidates_query_target(self):
        raw = _build_raw_sample(
            candidates=("c0", "c1", "c2"),
            query="repro c1", answer="c1",
        )
        s = mod.parse_sample(raw)
        assert s.candidates == ("c0", "c1", "c2")
        assert s.query == "repro c1"
        assert s.target == "c1"
        assert s.random_prefix == "REPLY:"

    def test_assigns_bin_from_n_chars(self):
        raw = _build_raw_sample(
            candidates=("a", "b"), query="x", answer="a",
            n_chars=20_000,
        )
        s = mod.parse_sample(raw)
        assert s.bin == "8K"

    def test_unknown_bin_falls_through_default(self):
        raw = _build_raw_sample(
            candidates=("a",), query="x", answer="a",
            n_chars=10_000,  # below the 16K floor
        )
        s = mod.parse_sample(raw, default_bin="?")
        assert s.bin == "?"

    def test_rejects_non_user_final(self):
        raw = _build_raw_sample(
            candidates=("a",), query="x", answer="a",
        )
        # Tamper: drop the final user message entirely.
        msgs = json.loads(raw["prompt"])
        msgs.pop()  # remove final user
        raw["prompt"] = json.dumps(msgs)
        with pytest.raises(ValueError, match="final message"):
            mod.parse_sample(raw)

    def test_rejects_no_assistants(self):
        raw = {
            "prompt": json.dumps([{"role": "user", "content": "q?"}]),
            "answer": "", "random_string_to_prepend": "", "n_chars": 0,
        }
        with pytest.raises(ValueError, match="no assistant"):
            mod.parse_sample(raw)


# ───────────────────────────────────────────────────── scoring


class TestScoreOutput:
    def test_perfect_match_with_prefix(self):
        score, ok = mod.score_output("PFX:hello", "PFX:hello", "PFX:")
        assert ok is True
        assert score == pytest.approx(1.0)

    def test_prefix_fail_zeroes_score(self):
        score, ok = mod.score_output(
            "no prefix here", "PFX:hello", "PFX:",
        )
        assert ok is False
        assert score == 0.0

    def test_no_prefix_skips_check(self):
        score, ok = mod.score_output("hello", "hello", "")
        assert ok is True
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        score, ok = mod.score_output(
            "PFX:hello world", "PFX:hello mars", "PFX:",
        )
        assert ok is True
        assert 0.0 < score < 1.0


# ───────────────────────────────────────────────────── synthetic samples


class TestSyntheticSample:
    def test_shape(self):
        s = mod.synthetic_sample()
        assert len(s.candidates) == 8
        assert s.target in s.candidates
        # Target index is reachable via helper
        assert mod._target_index(s) == 3  # default

    def test_target_idx_propagates(self):
        s = mod.synthetic_sample(target_idx=5)
        assert s.target == s.candidates[5]


# ───────────────────────────────────────────────────── prompt + helpers


class TestBuildPrompt:
    def test_orders_by_original_position(self):
        s = mod.synthetic_sample(n_candidates=5, target_idx=2)
        # Retrieved out-of-order
        sys_p, user_p = mod.build_prompt(s, [3, 0, 2])
        # Each "originally message N" appears in ascending N
        positions = [
            int(line.split("originally message ")[1].split(")")[0])
            for line in user_p.splitlines()
            if "originally message" in line
        ]
        assert positions == sorted(positions)
        assert sys_p == mod._SYSTEM_PROMPT

    def test_dedupes_repeated_indices(self):
        s = mod.synthetic_sample(n_candidates=4, target_idx=0)
        _, user_p = mod.build_prompt(s, [0, 0, 1, 0, 1])
        # Only candidates 0 and 1 should appear
        assert user_p.count("originally message 1)") == 1
        assert user_p.count("originally message 2)") == 1
        assert user_p.count("originally message 3)") == 0


def test_candidate_index_from_path():
    assert mod._candidate_index_from_path("a/b/message_0042.txt") == 42
    assert mod._candidate_index_from_path("/x/message_0000.txt") == 0
    assert mod._candidate_index_from_path("nope.py") is None


# ───────────────────────────────────────────────────── stub + retriever fakes


class _FakeRetriever:
    """Tiny in-process retriever for tests — returns candidate indices
    in a programmable order. The pipeline doesn't care that it isn't
    a real cosine search; it only needs ``retrieve()`` + ``close()``."""

    def __init__(self, order: tuple[int, ...]) -> None:
        self._order = order
        self.calls: int = 0

    def retrieve(self, sample, k: int) -> tuple[int, ...]:
        self.calls += 1
        return tuple(self._order[: max(k, 1)])

    def close(self) -> None:
        return None


class TestStubGenerator:
    def test_returns_programmed_output(self):
        g = mod.StubGenerator()
        mod._stub_set_output(g, "hello")
        assert g.complete(
            system="s", user="u", max_tokens=8, temperature=0.0,
        ) == "hello"

    def test_returns_empty_without_program(self):
        g = mod.StubGenerator()
        assert g.complete(
            system="s", user="u", max_tokens=8, temperature=0.0,
        ) == ""


# ───────────────────────────────────────────────────── pipeline plumbing


class TestEvaluateSample:
    def test_perfect_round_trip_via_stub(self):
        s = mod.synthetic_sample(target_idx=2, n_candidates=5)
        retriever = _FakeRetriever(order=(0, 2, 4, 1, 3))
        gen = mod.StubGenerator()
        mod._stub_set_output(gen, s.target)
        result = mod._evaluate_sample(
            s, retriever=retriever, generator=gen,
            top_k=3, temperature=0.0, max_output_tokens=512,
        )
        assert result.score == pytest.approx(1.0)
        assert result.prefix_pass is True
        assert result.target_in_topk is True
        # Target was at retrieval-rank 2 (index 1 in (0,2,4))
        assert result.target_rank == 2
        assert result.retrieved_indices == (0, 2, 4)

    def test_target_missing_from_topk_marked(self):
        s = mod.synthetic_sample(target_idx=4, n_candidates=8)
        retriever = _FakeRetriever(order=(0, 1, 2, 3))
        gen = mod.StubGenerator()
        mod._stub_set_output(gen, "wrong output")
        result = mod._evaluate_sample(
            s, retriever=retriever, generator=gen,
            top_k=4, temperature=0.0, max_output_tokens=128,
        )
        assert result.target_in_topk is False
        assert result.target_rank is None
        # No prefix pass → score zeroed
        assert result.score == 0.0
        assert result.prefix_pass is False

    def test_prefix_only_match_zeroes(self):
        """Output starts with prefix but content is wrong → low ratio,
        prefix_pass True."""
        s = mod.synthetic_sample(target_idx=0, n_candidates=4)
        retriever = _FakeRetriever(order=(0, 1, 2, 3))
        gen = mod.StubGenerator()
        # Prefix only, no real content
        mod._stub_set_output(gen, s.random_prefix + "totally unrelated")
        result = mod._evaluate_sample(
            s, retriever=retriever, generator=gen,
            top_k=4, temperature=0.0, max_output_tokens=128,
        )
        assert result.prefix_pass is True
        # Non-zero (some chars overlap) but well below 1
        assert 0.0 < result.score < 0.9


# ───────────────────────────────────────────────────── run + summary


def test_run_with_synthetic_iter():
    """Full ``run`` path using the ``samples_iter`` override so we
    don't touch HF or the real dataset."""
    samples = [mod.synthetic_sample(target_idx=i, bin="8K") for i in range(3)]
    retriever = _FakeRetriever(order=tuple(range(8)))
    gen = mod.StubGenerator()

    # Program the stub for each evaluation. Since _evaluate_sample
    # calls complete() once per sample we need to re-program before
    # the run() helper iterates — easiest path is a tiny generator
    # wrapper that pulls from a queue.
    class _QueueGen:
        def __init__(self, outputs):
            self._q = list(outputs)
        def complete(self, *, system, user, max_tokens, temperature):
            return self._q.pop(0) if self._q else ""

    queued = _QueueGen(outputs=[s.target for s in samples])
    report = mod.run(
        retriever=retriever, generator=queued,
        bins=("8K",), samples_per_bin={"8K": 3},
        samples_iter=samples, top_k=8,
    )
    assert "8K" in report.summaries
    assert report.summaries["8K"].n_samples == 3
    assert report.summaries["8K"].avg_score == pytest.approx(1.0)
    assert report.summaries["8K"].prefix_pass_rate == pytest.approx(1.0)


def test_summarize_aggregates_by_bin():
    rows = [
        mod.SampleResult(
            bin="8K", n_chars=20000, score=1.0, prefix_pass=True,
            retrieved_indices=(0,), target_in_topk=True, target_rank=1,
            retrieval_ms=1.0, generation_ms=2.0,
            model_output="x", target="x",
        ),
        mod.SampleResult(
            bin="8K", n_chars=20000, score=0.5, prefix_pass=True,
            retrieved_indices=(0,), target_in_topk=False, target_rank=None,
            retrieval_ms=3.0, generation_ms=4.0,
            model_output="x", target="y",
        ),
        mod.SampleResult(
            bin="32K", n_chars=80000, score=0.0, prefix_pass=False,
            retrieved_indices=(0,), target_in_topk=False, target_rank=None,
            retrieval_ms=5.0, generation_ms=6.0,
            model_output="", target="z",
        ),
    ]
    out = mod._summarize(rows)
    assert set(out) == {"8K", "32K"}
    assert out["8K"].avg_score == pytest.approx(0.75)
    assert out["8K"].prefix_pass_rate == pytest.approx(1.0)
    assert out["8K"].target_in_topk_rate == pytest.approx(0.5)
    assert out["32K"].avg_score == 0.0
    assert out["32K"].prefix_pass_rate == 0.0


def test_samples_per_bin_normalized():
    bins = ("8K", "32K")
    assert mod._samples_per_bin_normalized(10, bins) == {"8K": 10, "32K": 10}
    assert mod._samples_per_bin_normalized(
        {"8K": 5, "32K": 0}, bins
    ) == {"8K": 5, "32K": 0}
    # Missing keys default to 0
    assert mod._samples_per_bin_normalized({"8K": 5}, bins) == {
        "8K": 5, "32K": 0,
    }


# ───────────────────────────────────────────────────── CLI smoke


def test_smoke_subcommand_round_trips_score_one(capsys):
    """CLI ``--smoke`` exercises stub generator + a real retriever path
    (defaults to 'faiss' — for the test we override to longctx with a
    tiny chunk_tokens to avoid the SentenceTransformer load entirely…
    actually no, both retrievers load a model. Skip if not available.)
    """
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("rank_bm25")
    rc = mod.main([
        "--smoke",
        "--retriever", "faiss",
        # MiniLM is the AMD-baseline default; it's small and usually
        # cached in CI. Tests that lack the model skip via importorskip
        # when ST itself can't load.
    ])
    captured = capsys.readouterr()
    assert rc == 0
    # Either a table or JSON; the smoke summary should report at least
    # one bin and a non-empty header.
    assert "bin" in captured.out
    # Smoke samples are 8K-only
    assert "8K" in captured.out


def test_render_table_formats_summary():
    samples = [
        mod.SampleResult(
            bin="8K", n_chars=20000, score=0.76, prefix_pass=True,
            retrieved_indices=(0,), target_in_topk=True, target_rank=1,
            retrieval_ms=1.5, generation_ms=200.0,
            model_output="x", target="x",
        ),
    ]
    report = mod.E2EReport(
        config={}, summaries=mod._summarize(samples), samples=samples,
    )
    table = mod.render_table(report)
    assert "8K" in table
    assert "0.7600" in table
