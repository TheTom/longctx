"""Tests for MRCRRunner."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_fake_client(model="test-model", responses=None):
    """Build a fake LongCtxClient.ask() that returns canned responses."""
    from longctx.rag.client import LongCtxResponse

    responses = responses or []
    iterator = iter(responses)

    def _ask(query, candidates, top_k=8, max_tokens=4096, temperature=0.0):
        try:
            content = next(iterator)
        except StopIteration:
            content = ""
        return LongCtxResponse(
            content=content,
            retrieved_indices=list(range(min(top_k, len(candidates)))),
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.5,
        )

    fake = MagicMock()
    fake.model = model
    fake.ask = _ask
    return fake


def test_mrcr_runner_init(tmp_path):
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    assert r.data_dir == tmp_path


def test_mrcr_runner_load_raises_on_missing_data(tmp_path):
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    with pytest.raises(FileNotFoundError):
        r._load_8needle("8k", limit=10)


def test_mrcr_runner_run_scoring(tmp_path, fake_8needle_df, monkeypatch):
    from longctx.eval import MRCRRunner

    # Patch _load_8needle to return our fake DataFrame
    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )

    # Sample 0: gold answer is "PFXfirst response"; if we return it
    #           verbatim score should be ~1.0 with prefix_ok=True
    # Sample 1: gold "Q1Qbeta"; we return "Q1Qbeta" exactly = score 1.0
    fake = _make_fake_client(
        responses=["PFXfirst response", "Q1Qbeta"]
    )

    summary = r.run(fake, bin_name="8k", n=2, top_k=8, verbose=False)
    assert summary.n == 2
    assert summary.avg_score > 0.99
    assert summary.prefix_pass_rate == 1.0
    assert summary.bin == "8k"


def test_mrcr_runner_prefix_failure_zeros_score(
    tmp_path, fake_8needle_df, monkeypatch
):
    """If output doesn't start with prefix, score should be 0."""
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )
    fake = _make_fake_client(
        # First response missing the PFX prefix
        responses=["wrong start no prefix", "Q1Qbeta"]
    )
    summary = r.run(fake, bin_name="8k", n=2, top_k=8, verbose=False)
    assert summary.results[0].score == 0.0
    assert summary.results[0].starts_with_prefix is False
    # Second one still good
    assert summary.results[1].score > 0.99


def test_mrcr_runner_partial_match_scores_in_between(
    tmp_path, fake_8needle_df, monkeypatch
):
    """Partial-match output gets a fractional score."""
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )
    fake = _make_fake_client(
        responses=["PFXfirst respons", "Q1Qbeta"]
    )
    summary = r.run(fake, bin_name="8k", n=2, top_k=8, verbose=False)
    s0 = summary.results[0].score
    assert 0.0 < s0 < 1.0


def test_mrcr_runner_write_summary(
    tmp_path, fake_8needle_df, monkeypatch
):
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )
    fake = _make_fake_client(responses=["PFXfirst response", "Q1Qbeta"])
    summary = r.run(fake, bin_name="8k", n=2, top_k=8, verbose=False)

    out_path = tmp_path / "summary.json"
    MRCRRunner.write_summary(summary, out_path)
    data = json.loads(out_path.read_text())
    assert data["bin"] == "8k"
    assert data["n"] == 2
    assert "results" in data
    assert len(data["results"]) == 2
    assert "score" in data["results"][0]


def test_mrcr_runner_raises_when_no_samples_completed(
    tmp_path, fake_8needle_df, monkeypatch
):
    """If every sample is too short, runner should raise."""
    from longctx.eval import MRCRRunner

    pd = pytest.importorskip("pandas")
    empty_df = pd.DataFrame([], columns=fake_8needle_df.columns)
    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: empty_df,
    )
    fake = _make_fake_client(responses=[])
    with pytest.raises(RuntimeError):
        r.run(fake, bin_name="8k", n=2, top_k=8, verbose=False)


def test_mrcr_runner_skips_bad_json(
    tmp_path, fake_8needle_df, monkeypatch, capsys
):
    """Rows with malformed prompt JSON are skipped (verbose path covered)."""
    from longctx.eval import MRCRRunner

    pd = pytest.importorskip("pandas")
    df = fake_8needle_df.copy()
    df.at[0, "prompt"] = "{not json"
    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: df.head(limit),
    )
    fake = _make_fake_client(responses=["Q1Qbeta"])
    summary = r.run(fake, bin_name="8k", n=2, top_k=8, verbose=True)
    out = capsys.readouterr().out
    assert "skip i=0" in out
    # Only sample 1 completed
    assert summary.n == 1


def test_mrcr_runner_skips_when_too_few_assistant_msgs(
    tmp_path, fake_8needle_df, monkeypatch
):
    """top_k > len(asst_msgs) -> sample is skipped (line 141)."""
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )
    fake = _make_fake_client(responses=["PFXfirst response", "Q1Qbeta"])
    # Each fake row has 8 assistant messages; top_k=99 forces skip
    with pytest.raises(RuntimeError):
        r.run(fake, bin_name="8k", n=2, top_k=99, verbose=False)


def test_mrcr_runner_verbose_prints_progress(
    tmp_path, fake_8needle_df, monkeypatch, capsys
):
    """verbose=True prints per-sample lines + final summary line."""
    from longctx.eval import MRCRRunner

    r = MRCRRunner(tmp_path)
    monkeypatch.setattr(
        r, "_load_8needle",
        lambda bin_name, limit: fake_8needle_df.head(limit),
    )
    fake = _make_fake_client(responses=["PFXfirst response", "Q1Qbeta"])
    r.run(fake, bin_name="8k", n=2, top_k=8, verbose=True)
    out = capsys.readouterr().out
    assert "score=" in out
    assert "[summary]" in out


def test_mrcr_runner_loads_parquet(tmp_path):
    """_load_8needle reads parquet from data_dir/8needle/*.parquet."""
    from longctx.eval import MRCRRunner

    pd = pytest.importorskip("pandas")
    parquet_dir = tmp_path / "8needle"
    parquet_dir.mkdir()
    df = pd.DataFrame([
        {"n_chars": 20_000, "prompt": "[]", "answer": "x",
         "random_string_to_prepend": "x"},
        {"n_chars": 5_000, "prompt": "[]", "answer": "y",
         "random_string_to_prepend": "y"},
    ])
    df.to_parquet(parquet_dir / "part0.parquet")
    # Second parquet to exercise the concat branch
    df2 = pd.DataFrame([
        {"n_chars": 25_000, "prompt": "[]", "answer": "z",
         "random_string_to_prepend": "z"},
    ])
    df2.to_parquet(parquet_dir / "part1.parquet")

    r = MRCRRunner(tmp_path)
    out = r._load_8needle("8k", limit=10)
    # Bin 8k = [16_000, 32_000); only the 20k and 25k rows match
    assert len(out) == 2
