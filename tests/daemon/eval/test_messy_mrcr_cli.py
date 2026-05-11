"""CLI argparse tests for Messy MRCR.

Covers:
* default flags
* explicit overrides for needles / samples / bins / transforms / seed
* unknown bin / transform names exit non-zero
* ``--json`` flag emits valid JSON to stdout
* ``--no-write`` skips disk writes
* writes RESULTS.md + json when ``--no-write`` is absent

Heavy work (real embedder, real dataset) is mocked at the
``run_messy_mrcr`` boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from longctx_daemon.eval import messy_mrcr as mm
from longctx_daemon.eval.messy_mrcr import (
    DEFAULT_BINS,
    MessyMRCRReport,
    TRANSFORMS,
    _parse_args,
    main,
    report_to_dict,
)


# ─────────────────────────────────────── helpers

def _fake_report(transforms=("clean", "off-corpus"),
                 bins=("8K",)) -> MessyMRCRReport:
    return MessyMRCRReport(
        transforms=transforms, bins=bins, n_samples_per_cell=2,
        needles=8, embedder_id="fake/embedder", seed=0,
        matrix={(t, b): 0.5 for t in transforms for b in bins},
        per_sample=[],
    )


# ═══════════════════════════════════════ argparse

def test_default_args_pick_up_default_bins() -> None:
    args = _parse_args([])
    assert args.needles == 8
    assert args.samples == 20
    assert args.bins == ",".join(DEFAULT_BINS)
    assert args.transforms == ",".join(TRANSFORMS)
    assert args.seed == 0
    assert args.top_k == 5
    assert args.json is False


def test_args_override_needles() -> None:
    args = _parse_args(["--needles", "4"])
    assert args.needles == 4


def test_args_override_samples() -> None:
    args = _parse_args(["--samples", "100"])
    assert args.samples == 100


def test_args_override_bins() -> None:
    args = _parse_args(["--bins", "8K,1M"])
    assert args.bins == "8K,1M"


def test_args_override_transforms() -> None:
    args = _parse_args(["--transforms", "clean,typo"])
    assert args.transforms == "clean,typo"


def test_args_set_json_flag() -> None:
    args = _parse_args(["--json"])
    assert args.json is True


def test_args_invalid_needles_rejected() -> None:
    """argparse choices=[2,4,8] should reject 16."""
    with pytest.raises(SystemExit):
        _parse_args(["--needles", "16"])


# ═══════════════════════════════════════ main entry — bin / transform validation

def test_main_unknown_bin_exits_2(capsys) -> None:
    rc = main([
        "--bins", "8K,42M",
        "--samples", "1",
        "--transforms", "clean",
        "--no-write",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "42M" in err


def test_main_unknown_transform_exits_2(capsys) -> None:
    rc = main([
        "--bins", "8K",
        "--samples", "1",
        "--transforms", "clean,nonexistent",
        "--no-write",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nonexistent" in err


# ═══════════════════════════════════════ main entry — happy paths

def test_main_writes_results_md_and_json(tmp_path, capsys) -> None:
    """Default mode writes RESULTS.md and messy_mrcr_<date>.json."""
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    with patch.object(mm, "run_messy_mrcr", return_value=fake):
        rc = main([
            "--bins", "8K",
            "--transforms", "clean",
            "--samples", "1",
            "--out-dir", str(tmp_path),
        ])
    assert rc == 0
    assert (tmp_path / "RESULTS.md").exists()
    json_files = list(tmp_path.glob("messy_mrcr_*.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert payload["needles"] == 8
    assert payload["transforms"] == ["clean"]


def test_main_no_write_skips_disk(tmp_path) -> None:
    """``--no-write`` must not write anything."""
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    with patch.object(mm, "run_messy_mrcr", return_value=fake):
        rc = main([
            "--bins", "8K", "--transforms", "clean",
            "--samples", "1", "--out-dir", str(tmp_path),
            "--no-write",
        ])
    assert rc == 0
    # tmp_path is fresh from pytest; should still be empty.
    assert list(tmp_path.iterdir()) == []


def test_main_json_flag_emits_valid_json(capsys, tmp_path) -> None:
    """``--json`` writes a valid JSON document to stdout."""
    fake = _fake_report(transforms=("clean", "typo"), bins=("8K", "32K"))
    with patch.object(mm, "run_messy_mrcr", return_value=fake):
        rc = main([
            "--bins", "8K,32K", "--transforms", "clean,typo",
            "--samples", "1", "--out-dir", str(tmp_path),
            "--no-write", "--json",
        ])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["transforms"] == ["clean", "typo"]
    assert payload["bins"] == ["8K", "32K"]


def test_main_default_emits_matrix_table(capsys, tmp_path) -> None:
    """Without ``--json``, stdout has the markdown matrix table."""
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    with patch.object(mm, "run_messy_mrcr", return_value=fake):
        rc = main([
            "--bins", "8K", "--transforms", "clean",
            "--samples", "1", "--out-dir", str(tmp_path),
            "--no-write",
        ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "clean" in out
    assert "8K" in out
    assert "0.500" in out


def test_main_passes_seed_through(tmp_path) -> None:
    """The ``--seed`` flag should reach run_messy_mrcr."""
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return fake

    with patch.object(mm, "run_messy_mrcr", side_effect=_capture):
        rc = main([
            "--bins", "8K", "--transforms", "clean", "--samples", "1",
            "--seed", "42", "--out-dir", str(tmp_path), "--no-write",
        ])
    assert rc == 0
    assert captured["seed"] == 42


def test_main_passes_top_k_through(tmp_path) -> None:
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return fake

    with patch.object(mm, "run_messy_mrcr", side_effect=_capture):
        rc = main([
            "--bins", "8K", "--transforms", "clean", "--samples", "1",
            "--top-k", "10", "--out-dir", str(tmp_path), "--no-write",
        ])
    assert rc == 0
    assert captured["top_k"] == 10


def test_main_quiet_suppresses_progress(capsys, tmp_path) -> None:
    """--quiet should silence the [messy-mrcr] progress lines."""
    fake = _fake_report(transforms=("clean",), bins=("8K",))
    with patch.object(mm, "run_messy_mrcr", return_value=fake):
        rc = main([
            "--bins", "8K", "--transforms", "clean", "--samples", "1",
            "--out-dir", str(tmp_path), "--no-write", "--quiet",
        ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "[messy-mrcr]" not in err
