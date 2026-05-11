"""Tests for the ``longctx calibrate`` subcommand.

Drives the argparse layer + the table-rendering + the optional
``--write-config`` path. The actual sweep runs through a mocked
``sweep_floor`` so we don't need an embedder."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from longctx_daemon import cli
from longctx_daemon.eval.abstention import SweepReport


# ════════════════════════════════════════════════ argparse routing


def test_calibrate_registered_in_parser():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices")
               and a.choices is not None]
    assert "calibrate" in actions[0].choices


def test_calibrate_requires_corpus_dir():
    with pytest.raises(SystemExit) as ei:
        cli.main(["calibrate"])
    assert ei.value.code == 2


def test_calibrate_rejects_nonexistent_corpus(tmp_path, capsys):
    rc = cli.main(["calibrate", "--corpus-dir", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not a directory" in captured.err


# ════════════════════════════════════════════════ pretty + JSON output


def _fake_sweep_report() -> SweepReport:
    return SweepReport(
        points=[
            (0.30, 0.42, 0.00),
            (0.40, 0.55, 0.02),
            (0.50, 0.68, 0.05),
            (0.60, 0.81, 0.20),
            (0.70, 0.89, 0.45),
        ],
        recommended_floor=0.50,
        max_fnr=0.05,
    )


def test_calibrate_prints_table(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main(["calibrate", "--corpus-dir", str(corpus)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "0.30" in captured.out
    assert "0.50" in captured.out
    assert "0.70" in captured.out
    assert "recommended" in captured.out


def test_calibrate_emits_json_when_flag_set(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main([
            "calibrate", "--corpus-dir", str(corpus), "--json",
        ])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["recommended_floor"] == 0.50
    assert payload["max_fnr"] == 0.05
    assert len(payload["points"]) == 5


def test_calibrate_passes_floor_range_to_sweep(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with patch(
        "longctx_daemon.eval.abstention.sweep_floor",
        return_value=_fake_sweep_report(),
    ) as mock_sweep:
        cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--floor-min", "0.40",
            "--floor-max", "0.60",
            "--floor-step", "0.10",
        ])
    args, kwargs = mock_sweep.call_args
    floors = kwargs["floors"]
    # Inclusive endpoints
    assert floors[0] == pytest.approx(0.40)
    assert floors[-1] == pytest.approx(0.60)
    # Step honored
    assert all(
        abs(b - a - 0.10) < 1e-6
        for a, b in zip(floors, floors[1:])
    )


def test_calibrate_passes_max_fnr_to_sweep(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with patch(
        "longctx_daemon.eval.abstention.sweep_floor",
        return_value=_fake_sweep_report(),
    ) as mock_sweep:
        cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--max-fnr", "0.10",
        ])
    args, kwargs = mock_sweep.call_args
    assert kwargs["max_fnr"] == 0.10


# ════════════════════════════════════════════════ --write-config


def test_calibrate_dry_run_does_not_write(tmp_path, capsys):
    """Default behavior: print + exit. No config file should be touched."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg_path = tmp_path / "config.toml"

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main(["calibrate", "--corpus-dir", str(corpus)])
    assert rc == 0
    assert not cfg_path.exists()


def test_calibrate_write_config_creates_file(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg_path = tmp_path / "config.toml"

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--write-config", "--config", str(cfg_path),
        ])
    assert rc == 0
    assert cfg_path.exists()
    content = cfg_path.read_text()
    assert "relevance_floor" in content
    assert "0.5" in content


def test_calibrate_write_config_round_trips_via_load(tmp_path):
    """The written file must be readable by load_config."""
    from longctx_daemon.config import load_config

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg_path = tmp_path / "config.toml"

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--write-config", "--config", str(cfg_path),
        ])
    assert rc == 0
    loaded = load_config(cfg_path)
    assert loaded.index.relevance_floor == pytest.approx(0.50)


def test_calibrate_write_config_preserves_other_fields(tmp_path):
    """Pre-existing config values must be preserved when bumping
    relevance_floor — we're patching one field, not rewriting the
    file from defaults."""
    from longctx_daemon.config import (
        IndexConfigSection,
        ServiceConfig,
        default_config,
        write_config,
        load_config,
    )
    from dataclasses import replace

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg_path = tmp_path / "config.toml"

    base = default_config()
    pre = ServiceConfig(
        corpus=base.corpus, server=base.server,
        index=replace(base.index, embedder="custom/embedder",
                      chunk_size=4096),
        watcher=base.watcher, cleanup=base.cleanup, search=base.search,
        logging=base.logging,
    )
    write_config(cfg_path, pre)

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        rc = cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--write-config", "--config", str(cfg_path),
        ])
    assert rc == 0
    loaded = load_config(cfg_path)
    assert loaded.index.embedder == "custom/embedder"
    assert loaded.index.chunk_size == 4096
    assert loaded.index.relevance_floor == pytest.approx(0.50)


def test_calibrate_write_config_idempotent(tmp_path):
    """Running twice with the same recommendation produces the same
    on-disk content (no growing diff)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg_path = tmp_path / "config.toml"

    with patch("longctx_daemon.eval.abstention.sweep_floor",
               return_value=_fake_sweep_report()):
        cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--write-config", "--config", str(cfg_path),
        ])
        first = cfg_path.read_text()
        cli.main([
            "calibrate", "--corpus-dir", str(corpus),
            "--write-config", "--config", str(cfg_path),
        ])
        second = cfg_path.read_text()
    assert first == second
