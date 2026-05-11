"""Tests for the interactive project picker on top of ``longctx init``."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from longctx_daemon import cli
from longctx_daemon.discovery import DiscoveredProject


def _disc(name: str, ignored: bool = False) -> DiscoveredProject:
    return DiscoveredProject(
        name=name, root_path=Path(f"/tmp/{name}"),
        sentinel=".git", has_longctxignore=ignored,
    )


def test_picker_accept_defaults_returns_unignored(monkeypatch, capsys):
    """<Enter> with no toggles returns everything except .longctxignore."""
    discovered = [_disc("a"), _disc("b", ignored=True), _disc("c")]
    monkeypatch.setattr("builtins.input", lambda _: "")
    out = cli._interactive_pick_projects(discovered)
    assert {p.name for p in out} == {"a", "c"}


def test_picker_all_overrides_ignored(monkeypatch):
    """'all' selects everything including .longctxignore-marked."""
    discovered = [_disc("a"), _disc("b", ignored=True)]
    inputs = iter(["all", ""])  # 'all' then accept
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    out = cli._interactive_pick_projects(discovered)
    assert {p.name for p in out} == {"a", "b"}


def test_picker_none_then_toggle(monkeypatch):
    """'none' deselects all; then '1 3' toggles back specific items."""
    discovered = [_disc("a"), _disc("b"), _disc("c"), _disc("d")]
    inputs = iter(["none", "1 3", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    out = cli._interactive_pick_projects(discovered)
    assert {p.name for p in out} == {"a", "c"}


def test_picker_comma_separated_toggle(monkeypatch):
    """Comma-separated indices work too."""
    discovered = [_disc("a"), _disc("b"), _disc("c")]
    inputs = iter(["none", "1,3", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    out = cli._interactive_pick_projects(discovered)
    assert {p.name for p in out} == {"a", "c"}


def test_picker_quit_returns_none(monkeypatch):
    discovered = [_disc("a")]
    monkeypatch.setattr("builtins.input", lambda _: "q")
    out = cli._interactive_pick_projects(discovered)
    assert out is None


def test_picker_invalid_toggle_recovers(monkeypatch, capsys):
    """Garbage input doesn't crash; loops until something parseable."""
    discovered = [_disc("a"), _disc("b")]
    inputs = iter(["xyzzy", "1", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    out = cli._interactive_pick_projects(discovered)
    # Started with both selected; toggled 1 off → only b
    assert {p.name for p in out} == {"b"}


def test_picker_eof_treated_as_accept(monkeypatch):
    """EOFError on input (e.g. piped, no tty) defaults to accept."""
    discovered = [_disc("a"), _disc("b")]
    def _raise(_): raise EOFError()
    monkeypatch.setattr("builtins.input", _raise)
    out = cli._interactive_pick_projects(discovered)
    assert {p.name for p in out} == {"a", "b"}


def test_picker_empty_input_returns_empty():
    """No discovered projects → empty list, no prompt."""
    out = cli._interactive_pick_projects([])
    assert out == []
