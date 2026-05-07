"""End-to-end tests mirroring Sarah's user journey from PRD §2 + §7.

Each test maps a sentence in the PRD to a verifiable behavior. The sequence
matches what Sarah experiences using OpenCode + vllm-swift + longctx-svc.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _opencode_prefill(project_dir: Path, user_msg: str) -> str:
    """Mimic what an OpenCode harness sends in the prefill: system context
    that mentions the project's path + a user message."""
    return (
        f"# Workspace: {project_dir}\n"
        f"# Current file: {project_dir}/src/auth.ts\n"
        f"\n{user_msg}\n"
    )


# --- PRD §7.1: Single-project root ---

def test_sarah_first_message_detects_scope(client, project_dir):
    """'She launches OpenCode in ~/dev/myapp. First message: why is auth
    failing?' → vllm-swift parses paths, detects project root."""
    prefill = _opencode_prefill(
        project_dir,
        "why is auth failing in the docker build?",
    )
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "opencode:abc123"},
        json={
            "prefill_text": prefill,
            "query": "why is auth failing in the docker build?",
            "top_k": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scope_path"] is not None
    assert data["scope_sentinel"] == "package.json"
    assert data["session_id"] == "opencode:abc123"
    # Hot scope indexes in the first request (synchronous in MVP).
    assert data["scope_status"] in ("ready", "empty")


def test_response_carries_debug_headers(client, project_dir):
    """PRD §5.9: response headers carry the Sarah-visible debug info."""
    prefill = _opencode_prefill(project_dir, "auth flow?")
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "opencode:dbg"},
        json={"prefill_text": prefill, "query": "auth flow?", "top_k": 4},
    )
    h = response.headers
    assert h["x-longctx-session"] == "opencode:dbg"
    assert h["x-longctx-scope"]
    assert "x-longctx-chunks-used" in h
    assert "x-longctx-scope-status" in h


def test_chunks_cite_real_files_at_real_lines(client, project_dir):
    """'Subsequent answers cite real files at real line numbers.' Each
    chunk has file_path + start_line + end_line."""
    prefill = _opencode_prefill(project_dir, "auth flow")
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "opencode:cite"},
        json={"prefill_text": prefill, "query": "auth flow", "top_k": 4},
    )
    data = response.json()
    if data["chunks"]:
        for c in data["chunks"]:
            assert c["file_path"]
            assert isinstance(c["start_line"], int)
            assert isinstance(c["end_line"], int)
            assert c["start_line"] >= 1
            assert c["end_line"] >= c["start_line"]
            assert Path(c["file_path"]).exists()


# --- PRD §7.5: Concurrent sessions, separate projects ---

def test_two_sessions_two_projects_no_crosstalk(client, tmp_path):
    """'OpenCode session A in ~/dev/foo and Hermes session B in
    ~/dev/bar. No cross-contamination.'"""
    foo = tmp_path / "foo"
    bar = tmp_path / "bar"
    for d in (foo, bar):
        d.mkdir()
        (d / "package.json").write_text(f'{{"name":"{d.name}"}}')
        (d / "main.ts").write_text(
            f"// {d.name}-specific content"
        )
    # Session A → foo
    rA = client.post(
        "/retrieve",
        headers={"x-session-affinity": "session-A"},
        json={
            "prefill_text": f"see {foo}/main.ts",
            "query": "main",
            "top_k": 4,
        },
    )
    # Session B → bar
    rB = client.post(
        "/retrieve",
        headers={"x-session-affinity": "session-B"},
        json={
            "prefill_text": f"see {bar}/main.ts",
            "query": "main",
            "top_k": 4,
        },
    )
    assert rA.json()["scope_hash"] != rB.json()["scope_hash"]


# --- PRD §7.6: Concurrent sessions, same project share index ---

def test_two_sessions_same_project_share_scope(client, project_dir):
    rA = client.post(
        "/retrieve",
        headers={"x-session-affinity": "alice"},
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    rB = client.post(
        "/retrieve",
        headers={"x-session-affinity": "bob"},
        json={
            "prefill_text": f"see {project_dir}/src/billing.ts",
            "query": "billing", "top_k": 4,
        },
    )
    # Same scope_hash means they share the underlying index
    assert rA.json()["scope_hash"] == rB.json()["scope_hash"]


# --- PRD §5.5: Stateless fallback / no session header ---

def test_no_session_header_ephemeral(client, project_dir):
    """'No session header → ephemeral request, no caching.'"""
    response = client.post(
        "/retrieve",
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] is None
    assert response.headers["x-longctx-session"] == "ephemeral"


# --- PRD §7.3: Ambiguous root → no scope, no retrieval ---

def test_no_path_in_prefill_returns_no_scope(client):
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "stuck"},
        json={
            "prefill_text": "how do i fix this auth bug?",
            "query": "auth bug",
            "top_k": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scope_path"] is None
    assert data["scope_status"] == "no-scope"
    assert data["chunks"] == []


# --- PRD §5.8: Manual override ---

def test_explicit_scope_override(client, project_dir):
    """'longctx-scope ./apps/billing' wins over prefill detection."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "manual"},
        json={
            "prefill_text": "no path mentioned here",
            "query": "auth",
            "top_k": 4,
            "explicit_scope": str(project_dir),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scope_sentinel"] == "explicit"
    # scope_path equality is up to canonicalization (case-fold on macOS)
    assert Path(data["scope_path"]).name.lower() == project_dir.name.lower()


# --- PRD §5.9: /longctx status (text/plain Sarah-visible rendering) ---

def test_status_json_default(client, project_dir):
    # warm up by issuing a retrieve so there's state to show
    client.post(
        "/retrieve",
        headers={"x-session-affinity": "stat"},
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 2,
        },
    )
    response = client.get("/longctx/status")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "local-only"
    assert data["sessions_active"] >= 1
    assert data["scopes_total"] >= 1


def test_status_text_plain_sarah_format(client, project_dir):
    """PRD §5.9: GET /longctx/status with Accept: text/plain returns the
    Sarah-visible block."""
    client.post(
        "/retrieve",
        headers={"x-session-affinity": "stat-txt"},
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "auth", "top_k": 2,
        },
    )
    response = client.get(
        "/longctx/status",
        headers={"accept": "text/plain"},
    )
    assert response.status_code == 200
    text = response.text
    assert "[longctx] mode: local-only" in text
    assert "[longctx] session:" in text
    assert "[longctx] scope:" in text
    assert "[longctx] indexed:" in text
    assert "[longctx] memory:" in text


# --- PRD §7.7: .gitignore respected ---

def test_gitignored_files_not_indexed(client, project_dir):
    """The dist/ignored.js must not appear in any retrieval result."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "gi"},
        json={
            "prefill_text": f"see {project_dir}/src/auth.ts",
            "query": "ignored", "top_k": 8,
        },
    )
    assert response.status_code == 200
    for c in response.json()["chunks"]:
        assert "dist/ignored.js" not in c["file_path"]


# --- Healthz ---

def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- UX fallback: default_scope when no path detected ---

def test_default_scope_fires_when_no_path_in_message(client, project_dir):
    """Tool-using agents (Hermes etc.) often don't put absolute paths
    in user messages. With `default_scope` set, retrieval still fires
    against the engine's working directory."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "default-scope-test"},
        json={
            "prefill_text": "what does the app do?",  # no abs path
            "query": "what does the app do",
            "top_k": 4,
            "default_scope": str(project_dir),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scope_status"] == "ready"
    assert data["scope_path"] is not None
    assert len(data["chunks"]) > 0


def test_path_in_message_overrides_default_scope(client, project_dir, tmp_path):
    """When the user explicitly mentions a path, that wins over
    default_scope (the user's intent is more specific)."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "package.json").write_text('{"name":"other"}')
    (other / "main.ts").write_text("// other content\n")
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "override-test"},
        json={
            "prefill_text": f"explain {project_dir}/src/auth.ts",
            "query": "auth",
            "top_k": 4,
            "default_scope": str(other),  # would point elsewhere
        },
    )
    data = response.json()
    # The mentioned path's project should win, not `default_scope`
    assert data["scope_status"] == "ready"
    assert "myapp" in data["scope_path"]


def test_default_scope_unused_when_no_query_or_unreachable(client, tmp_path):
    """default_scope = nonexistent path → no-scope (don't crash)."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "missing-default"},
        json={
            "prefill_text": "what does the app do?",
            "query": "anything",
            "top_k": 4,
            "default_scope": "/this/does/not/exist",
        },
    )
    # Service must not 500 even on a bad fallback path
    assert response.status_code == 200


def test_default_scope_omitted_falls_through_to_no_scope(client):
    """No path mention + no default_scope = no-scope (existing behavior)."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "vanilla-no-scope"},
        json={
            "prefill_text": "what is 2 + 2?",
            "query": "math",
            "top_k": 4,
        },
    )
    data = response.json()
    assert data["scope_status"] == "no-scope"
    assert data["chunks"] == []
