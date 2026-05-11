"""HTTP integration tests for path-cluster scope detection.

Covers the "automatic engagement when sentinel-walk fails" path: agents
working in fresh dirs that haven't been git-init'd, scratch projects,
monorepo subdirs where the sentinel walks too high.

The cluster tracker is per-session — these tests drive multi-turn
sequences to verify phase progression (WATCH → SPECULATIVE → ACTIVE)
and that the deliberately-killed engine-cwd fallback stays killed.
"""
from __future__ import annotations

from pathlib import Path


def _fresh_project(tmp_path: Path) -> Path:
    """Build a tree with NO sentinel file — simulates a fresh / scratch
    dir the agent is working in before running ``git init``.
    """
    root = tmp_path / "scratch-project"
    root.mkdir()
    (root / "index.html").write_text(
        "<!doctype html><html><body><canvas id=game></canvas></body></html>"
    )
    (root / "sketch.js").write_text(
        "// alpha entry point\nfunction setup() { /* alpha alpha */ }\n"
    )
    (root / "style.css").write_text("body { margin: 0; background: black; }\n")
    (root / "engine.js").write_text(
        "// game loop\nfunction step() { /* beta tick */ }\n"
    )
    (root / "player.js").write_text(
        "// player logic\nfunction shoot() { /* gamma fire */ }\n"
    )
    return root


def test_sentinel_walk_still_wins_when_available(client, project_dir):
    """Sanity: when a sentinel root is reachable, cluster detection does
    NOT clobber it. project_dir has package.json → sentinel-walk fires."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "sentinel-wins"},
        json={
            "prefill_text": f"open {project_dir}/src/auth.ts",
            "query": "auth",
            "top_k": 4,
        },
    )
    data = response.json()
    assert data["scope_status"] in ("ready", "empty")
    assert data["scope_sentinel"] == "package.json"


def test_no_session_no_cluster_pinning(client, tmp_path):
    """Without a session id, cluster tracking has nowhere to live —
    cumulative mentions can't accrue. Single-turn detection should
    return no-scope and never pin to a default fallback."""
    fresh = _fresh_project(tmp_path)
    response = client.post(
        "/retrieve",
        json={
            "prefill_text": "what is in the game?",   # no paths, no session
            "query": "anything",
            "top_k": 4,
            "default_scope": str(fresh),
        },
    )
    data = response.json()
    # No session ⇒ no tracker; no path mentions ⇒ no sentinel scope.
    # default_scope must NOT auto-bind (the 2026-05-11 fix).
    assert data["scope_status"] == "no-scope"


def test_cluster_promotes_after_sustained_mentions(client, tmp_path):
    """Multi-turn: agent works in a fresh dir over several turns. Cluster
    tracker accumulates path mentions; by turn 2 the working scope auto-
    activates without ever needing a sentinel file in the dir."""
    fresh = _fresh_project(tmp_path)
    sid_headers = {"x-session-affinity": "fresh-dir-session"}

    # Turn 1: agent reads a few files. Hits speculate threshold (3 files
    # default) but not active yet (defaults: 5 files OR 2 sustained turns).
    prefill_t1 = (
        f"tool_call: read({fresh}/index.html)\n"
        f"tool_result: <body><canvas>...\n"
        f"tool_call: read({fresh}/sketch.js)\n"
        f"tool_result: function setup() {{ ... }}\n"
        f"tool_call: read({fresh}/engine.js)\n"
        f"tool_result: function step() {{ ... }}\n"
    )
    r1 = client.post(
        "/retrieve",
        headers=sid_headers,
        json={"prefill_text": prefill_t1, "query": "what runs the game?",
              "top_k": 4},
    )
    d1 = r1.json()
    # Turn 1: SPECULATIVE — no active scope yet, so scope_status is
    # no-scope from the /retrieve perspective. (Speculative warm may
    # have kicked off in the background.)
    assert d1["scope_status"] == "no-scope"

    # Turn 2: another mention of a file in the same dir. With 4+ unique
    # files cumulatively AND sustained=2 turns, the cluster promotes to
    # ACTIVE.
    prefill_t2 = prefill_t1 + (
        f"tool_call: read({fresh}/player.js)\n"
        f"tool_result: function shoot() {{ ... }}\n"
    )
    r2 = client.post(
        "/retrieve",
        headers=sid_headers,
        json={"prefill_text": prefill_t2, "query": "what shoots?",
              "top_k": 4},
    )
    d2 = r2.json()
    # Now scope should be the fresh dir, sentinel=cluster (not package.json).
    assert d2["scope_path"] is not None
    assert d2["scope_sentinel"] == "cluster"
    # macOS canonicalizes to lowercase — compare lowercased.
    assert str(fresh).lower() in d2["scope_path"].lower()


def test_single_path_mention_does_not_activate(client, tmp_path):
    """One file mentioned once is too thin a signal to bind a scope.
    Cluster's min_files threshold (3 default) keeps single-shot mentions
    from causing accidental engagement."""
    fresh = _fresh_project(tmp_path)
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "single-mention"},
        json={
            "prefill_text": f"check {fresh}/sketch.js once",
            "query": "what is sketch",
            "top_k": 4,
        },
    )
    data = response.json()
    # Single file + no sentinel above = no scope. Cluster threshold not met.
    # (One mention of a file under no sentinel walks up to /, which is
    # filtered by MIN_CLUSTER_SCOPE_DEPTH so we don't bind to "/".)
    assert data["scope_status"] == "no-scope"


def test_cluster_session_isolation(client, tmp_path):
    """Two sessions working in two different scratch dirs must not bleed
    observations into each other. Session A's tracker doesn't know about
    session B's files."""
    a_parent = tmp_path / "a-side"
    a_parent.mkdir()
    a = _fresh_project(a_parent)
    # Make second project — fresh tree to avoid path overlap.
    b_root = tmp_path / "b-side" / "other-project"
    b_root.mkdir(parents=True)
    for n in ("one.js", "two.js", "three.js", "four.js"):
        (b_root / n).write_text(f"// {n}\n")

    a_prefill = "".join(
        f"read({a}/{name})\n"
        for name in ("index.html", "sketch.js", "engine.js", "player.js")
    )
    b_prefill = "".join(
        f"read({b_root}/{name})\n"
        for name in ("one.js", "two.js", "three.js", "four.js")
    )
    # Two turns each so both clusters get to ACTIVE.
    for prefill, sid, _ in [(a_prefill, "iso-a", a),
                            (a_prefill, "iso-a", a),
                            (b_prefill, "iso-b", b_root),
                            (b_prefill, "iso-b", b_root)]:
        client.post(
            "/retrieve",
            headers={"x-session-affinity": sid},
            json={"prefill_text": prefill, "query": "anything", "top_k": 4},
        )

    # Verify A's session bound to A's dir, not B's.
    ra = client.post(
        "/retrieve",
        headers={"x-session-affinity": "iso-a"},
        json={"prefill_text": a_prefill, "query": "anything", "top_k": 1},
    )
    da = ra.json()
    assert da["scope_path"] is not None
    assert str(a).lower() in da["scope_path"].lower()
    # And B's bound to B's dir.
    rb = client.post(
        "/retrieve",
        headers={"x-session-affinity": "iso-b"},
        json={"prefill_text": b_prefill, "query": "anything", "top_k": 1},
    )
    db = rb.json()
    assert db["scope_path"] is not None
    assert "other-project" in db["scope_path"].lower()


def test_default_scope_no_longer_pins_engine_cwd(client, project_dir):
    """The killed-fallback contract: an explicit default_scope MUST NOT
    auto-bind when the prefill has zero path mentions. This was the
    Hermes / vllm-swift bug — engine cwd became scope-of-last-resort and
    poisoned the index with unrelated source."""
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "no-cwd-pin"},
        json={
            "prefill_text": "explain how things work",   # zero paths
            "query": "anything",
            "top_k": 4,
            "default_scope": str(project_dir),
        },
    )
    data = response.json()
    assert data["scope_status"] == "no-scope"
    assert data["scope_path"] is None


def test_cluster_walks_up_to_sentinel_when_present(client, project_dir):
    """Convergence test: when the cluster ancestor is INSIDE a sentinel-
    rooted project, the resolved scope should be the sentinel root, not
    the deeper cluster ancestor. Otherwise we'd fragment the index
    across /repo (sentinel path) and /repo/src (cluster path) — same
    project, two scope_hashes, double-indexed.

    Setup: turn 1 mentions enough files under /myapp/ to immediately
    activate the cluster AND sentinel-walk finds package.json. Turn 2
    has NO path mentions, so detect_scope returns None and we fall
    through to cluster. The cluster ancestor sits inside /myapp; the
    sentinel walk-up should pull us back to /myapp (matching turn 1)
    instead of pinning a deeper subdir.
    """
    src = project_dir / "src"
    # Turn 1: 5 unique paths under /myapp/ — clears active_min_files in
    # ONE turn so the cluster is ACTIVE going into turn 2. Sentinel-walk
    # finds /myapp at the same time (package.json), so the cluster work
    # is purely backup history for now.
    prefill_t1 = (
        f"open({src}/auth.ts)\n"
        f"open({src}/billing.ts)\n"
        f"open({project_dir}/README.md)\n"
        f"open({src}/extra_one.ts)\n"
        f"open({src}/extra_two.ts)\n"
    )
    sid = "convergence-test"
    r1 = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": prefill_t1, "query": "auth", "top_k": 4},
    )
    d1 = r1.json()
    assert d1["scope_sentinel"] == "package.json"
    sentinel_hash = d1["scope_hash"]

    # Turn 2: prefill has no paths AT ALL. detect_scope returns None.
    # Tracker still has /myapp ACTIVE from turn 1; cluster ancestor is
    # likely /myapp itself (5 unique under it). Sentinel walk-up finds
    # package.json AT /myapp → same hash as turn 1.
    r2 = client.post(
        "/retrieve",
        headers={"x-session-affinity": sid},
        json={"prefill_text": "now fix the bug", "query": "fix it",
              "top_k": 4},
    )
    d2 = r2.json()
    assert d2["scope_sentinel"] == "package.json"   # walked to sentinel
    assert d2["scope_hash"] == sentinel_hash         # same scope as turn 1


def test_path_mentions_with_sentinel_still_route_correctly(client, project_dir):
    """Confidence test: with both a working session AND a sentinel-rooted
    project, the sentinel-based scope wins — cluster doesn't second-guess
    it. We don't want the cluster heuristic to flap an established
    project root just because mentions are concentrated in a sub-dir."""
    src = project_dir / "src"
    prefill = (
        f"read({src}/auth.ts)\n"
        f"read({src}/billing.ts)\n"
        f"read({project_dir}/README.md)\n"
    )
    response = client.post(
        "/retrieve",
        headers={"x-session-affinity": "sentinel-respected"},
        json={"prefill_text": prefill, "query": "auth flow", "top_k": 4},
    )
    data = response.json()
    assert data["scope_sentinel"] == "package.json"
    assert data["scope_status"] in ("ready", "empty")
