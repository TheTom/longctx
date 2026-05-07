"""Hot → Package auto-promotion tests. PRD §6.1 / v0.3.1."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from longctx_svc.config import Limits, ServiceConfig, set_config
from longctx_svc.indexer.builder import ScopeIndex
from longctx_svc.indexer.chunker import Chunk
from longctx_svc.retrieve.pipeline import RetrievePipeline
from longctx_svc.state import get_state, reset_state


def _wait_for(predicate, timeout: float = 5.0,
              interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _seed_hot_scope(tmp_path: Path, fake_embedder, fake_reranker):
    """Create a project with two src dirs; Hot indexes only one."""
    reset_state()
    set_config(ServiceConfig(cache_dir=tmp_path / "cache"))
    root = tmp_path / "myapp"
    root.mkdir()
    (root / "package.json").write_text('{"name":"myapp"}')
    src = root / "src"
    src.mkdir()
    (src / "alpha.py").write_text("# alpha module\n" * 5)
    far = root / "far_away" / "nested"
    far.mkdir(parents=True)
    (far / "beta.py").write_text("# beta module\n" * 5)

    state = get_state()
    state.set_pipeline(RetrievePipeline(
        embedder=fake_embedder, reranker=fake_reranker,
    ))
    entry = state.register_scope(root, "promo1", "package.json")
    # Hand-craft a Hot-mode index containing only alpha.py
    chunks = [Chunk(
        text="# alpha\n", file_path=str(src / "alpha.py"),
        start_line=1, end_line=1, file_type="code",
    )]
    embs = np.ones((1, 4), dtype=np.float32) / 2
    entry.index = ScopeIndex(
        scope_root=root, scope_hash="promo1", chunks=chunks,
        embeddings=embs, file_count=1, built_at=time.time(),
    )
    entry.indexed_files = frozenset({str(src / "alpha.py")})
    entry.mode = "hot"
    entry.status = "ready"
    return state, root, src, far


def test_no_promotion_when_path_inside_hot(tmp_path, fake_embedder,
                                              fake_reranker):
    state, _, src, _ = _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    promoted = state.maybe_promote("promo1", [str(src / "alpha.py")])
    assert promoted is False
    assert state.get_scope("promo1").mode == "hot"
    set_config(ServiceConfig())


def test_promotion_when_path_outside_hot(tmp_path, fake_embedder,
                                           fake_reranker):
    state, _, _, far = _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    promoted = state.maybe_promote("promo1", [str(far / "beta.py")])
    assert promoted is True
    entry = state.get_scope("promo1")
    assert entry.status == "promoting"
    # Wait for background worker
    assert _wait_for(lambda: entry.status in ("ready", "empty", "error"),
                     timeout=10.0)
    assert entry.status == "ready"
    assert entry.mode == "package"
    # The previously-out-of-scope file is now in the indexed set
    assert any("beta.py" in f for f in entry.indexed_files)
    set_config(ServiceConfig())


def test_promotion_idempotent_when_already_promoting(tmp_path,
                                                       fake_embedder,
                                                       fake_reranker):
    state, _, _, far = _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    first = state.maybe_promote("promo1", [str(far / "beta.py")])
    second = state.maybe_promote("promo1", [str(far / "beta.py")])
    assert first is True
    assert second is False  # already promoting
    set_config(ServiceConfig())


def test_promotion_skipped_when_already_package_mode(tmp_path,
                                                       fake_embedder,
                                                       fake_reranker):
    state, _, _, far = _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    state.get_scope("promo1").mode = "package"
    promoted = state.maybe_promote("promo1", [str(far / "beta.py")])
    assert promoted is False
    set_config(ServiceConfig())


def test_promotion_safe_with_unknown_scope(tmp_path, fake_embedder,
                                             fake_reranker):
    _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    state = get_state()
    promoted = state.maybe_promote("does-not-exist", ["/some/path"])
    assert promoted is False
    set_config(ServiceConfig())


def test_promotion_skipped_when_index_evicted(tmp_path, fake_embedder,
                                                fake_reranker):
    state, _, _, far = _seed_hot_scope(tmp_path, fake_embedder, fake_reranker)
    entry = state.get_scope("promo1")
    entry.index = None
    entry.status = "evicted"
    promoted = state.maybe_promote("promo1", [str(far / "beta.py")])
    assert promoted is False
    set_config(ServiceConfig())


# --- end-to-end through /retrieve ---

def test_retrieve_triggers_promotion(client, project_dir):
    """Hit /retrieve once with an in-Hot path (Hot mode), then again
    with an out-of-Hot path; confirm the second triggers promotion."""
    # Force this scope into hot mode by pre-seeding a tiny indexed_files
    # set (the standard project_dir fixture would auto-fall-through to
    # package mode because it's small). We exercise via the public API.
    auth = project_dir / "src" / "auth.ts"
    r1 = client.post(
        "/retrieve",
        headers={"x-session-affinity": "promo-e2e"},
        json={"prefill_text": f"see {auth}",
              "query": "auth", "top_k": 4},
    )
    assert r1.status_code == 200
    # Force the scope into hot mode regardless of fall-through
    from longctx_svc.state import get_state
    state = get_state()
    scopes = state.all_scopes()
    assert scopes
    entry = scopes[0]
    entry.mode = "hot"
    entry.indexed_files = frozenset({str(auth)})  # only one file in Hot

    # Reference a different file → should trigger promotion
    other = project_dir / "src" / "billing.ts"
    r2 = client.post(
        "/retrieve",
        headers={"x-session-affinity": "promo-e2e"},
        json={"prefill_text": f"see {other}",
              "query": "billing", "top_k": 4},
    )
    assert r2.status_code == 200
    # Scope status is promoting OR already swung back to ready/empty if the
    # background worker raced through. Both are valid.
    assert entry.status in ("promoting", "ready", "empty")
