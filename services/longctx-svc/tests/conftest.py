"""Shared pytest fixtures for longctx-svc tests."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Disable the background janitor across the test suite — we exercise
# eviction synchronously via state.evict_idle_indexes().
os.environ.setdefault("LONGCTX_NO_JANITOR", "1")


@pytest.fixture
def fake_embedder():
    """Returns deterministic 4-dim embeddings keyed by substring presence."""

    class _FakeEmbedder:
        def encode(self, texts, convert_to_numpy=True,
                   normalize_embeddings=True, batch_size=32, **_):
            out = []
            for t in texts:
                low = t.lower()
                vec = np.array([
                    1.0 if "alpha" in low else 0.0,
                    1.0 if "beta" in low else 0.0,
                    1.0 if "gamma" in low else 0.0,
                    1.0 if "auth" in low else 0.5,
                ], dtype=np.float32)
                n = np.linalg.norm(vec) + 1e-8
                out.append(vec / n)
            return np.stack(out)

    return _FakeEmbedder()


@pytest.fixture
def fake_reranker():
    """CrossEncoder mock that scores by token overlap."""
    class _FakeReranker:
        def predict(self, pairs, batch_size=16, show_progress_bar=False, **_):
            scores = []
            for q, c in pairs:
                qs = set(q.lower().split())
                cs = set(c.lower().split())
                scores.append(float(len(qs & cs)))
            return np.array(scores, dtype=np.float32)

    return _FakeReranker()


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Build a small fake project tree with a sentinel file."""
    root = tmp_path / "myapp"
    root.mkdir()
    (root / "package.json").write_text('{"name": "myapp"}\n')
    (root / "README.md").write_text(
        "# myapp\n\nFirst paragraph about alpha.\n\n"
        "Second paragraph about beta usage.\n"
    )
    src = root / "src"
    src.mkdir()
    (src / "auth.ts").write_text(
        "export function authMiddleware() {\n"
        "  // alpha auth flow\n"
        "  return true;\n"
        "}\n"
    )
    (src / "billing.ts").write_text(
        "export function chargeUser() {\n"
        "  // beta billing flow\n"
        "  return false;\n"
        "}\n"
    )
    # gitignore that drops dist/
    (root / ".gitignore").write_text("dist/\n*.log\n")
    dist = root / "dist"
    dist.mkdir()
    (dist / "ignored.js").write_text("// should be skipped\n")
    return root


@pytest.fixture(autouse=True)
def fresh_state():
    """Reset the singleton service state between tests."""
    from longctx_svc.state import reset_state
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client(fake_embedder, fake_reranker):
    """FastAPI TestClient with the SentenceTransformer + CrossEncoder
    constructors patched so tests run with deterministic fakes (no
    downloads, no GPU)."""
    with patch("sentence_transformers.SentenceTransformer",
               return_value=fake_embedder), \
         patch("sentence_transformers.CrossEncoder",
               return_value=fake_reranker):
        from longctx_svc.app import app
        from longctx_svc.retrieve.pipeline import RetrievePipeline
        from longctx_svc.state import get_state
        get_state().set_pipeline(RetrievePipeline(
            embedder=fake_embedder,
            reranker=fake_reranker,
        ))
        with TestClient(app) as c:
            yield c
