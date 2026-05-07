"""Indexer + retrieve unit tests with mocked embedder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from longctx_svc.indexer.builder import build_index
from longctx_svc.retrieve.pipeline import RetrievePipeline


def test_build_index_chunks_files(project_dir: Path, fake_embedder):
    files = [project_dir / "src" / "auth.ts",
             project_dir / "src" / "billing.ts"]
    idx = build_index(project_dir, files, "scope-1", embedder=fake_embedder)
    assert idx.chunk_count >= 2
    assert idx.embeddings is not None
    assert idx.embeddings.shape[0] == idx.chunk_count
    assert idx.file_count == 2
    # Embeddings normalized to unit length
    norms = np.linalg.norm(idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_index_empty_files(project_dir: Path, fake_embedder):
    idx = build_index(project_dir, [], "scope-1", embedder=fake_embedder)
    assert idx.chunk_count == 0
    assert idx.embeddings is None
    assert idx.file_count == 0


def test_retrieve_returns_top_k(project_dir: Path, fake_embedder):
    files = [project_dir / "src" / "auth.ts",
             project_dir / "src" / "billing.ts"]
    idx = build_index(project_dir, files, "scope-1", embedder=fake_embedder)
    p = RetrievePipeline(embedder=fake_embedder, reranker=None,
                         use_multi_query=False)
    result = p.retrieve("auth flow", idx, top_k=1)
    assert len(result.chunks) == 1
    assert "auth" in result.chunks[0].text.lower()


def test_retrieve_uses_multi_query(project_dir: Path, fake_embedder):
    """Multi-query expansion fires above multiquery_min_chunks. Lower the
    threshold for this small fixture so the gate trips."""
    from longctx_svc.config import Limits, ServiceConfig, set_config
    set_config(ServiceConfig(limits=Limits(multiquery_min_chunks=1)))
    try:
        files = [project_dir / "src" / "auth.ts"]
        idx = build_index(project_dir, files, "scope-1", embedder=fake_embedder)
        p = RetrievePipeline(embedder=fake_embedder, reranker=None,
                             use_multi_query=True, n_paraphrases=3)
        result = p.retrieve("auth", idx, top_k=1)
        assert len(result.paraphrases) >= 2
    finally:
        set_config(ServiceConfig())


def test_retrieve_with_rerank(project_dir: Path, fake_embedder, fake_reranker):
    """Rerank fires when the scope is large enough (>= rerank_min_chunks).
    Lower the threshold for this small fixture so the gate trips."""
    from longctx_svc.config import Limits, ServiceConfig, set_config
    set_config(ServiceConfig(limits=Limits(rerank_min_chunks=1)))
    try:
        files = [project_dir / "src" / "auth.ts",
                 project_dir / "src" / "billing.ts"]
        idx = build_index(project_dir, files, "scope-1", embedder=fake_embedder)
        p = RetrievePipeline(embedder=fake_embedder, reranker=fake_reranker,
                             use_multi_query=False)
        result = p.retrieve("auth flow", idx, top_k=1, use_rerank=True)
        assert result.used_rerank is True
    finally:
        set_config(ServiceConfig())


def test_retrieve_handles_empty_index(fake_embedder):
    from longctx_svc.indexer.builder import ScopeIndex
    empty_idx = ScopeIndex(scope_root=Path("/x"), scope_hash="x",
                            chunks=[], embeddings=None)
    p = RetrievePipeline(embedder=fake_embedder, reranker=None)
    result = p.retrieve("anything", empty_idx, top_k=8)
    assert result.chunks == []
    assert result.scores == []


# --- Scale-aware retrieval gates (Tom 2026-05-07: 7s /retrieve was too slow) ---

def test_retrieve_skips_rerank_when_below_threshold(fake_embedder, tmp_path):
    """Small scopes (<rerank_min_chunks) skip the cross-encoder for speed.
    cosine R@8 is already strong there; the rerank's 5s CPU pass isn't
    worth it on a 50-file repo."""
    import time as _time
    import numpy as np
    from longctx_svc.config import Limits, ServiceConfig, set_config
    from longctx_svc.indexer.builder import ScopeIndex
    from longctx_svc.indexer.chunker import Chunk
    from longctx_svc.retrieve.pipeline import RetrievePipeline

    set_config(ServiceConfig(
        limits=Limits(rerank_min_chunks=200, multiquery_min_chunks=500),
    ))

    class _FakeReranker:
        def __init__(self):
            self.calls = 0
        def predict(self, pairs, **_):
            self.calls += 1
            return np.zeros(len(pairs), dtype=np.float32)

    rr = _FakeReranker()
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=rr)

    # Small index (10 chunks) → rerank should be skipped
    chunks = [Chunk(text=f"chunk {i}", file_path="/p/x.py",
                    start_line=i, end_line=i, file_type="code")
              for i in range(10)]
    embs = np.random.randn(10, 4).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    idx = ScopeIndex(scope_root=tmp_path, scope_hash="small",
                     chunks=chunks, embeddings=embs,
                     file_count=10, built_at=_time.time())

    res = pipe.retrieve("anything", idx, top_k=4)
    assert rr.calls == 0, "reranker fired on a small scope"
    assert res.used_rerank is False
    set_config(ServiceConfig())


def test_retrieve_uses_rerank_when_above_threshold(fake_embedder, tmp_path):
    """Big scope (>=rerank_min_chunks) → rerank fires."""
    import time as _time
    import numpy as np
    from longctx_svc.config import Limits, ServiceConfig, set_config
    from longctx_svc.indexer.builder import ScopeIndex
    from longctx_svc.indexer.chunker import Chunk
    from longctx_svc.retrieve.pipeline import RetrievePipeline

    set_config(ServiceConfig(
        limits=Limits(rerank_min_chunks=50),  # easy threshold for the test
    ))

    class _FakeReranker:
        def __init__(self):
            self.calls = 0
        def predict(self, pairs, **_):
            self.calls += 1
            return np.array([float(len(p[1])) for p in pairs],
                             dtype=np.float32)

    rr = _FakeReranker()
    pipe = RetrievePipeline(embedder=fake_embedder, reranker=rr)
    chunks = [Chunk(text=f"chunk {i}", file_path="/p/x.py",
                    start_line=i, end_line=i, file_type="code")
              for i in range(100)]
    embs = np.random.randn(100, 4).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    idx = ScopeIndex(scope_root=tmp_path, scope_hash="big",
                     chunks=chunks, embeddings=embs,
                     file_count=100, built_at=_time.time())

    res = pipe.retrieve("anything", idx, top_k=4)
    assert rr.calls == 1, "reranker did not fire on a large scope"
    assert res.used_rerank is True
    set_config(ServiceConfig())


def test_multi_query_skipped_below_threshold(fake_embedder, tmp_path):
    """multiquery_min_chunks gates the 4× embed expansion."""
    import time as _time
    import numpy as np
    from longctx_svc.config import Limits, ServiceConfig, set_config
    from longctx_svc.indexer.builder import ScopeIndex
    from longctx_svc.indexer.chunker import Chunk
    from longctx_svc.retrieve.pipeline import RetrievePipeline

    set_config(ServiceConfig(
        limits=Limits(multiquery_min_chunks=500),
    ))
    pipe = RetrievePipeline(
        embedder=fake_embedder, reranker=None, use_multi_query=True,
    )
    chunks = [Chunk(text=f"chunk {i}", file_path="/p/x.py",
                    start_line=i, end_line=i, file_type="code")
              for i in range(30)]
    embs = np.random.randn(30, 4).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    idx = ScopeIndex(scope_root=tmp_path, scope_hash="small-mq",
                     chunks=chunks, embeddings=embs,
                     file_count=30, built_at=_time.time())

    res = pipe.retrieve("auth flow", idx, top_k=4)
    # paraphrases collapsed to just the original query
    assert res.paraphrases == ["auth flow"]
    set_config(ServiceConfig())
