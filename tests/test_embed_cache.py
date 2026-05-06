"""Tests for the disk-backed embedding cache + GPU autodetect."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from longctx.rag.embed_cache import EmbedCache, _default_cache_dir
from longctx.rag.pipeline import _resolve_device


# --- _resolve_device ---

def test_resolve_device_explicit_passthrough():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("mps") == "mps"


def test_resolve_device_auto_falls_through_to_cpu():
    """When neither CUDA nor MPS is reachable, fall back to CPU."""
    with patch.dict("sys.modules", {"torch": None}):
        # Force torch import to fail
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _resolve_device("auto") == "cpu"


def test_resolve_device_auto_picks_cuda_when_available():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _resolve_device("auto") == "cuda"


def test_resolve_device_auto_picks_mps_when_only_mps_available():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _resolve_device("auto") == "mps"


# --- EmbedCache ---

def test_cache_disabled_when_dir_is_none():
    c = EmbedCache(cache_dir=None)
    assert not c.enabled
    assert c.get("anything") is None


def test_cache_set_and_get_roundtrip(tmp_path):
    c = EmbedCache(cache_dir=tmp_path, embedder_name="test-model")
    assert c.enabled
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c.set("hello world", emb)
    out = c.get("hello world")
    assert out is not None
    np.testing.assert_array_equal(out, emb)


def test_cache_miss_returns_none(tmp_path):
    c = EmbedCache(cache_dir=tmp_path)
    assert c.get("never seen") is None


def test_cache_get_handles_load_error(tmp_path):
    """Corrupt cache file should return None, not raise."""
    c = EmbedCache(cache_dir=tmp_path)
    path = c._path_for("victim")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a numpy file")
    assert c.get("victim") is None


def test_cache_set_handles_oserror(tmp_path, monkeypatch):
    """Disk errors during set should be silent."""
    c = EmbedCache(cache_dir=tmp_path)

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", boom)
    # Should not raise
    c.set("key", np.zeros(3))


def test_cache_invalid_dir_disables_cache(monkeypatch):
    """If we can't create the cache dir, the cache should silently disable."""
    def boom(*a, **kw):
        raise OSError("read-only")
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "mkdir", boom)
    c = EmbedCache(cache_dir="/tmp/nonexistent-readonly-path-xyz")
    assert not c.enabled


def test_default_cache_dir_uses_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("LONGCTX_CACHE_DIR", str(tmp_path))
    assert _default_cache_dir() == tmp_path


def test_default_cache_dir_falls_through_to_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("LONGCTX_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _default_cache_dir() == tmp_path / "longctx"


# --- encode_with_cache ---

class _CountingEmbedder:
    """Mock embedder. Tracks how many times encode() was called and
    on how many texts."""

    def __init__(self):
        self.encode_calls = 0
        self.texts_seen: list[str] = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               batch_size=64, **_):
        self.encode_calls += 1
        self.texts_seen.extend(texts)
        return np.array(
            [[hash(t) % 1000 / 1000.0, 0.1, 0.2] for t in texts],
            dtype=np.float32,
        )


def test_encode_with_cache_hits_cache_on_second_call(tmp_path):
    c = EmbedCache(cache_dir=tmp_path)
    embedder = _CountingEmbedder()
    texts = ["alpha alpha", "beta beta", "gamma gamma"]
    out1 = c.encode_with_cache(embedder, texts)
    assert embedder.encode_calls == 1
    assert len(embedder.texts_seen) == 3

    # Second call with identical texts: cache should hit; embedder NOT called
    out2 = c.encode_with_cache(embedder, texts)
    assert embedder.encode_calls == 1  # unchanged
    np.testing.assert_array_equal(out1, out2)


def test_encode_with_cache_partial_hit(tmp_path):
    """Mixing cached and uncached chunks: only the new ones get embedded."""
    c = EmbedCache(cache_dir=tmp_path)
    embedder = _CountingEmbedder()
    c.encode_with_cache(embedder, ["a", "b"])  # warms cache
    embedder_calls_after_warm = embedder.encode_calls
    seen_after_warm = list(embedder.texts_seen)

    # New batch: 'a' cached, 'c' new
    out = c.encode_with_cache(embedder, ["a", "c", "b"])
    assert out.shape == (3, 3)
    # Only 'c' should be newly embedded (1 additional encode call)
    assert embedder.encode_calls == embedder_calls_after_warm + 1
    new_texts = embedder.texts_seen[len(seen_after_warm):]
    assert new_texts == ["c"]


def test_encode_with_cache_disabled_falls_through(tmp_path):
    """When cache_dir=None, every call hits the embedder fresh."""
    c = EmbedCache(cache_dir=None)
    embedder = _CountingEmbedder()
    c.encode_with_cache(embedder, ["a", "b"])
    c.encode_with_cache(embedder, ["a", "b"])
    assert embedder.encode_calls == 2  # cache disabled, both calls land


# --- end-to-end: pipeline uses cache ---

def test_pipeline_passes_cache_through_to_retrieve(tmp_path):
    """Verify RetrievalPipeline.retrieve() actually exercises the cache."""
    fake_emb = _CountingEmbedder()
    with patch(
        "sentence_transformers.SentenceTransformer", return_value=fake_emb
    ):
        from longctx import RetrievalPipeline
        p = RetrievalPipeline(cache_dir=tmp_path, device="cpu")

    candidates = ["alpha doc", "beta doc", "gamma doc"]
    p.retrieve(query="alpha", candidates=candidates, top_k=1)
    p.retrieve(query="alpha", candidates=candidates, top_k=1)
    # Two retrieve calls but candidates only embedded once (plus 2 query embeds)
    # query embeds aren't cached, so total encode calls = 1 (cands) + 2 (queries)
    assert fake_emb.encode_calls == 3
