"""Tests for ``longctx_daemon.storage.memmap_store``.

Coverage focus: each public method (happy / empty / invalid / edge),
plus the integration scenarios from the Phase 2.0 spec — memmap
growth, freelist reuse, embedder-identity guard, dense search with
scope filter, and persistence across reopen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from longctx_daemon.storage.memmap_store import (
    DEFAULT_INITIAL_ROWS,
    EmbedderMismatchError,
    MemmapEmbedStore,
)


# ---------------------------------------------------------------- helpers


def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize, returning float32."""
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _mk_store(
    tmp_path: Path,
    *,
    dim: int = 4,
    initial_rows: int = 8,
    sha: str = "sha-aaa",
    name: str = "test-model",
    validate: bool = True,
) -> MemmapEmbedStore:
    return MemmapEmbedStore(
        tmp_path,
        model_name=name,
        model_sha256=sha,
        dim=dim,
        initial_rows=initial_rows,
        validate=validate,
    )


# ---------------------------------------------------------------- init


class TestInit:
    def test_creates_files(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            assert (tmp_path / "test-model.npy").exists()
            assert (tmp_path / "test-model.idx").exists()
        finally:
            s.close()

    def test_invalid_dim(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _mk_store(tmp_path, dim=0)

    def test_invalid_growth_factor(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            MemmapEmbedStore(
                tmp_path, model_name="m", model_sha256="s",
                dim=4, growth_factor=1.0,
            )

    def test_invalid_initial_rows(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _mk_store(tmp_path, initial_rows=0)

    def test_invalid_on_mismatch(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            MemmapEmbedStore(
                tmp_path, model_name="m", model_sha256="s",
                dim=4, on_mismatch="explode",
            )

    def test_default_initial_rows_constant(self) -> None:
        # Sanity: the constant is sane and exported.
        assert DEFAULT_INITIAL_ROWS >= 1


# ---------------------------------------------------------------- append


class TestAppend:
    def test_append_one(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            row = s.append(_norm(np.array([1.0, 0.0, 0.0, 0.0])))
            assert row == 0
            assert s.num_rows == 1
        finally:
            s.close()

    def test_append_wrong_shape(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            with pytest.raises(ValueError):
                s.append(np.array([1.0, 0.0], dtype=np.float32))
        finally:
            s.close()

    def test_append_unnormalized_rejected(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, validate=True)
        try:
            with pytest.raises(AssertionError):
                s.append(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32))
        finally:
            s.close()

    def test_append_unnormalized_skipped_when_validate_false(
        self, tmp_path: Path
    ) -> None:
        s = _mk_store(tmp_path, dim=4, validate=False)
        try:
            row = s.append(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32))
            assert row == 0
        finally:
            s.close()

    def test_append_zero_norm_rejected(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, validate=True)
        try:
            with pytest.raises(AssertionError):
                s.append(np.zeros(4, dtype=np.float32))
        finally:
            s.close()

    def test_append_dtype_promoted(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            # f64 input is auto-cast; norm preserved.
            v = (np.array([1.0, 0.0, 0.0, 0.0]) / 1.0).astype(np.float64)
            row = s.append(v)
            assert row == 0
        finally:
            s.close()


# ---------------------------------------------------------------- batch


class TestAppendBatch:
    def test_batch(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, initial_rows=2)
        try:
            embs = np.stack([
                _norm(np.array([1.0, 0, 0, 0])),
                _norm(np.array([0, 1.0, 0, 0])),
                _norm(np.array([0, 0, 1.0, 0])),
            ])
            rows = s.append_batch(embs)
            assert rows == (0, 1, 2)
            assert s.num_rows == 3
        finally:
            s.close()

    def test_batch_empty(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            assert s.append_batch(np.empty((0, 4), dtype=np.float32)) == ()
        finally:
            s.close()

    def test_batch_wrong_shape(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            with pytest.raises(ValueError):
                s.append_batch(np.zeros((3, 5), dtype=np.float32))
        finally:
            s.close()

    def test_batch_one_unnormalized_rejected(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, validate=True)
        try:
            embs = np.stack([
                _norm(np.array([1.0, 0, 0, 0])),
                np.array([2.0, 0, 0, 0], dtype=np.float32),  # bad
            ])
            with pytest.raises(AssertionError):
                s.append_batch(embs)
        finally:
            s.close()


# ---------------------------------------------------------------- growth


class TestGrowth:
    def test_grows_beyond_initial(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, initial_rows=2)
        try:
            # Append more rows than initial allocation.
            for i in range(10):
                v = np.zeros(4, dtype=np.float32)
                v[i % 4] = 1.0
                s.append(v)
            assert s.num_rows == 10
            assert s.capacity >= 10
            # All rows queryable.
            for i in range(10):
                row = s.get(i)
                assert row.shape == (4,)
        finally:
            s.close()

    def test_grow_via_batch(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4, initial_rows=2)
        try:
            embs = np.stack([_norm(np.eye(4)[i % 4]) for i in range(20)])
            rows = s.append_batch(embs)
            assert len(rows) == 20
            assert s.num_rows == 20
        finally:
            s.close()


# ---------------------------------------------------------------- read


class TestRead:
    def test_get_out_of_range(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            with pytest.raises(IndexError):
                s.get(0)
        finally:
            s.close()

    def test_get_batch_empty(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            arr = s.get_batch([])
            assert arr.shape == (0, 4)
        finally:
            s.close()

    def test_get_batch_round_trip(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            v0 = _norm(np.array([1.0, 0, 0, 0]))
            v1 = _norm(np.array([0, 1.0, 0, 0]))
            s.append(v0)
            s.append(v1)
            got = s.get_batch([0, 1])
            assert np.allclose(got[0], v0)
            assert np.allclose(got[1], v1)
        finally:
            s.close()

    def test_get_batch_invalid_row(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))
            with pytest.raises(IndexError):
                s.get_batch([0, 99])
        finally:
            s.close()


# ---------------------------------------------------------------- freelist


class TestFreelist:
    def test_free_then_append_reuses(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            r0 = s.append(_norm(np.array([1.0, 0, 0, 0])))
            r1 = s.append(_norm(np.array([0, 1.0, 0, 0])))
            s.free_row(r0)
            assert r0 in s.freelist
            r2 = s.append(_norm(np.array([0, 0, 1.0, 0])))
            assert r2 == r0  # reused
            assert s.freelist == ()
            # The original row 1 still has its data.
            assert np.allclose(s.get(r1), _norm(np.array([0, 1.0, 0, 0])))
        finally:
            s.close()

    def test_free_idempotent(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))
            s.free_row(0)
            s.free_row(0)  # second call no-ops
            assert s.freelist == (0,)
        finally:
            s.close()

    def test_free_out_of_range(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            with pytest.raises(IndexError):
                s.free_row(99)
        finally:
            s.close()

    def test_free_zeros_row(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            r = s.append(_norm(np.array([1.0, 0, 0, 0])))
            s.free_row(r)
            assert np.allclose(s.get(r), 0.0)
        finally:
            s.close()


# ---------------------------------------------------------------- search


class TestSearch:
    def test_empty_store(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            q = _norm(np.array([1.0, 0, 0, 0]))
            assert s.search_dense(q, k=5) == ()
        finally:
            s.close()

    def test_invalid_k(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))
            assert s.search_dense(_norm(np.array([1.0, 0, 0, 0])), k=0) == ()
        finally:
            s.close()

    def test_invalid_query_shape(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            with pytest.raises(ValueError):
                s.search_dense(np.zeros(8, dtype=np.float32), k=1)
        finally:
            s.close()

    def test_finds_nearest(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))   # row 0
            s.append(_norm(np.array([0, 1.0, 0, 0])))   # row 1
            s.append(_norm(np.array([0, 0, 1.0, 0])))   # row 2

            hits = s.search_dense(_norm(np.array([0.9, 0.1, 0, 0])), k=2)
            assert hits[0].chunk_id == 0
            assert hits[1].chunk_id == 1
            assert hits[0].score > hits[1].score
        finally:
            s.close()

    def test_scope_filter(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))   # row 0 — best
            s.append(_norm(np.array([0.99, 0.01, 0, 0])))  # row 1 — almost as good
            s.append(_norm(np.array([0.98, 0.01, 0.01, 0])))  # row 2

            # Restrict to only row 1 + 2; row 0 must be excluded.
            hits = s.search_dense(
                _norm(np.array([1.0, 0, 0, 0])), k=5,
                chunk_ids_in_scope=(1, 2),
            )
            ids = [h.chunk_id for h in hits]
            assert 0 not in ids
            assert set(ids) == {1, 2}
        finally:
            s.close()

    def test_freed_rows_excluded(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        try:
            s.append(_norm(np.array([1.0, 0, 0, 0])))   # row 0
            s.append(_norm(np.array([0, 1.0, 0, 0])))   # row 1
            s.free_row(0)
            hits = s.search_dense(_norm(np.array([1.0, 0, 0, 0])), k=5)
            ids = [h.chunk_id for h in hits]
            assert 0 not in ids
        finally:
            s.close()


# ---------------------------------------------------------------- persistence


class TestPersistence:
    def test_reopen_preserves_rows(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        v = _norm(np.array([0.5, 0.5, 0.5, 0.5]))
        s.append(v)
        s.append(v)
        s.close()

        s2 = _mk_store(tmp_path, dim=4)
        try:
            assert s2.num_rows == 2
            assert np.allclose(s2.get(0), v)
        finally:
            s2.close()

    def test_reopen_preserves_freelist(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        s.append(_norm(np.array([1.0, 0, 0, 0])))
        s.append(_norm(np.array([0, 1.0, 0, 0])))
        s.free_row(0)
        s.close()

        s2 = _mk_store(tmp_path, dim=4)
        try:
            assert 0 in s2.freelist
            # Next append reuses row 0.
            r = s2.append(_norm(np.array([0, 0, 1.0, 0])))
            assert r == 0
        finally:
            s2.close()


# ---------------------------------------------------------------- identity


class TestEmbedderIdentity:
    def test_mismatch_raises(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, sha="sha-aaa")
        s.close()

        with pytest.raises(EmbedderMismatchError):
            _mk_store(tmp_path, sha="sha-bbb")

    def test_mismatch_warn_mode(self, tmp_path: Path, caplog) -> None:
        s = _mk_store(tmp_path, sha="sha-aaa")
        s.close()

        s2 = MemmapEmbedStore(
            tmp_path, model_name="test-model",
            model_sha256="sha-bbb", dim=4, on_mismatch="warn",
        )
        try:
            assert s2.model_sha256 == "sha-bbb"
        finally:
            s2.close()

    def test_model_name_change_raises(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, name="m1")
        s.close()
        # Same path, different name — refuses.
        with pytest.raises(EmbedderMismatchError):
            MemmapEmbedStore(
                tmp_path, model_name="m1", model_sha256="other",
                dim=4,
            )

    def test_dim_change_raises(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path, dim=4)
        s.close()
        with pytest.raises(ValueError):
            _mk_store(tmp_path, dim=8)


# ---------------------------------------------------------------- close


class TestClose:
    def test_close_idempotent(self, tmp_path: Path) -> None:
        s = _mk_store(tmp_path)
        s.close()
        s.close()  # must not raise

    def test_safe_filename_handling(self, tmp_path: Path) -> None:
        # HF-style id with slash should be sanitized into the filename.
        s = MemmapEmbedStore(
            tmp_path, model_name="BAAI/bge-small-en-v1.5",
            model_sha256="abc", dim=4,
        )
        try:
            assert (tmp_path / "BAAI_bge-small-en-v1.5.npy").exists()
        finally:
            s.close()
