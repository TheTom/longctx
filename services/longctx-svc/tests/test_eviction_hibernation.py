"""Tier 4 of the V3 rescue stack: cold storage / session hibernation.

Round-trips a populated session through the disk-snapshot path and
asserts the resumed store produces the same retrieval results. Also
covers the safety guards: schema-version mismatch, embedder fingerprint
mismatch, empty-session no-op, and the disk-LRU cleanup.

These tests use the same fake embedder as ``test_eviction_store.py`` so
no model weights are downloaded.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from longctx_svc.eviction_store import (
    EmbedderMismatch,
    EvictedChunk,
    EvictionStore,
    HIBERNATION_SCHEMA_VERSION,
    SnapshotSchemaMismatch,
)


class _FakeEmbedder:
    """4-dim deterministic embedder. Shared with test_eviction_store but
    inlined here so the file is self-contained."""

    name = "fake-embedder-v1"

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.lower()[:8]):
                out[i, j % 4] += float(ord(ch))
        return out


def _ev(text: str, layer: int = 0, score: float = 0.0) -> EvictedChunk:
    return EvictedChunk(
        text=text, token_range=(0, len(text)),
        layer=layer, score=score,
    )


@pytest.fixture
def store(tmp_path):
    """A store rooted at tmp_path so each test gets an isolated cold-
    storage directory."""
    return EvictionStore(
        embedder=_FakeEmbedder(),
        hibernation_dir=tmp_path / "snapshots",
    )


# ------------------------------------------------------------ round trip

def test_hibernate_then_resume_preserves_chunks(store):
    """End-to-end: write → hibernate → assert RAM is empty + disk has
    snapshot → resume → retrieve returns same content."""
    store.write("sess-A", [
        _ev("the secret password is ZEBRA42", layer=2, score=0.9),
        _ev("apple banana cherry", layer=5, score=0.3),
        _ev("zzz unrelated noise", layer=1, score=0.1),
    ])
    # Pre-hibernation retrieval baseline
    pre = store.retrieve("sess-A", "what is the password", top_k=1)
    assert len(pre) == 1
    assert "ZEBRA42" in pre[0].text

    # Hibernate
    assert store.hibernate("sess-A") is True
    # RAM should be empty for this session
    assert store.stats()["per_session"].get("sess-A", 0) == 0
    # Disk snapshot should be present
    assert store.is_hibernated("sess-A")

    # Resume
    n = store.resume("sess-A")
    assert n == 3
    # Same query → same top hit
    post = store.retrieve("sess-A", "what is the password", top_k=1)
    assert len(post) == 1
    assert "ZEBRA42" in post[0].text
    # Metadata preserved
    assert post[0].score == pytest.approx(0.9)
    assert post[0].layer == 2


def test_hibernate_empty_session_is_noop(store):
    """No chunks → nothing worth persisting → return False, no disk write."""
    # Session that was never written to
    assert store.hibernate("never-written") is False
    assert not store.is_hibernated("never-written")
    # Session that was cleared
    store.write("sess-X", [_ev("foo")])
    store.clear("sess-X")
    assert store.hibernate("sess-X") is False
    assert not store.is_hibernated("sess-X")


def test_resume_missing_snapshot_returns_zero(store):
    """No snapshot on disk → resume returns 0, no crash."""
    assert store.resume("nonexistent-session") == 0


def test_maybe_resume_autoresumes(store):
    """Convenience: maybe_resume loads from disk only when RAM is empty
    AND snapshot exists."""
    store.write("sess-AR", [_ev("autoresumed chunk one"), _ev("two")])
    store.hibernate("sess-AR")

    # In RAM after auto-resume
    loaded = store.maybe_resume("sess-AR")
    assert loaded == 2
    # Calling again is a no-op (already in RAM)
    assert store.maybe_resume("sess-AR") == 0


def test_maybe_resume_skips_when_already_in_ram(store):
    """RAM hit → maybe_resume must not touch disk."""
    store.write("sess-RAM", [_ev("hot chunk")])
    # Not hibernated yet — maybe_resume is a no-op
    assert store.maybe_resume("sess-RAM") == 0
    # Chunks still in RAM, retrievable
    hits = store.retrieve("sess-RAM", "hot", top_k=1)
    assert len(hits) == 1


# ----------------------------------------------------------- safety guards

def test_resume_refuses_schema_mismatch(store, tmp_path):
    """Snapshot from a future / past schema version must not silently
    deserialise — must raise SnapshotSchemaMismatch so caller can decide."""
    store.write("sess-S", [_ev("payload")])
    store.hibernate("sess-S")
    # Mutate the on-disk metadata to claim a different schema.
    meta_path = store._session_dir("sess-S") / "metadata.json"  # noqa: SLF001
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = HIBERNATION_SCHEMA_VERSION + 99
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(SnapshotSchemaMismatch):
        store.resume("sess-S")


def test_resume_refuses_embedder_fingerprint_mismatch(tmp_path):
    """Snapshot built with embedder-A must not load into a store whose
    current embedder is embedder-B. Embeddings aren't comparable across
    models — silent acceptance would produce nonsense top-K rankings."""
    # Store A: writes + hibernates with fake-embedder-v1
    emb_a = _FakeEmbedder()
    store_a = EvictionStore(
        embedder=emb_a, hibernation_dir=tmp_path / "shared",
    )
    store_a.write("sess-FP", [_ev("hello world")])
    store_a.hibernate("sess-FP")

    # Store B: same disk path, different embedder fingerprint
    class _OtherEmbedder(_FakeEmbedder):
        name = "fake-embedder-v2"

    store_b = EvictionStore(
        embedder=_OtherEmbedder(), hibernation_dir=tmp_path / "shared",
    )
    # Trigger fingerprint capture on store_b
    store_b._ensure_embedder()  # noqa: SLF001
    # Force the fingerprint to differ — _compute_embedder_fingerprint uses
    # falls through the attribute list; both fakes share class so we
    # explicitly mismatch by setting the attr the helper reads first.
    store_b._embedder_fingerprint = "fake-embedder-v2"  # noqa: SLF001
    with pytest.raises(EmbedderMismatch):
        store_b.resume("sess-FP")


def test_resume_pins_fingerprint_when_embedder_not_loaded(tmp_path):
    """When the resuming store has never loaded an embedder, accept the
    snapshot's fingerprint. (First retrieve will validate later.)"""
    emb_a = _FakeEmbedder()
    store_a = EvictionStore(
        embedder=emb_a, hibernation_dir=tmp_path / "shared",
    )
    store_a.write("sess-FP2", [_ev("hello")])
    store_a.hibernate("sess-FP2")

    # Fresh store, no embedder loaded yet
    store_b = EvictionStore(
        embedder=None, hibernation_dir=tmp_path / "shared",
    )
    # _embedder_fingerprint is None before any embedder use
    assert store_b._embedder_fingerprint is None  # noqa: SLF001
    # Should accept the snapshot without raising
    n = store_b.resume("sess-FP2", strict_fingerprint=False)
    assert n == 1


# ----------------------------------------------------------- idle janitor

def test_evict_idle_with_hibernate_writes_snapshot(store):
    """The janitor's default behavior: stale session goes to cold storage
    instead of being dropped."""
    store.write("sess-idle", [_ev("payload one"), _ev("payload two")])
    # Backdate last_access far enough to count as stale
    with store._lock:  # noqa: SLF001
        store._sessions["sess-idle"].last_access = time.time() - 9999  # noqa: SLF001
    n = store.evict_idle(ttl_seconds=10, hibernate=True)
    assert n == 1
    # Should be hibernated, not just dropped
    assert store.is_hibernated("sess-idle")
    assert store.stats()["per_session"].get("sess-idle", 0) == 0


def test_evict_idle_hibernate_false_keeps_legacy_behavior(store):
    """Legacy hard-drop path still works when hibernate=False."""
    store.write("sess-drop", [_ev("payload")])
    with store._lock:  # noqa: SLF001
        store._sessions["sess-drop"].last_access = time.time() - 9999  # noqa: SLF001
    n = store.evict_idle(ttl_seconds=10, hibernate=False)
    assert n == 1
    assert not store.is_hibernated("sess-drop")


# -------------------------------------------------------------- forget + LRU

def test_forget_hibernated_removes_snapshot(store):
    """Explicit delete clears disk state."""
    store.write("sess-forget", [_ev("x")])
    store.hibernate("sess-forget")
    assert store.is_hibernated("sess-forget")
    assert store.forget_hibernated("sess-forget") is True
    assert not store.is_hibernated("sess-forget")
    # Idempotent
    assert store.forget_hibernated("sess-forget") is False


def test_disk_lru_cleanup_drops_old_snapshots(store):
    """Snapshots older than ttl_days are removed."""
    store.write("sess-old", [_ev("old payload")])
    store.write("sess-new", [_ev("new payload")])
    store.hibernate("sess-old")
    store.hibernate("sess-new")
    # Backdate sess-old's metadata to 60 days ago
    meta_path = store._session_dir("sess-old") / "metadata.json"  # noqa: SLF001
    meta = json.loads(meta_path.read_text())
    meta["hibernated_at"] = time.time() - 60 * 86400
    meta_path.write_text(json.dumps(meta))

    removed = store.disk_lru_cleanup(ttl_days=30.0)
    assert removed == 1
    assert not store.is_hibernated("sess-old")
    assert store.is_hibernated("sess-new")


# ----------------------------------------------------------- session-id safety

def test_session_id_with_funky_chars_round_trips(store):
    """Session ids with filesystem-unfriendly characters still snapshot
    + resume cleanly (path-sanitised + hash-suffixed)."""
    weird = "sess/with:weird?chars/and slashes"
    store.write(weird, [_ev("payload")])
    assert store.hibernate(weird) is True
    assert store.is_hibernated(weird)
    n = store.resume(weird)
    assert n == 1
