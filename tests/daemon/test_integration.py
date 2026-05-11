"""End-to-end Phase 2.0 integration tests across all components.

Each test wires the REAL ``SqliteChunkStore`` + ``MemmapEmbedStore``
+ ``Indexer`` + ``Searcher`` (no fakes) against a synthetic on-disk
corpus. Embedder is a tiny deterministic mock — fast, no model
download, but exercises the full data flow including the actual
storage layer and the searcher's scope-decision + RRF fusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from longctx.rag.chunker import Chunker
from longctx_daemon.indexer import Indexer, IndexerConfig
from longctx_daemon.searcher import Searcher, SearcherConfig
from longctx_daemon.storage.memmap_store import MemmapEmbedStore
from longctx_daemon.storage.sqlite_store import SqliteChunkStore


# --------------------------------------------------------------- fixtures

class _DeterministicEmbedder:
    """Tiny embedder used across the integration tests.

    Embeds text into a 4-dim vector keyed on substring presence so the
    cosine ranking is predictable. L2-normalizes per the EmbedStore
    contract. ``fake_sha`` lets the indexer record a stable identity
    without touching the real HF cache."""

    fake_sha = "0" * 64
    model_name = "deterministic-test-embedder"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    @property
    def max_seq_length(self) -> int:
        return 128

    @max_seq_length.setter
    def max_seq_length(self, _v: int) -> None:
        pass

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_,
    ) -> np.ndarray:
        out = []
        for t in texts:
            tl = t.lower()
            v = np.array([
                1.0 if "needle" in tl else 0.0,
                1.0 if "auth" in tl else 0.0,
                1.0 if "billing" in tl else 0.0,
                0.5,
            ], dtype=np.float32)
            n = float(np.linalg.norm(v)) + 1e-8
            out.append(v / n)
        return np.stack(out)


@pytest.fixture
def stack(tmp_path: Path):
    """Real stores + indexer + searcher pointed at a synthetic corpus."""
    chunk_store = SqliteChunkStore(tmp_path / "index.db")
    embedder = _DeterministicEmbedder()
    embed_store = MemmapEmbedStore(
        tmp_path / "embeddings",
        model_name=embedder.model_name,
        model_sha256=embedder.fake_sha,
        dim=embedder.dim,
    )
    chunker = Chunker(tokens_per_chunk=64, respect_sentences=False)
    indexer = Indexer(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        chunker=chunker,
        config=IndexerConfig(embed_batch_size=8),
    )
    searcher = Searcher(
        chunk_store=chunk_store,
        embed_store=embed_store,
        embedder=embedder,
        config=SearcherConfig(),
    )

    yield {
        "chunk_store": chunk_store,
        "embed_store": embed_store,
        "indexer": indexer,
        "searcher": searcher,
        "embedder": embedder,
    }
    chunk_store.close()
    embed_store.close()


def _build_corpus(root: Path, files: dict[str, str]) -> None:
    """Write a synthetic project under ``root``. Marks it as a project
    via a ``.git`` sentinel so future Phase 2.1 auto-discovery picks
    it up too."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


# ---------------------------------------------------- happy-path search

def test_index_then_search_returns_planted_chunk(stack, tmp_path):
    """Full pipeline: build corpus, index it, search for a planted
    needle, verify the right file:line citation comes back."""
    proj = tmp_path / "myapp"
    _build_corpus(proj, {
        "src/auth.py": "def auth_middleware():\n    return True\n",
        "src/billing.py": "def charge():\n    return False\n",
        "docs/notes.md": (
            "# Notes\n\n"
            "the secret needle is planted in this paragraph for testing.\n"
        ),
    })

    stack["indexer"].add_project("myapp", proj)
    scan = stack["indexer"].full_scan("myapp")
    assert scan.n_files == 3
    assert scan.n_chunks_total >= 3

    result = stack["searcher"].search(
        query="needle",
        cwd=str(proj),
        max_tokens=4096,
    )

    assert result.chunks, "expected at least one chunk"
    paths = [c.citation.file_path for c in result.chunks]
    assert any("notes.md" in p for p in paths), \
        f"needle file should rank; got {paths}"
    assert result.scope_decision.primary_project == "myapp"


# ---------------------------------------------------- incremental update

def test_incremental_update_reuses_chunks(stack, tmp_path):
    """Edit one line in a long file; verify ``update_file`` reuses the
    chunks whose ``content_hash`` didn't change. We need a file long
    enough that one edit doesn't ripple through every chunk's overlap
    region — 400 lines comfortably exceeds the default overlap window."""
    proj = tmp_path / "myapp"
    body = "\n".join(f"def func_{i:03d}(): return {i}" for i in range(400))
    _build_corpus(proj, {"src/auth.py": body})

    stack["indexer"].add_project("myapp", proj)
    stack["indexer"].full_scan("myapp")

    # Edit ONE line deep in the file
    target = proj / "src/auth.py"
    text = target.read_text()
    text = text.replace(
        "def func_010(): return 10",
        "def func_010(): return 999  # edited",
    )
    target.write_text(text)

    upd = stack["indexer"].update_file("myapp", target)
    assert upd.n_chunks_total > 0
    assert upd.n_chunks_reused > 0, (
        "incremental should reuse most chunks; got "
        f"{upd.n_chunks_reused}/{upd.n_chunks_total} reused"
    )
    assert upd.n_chunks_new + upd.n_chunks_reused == upd.n_chunks_total
    # And the touched chunk should be small relative to the corpus —
    # if everything got re-embedded the incremental path is broken.
    assert upd.n_chunks_new < upd.n_chunks_total


# ---------------------------------------------------- delete cascade

def test_delete_file_removes_chunks(stack, tmp_path):
    """Deleting a file via the indexer drops its chunks (plus frees
    embedding rows)."""
    proj = tmp_path / "myapp"
    _build_corpus(proj, {
        "src/a.py": "def aaaa(): pass",
        "src/b.py": "def bbbb(): pass",
    })
    stack["indexer"].add_project("myapp", proj)
    stack["indexer"].full_scan("myapp")

    before = stack["chunk_store"].chunk_count()
    assert before >= 2

    stack["indexer"].delete_file("myapp", "src/a.py")

    after = stack["chunk_store"].chunk_count()
    assert after < before


# ---------------------------------------------------- multi-project routing

def test_search_scopes_to_cwd_project(stack, tmp_path):
    """Two projects; cwd-walk picks the right primary."""
    auth = tmp_path / "auth-svc"
    billing = tmp_path / "billing-svc"
    _build_corpus(auth, {
        "src/auth.py": "def auth_handler(): pass",
        "README.md": "auth service handles login",
    })
    _build_corpus(billing, {
        "src/charge.py": "def charge_card(): pass",
        "README.md": "billing service handles invoicing",
    })

    stack["indexer"].add_project("auth-svc", auth)
    stack["indexer"].full_scan("auth-svc")
    stack["indexer"].add_project("billing-svc", billing)
    stack["indexer"].full_scan("billing-svc")

    # cwd inside auth-svc → primary should be auth-svc
    auth_result = stack["searcher"].search(
        query="auth", cwd=str(auth / "src"),
    )
    assert auth_result.scope_decision.primary_project == "auth-svc"
    auth_paths = [c.citation.project for c in auth_result.chunks]
    assert auth_paths.count("auth-svc") >= auth_paths.count("billing-svc")

    # cwd inside billing-svc → primary should flip
    billing_result = stack["searcher"].search(
        query="billing", cwd=str(billing / "src"),
    )
    assert billing_result.scope_decision.primary_project == "billing-svc"


def test_explicit_project_arg_overrides_cwd(stack, tmp_path):
    """``project=`` arg takes precedence over cwd-walk."""
    auth = tmp_path / "auth-svc"
    billing = tmp_path / "billing-svc"
    _build_corpus(auth, {"x.py": "def auth(): pass"})
    _build_corpus(billing, {"y.py": "def billing(): pass"})

    stack["indexer"].add_project("auth-svc", auth)
    stack["indexer"].full_scan("auth-svc")
    stack["indexer"].add_project("billing-svc", billing)
    stack["indexer"].full_scan("billing-svc")

    result = stack["searcher"].search(
        query="anything",
        cwd=str(auth),               # cwd says auth
        project="billing-svc",       # explicit override
    )
    assert result.scope_decision.primary_project == "billing-svc"
    assert result.scope_decision.primary_source == "explicit_project"


# ---------------------------------------------------- persistence

def test_index_survives_close_and_reopen(tmp_path):
    """Close stores; reopen; chunks + BM25 + memmap all intact."""
    proj = tmp_path / "myapp"
    _build_corpus(proj, {
        "src/a.py": "def alpha(): pass\ndef beta(): pass\n" * 10,
    })

    embedder = _DeterministicEmbedder()
    chunk_store = SqliteChunkStore(tmp_path / "index.db")
    embed_store = MemmapEmbedStore(
        tmp_path / "embeddings",
        model_name=embedder.model_name,
        model_sha256=embedder.fake_sha,
        dim=embedder.dim,
    )
    indexer = Indexer(
        chunk_store=chunk_store, embed_store=embed_store,
        embedder=embedder, chunker=Chunker(tokens_per_chunk=64),
        config=IndexerConfig(),
    )
    indexer.add_project("myapp", proj)
    indexer.full_scan("myapp")
    n_chunks = chunk_store.chunk_count()
    assert n_chunks > 0
    chunk_store.close()
    embed_store.close()

    # Reopen fresh
    chunk_store2 = SqliteChunkStore(tmp_path / "index.db")
    embed_store2 = MemmapEmbedStore(
        tmp_path / "embeddings",
        model_name=embedder.model_name,
        model_sha256=embedder.fake_sha,
        dim=embedder.dim,
    )
    assert chunk_store2.chunk_count() == n_chunks

    searcher = Searcher(
        chunk_store=chunk_store2, embed_store=embed_store2,
        embedder=embedder, config=SearcherConfig(),
    )
    result = searcher.search(query="alpha", cwd=str(proj))
    assert result.chunks, "search should work after reopen"

    chunk_store2.close()
    embed_store2.close()


# ---------------------------------------------------- freshness signals

def test_freshness_marks_fresh_when_no_pending(stack, tmp_path):
    """No watcher; pending_updates always 0; is_fully_fresh always True."""
    proj = tmp_path / "myapp"
    _build_corpus(proj, {"x.py": "def x(): pass"})
    stack["indexer"].add_project("myapp", proj)
    stack["indexer"].full_scan("myapp")

    result = stack["searcher"].search(query="x", cwd=str(proj))
    assert result.freshness.is_fully_fresh is True
    assert result.freshness.pending_updates == 0
    assert result.freshness.indexed_through  # ISO string set
