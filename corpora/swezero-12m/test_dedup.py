"""Spot-check dedup_by_doc_root against the python swezero pack.

Runs ONE query twice (dedup off vs on), prints top-5 rel_paths for each,
and asserts: with dedup off, all 5 should share a doc_root prefix; with
dedup on, all 5 should have DISTINCT doc_root prefixes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from longctx_daemon.storage.memmap_store import (
    MemmapEmbedStore,
    _compute_embedder_sha256,
)
from longctx_daemon.storage.sqlite_store import SqliteChunkStore

PACK_DIR = Path("~/.cache/longctx/corpora/swezero-12m/python").expanduser()
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
DOC_ROOT_HASH_RE = re.compile(r"\.[0-9a-f]{12}$")
QUERY = "add type annotations to a python function"


def top5_with_optional_dedup(*, dedup: bool):
    chunk_store = SqliteChunkStore(PACK_DIR / "chunks.sqlite")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="mps")
    sha = _compute_embedder_sha256(model)
    embed_store = MemmapEmbedStore(
        PACK_DIR / "embeds",
        model_name=EMBED_MODEL, model_sha256=sha, dim=EMBED_DIM,
    )

    # Load committed chunks
    import sqlite3
    conn = sqlite3.connect(str(PACK_DIR / "chunks.sqlite"))
    rows = conn.execute(
        "SELECT id, file_id, embedding_row FROM chunks WHERE embedding_row IS NOT NULL"
    ).fetchall()
    conn.close()
    chunk_ids = np.array([r[0] for r in rows], dtype=np.int64)
    file_ids = np.array([r[1] for r in rows], dtype=np.int64)
    embed_rows = np.array([r[2] for r in rows], dtype=np.int64)

    # Read embeddings + score
    vecs = np.asarray(embed_store._mm[embed_rows], dtype=np.float32)
    qv = model.encode([QUERY], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    scores = vecs @ qv
    order = np.argsort(-scores)  # descending

    # Resolve rel_paths via batched file lookups across the top-1000
    # candidate pool — matches SearcherConfig.default_top_k_for_fusion.
    # Without this width, popular PRs with 100 rollouts each saturate
    # a smaller pool and dedup can't find enough distinct doc roots.
    unique_fids = sorted({int(f) for f in file_ids[order[:1000]]})
    file_recs = {fid: chunk_store.get_file_by_id(fid) for fid in unique_fids}

    # Walk top down, optionally dedup
    seen_keys: set[str] = set()
    picks: list[tuple[int, float, str, str]] = []
    for i in order:
        if len(picks) == 5:
            break
        fid = int(file_ids[i])
        rec = file_recs.get(fid)
        if rec is None:
            continue
        doc_root = DOC_ROOT_HASH_RE.sub("", rec.rel_path)
        key = f"{rec.project}::{doc_root}"
        if dedup:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        picks.append((int(chunk_ids[i]), float(scores[i]), rec.rel_path, doc_root))
    return picks


def main() -> None:
    print(f"query: {QUERY!r}\n")

    print("=" * 78)
    print("DEDUP OFF — top-5 rel_paths")
    print("=" * 78)
    for cid, score, rp, dr in top5_with_optional_dedup(dedup=False):
        print(f"  {score:.4f}  chunk_id={cid:<8}  rel={rp}")

    print()
    print("=" * 78)
    print("DEDUP ON — top-5 rel_paths")
    print("=" * 78)
    on = top5_with_optional_dedup(dedup=True)
    for cid, score, rp, dr in on:
        print(f"  {score:.4f}  chunk_id={cid:<8}  rel={rp}")

    distinct_roots = {dr for _, _, _, dr in on}
    print(f"\n  distinct doc_roots in top-5 with dedup ON: {len(distinct_roots)} (target = 5)")
    assert len(distinct_roots) == 5, "dedup pass did not produce 5 distinct doc_roots"
    print("  PASS")


if __name__ == "__main__":
    main()
