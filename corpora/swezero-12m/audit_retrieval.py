"""Manual quality audit for swezero-12m packs.

Loads one pack, encodes a small set of canonical queries, prints top-5
retrieved chunks with similarity scores. Eyeball judgement — does the
top-5 look semantically relevant to the query?

Usage:
    python audit_retrieval.py --pack python
    python audit_retrieval.py --pack go --queries my_queries.txt
"""
from __future__ import annotations

import argparse
import json
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

DEFAULT_PACK_ROOT = Path('~/.cache/longctx/corpora/swezero-12m').expanduser()
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBED_DIM = 384

# Canonical sample queries — code-agent intent shapes the corpus should answer
DEFAULT_QUERIES = [
    'fix a flaky test that intermittently fails on CI',
    'add type annotations to a python function',
    'rename a variable across the entire codebase',
    'debug a null pointer exception in production',
    'write a unit test for a regex parser',
    'add error handling to a database query',
    'optimize a slow nested loop with O(n^2) complexity',
    'refactor a long function into smaller helpers',
    'fix an off-by-one bug in pagination logic',
    'add logging to track an intermittent issue',
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--pack', required=True,
                   help='pack name: python, go, rust, javascript, typescript, java, misc')
    p.add_argument('--root', type=Path, default=DEFAULT_PACK_ROOT)
    p.add_argument('--queries', type=Path, default=None,
                   help='optional newline-separated query file; default = built-in 10')
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()

    pack_dir = args.root / args.pack
    if not pack_dir.exists():
        sys.exit(f'pack dir not found: {pack_dir}')

    print(f'loading {args.pack} pack from {pack_dir}')
    chunk_store = SqliteChunkStore(pack_dir / 'chunks.sqlite')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device='mps')
    sha = _compute_embedder_sha256(model)
    embed_store = MemmapEmbedStore(
        pack_dir / 'embeds',
        model_name=EMBED_MODEL,
        model_sha256=sha,
        dim=EMBED_DIM,
    )

    n_rows = embed_store.num_rows
    print(f'pack stats: {n_rows:,} embedded rows (per memmap)')

    # Start from sqlite-committed chunks (which is the authoritative set).
    # Concurrent Pass 2 writes embed-then-sqlite, so memmap may have rows
    # that don't yet have chunk metadata — those aren't searchable.
    print('loading committed chunks from sqlite...')
    import sqlite3
    conn = sqlite3.connect(str(pack_dir / 'chunks.sqlite'))
    rows = conn.execute(
        'SELECT id, embedding_row, text FROM chunks WHERE embedding_row IS NOT NULL'
    ).fetchall()
    conn.close()
    print(f'committed chunks: {len(rows):,}')
    if not rows:
        sys.exit('no committed chunks yet — retry in a minute')

    chunk_ids = np.array([r[0] for r in rows], dtype=np.int64)
    embed_rows = np.array([r[1] for r in rows], dtype=np.int64)
    texts = [r[2] for r in rows]

    print(f'loading {len(embed_rows):,} embeddings into RAM...')
    # Bulk-read all relevant rows. For 100K-500K chunks this is fast and bounded.
    vecs = np.asarray(embed_store._mm[embed_rows], dtype=np.float32)

    queries = (args.queries.read_text().splitlines() if args.queries else DEFAULT_QUERIES)
    queries = [q.strip() for q in queries if q.strip()]

    for q in queries:
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        scores = vecs @ qv  # cosine on normalized vectors
        top_idx = np.argpartition(-scores, args.top_k)[:args.top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        print(f'\n{"=" * 78}\nQUERY: {q}\n{"=" * 78}')
        for rank, i in enumerate(top_idx, 1):
            preview = ' '.join(texts[i].split())[:280]
            print(f'  #{rank}  score={scores[i]:.4f}  {preview}')


if __name__ == '__main__':
    main()
