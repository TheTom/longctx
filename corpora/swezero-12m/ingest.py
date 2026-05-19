"""SWE-ZERO 12M-trajectories → per-language longctx packs.

Reads the parquet shards from a local SWE-ZERO-12M-trajectories checkout,
splits rollouts by repo language via the sidecar `repo_languages.json`,
and writes one (SqliteChunkStore, MemmapEmbedStore) pair per pack.

Designed to run on M5 Max with bge-small-en-v1.5. Two-pass plan:

  Pass 1 (PR-level)         122,908 chunks   ~10 min          --pass=pr
  Pass 2 (rollout-level)  12,290,800 chunks   ~1-2 hours      --pass=rollout (default)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.dataset as pads

# Add the longctx-cleanup repo root to sys.path so longctx_daemon imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from longctx_daemon.storage.memmap_store import (
    MemmapEmbedStore,
    _compute_embedder_sha256,
)
from longctx_daemon.storage.sqlite_store import SqliteChunkStore
from longctx_daemon.types import Chunk, FileRecord, Project

# ----- constants -----------------------------------------------------------

TOP_LANGUAGES = {'Go', 'Python', 'Rust', 'JavaScript', 'TypeScript', 'Java'}
MISC_PACK = 'misc'

DEFAULT_DATA_DIR = Path('~/models/SWE-ZERO-12M-trajectories/data').expanduser()
DEFAULT_OUT_DIR = Path('~/.cache/longctx/corpora/swezero-12m').expanduser()
REPO_LANG_FILE = HERE / 'repo_languages.json'

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBED_DIM = 384
EMBED_BATCH = 256
MAX_CHUNK_CHARS = 8000  # ~2K tokens, bge-small ctx window is 512 tokens but
                        # encode() truncates internally

# Regex to pull bash blocks out of mini-swe-agent-1 assistant messages.
# Matches ```bash ... ``` and the looser ``` ... ``` form some agents use.
BASH_BLOCK_RE = re.compile(r'```(?:bash|sh|shell)?\n(.*?)\n```', re.DOTALL)


# ----- repo → pack ---------------------------------------------------------

def load_pack_assignment() -> dict[str, str]:
    """Return {repo: pack_name}. Top-6 by rollout-count get their own pack.
    Everything else (16 langs + 10 untagged repos) folds into 'misc'."""
    repo_lang = json.loads(REPO_LANG_FILE.read_text())
    out = {}
    for repo, lang in repo_lang.items():
        if lang in TOP_LANGUAGES:
            out[repo] = lang.lower()
        else:
            out[repo] = MISC_PACK
    return out


# ----- chunk text strategies (PRD open question #1) ------------------------

def extract_bash_commands(messages: list[dict]) -> list[str]:
    """Pull every ```bash ... ``` block out of assistant messages."""
    cmds: list[str] = []
    for m in messages:
        if m.get('role') != 'assistant':
            continue
        for match in BASH_BLOCK_RE.finditer(m.get('content', '') or ''):
            cmd = match.group(1).strip()
            if cmd:
                cmds.append(cmd)
    return cmds


def first_user_message(messages: list[dict]) -> str:
    """Return the first user-role message body, or '' if absent."""
    for m in messages:
        if m.get('role') == 'user':
            return (m.get('content') or '').strip()
    return ''


def trajectory_to_chunk_text(row: dict, strategy: str) -> str:
    """Produce the text we'll embed for one rollout.

    Strategies (per PRD open question #1):
      - 'command-summary' (default): repo + instance + first user msg +
        bash commands in order. Captures the agent's arc tightly.
      - 'first-user-only': repo + instance + first user message only.
      - 'full': full trajectory including bash stdout (verbose).
    """
    repo = row.get('repo', '')
    instance = row.get('instance_id', '')
    messages = row.get('messages', []) or []
    header = f'repo: {repo}\ninstance: {instance}\n'

    if strategy == 'command-summary':
        first = first_user_message(messages)
        cmds = extract_bash_commands(messages)
        body = f'{first[:1500]}\n\nbash:\n' + '\n'.join(cmds)
        return (header + body)[:MAX_CHUNK_CHARS]
    if strategy == 'first-user-only':
        return (header + first_user_message(messages))[:MAX_CHUNK_CHARS]
    if strategy == 'full':
        parts = [f"[{m['role']}] {m.get('content', '')}" for m in messages]
        return (header + '\n'.join(parts))[:MAX_CHUNK_CHARS]
    raise ValueError(f'unknown chunk strategy: {strategy}')


def pr_to_chunk_text(rollouts: list[dict]) -> str:
    """Aggregate N rollouts for one PR into a single embed-able document.

    Per PRD: repo + instance_id + first user message + bash-command
    summary across all rollouts (set + frequency for the high-recurrence
    strategies the agent converged on).
    """
    repo = rollouts[0]['repo']
    instance = rollouts[0]['instance_id']
    first_user = first_user_message(rollouts[0].get('messages') or [])

    # Aggregate bash commands across all rollouts with frequency
    freq: defaultdict[str, int] = defaultdict(int)
    for r in rollouts:
        for cmd in extract_bash_commands(r.get('messages') or []):
            # Normalize whitespace so near-duplicates collapse
            key = ' '.join(cmd.split())
            freq[key] += 1

    # Top-N most-recurrent commands, sorted by frequency desc
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:60]
    cmd_lines = [f'[{n}x] {cmd[:300]}' for cmd, n in top]

    header = f'repo: {repo}\ninstance: {instance}\nn_rollouts: {len(rollouts)}\n'
    body = f'\nfirst user:\n{first_user[:1500]}\n\nbash command summary:\n' + '\n'.join(cmd_lines)
    return (header + body)[:MAX_CHUNK_CHARS]


# ----- per-pack store init -------------------------------------------------

class PackStores:
    """Bundles a (SqliteChunkStore, MemmapEmbedStore) pair for one pack.
    Owns the project + sentinel-file lifecycle so chunks can be written."""

    def __init__(
        self,
        pack: str,
        out_dir: Path,
        embedder_model: str,
        embedder_sha: str,
    ) -> None:
        pack_dir = out_dir / pack
        pack_dir.mkdir(parents=True, exist_ok=True)
        self.pack = pack
        self.project_name = f'swezero-{pack}'

        self.chunk_store = SqliteChunkStore(pack_dir / 'chunks.sqlite')
        self.embed_store = MemmapEmbedStore(
            pack_dir / 'embeds',
            model_name=embedder_model,
            model_sha256=embedder_sha,
            dim=EMBED_DIM,
            initial_rows=200_000,
        )

        # Register the project lazily (idempotent)
        if not self.chunk_store.get_project(self.project_name):
            self.chunk_store.upsert_project(Project(
                name=self.project_name,
                root_path=str(pack_dir),
                last_full_scan_at=int(time.time()),
            ))

        # Per-(repo, instance) FileRecord cache: rel_path -> file_id
        self._file_id_cache: dict[str, int] = {}
        self._embedder_model = embedder_model
        self._embedder_sha = embedder_sha

    def file_id_for(self, rel_path: str, content_hash: str, size: int) -> int:
        """Get-or-create a FileRecord for this rel_path. Caches the id."""
        if rel_path in self._file_id_cache:
            return self._file_id_cache[rel_path]
        fid = self.chunk_store.upsert_file(FileRecord(
            id=0,
            project=self.project_name,
            rel_path=rel_path,
            mtime=int(time.time()),
            size_bytes=size,
            content_hash=content_hash,
        ))
        self._file_id_cache[rel_path] = fid
        return fid


# ----- embedding -----------------------------------------------------------

def make_embedder() -> tuple[callable, str]:
    """Return (encode_fn, model_sha256). encode_fn: list[str] -> np.float32[N, EMBED_DIM]."""
    from sentence_transformers import SentenceTransformer

    print(f'loading {EMBED_MODEL}...')
    # 'mps' = Apple Metal Performance Shaders (M-series GPU). Falls back
    # to CPU silently if mps isn't available.
    model = SentenceTransformer(EMBED_MODEL, device='mps')
    sha = _compute_embedder_sha256(model)
    print(f'embedder sha256: {sha[:16]}...')

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # required by MemmapEmbedStore validate=True
        ).astype(np.float32)

    return encode, sha


# ----- main streaming loops ------------------------------------------------

def iter_rollouts(data_dir: Path, exit_filter: str | None,
                  shard_limit: int | None) -> Iterable[dict]:
    """Stream rollouts from parquet shards. Memory-bounded via batches."""
    files = sorted(data_dir.glob('train-*.parquet'))
    if shard_limit:
        files = files[:shard_limit]
    print(f'reading {len(files)} parquet shards from {data_dir}')
    dataset = pads.dataset(files, format='parquet')
    columns = ['instance_id', 'repo', 'messages', 'exit_status']
    for batch in dataset.to_batches(columns=columns, batch_size=2000):
        for row in batch.to_pylist():
            if exit_filter and row.get('exit_status') != exit_filter:
                continue
            yield row


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8', errors='replace')).hexdigest()


def _flush_batch(
    pack_stores: PackStores,
    rows_and_texts: list[tuple[dict, str, str]],   # (row, rel_path, text)
    encode,
) -> int:
    """Embed + write a batch of (row, rel_path, text). Returns count written."""
    if not rows_and_texts:
        return 0

    texts = [t for _, _, t in rows_and_texts]
    vecs = encode(texts)
    rows_assigned = pack_stores.embed_store.append_batch(vecs)

    chunks_to_insert: list[Chunk] = []
    for (row, rel_path, text), embed_row in zip(rows_and_texts, rows_assigned):
        chash = _sha256_str(text)
        size = len(text.encode('utf-8'))
        file_id = pack_stores.file_id_for(rel_path, chash, size)
        chunks_to_insert.append(Chunk(
            id=0,
            file_id=file_id,
            chunk_index=0,
            start_offset=0,
            end_offset=size,
            start_line=1,
            end_line=1,
            token_count=max(1, size // 4),
            content_hash=chash,
            text=text,
            embedder_model=pack_stores._embedder_model,
            embedder_sha256=pack_stores._embedder_sha,
            embedding_row=int(embed_row),
        ))
    pack_stores.chunk_store.upsert_chunks(chunks_to_insert)
    return len(chunks_to_insert)


def ingest_rollout_level(args) -> None:
    """Pass 2: one chunk per rollout, partitioned by language pack."""
    pack_of = load_pack_assignment()
    selected_packs = (
        set(args.packs.split(',')) if args.packs else None
    )

    encode, embedder_sha = make_embedder()

    stores: dict[str, PackStores] = {}
    # Stage rollouts per-pack until BUF_LIMIT, then flush as one embed batch.
    buf: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)

    n_seen = 0
    n_written = 0
    t0 = time.time()

    for row in iter_rollouts(args.data_dir, args.exit_filter, args.shard_limit):
        n_seen += 1
        pack = pack_of.get(row['repo'], MISC_PACK)
        if selected_packs and pack not in selected_packs:
            continue

        text = trajectory_to_chunk_text(row, args.chunk_strategy)
        if not text.strip():
            continue
        # Unique rel_path per rollout: instance_id + content-hash prefix
        chash_prefix = _sha256_str(text)[:12]
        rel_path = f"{row['repo']}/{row['instance_id']}.{chash_prefix}"
        buf[pack].append((row, rel_path, text))

        if len(buf[pack]) >= EMBED_BATCH:
            if pack not in stores:
                stores[pack] = PackStores(pack, args.out_dir, EMBED_MODEL, embedder_sha)
            n_written += _flush_batch(stores[pack], buf[pack], encode)
            buf[pack].clear()

        if n_seen % 50_000 == 0:
            rate = n_seen / (time.time() - t0)
            print(f'  seen={n_seen:,} written={n_written:,} '
                  f'rate={rate:.0f}/s active_packs={len(stores)}')

    # Flush remaining
    for pack, items in buf.items():
        if items:
            if pack not in stores:
                stores[pack] = PackStores(pack, args.out_dir, EMBED_MODEL, embedder_sha)
            n_written += _flush_batch(stores[pack], items, encode)

    elapsed = time.time() - t0
    print(f'\ndone. seen={n_seen:,} written={n_written:,} '
          f'elapsed={elapsed/60:.1f}min rate={n_written/elapsed:.0f}/s')


def ingest_pr_level(args) -> None:
    """Pass 1: aggregate N rollouts per PR into one chunk."""
    pack_of = load_pack_assignment()
    selected_packs = set(args.packs.split(',')) if args.packs else None

    # Group rollouts by (repo, instance_id). Memory-bounded by streaming
    # all instances first, then iterating groups.
    print('grouping rollouts by PR (this is the memory-heavy step)...')
    pr_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_rollouts = 0
    for row in iter_rollouts(args.data_dir, args.exit_filter, args.shard_limit):
        pack = pack_of.get(row['repo'], MISC_PACK)
        if selected_packs and pack not in selected_packs:
            continue
        pr_buckets[(row['repo'], row['instance_id'])].append(row)
        n_rollouts += 1
    print(f'grouped {n_rollouts:,} rollouts into {len(pr_buckets):,} PR buckets')

    encode, embedder_sha = make_embedder()
    stores: dict[str, PackStores] = {}
    buf: dict[str, list[tuple[dict, str, str]]] = defaultdict(list)

    n_written = 0
    t0 = time.time()
    for (repo, instance), rollouts in pr_buckets.items():
        pack = pack_of.get(repo, MISC_PACK)
        text = pr_to_chunk_text(rollouts)
        if not text.strip():
            continue
        # PR-level rel_path: no rollout hash; one chunk per PR
        rel_path = f'{repo}/{instance}'
        # Synthesize a row dict so the flush helper has consistent input
        row = {'repo': repo, 'instance_id': instance, 'messages': []}
        buf[pack].append((row, rel_path, text))

        if len(buf[pack]) >= EMBED_BATCH:
            if pack not in stores:
                stores[pack] = PackStores(pack, args.out_dir, EMBED_MODEL, embedder_sha)
            n_written += _flush_batch(stores[pack], buf[pack], encode)
            buf[pack].clear()

    for pack, items in buf.items():
        if items:
            if pack not in stores:
                stores[pack] = PackStores(pack, args.out_dir, EMBED_MODEL, embedder_sha)
            n_written += _flush_batch(stores[pack], items, encode)

    elapsed = time.time() - t0
    print(f'\ndone. PRs={len(pr_buckets):,} written={n_written:,} '
          f'elapsed={elapsed/60:.1f}min')


# ----- CLI -----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument('--pass', dest='pass_kind', choices=['pr', 'rollout'],
                   default='rollout')
    p.add_argument('--chunk-strategy',
                   choices=['command-summary', 'first-user-only', 'full'],
                   default='command-summary')
    p.add_argument('--exit-filter', default=None,
                   help="optional: 'Submitted' to keep only completed rollouts (loses ~97%% of corpus)")
    p.add_argument('--packs', default=None,
                   help='comma-separated subset; default = all 7 packs')
    p.add_argument('--shard-limit', type=int, default=None,
                   help='dev: cap parquet shards read (e.g., 5 = 50K rollouts)')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.pass_kind == 'pr':
        ingest_pr_level(args)
    else:
        ingest_rollout_level(args)


if __name__ == '__main__':
    main()
