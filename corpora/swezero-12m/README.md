# SWE-ZERO-12M corpus — longctx pack builder

Source: [`AlienKevin/SWE-ZERO-12M-trajectories`](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories)
(Apache 2.0, 36 GB advertised / 34 GB on disk, 12,290,800 rollouts across 3,222
repos and 16+ languages).

## Files in this directory

| File | Purpose |
|---|---|
| `repo_languages.json` | Canonical repo → primary-language map for the 3,222 repos (3,212 tagged + 10 untagged), enriched via GitHub API `/repos/{owner}/{repo}` on 2026-05-18 |
| `ingest.py` | Two-pass ingest: PR-level (122,908 chunks) and rollout-level (12,290,800 chunks). Partitions output into one (SqliteChunkStore, MemmapEmbedStore) pack per top-6 language plus a `misc` pack for the long tail |
| `README.md` | This file |

## Pack scheme

| Pack | Rollouts | % of corpus | Embed memmap |
|---|---|---|---|
| `swezero-go`         | 4,395,600 | 35.8% | 6.29 GB |
| `swezero-python`     | 2,836,200 | 23.1% | 4.06 GB |
| `swezero-rust`       | 1,149,700 |  9.4% | 1.64 GB |
| `swezero-javascript` | 1,129,500 |  9.2% | 1.62 GB |
| `swezero-typescript` | 1,124,400 |  9.1% | 1.61 GB |
| `swezero-java`       |   718,500 |  5.8% | 1.03 GB |
| `swezero-misc` (C, C++, Elixir, PHP, Julia, Kotlin, Scala, Clojure, Dart, Swift, OCaml, R, Lua, HTML, PLpgSQL, untagged) | ~937K | 7.6% | ~1.37 GB |
| **total all packs** | **12,290,800** | **100%** | **~17.6 GB** |

A Python-only user installs `swezero-python` and loads ~4 GB of embeddings
instead of the monolithic 19 GB. The `misc` pack covers the long tail in one
piece so users don't need 16 separate tiny packs.

## Two-pass plan

Matches the PRD validation substrate section:

**Pass 1 — PR-level (122,908 chunks).** Aggregate all rollouts for one PR into
one chunk: `repo + instance_id + first user message + summary of bash commands
across all rollouts`. Validates the pipeline end-to-end at a tractable scale
(comparable to a large Python codebase). Embed time target: < 10 min.

**Pass 2 — rollout-level (12,290,800 chunks).** Each (PR, rollout) is its own
chunk. Headline 12M-chunk scale; closes `longctx Architecture` open item #5
(latency-at-scale). Embed time target: < 2 hours on M5 Max GPU.

Run Pass 1 first, audit retrieval quality on a 50-query sample (≥ 60% useful
top-5), then commit to Pass 2.

## Chunk text strategy (PRD open question #1)

`ingest.py` accepts `--chunk-strategy {command-summary,first-user-only,full}`.

Default `command-summary` follows the PRD lean: extract only assistant bash
commands across the trajectory, drop verbose bash stdout. Tighter semantic
space than `full`, retains more agent-arc context than `first-user-only`.

Worth re-evaluating on Pass 1 if retrieval quality is weak.

## Usage

```bash
# Pass 1 (PR-level, validates pipeline)
python ingest.py --pass pr --chunk-strategy command-summary

# Pass 2 (rollout-level, headline scale)
python ingest.py --pass rollout --chunk-strategy command-summary

# Subset of packs (smoke test on one language)
python ingest.py --pass rollout --packs python

# Drop incompletes (loses ~97% of corpus — generally not recommended)
python ingest.py --pass rollout --exit-filter Submitted
```

Output goes to `~/.cache/longctx/corpora/swezero-12m/<pack>/`.

## Implementation status (2026-05-18)

`ingest.py` is a **structural skeleton**. TODO blocks to fill in:

1. `trajectory_to_chunk_text`: parse `mini-swe-agent-1` format (extract
   ```` ```bash ... ``` ```` from assistant messages)
2. `pr_to_chunk_text`: aggregate distinct bash commands across the 100
   rollouts of one PR (set + frequency)
3. `open_pack_stores`: wire to `longctx_daemon.storage.sqlite_store` +
   `memmap_store` constructors
4. `make_embedder`: `SentenceTransformer('BAAI/bge-small-en-v1.5',
   device='mps')` with batched encode + normalized embeddings
5. `flush`: actually write to the per-pack stores

The shape compiles and the CLI/data-flow scaffolding works. Filling in the
five TODO blocks is what turns this into a runnable pipeline.
