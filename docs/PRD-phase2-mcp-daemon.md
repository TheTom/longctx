# Phase 2 — MCP Server, Persistent Index, Multi-Root Daemon

**Status:** Addendum to `longctx-selector-spec.md`. Phase 1 ships the retrieval pipeline as a library (`CoarseFilter`, `Chunker`, `RetrievalPipeline.retrieve_chunked` integration, `longctx-svc` BM25 fusion lane). Phase 2 ships it as an always-on daemon with an MCP interface, exposed to any agent on the machine.

**Audience:** A coding agent implementing on top of the merged Phase 1 codebase.

**Driving usage pattern:** A developer opens multiple terminals, each running an agent (Claude Code, Cursor, opencode, Hermes, Pi, custom OSS harness) cd'd into different projects under `~/dev/`. Sessions sometimes reference each other. The user does not want to think about which project is "the corpus" — longctx auto-discovers projects, each session gets its own scope from its cwd, the inference server is location-agnostic and unaware of retrieval. Configuration is "install once, run forever".

---

## 1. The shape of the deliverable

After Phase 2, the user experience is:

```bash
# Install once
brew install longctx
longctx init                         # auto-discovers projects under ~/dev, writes config
longctx service install              # registers launchd plist (macOS) or systemd unit (Linux)
longctx service start

# Use forever
# Open Claude Code / Cursor / opencode / Hermes / any MCP-aware agent
# Agent automatically discovers longctx via MCP
# Agent passes its cwd; longctx scopes retrieval to the right project
# User edits files; watcher updates the index; agent always sees current state
```

The user runs `longctx` commands twice in their lifetime: install + init. After that, longctx is infrastructure they don't think about.

Sub-phases (ship in order; each is independently useful):

| Phase | Scope | Effort |
|---|---|---|
| **2.0** | Single-project, foreground, stdio-MCP only, persistent SQLite + memmap | 2-3 days |
| **2.1** | Multi-root auto-discovery, daemon mode, SSE transport, singleton, port discovery | 1 week |
| **2.2** | Watcher with agent-pace updates + staleness signaling | 4 days |
| **2.3** | macOS launchd integration + first-time wizard | 3 days |
| **2.4** | Linux systemd, Windows Service, hardening, full test matrix | 1 week |

Total realistic: **3-4 weeks for one engineer, ~2 weeks with an agent.** Not 9 days. MCP SSE reconnection, launchd quirks, file watcher reliability across platforms, atomic-rename detection — each is more work than it looks.

---

## 2. Architecture — three roles, cleanly separated

```
┌─────────────────────┐  MCP   ┌──────────────────┐
│  agent              │───────▶│ longctx daemon   │
│  (claude-code,      │        │ (single process, │
│   cursor, opencode, │◀───────│  ~/dev indexed)  │
│   hermes, pi, ...)  │ chunks │                  │
└──────────┬──────────┘        └──────────────────┘
           │ OpenAI HTTP
           │ (system + chunks + user msg)
           ▼
┌─────────────────────┐
│  inference server   │  knows nothing about retrieval
│  (vllm-swift, llama-│  serves any model, location-agnostic
│  server, ollama,    │
│  vllm, sglang ...)  │
└─────────────────────┘
```

| component | knows about | does NOT know |
|---|---|---|
| **agent** | session cwd, user intent, context budget, when/whether to retrieve | how chunks are embedded, where projects live |
| **longctx daemon** | all projects under `~/dev`, embeddings, chunk index, file watching | which model the agent uses, how splicing happens |
| **inference server** | how to run a model | retrieval, projects, sessions |

The agent is the orchestrator. The inference server is a dumb model. The daemon is a search index. They don't talk to each other; only the agent does. This is why launching `vllm-swift` from any directory is fine — it has no opinion about projects.

Implications for legacy v0.3 svc design:

- `--enable-longctx` flag on inference servers becomes legacy / optional, not the main story
- `--longctx-scope <path>` goes away — scope comes from the agent's cwd, not the server's launch dir
- Splice budget caps inside `response_rewriter` go away — the agent owns its own context budget
- The Phase 1 BM25 fusion lane wired into `longctx-svc` still works for headless setups but is not the primary path

Daemon properties:

- **One process, many clients.** Multiple agent sessions share the daemon
- **Persistent across restarts.** Index survives daemon restart; only changed files re-embed
- **Concurrent reads, serialized embeds.** Search is parallel; embedding is one-at-a-time (model is the bottleneck)
- **Resource bounded.** Memory footprint targets <2GB for ~13M-token corpus regardless of session count

---

## 3. Auto-discovery, multi-root corpus, per-session scope

### 3.1 Auto-discovery is the default

`longctx init` walks the user's `parent_dir` (default `~/dev`) and picks up every directory containing a sentinel file. Sentinels in priority order:

```
.git, pyproject.toml, package.json, pnpm-workspace.yaml,
Cargo.toml, go.mod, Package.swift, WORKSPACE, BUILD.bazel,
pom.xml, Gemfile, .longctxinclude
```

Default behavior: index every match. User edits config later only to **remove** things, never to add.

A `.longctxignore` file at any project root marks that project as "don't index, even if discovered". Same syntax as `.gitignore`.

### 3.2 Config file format (defaults, not requirements)

`~/.config/longctx/config.toml` (Linux) / `~/Library/Application Support/longctx/config.toml` (macOS) / `%APPDATA%\longctx\config.toml` (Windows). Resolved via `platformdirs`.

```toml
[corpus]
parent_dir = "~/dev"

# Auto-discover via sentinels (default true). When true, `include`
# acts as a deny-list against discovered projects. When false,
# `include` is the explicit allow-list.
auto_discover = true

# Sentinel file/directory names that mark a project root
sentinels = [
    ".git", "pyproject.toml", "package.json", "pnpm-workspace.yaml",
    "Cargo.toml", "go.mod", "Package.swift", "WORKSPACE", "pom.xml",
    "Gemfile", ".longctxinclude",
]

# Optional explicit list — only used if auto_discover = false
include = []

# Always exclude these patterns (relative to any indexed directory)
exclude = [
    "**/node_modules", "**/.git/objects", "**/__pycache__",
    "**/.venv", "**/venv", "**/.pytest_cache",
    "**/target", "**/build", "**/dist", "**/.next", "**/.svelte-kit",
    "**/*.lock", "**/*.min.js", "**/*.map",
]

# Always-excluded secrets patterns — NOT user-removable without
# the explicit override flag below
secret_patterns = [
    ".env*", "*.key", "*.pem", "id_rsa*", "id_ed25519*",
    "*.p12", "*.pfx", "**/secrets/**", "**/credentials/**",
    "**/.aws/**", "**/.ssh/**", "**/.gnupg/**",
]
allow_secret_patterns = false   # set true with --i-know-what-im-doing

# Refuse to index any PROJECT whose root-directory name matches these.
# Distinct from secret_patterns: secret_patterns is a per-file glob
# (matches `~/dev/myapp/.env`, `~/dev/anything/secrets/foo.txt`, etc.);
# forbidden_dirs is a project-root check (refuses `~/dev/secrets/`
# as a project but would still index `~/dev/myapp/secrets/foo.txt`
# only if its content didn't match secret_patterns).
forbidden_dirs = ["secrets", "credentials", ".aws", ".ssh", ".gnupg"]

respect_gitignore = true
max_file_size_kb = 1024

# File extensions to index. Empty = all text-detected files.
include_extensions = []

[server]
# Preferred ports; daemon walks forward if taken (see §8)
mcp_port_preferred = 8765
status_port_preferred = 8766
mcp_host = "127.0.0.1"

# MCP transports to enable. "sse" supports multi-client; "streamable-http"
# is the newer spec; "stdio" is for one-client-per-process tools.
mcp_transports = ["sse", "streamable-http"]

[index]
embedder = "BAAI/bge-small-en-v1.5"
chunk_size = 2048
chunk_overlap = 128

# Tier 3 — disk budget cap. When > 0, eviction kicks in to keep
# total cache size under this many GB; oldest-queried projects go
# first. Default 0 = unlimited. Don't surprise users with missing
# data; let power users opt into the cap explicitly.
disk_budget_gb = 0

[watcher]
debounce_ms = 200                # agent-pace, not human-pace
queue_size = 10000
periodic_mtime_sweep_seconds = 30  # watcher-missed-it fallback

[cleanup]
# Tier 1 — drop projects whose root_path has been missing for this
# many days. Grace handles network-drive unmount + USB removal +
# `git worktree remove` cases. 0 = drop immediately, no grace.
missing_root_grace_days = 7
periodic_check_seconds = 3600

# Tier 2 — pause file watchers for projects not queried in this
# many days. Saves inotify slots / FSEvents subs. Periodic mtime
# sweep still catches changes if the user rotates back. -1 disables.
watcher_idle_pause_days = 30

[search]
# Default time the search call will block waiting for in-flight
# index updates to drain before running. Agents that want
# immediacy can pass 0; agents that just wrote files can pass more.
default_wait_for_quiescence_ms = 500

[logging]
level = "INFO"
file_logging = true
```

### 3.3 Per-session scope from cwd

Every MCP call carries the agent's `cwd` (or active-file path). The daemon walks up from cwd to the nearest sentinel → that's the session's "primary project". Search results are biased toward it; other projects remain searchable.

```
search_codebase(query="...", cwd="/Users/tom/dev/longctx/src/foo.py")
   → primary_project = "longctx"
   → returns: longctx-weighted top-K, with high-scoring cross-project hits
```

If `cwd` is outside any indexed project (e.g., `~/dev/obsidian` is the launch dir but not a sentinel-marked project), no primary → fan out across all indexed projects with global ranking.

### 3.4 Sticky session context

Each MCP connection has a session ID. Agent can call `set_active_project("longctx")` once; subsequent searches honor it without re-passing cwd. Three terminals → three independent session contexts in the daemon, no cross-contamination. Connection drops → context drops.

### 3.5 Cross-project query routing

When the query mentions a project name explicitly — `"in longctx"`, `"the mlx-swift-lm centroid"`, `"in the auth-svc"` — that overrides session context. Cheap pre-filter: detect `<project_name>` substrings in the query, weight matching projects up.

**Tiebreaker rules** (avoid misrouting on common-word project names):

1. **Project name matches session's primary project** → no override (user is using the name conversationally inside its own scope, not pointing elsewhere).
2. **Project name appears with a syntactic cue** — `"in <name>"`, `"the <name> repo"`, `"<name> module"`, `"<name>'s ..."`, etc. → override fires.
3. **Project name appears bare AND is a common English word** (matches a small dictionary like "core", "auth", "lib", "api", "client", "server", "tools", "utils", "platform") → require the syntactic cue from rule 2; otherwise treat as a content word and don't override.
4. **Project name is distinctive** (any name that isn't in the common-word list) → bare mention is enough to override.

| query | session cwd | scope |
|---|---|---|
| `"where does longctx do multi-query fusion"` | longctx | longctx (rule 1: matches session) |
| `"how does mlx-swift handle centroid"` | longctx | mlx-swift-lm (rule 4: distinctive) |
| `"where do I handle auth"` | myapp | myapp (rule 3: "auth" is a common word, no cue → not a project mention) |
| `"in the auth project, where do we rate-limit"` | myapp | auth (rule 2: syntactic cue overrides commonality) |
| `"compare auth in longctx vs mlx-swift-lm"` | anywhere | both, fanned (rule 2: "in <name>" cue) |

### 3.6 Multi-project ranking normalization

When fanning across projects, RRF rank-based fusion already normalizes implicitly (a small project's chunks aren't drowned by a large project's score magnitudes). For per-project hit-rate parity, also reweight by `1/log(num_chunks_in_project)` so a 5K-chunk project doesn't dominate a 50-chunk project on irrelevant queries.

### 3.7 Live project add/remove

The watcher monitors `parent_dir` itself. `mkdir ~/dev/newthing && cd newthing && git init` → sentinel detected → new project queued for indexing in background. `rm -rf ~/dev/oldthing` → daemon drops it from the index. No `longctx reload` needed for project-set changes.

### 3.8 Ad-hoc out-of-tree projects

If `claude-code` runs in `~/code/something-else` (outside `parent_dir`), the agent can call `add_project(path, persist=False)` — daemon indexes that path for the session.

**Storage semantics for `persist=False`:**

- Index lives **on disk** under `~/.cache/longctx/sessions/<session_id>/`, not in memory. Re-using an in-progress index across reconnects from the same agent is cheap.
- A `session_id` column on the `projects` table marks session-bound projects.
- On daemon startup, GC any session-bound project whose creating session is no longer alive (no MCP connection holding it). Handles crash / orphan cases without leaving stale state.
- On clean disconnect, GC immediately.
- `persist=True` removes the `session_id` and writes the project to the user's config so it survives daemon restart.

The on-disk path is chosen because (a) re-indexing on every reconnect is too slow for large ad-hoc projects, (b) GC-on-startup handles orphans correctly, and (c) it reuses the same `ChunkStore` machinery without a separate in-memory implementation.

### 3.9 Config precedence

```
1. Per-call MCP arguments (cwd, project_hint)
2. Sticky session state (set_active_project)
3. CLI flags (--corpus-dir, etc. for direct invocations)
4. Environment variables (LONGCTX_*)
5. Config file
6. Auto-discovery defaults
```

### 3.10 Reload without restart

`longctx reload` sends SIGHUP. Daemon re-reads config, diffs against current state, reconciles. Existing MCP connections stay connected. Embedder change requires explicit confirmation (see §5.5) — typo in a config file should not silently nuke an 8-hour embed.

---

## 4. Persistent index

### 4.1 Storage layout

```
~/.cache/longctx/
├── index.db                    # SQLite (WAL): chunks, files, projects
├── embeddings/                 # numpy memmap files
│   ├── bge-small-en-v1.5.npy   # one file per embedder model
│   └── bge-small-en-v1.5.idx
├── bm25_stats.pkl              # pickled BM25Okapi (validated against chunks on start)
├── server.info                 # JSON, runtime port + pid (see §8)
└── server.lock                 # flock-based singleton
```

Files at mode 0600. Directory at 0700. No world-readable indexes.

### 4.2 Storage as a Protocol, not a hard-coded backend

```python
class ChunkStore(Protocol):
    def search_dense(self, query_emb: np.ndarray, k: int, filter: ScopeFilter) -> list[Hit]: ...
    def search_lexical(self, query_terms: list[str], k: int, filter: ScopeFilter) -> list[Hit]: ...
    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def delete_file(self, file_id: int) -> None: ...
    def list_projects(self) -> list[Project]: ...
    # ...
```

Today's implementation: `SqliteChunkStore` + `MemmapEmbedStore`. Future: `LanceChunkStore`, `QdrantChunkStore` for >100M token corpora. Same interface; only storage swaps.

### 4.3 SQLite schema

```sql
CREATE TABLE projects (
    name        TEXT PRIMARY KEY,
    root_path   TEXT NOT NULL UNIQUE,
    last_full_scan_at  INTEGER NOT NULL
);

CREATE TABLE files (
    id          INTEGER PRIMARY KEY,
    project     TEXT NOT NULL REFERENCES projects(name),
    rel_path    TEXT NOT NULL,
    mtime       INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(project, rel_path)
);
CREATE INDEX idx_files_project ON files(project);

CREATE TABLE chunks (
    id              INTEGER PRIMARY KEY,
    file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    start_offset    INTEGER NOT NULL,
    end_offset      INTEGER NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    text            TEXT NOT NULL,
    embedder_model  TEXT NOT NULL,
    embedder_sha256 TEXT NOT NULL,    -- actual model file hash
    embedding_row   INTEGER             -- index into memmap, NULL if not yet embedded
);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

### 4.4 Embedding identity = HF id + actual SHA256

Storing only the HF model id is unsafe — upstream model swaps invalidate caches silently. Store the actual model-file SHA256 alongside. On daemon start, compute the SHA256 of the loaded embedder; if it differs from the recorded value, that's a model-version change and chunks need re-embedding.

### 4.5 Incremental update protocol

Watcher fires for a changed file:

```python
def update_file(file_path: Path):
    project = find_owning_project(file_path)
    rel_path = file_path.relative_to(project.root)
    new_content = read_text(file_path)
    new_hash = sha256(new_content)

    existing = db.get_file(project.name, rel_path)
    if existing and existing.content_hash == new_hash:
        return  # no actual content change (touch, etc.)

    new_chunks = chunker.chunk(new_content, file_path)
    old_by_hash = {c.content_hash: c for c in db.get_chunks(file_id=existing.id)}
    chunks_to_add, chunks_to_keep = [], []
    for nc in new_chunks:
        if nc.content_hash in old_by_hash:
            chunks_to_keep.append((nc, old_by_hash[nc.content_hash]))
        else:
            chunks_to_add.append(nc)

    new_embs = embedder.embed_batch([c.text for c in chunks_to_add])

    with db.transaction():
        db.delete_chunks(file_id=existing.id)
        for nc, old in chunks_to_keep:
            db.insert_chunk(nc, embedding_row=old.embedding_row)
        for nc, emb in zip(chunks_to_add, new_embs):
            row = memmap.append(emb)
            db.insert_chunk(nc, embedding_row=row)
        db.update_file(project.name, rel_path, ...)

    bm25_stats.invalidate(file_id=existing.id)
```

Properties:

- **Content-addressed reuse** — if a 10K-line file changes one line, only chunks containing that line re-embed
- **Atomic update via DB transaction** — concurrent searches see either old or new state, never partial
- **Memmap rows reused** for kept chunks (no rewrite); new rows appended for added chunks
- **BM25 stats persisted** as a pickled `BM25Okapi` at `bm25_stats.pkl`. On daemon startup: load from disk, validate that the chunk set matches the SQLite `chunks` count + content_hash digest. If validation fails, rebuild from scratch (~2-10s on 13.4M tokens). On incremental updates, mark dirty + rebuild lazily in the background; persist the rebuilt index back to disk after each successful pass.

### 4.6 Embedder change requires explicit confirmation

User edits config → `embedder = "BAAI/bge-base-en-v1.5"` → `longctx reload`:

- Daemon detects a different embedder
- Refuses to silently rebuild N hours of embeddings
- Logs a clear WARN with chunk count + estimated time
- Requires `longctx reembed --confirm` to proceed
- During the reembed, search keeps working against old embeddings; atomic swap on completion

---

## 5. Watcher — agent-pace, not human-pace

### 5.1 Library choice

Use **`watchfiles`** (Rust-backed, faster, better atomic-rename detection), not `watchdog`. High event rates (refactor storms, `git checkout`) are common in agent workflows; the Python-only watcher drops events under load.

### 5.2 Filter chain

Per the corpus config, filters applied in order (cheap-to-expensive) to every event:

```python
def should_index(file_path: Path) -> bool:
    project = find_owning_project(file_path)
    if project is None: return False
    rel = file_path.relative_to(project.root)

    # Secret patterns — always-excluded
    for pattern in config.secret_patterns:
        if rel.match(pattern):
            return False

    # User exclude patterns
    for pattern in config.exclude_patterns:
        if rel.match(pattern):
            return False

    # Gitignore
    if config.respect_gitignore and gitignore_matchers[project.name].is_ignored(rel):
        return False

    # Extension allowlist
    if config.include_extensions and file_path.suffix not in config.include_extensions:
        return False

    # Size cap
    try:
        if file_path.stat().st_size > config.max_file_size_kb * 1024:
            return False
    except OSError:
        return False

    # Text detection (skip binary)
    if not is_text_file(file_path): return False

    return True
```

### 5.3 Debounce + batched embed

Default debounce **200ms** (agent-pace). Worker thread wakes when events are ready, **drains all pending events**, groups chunks into one batched embed pass, commits atomically. Avoids per-file model loop.

A 20-file refactor → ~40K tokens → ~20 chunks → batched embed on MPS ~100-200ms. Index fully fresh in <500ms after the last file write.

Bounded queue (default 10K events). On overflow:

- Don't drop silently
- Log WARN
- Flip the affected project into "rescan-pending" mode
- Worker walks the project, diffs file mtimes against DB, schedules updates

### 5.4 Full event vocabulary

| event | handling |
|---|---|
| CREATE | new file → chunk → batched embed → insert |
| MODIFY | re-chunk → diff old/new content hashes → embed only new chunks |
| DELETE | drop chunks + free memmap rows |
| RENAME | update `files.rel_path`, keep chunks (text unchanged → no re-embed) |
| editor atomic-rename (vim-style) | dedupe rapid DELETE+CREATE on same path within debounce → treat as MODIFY |

Cheap RENAME handling matters: agents do `git mv` and bulk renames during refactors. Re-embedding 50 unchanged files because of a rename would waste minutes.

### 5.5 Periodic mtime sweep — watcher-missed-it fallback

Even good watchers miss events under heavy load (FSEvents coalesces, inotify exhausts). Every project gets a periodic mtime walk every 30s (configurable) that diffs file mtimes against DB and catches misses. Cheap because mtime stat is fast — only chunks files whose mtime > recorded.

The same loop also performs the cleanup checks from §12.4 — Tier 1 missing-root detection and Tier 3 disk-budget enforcement piggyback on the sweep so we don't pay for two separate periodic walkers.

### 5.6 Cold-project watcher pause

Projects not queried for `watcher_idle_pause_days` (default 30) AND with no recent file events get their watch unsubscribed. Frees OS resources (inotify slots have a per-process cap; FSEvents subscriptions accumulate cost). Periodic mtime sweep still catches changes if the user rotates back; first query post-rotation pays a one-time mtime-walk before serving fresh results. Disk + index unchanged. See §12.4 Tier 2 for the full story.

### 5.6 New-directory detection

`mkdir src/auth && touch src/auth/middleware.ts` must register the new directory + its descendants. `watchfiles` handles nested-mkdir on macOS; `watchdog` is shaky there.

### 5.7 Documented limitations

Watch loops on Docker volumes, bind mounts, network drives are unreliable across all platforms. Document. Provide `longctx reindex <path>` for explicit re-walk when the watcher can't help.

---

## 6. MCP server interface

### 6.1 Transports

Support **all three** common transports:

- **stdio** — one client per process. Suitable for Claude Desktop's command-launch mode. `longctx mcp-stdio` reads the daemon's `server.info` and bridges stdio↔SSE to the running daemon.
- **sse** — HTTP server, multi-client. The daemon mode default.
- **streamable-http** — newer MCP spec. Some clients only support this; ship both alongside `sse`.

Don't pick one and lose a client ecosystem.

### 6.2 Tools exposed

```python
@mcp_server.tool
def search_codebase(
    query: str,
    cwd: Optional[str] = None,
    project: Optional[str] = None,
    max_tokens: int = 4096,
    max_results: Optional[int] = None,
    wait_for_quiescence_ms: Optional[int] = None,
) -> SearchResponse:
    """Search the indexed codebase for code, comments, or documentation
    relevant to the query. Returns top-matching chunks with file paths,
    line numbers, and relevance scores.

    Args:
        query: Natural language description of what to find. A question,
            a code-pattern description, a feature name, or an identifier.
        cwd: Optional path the agent is currently working in. Used to bias
            results toward the active project. If None, falls back to the
            session's sticky active project (set via set_active_project).
        project: Optional project name to restrict search to. Available
            projects can be enumerated via list_projects.
        max_tokens: Cap the total token count of returned chunks. Default
            4096. Agents with smaller context windows pass less.
        max_results: Optional cap on chunk count. Use max_tokens preferentially;
            this is a fallback for callers that prefer count-based limits.
        wait_for_quiescence_ms: Block up to N ms waiting for in-flight index
            updates to drain before searching. Default 500. Pass 0 for
            zero-wait; pass 2000 if the agent just wrote many files.

    Returns:
        SearchResponse with:
            chunks: list of {project, file_path, start_line, end_line,
                              text, relevance_score}
            stale_files: list of files whose mtime > last_indexed_at
            pending_updates: queue size at response time (0 = fully fresh)
            indexed_through: ISO timestamp of last completed index update
    """
    ...

@mcp_server.tool
def find_related(file_path: str, line: Optional[int] = None,
                 max_results: int = 5) -> list[dict]:
    """Find chunks semantically similar to the chunk at file_path:line
    (or to the whole file if line is None).

    Phase 2 implementation: dense embedding similarity only. The chunk at
    file_path:line is embedded; the top-K nearest other chunks are
    returned. Useful for "show me similar implementations in other parts
    of the codebase" or "what else looks like this".

    NOT a static-analysis tool — it does not parse callers/callees,
    follow imports, or resolve symbol references. Static-analysis-aware
    variants (callers_of, callees_of, references_to) are planned for
    Phase 3+ and will be separate tools so the agent never has to guess
    which kind of "related" it asked for."""
    ...

@mcp_server.tool
def list_projects() -> list[dict]:
    """List all projects currently indexed, with stats:
    name, root_path, file_count, chunk_count, token_count, last_updated."""
    ...

@mcp_server.tool
def set_active_project(project: str) -> None:
    """Set the sticky active project for this MCP session. Subsequent
    search_codebase calls without cwd or project args use this scope."""
    ...

@mcp_server.tool
def add_project(path: str, persist: bool = False) -> dict:
    """Index a new directory. persist=True writes to config (survives
    daemon restart); persist=False indexes for this session only."""
    ...

@mcp_server.tool
def wait_for_quiescence(project: Optional[str] = None,
                         timeout_ms: int = 2000) -> dict:
    """Block until the index has no pending updates for the given project
    (or all projects). Returns when queue is empty or timeout fires.
    Useful between 'I just wrote N files' and 'now search for them'."""
    ...

@mcp_server.tool
def index_status() -> dict:
    """Current daemon state: status, total_chunks, pending_updates,
    embedder_model + sha256, last_full_scan, per-project stats."""
    ...
```

### 6.3 Search response invariants

Every `search_codebase` response carries:

- `chunks` — the actual hits
- `is_fully_fresh: bool` — `True` iff `pending_updates == 0` and `stale_files` is empty. The agent's first check is one comparison.
- `stale_files` — list of files whose mtime is later than their last index time (provided for completeness so an agent that wants to know *which* files are stale can act granularly)
- `pending_updates` — queue size when the search ran
- `indexed_through` — ISO timestamp of the most recent committed index update

The agent never has to guess freshness. Common usage: check `is_fully_fresh`; if `False`, call `wait_for_quiescence` and re-search, or warn the user.

### 6.4 Tool description discipline

The agent only knows what the docstring says. Three rules:

1. **Lead with what the tool does, in one sentence.**
2. **Document inputs with usage examples** (the kinds of queries that work).
3. **Surface the indexed scope dynamically via `list_projects`**, not via mutating tool docstrings between connections (MCP clients cache descriptions; that hack is fragile).

### 6.5 Headline invariant

**Between any two agent rounds, the index reflects every filesystem change the agent made in the previous round.** Watcher debounce + batched embed + auto-quiescence on search are the mechanisms; the spec acceptance test asserts this end-to-end.

---

## 7. Concurrency model

### 7.1 Read concurrency

Search queries are read-only against the index. Multiple concurrent searches run in parallel:

- BM25 lookup: GIL-released on string ops, parallel via threads
- Embedding lookup: numpy memmap reads are GIL-free
- RRF fusion: Python, GIL-bound but quick
- Reranker (if enabled): PyTorch model, GIL-released during forward, batches multiple requests
- LLM picker (if used): network I/O, fully concurrent

Target: 10+ concurrent search requests with no per-request slowdown until embedder/reranker batch limits hit.

### 7.2 Write concurrency

Index updates serialized through a single watcher worker thread. Reads proceed concurrently with writes via SQLite WAL mode and atomic memmap appends.

### 7.3 Embedder concurrency

Single embedder instance, accessed under a lock for batching. Search queries that need to embed the query text also acquire briefly. Single-instance embedder is fine for Phase 2; pool only if profiling shows contention.

### 7.4 Per-session state

Each MCP connection gets its own session struct in the daemon: `{session_id, active_project, cwd_history, last_search_at}`. Sessions don't share state — three terminals = three independent contexts.

---

## 8. Port discovery and service info

### 8.1 Try preferred → walk forward → record actual

```python
def bind_port(preferred: int, max_tries: int = 20) -> tuple[socket.socket, int]:
    for offset in range(max_tries):
        port = preferred + offset
        try:
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            sock.listen()
            return sock, port
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                continue
            raise
    raise RuntimeError(f"no free port in {preferred}..{preferred + max_tries}")
```

Defaults: MCP 8765, status 8766. If 8765 taken → walks to 8767 (preserving MCP/status pair adjacency) etc. Status port walks independently if its slot is taken.

### 8.2 Runtime info file

`~/.cache/longctx/server.info` written atomically on bind, deleted on shutdown:

```json
{
  "pid": 12345,
  "started_at": "2026-05-09T14:33:01Z",
  "mcp_port": 8767,
  "mcp_transports": ["sse", "streamable-http"],
  "status_port": 8768,
  "version": "0.2.0"
}
```

`longctx status` reads this file. Stale info file from crashed daemon is detected (`os.kill(pid, 0)` raises) and reclaimed on next start.

### 8.3 Singleton via flock

Port walking is for "port taken by someone else", **not** for "another longctx is running". Two concurrent daemons = bad (two index writers, races). Order:

1. `flock(server.lock)` exclusive non-blocking
2. Lock acquired → bind ports, write `server.info`
3. Lock contended → check existing PID alive → exit with "daemon already running on port N" or reclaim if stale

### 8.4 MCP clients discover via the info file, not hardcoded ports

Agent config doesn't pin a port:

```json
{
  "mcpServers": {
    "longctx": {"command": "longctx", "args": ["mcp-stdio"]}
  }
}
```

`longctx mcp-stdio` is a **bridge subcommand**, not a standalone daemon mode. When invoked it:

1. Reads `~/.cache/longctx/server.info` to find the running daemon's port + transport
2. Opens an SSE (or streamable-http) connection to that daemon
3. Bridges stdio ↔ HTTP for the calling MCP client
4. Forwards every MCP message bidirectionally; exits when the client disconnects

It does NOT start a new daemon, hold the singleton lock, or maintain its own index. The client thinks it's talking to a stdio-MCP server; under the hood, it's all one shared daemon. Decouples client config from runtime port and preserves the single-writer index model.

If `server.info` is missing or stale, `longctx mcp-stdio` prints a clear error suggesting `longctx service start`.

For HTTP-spoken clients (curl, Hermes, custom OSS harnesses): expose `longctx port [mcp|status]` CLI that prints just the port. Or read `server.info` directly. HTTP API endpoints documented in `docs/http-api.md` mirror the MCP tool surface — same JSON shapes.

### 8.5 Status report shows actual ports

```
longctx daemon: running (PID 12345, uptime 3h 14m)
mcp:    sse on 127.0.0.1:8767  (configured 8765, walked +2)
status: http on 127.0.0.1:8768
```

The `walked +N` flag tells the user without making them dig.

---

## 9. Daemon mode and service integration

### 9.1 Process model

```bash
longctx serve              # foreground
longctx serve --daemon     # detach, write PID file
```

Singleton via `~/.cache/longctx/server.lock` (flock).

### 9.2 Signals

| Signal | Behavior |
|---|---|
| SIGTERM | Graceful shutdown: reject new MCP connections, send shutdown frames to existing clients, wait up to 10s for in-flight retrievals, hard-kill after timeout, flush index, exit |
| SIGINT | Same as SIGTERM |
| SIGHUP | Reload config without dropping MCP connections |
| SIGUSR1 | Force full re-index (debugging) |

### 9.3 Service installer

```bash
longctx service install       # platform-aware: launchd / systemd / Windows Service
longctx service start
longctx service stop
longctx service status
longctx service uninstall
longctx service logs
```

**macOS (launchd):**
- `~/Library/LaunchAgents/com.tomturney.longctx.plist`
- `RunAtLoad=true`, `KeepAlive=true`
- Logs at `~/Library/Logs/longctx/{stdout.log,stderr.log}`
- Loaded via `launchctl load`

**Linux (systemd user unit):**
- `~/.config/systemd/user/longctx.service`
- `systemctl --user daemon-reload && systemctl --user enable --now longctx`
- Logs to journald, `journalctl --user -u longctx`

**Windows:**
- `pywin32` Windows Service registration, or scheduled-task-at-login fallback
- Logs at `%LOCALAPPDATA%\longctx\logs\`

Use `platformdirs` + `python-daemon` (or platform-specific equivalents); do not roll these.

---

## 10. Status and health

### 10.1 HTTP status

`GET /health` on the status port:

```json
{
  "status": "ready",
  "version": "0.2.0",
  "uptime_seconds": 3847,
  "projects": [
    {"name": "obsidian", "chunks": 762, "tokens_approx": 1543000, "last_updated": "..."},
    {"name": "longctx", "chunks": 124, "tokens_approx": 248000, "last_updated": "..."}
  ],
  "watcher": {"queued": 0, "events_processed_total": 1284},
  "embedder": {"model": "BAAI/bge-small-en-v1.5", "sha256": "..."},
  "memory_rss_mb": 5872,
  "indexed_through": "2026-05-09T14:33:01Z"
}
```

### 10.2 CLI status

```bash
longctx status
# longctx daemon: running (PID 12345, uptime 1h 4m)
# Indexed projects:
#   obsidian       762 chunks    ~1.5M tokens   updated 3h ago
#   longctx        124 chunks    ~248K tokens   updated 12m ago
#   vllm-swift   3,248 chunks    ~6.5M tokens   updated 2d ago
# Total: 4,134 chunks, ~8.3M tokens
# Watcher: 0 queued, 1,284 events processed
# MCP server: sse on 127.0.0.1:8767 (configured 8765, walked +2)
# Embedder: bge-small-en-v1.5 (sha256: a1b2…)
# Memory: 5.7 GB RSS
```

If the daemon isn't running, output gives clear instructions.

---

## 11. First-time setup wizard

```bash
longctx init
```

Interactive (or `--non-interactive` with sensible defaults):

```
$ longctx init
Welcome to longctx. Let's set up your codebase index.

Where do your projects live? [~/dev]:
Discovering projects...
Found 12 sentinel-marked directories. Which should longctx index?
  [✓] obsidian
  [✓] longctx
  [✓] vllm-swift
  [✓] turboquant_plus
  [✓] mlx-swift-lm
  [ ] node_modules    (excluded by default)
  [ ] tmp             (excluded by name)
  ...
(space to toggle, enter to confirm)

Configuration written to ~/.config/longctx/config.toml.

Indexing 5 projects (~13.4M tokens). First run downloads the
embedder model (~130 MB) and embeds all chunks; this takes
5–15 minutes on a laptop. The daemon will be available for
search throughout — agents will get partial results until
indexing completes.

[████████████████████░░░░] 80%   eta 2m 14s

Indexing complete. 4,134 chunks across 3,287 files.

Install as a background service so longctx starts on login? [Y/n]:
launchd plist installed.

Start the service now? [Y/n]:
longctx daemon started. mcp on 127.0.0.1:8765, status on :8766.

You're set up. Open Claude Code / Cursor / opencode and the
search_codebase tool will be available automatically.

For help: longctx --help        For status: longctx status
```

**Critical:** the wizard tells the truth about timing. "5–15 minutes on a laptop" is the honest range; promising "40 seconds" loses users when reality hits 8 minutes.

The daemon starts serving search requests **before** indexing completes — partial results with `pending_updates > 0` are returned. The wizard makes that explicit.

### 11.1 Non-interactive defaults

`longctx init --non-interactive` (e.g. for CI / automated provisioning) uses these defaults without prompting:

- `parent_dir = ~/dev`
- `auto_discover = true` — auto-include every sentinel-marked subdirectory (no manual confirmation)
- Skip service install (run `longctx service install` separately if wanted)
- Skip daemon start (run `longctx service start` separately)
- Skip initial indexing (the daemon will index incrementally on first start)

Override via flags: `--parent-dir`, `--no-auto-discover`, `--include`, `--with-service`, `--start-daemon`. Override via env: `LONGCTX_PARENT_DIR`, `LONGCTX_AUTO_DISCOVER`, etc.

The non-interactive path is intentionally conservative: it produces a working config but does NOT take any side-effecting action (service registration, daemon start, multi-minute embedding pass) unless explicitly requested. CI scripts that want the whole flow chain `longctx init --non-interactive && longctx service install && longctx service start`.

---

## 12. Resource bounds and safety

### 12.1 Memory targets (13.4M-token corpus on M5 Max)

| Component | RSS | Notes |
|---|---|---|
| Python interpreter + libs | ~300 MB | baseline |
| bge-small embedder (FP16) | ~250 MB | resident |
| SQLite WAL buffers | ~100 MB | bounded |
| BM25 stats (in-memory) | ~150 MB | proportional |
| Memmap embedding cache | ~50 MB | OS-managed |
| Watcher event buffers | ~10 MB | bounded |
| HTTP server overhead | ~50 MB | per-connection state |
| **Total target** | **<2 GB** | |

If RSS > 4GB: log WARN + diagnostic dump. If > 8GB: refuse new connections, log CRITICAL.

### 12.2 Disk targets

- SQLite chunks: ~30 MB for 13.4M tokens
- Memmap embeddings: ~10 MB at 384-dim
- BM25 stats: ~50 MB
- **Total: ~100 MB for 13.4M tokens.**

Linear scaling; ~750 MB at 100M tokens; switch to LanceDB/Qdrant somewhere before 1B.

### 12.3 Refuse-to-do-stupid checks

- Refuse to index `$HOME` or `/` directly
- Refuse to index any directory containing > N files (configurable, default 50,000)
- Refuse to start if the configured `parent_dir` doesn't exist
- Refuse to start if SQLite version < 3.37 (WAL semantics differ)
- Refuse to index `forbidden_dirs` (`secrets`, `credentials`, `.aws`, `.ssh`, `.gnupg`) without `--i-know-what-im-doing`
- Warn if total estimated tokens > 100M (proceed with explicit confirm)
- Warn if config-changed embedder will trigger massive re-embed (require `longctx reembed --confirm`)

### 12.4 Cleanup and resource hygiene

Three-tier story for keeping the index from accumulating cruft over time. **Conservative defaults**: don't lose user data. **Explicit knobs** for power users.

For most users: 100MB / 13M tokens × 20 projects ≈ 2GB total. Below the disk-pain floor; they never need to think about cleanup. For users with many monorepos, decade-long workflow histories, or Time-Machine-synced caches, the knobs matter.

#### 12.4.1 Tier 1 — auto-drop missing root_path (always-on)

Hourly check (piggybacked on the periodic mtime sweep, §5.5): for each indexed project, verify `root_path.exists()`. If missing for **>7 days continuously** (config: `cleanup.missing_root_grace_days`), drop chunks + embeddings + BM25 entries.

The 7-day grace handles:

- network-drive unmount (the project lives on `/Volumes/work` and the user is currently away from the office)
- USB drive temporarily removed
- `git worktree remove` cleanup
- `git checkout` to a branch where the directory doesn't exist

Don't nuke real data because someone's external drive is unplugged. Track "first observed missing" timestamp per project; reset on first re-appearance.

The watcher's parent-dir monitoring (§3.7) handles the fast path when projects are deleted while the daemon is running. Tier 1 catches the slow path (daemon restarted, project was already gone).

#### 12.4.2 Tier 2 — watcher pause for cold projects (always-on)

Project not queried for `cleanup.watcher_idle_pause_days` (default 30) AND with no recent file events → unsubscribe its file-system watch. Pure OS-resource hygiene:

- Linux inotify has a default per-user cap (~8192 watches); a 50-project user can hit it
- macOS FSEvents subscriptions accumulate per-volume cost
- Windows ReadDirectoryChangesW handles are bounded

Disk + index data are unchanged. Periodic mtime sweep (§5.5) still catches changes if the user rotates back. First query post-rotation pays a one-time mtime-walk cost (~seconds) before serving fresh results, then the watcher re-subscribes.

#### 12.4.3 Tier 3 — disk-budget cap with LRU eviction (opt-in)

Config:
```toml
[index]
disk_budget_gb = 0   # 0 = unlimited (default)
```

When set > 0 and total `~/.cache/longctx/` size exceeds the budget, evict projects in **LRU order by `last_query_at`** until under budget. Evicted projects drop from the index but their entry stays in config — next query against that project re-indexes from scratch.

Default `0` (unlimited): don't surprise users with missing data. Power users who care about disk set the cap explicitly. Eviction logged at WARN with the project name and freed bytes.

#### 12.4.4 Manual cleanup command

```bash
longctx clean                    # dry-run: show what would be removed
longctx clean --idle 90d         # drop projects not queried in 90+ days
longctx clean --orphans          # session-bound projects from dead sessions
longctx clean --missing          # missing root_path, no grace period
longctx clean --all              # all of the above
longctx clean --idle 90d --yes   # skip the "are you sure" prompt
```

Always dry-runs and confirms unless `--yes`. `--orphans` also runs implicitly on daemon startup (§3.8).

#### 12.4.5 What we don't auto-drop

Conservative-by-default principles:

- **Projects by query staleness alone.** A user might rotate back; the data is small. Manual `longctx clean --idle Nd` if they care.
- **Embeddings of recently-modified files**, even if the project is otherwise idle.
- **The bge-small model itself**, even if no projects are using it currently — re-download is slow and unpleasant.
- **`server.info`, `server.lock`** — managed by the daemon lifecycle, not cleanup.

When in doubt: keep data. Disk is cheap; surprise data loss is expensive.

---

## 13. Privacy and secrets posture

### 13.1 Local-only by default

- All sockets bind to `127.0.0.1` — never `0.0.0.0`
- Index files at 0600, parent dir at 0700
- No telemetry, no network calls outside localhost
- `[longctx] mode: local-only` reported in status output

### 13.2 Secret-pattern always-exclude

Default-excluded patterns are NOT user-removable from config without an explicit `allow_secret_patterns = true` and a `--i-know-what-im-doing` CLI flag:

```
.env*  *.key  *.pem  id_rsa*  id_ed25519*  *.p12  *.pfx
**/secrets/**  **/credentials/**  **/.aws/**  **/.ssh/**  **/.gnupg/**
```

### 13.3 Forbidden directory names

`secrets`, `credentials`, `.aws`, `.ssh`, `.gnupg` — refuse to index any directory whose name matches, even if the user added it explicitly. Override only via the same `--i-know-what-im-doing` path.

### 13.4 Auth on non-localhost binds

If the user re-configures `mcp_host` to a non-localhost address, the daemon REQUIRES a token in the `mcp_auth_token` config field and rejects unauthenticated connections. No accidental LAN exposure.

### 13.5 The "secrets in indexed files" mitigation

Even with the patterns above, secrets can leak into committed files (example.env, tests/fixtures). Document the risk in README + `longctx init`. Recommend running `truffleHog` or `gitleaks` over the corpus before exposing the daemon to a shared agent context.

---

## 14. Logging

### 14.1 Format

JSON lines to stdout (when `transport=stdio`, redirect to stderr to keep MCP traffic clean) AND to log file:

```json
{"ts": "2026-05-09T14:33:01.234Z", "level": "INFO", "component": "watcher",
 "msg": "Updated chunk", "file": "longctx/coarse_filter.py",
 "chunks_added": 2, "chunks_kept": 12, "embed_ms": 87}
```

### 14.2 Levels

- **DEBUG** — per-event/per-chunk; off by default
- **INFO** — file updates, server lifecycle, config reloads, MCP connections
- **WARN** — dropped events, refused connections, degraded operation
- **ERROR** — failed updates, embedder failures, MCP protocol errors
- **CRITICAL** — index corruption, OOM, refuse-to-do-stupid trips

### 14.3 Rotation

Daily rotation, keep 7 days, gzip older. Use `loguru` defaults.

### 14.4 Per-call MCP trace — the single most useful artifact

Every MCP tool call emits one structured `INFO` line capturing the entire request/response shape:

```json
{
  "ts": "2026-05-09T14:33:01.234Z",
  "level": "INFO",
  "component": "mcp",
  "trace_id": "01H8XKY7M3A2RF...",
  "session_id": "ses_abc123",
  "connection_id": "conn_xyz789",
  "client": {"name": "opencode", "version": "0.4.2"},
  "tool": "search_codebase",
  "args": {
    "query": "where is the auth middleware",
    "cwd": "/Users/tom/dev/myapp",
    "max_tokens": 4096,
    "wait_for_quiescence_ms": 500
  },
  "scope": {
    "primary_project": "myapp",
    "primary_source": "cwd_walk_to_sentinel",
    "fanout_projects": ["myapp"],
    "cross_project_pattern_matched": null,
    "active_project_sticky": null
  },
  "latency_ms": {
    "wait_quiescence": 12,
    "embed_query": 18,
    "bm25_score": 7,
    "dense_score": 24,
    "rrf_fuse": 1,
    "fetch_chunks": 3,
    "total": 65
  },
  "result": {
    "chunk_count": 5,
    "files": [
      "src/auth/middleware.ts:42-89",
      "src/auth/types.ts:1-30",
      "tests/auth.test.ts:120-178"
    ],
    "is_fully_fresh": true,
    "pending_updates": 0,
    "indexed_through": "2026-05-09T14:33:00.500Z"
  }
}
```

One line tells you: who asked, what they asked, what scope longctx picked and *why*, how long each stage took, what came back, and whether it was fresh. **This is the primary tool for verifying the daemon matches user expectations during real-harness testing.**

### 14.5 Trace IDs for correlation

Every MCP call generates a `trace_id` (ULID). All sub-events propagate it: watcher updates triggered by `wait_for_quiescence`, embedder forward passes, BM25 lookups, atomic-update transactions. `grep 01H8XKY7M3A2RF longctx.log` returns the full causal chain end-to-end.

### 14.6 Per-harness identification

MCP's `initialize` request carries `clientInfo: {name, version}`. Daemon records this on connection establishment and includes it in every per-call log. Lets you slice queries by harness without parsing connection-creation events:

```bash
# all Pi queries
jq 'select(.client.name == "Pi")' ~/Library/Logs/longctx/longctx.log

# all opencode queries that returned stale results
jq 'select(.client.name == "opencode" and .result.is_fully_fresh == false)' \
   ~/Library/Logs/longctx/longctx.log

# latency p95 by harness
jq -s 'group_by(.client.name)
       | map({client: .[0].client.name,
              p95_ms: (sort_by(.latency_ms.total)[(length * 0.95 | floor)].latency_ms.total)})' \
   ~/Library/Logs/longctx/longctx.log
```

Harnesses that don't send `clientInfo` (older MCP versions) get logged as `{"name": "unknown"}` plus User-Agent inferred from transport headers when available.

### 14.7 Live tail — `longctx watch`

Pretty-printed real-time stream of MCP activity. The "debug TV" for what agents are doing right now:

```
$ longctx watch
14:33:01  opencode/0.4.2  search_codebase  "where is the auth middleware"
                          scope=myapp (cwd_walk_to_sentinel) → 5 chunks (65ms, fresh)
                          ↳ src/auth/middleware.ts:42-89
                          ↳ src/auth/types.ts:1-30
                          ↳ tests/auth.test.ts:120-178

14:33:08  Pi/2.1.0        list_projects
                          → 8 projects (4ms)

14:33:14  Hermes/0.9.5    search_codebase  "qwen3 attention forward"
                          scope=mlx-swift-lm (cross_project_pattern: "qwen3")
                          → 5 chunks (122ms, 1 stale_file)
                          ↳ Libraries/MLXLLM/Models/Qwen3.swift:89-156
                          ↳ ...
```

Color-coded by harness. `--verbose` shows trace IDs and latency breakdown. `--filter client=opencode` for a single harness. `--since 1h` to backfill recent activity. Critical for hermeticity testing — keep it open in one terminal while running real agents in others.

### 14.8 Replay log + `longctx replay`

A separate file `~/.cache/longctx/interactions.jsonl` captures full request/response payloads (independent of the operational log's redaction config) for replay-based regression testing:

```bash
# Capture today's activity into a fixture
cp ~/.cache/longctx/interactions.jsonl tests/fixtures/2026-05-09-real-traffic.jsonl

# After tweaking the chunker / embedder / RRF weights, replay
longctx replay tests/fixtures/2026-05-09-real-traffic.jsonl
# → diff: queries where top-K changed, latency deltas, freshness regressions
```

Useful for "did this change improve or degrade what real agents actually asked yesterday" — far more meaningful than synthetic NIAH alone.

#### 14.8.1 Replay log retention

`interactions.jsonl` is unbounded by default — on a heavily-used daemon it would grow indefinitely. Retention policy:

- **Active file rotates at 1 GB or 30 days**, whichever first
- **Rotated shards gzip + retain 7 days**, then deleted
- Override via config:

```toml
[logging]
replay_retention_days = 7         # rotated-shard retention
replay_active_max_gb = 1.0        # active-file rotation trigger
replay_active_max_days = 30       # active-file rotation trigger (whichever first)
```

- Manual sweep: `longctx clean --replay-older-than 14d` drops shards older than N days regardless of policy

Set `replay_retention_days = 0` to disable rotation entirely (capture everything forever — useful when actively building a regression corpus). Set `replay_active_max_gb = 0` to disable replay logging altogether for users who don't want it.

### 14.9 Privacy-aware logging

Default: log query text + cwd in plain. Tom's localhost-only single-user case is fine. Three opt-in redaction knobs for sharing/analysis:

```toml
[logging]
redact_query_text = false        # → "<query of 47 chars>"
redact_cwd = false               # → "<cwd>"
log_chunk_text = false           # never default on; way too verbose + sensitive
log_chunk_paths = true           # file:line citations stay (the useful part)
```

If a user wants to share a daemon log for debugging, redaction strips queries/paths first. The interactions.jsonl replay file ignores these knobs (always full payload) — it's not for sharing, it's for local replay.

### 14.10 What the operational log alone tells you about a harness

> **Descriptive, not prescriptive.** The implementing agent does not build §14.10 — it explains what insights become available from the §14.4–14.9 logging output. Included here so reviewers and future readers understand why the structured-trace work is worth the implementation cost. If you're an implementing agent: skip to §15.

For each new harness Tom tests (Pi, Hermes, opencode), 5 minutes of `longctx watch` reveals:

- **Is the harness MCP-discovering longctx at all?** Connection event in the log.
- **Does it pass `cwd`?** Look at the `args.cwd` field across calls.
- **Does it set `set_active_project`?** `tool=set_active_project` lines show this.
- **Does it use `wait_for_quiescence` after edits?** Watch for it post-`write_file` calls.
- **Does it batch queries or call sequentially?** Connection-id grouping + timestamp deltas.
- **Does it respect `max_tokens` budgets?** `args.max_tokens` distribution.
- **What's its latency tolerance?** Look at p95 of `total_ms` per harness; rough proxy for the harness's own retry/timeout config.

Each of these answers a "is the harness actually using longctx the way users expect?" question. Without the structured trace, all of this is guesswork.

---

## 15. Testing

### 15.1 Unit

- Config parser: round-trip various TOML configs, edge cases, env override precedence
- Auto-discovery: directory tree → expected project list, edge cases (sentinel inside .gitignore, multiple sentinels in one dir)
- Filter chain: known-positive and known-negative paths
- Debouncer: simulated event storms, verify coalescing and atomic-rename dedup
- Incremental update: file mutation scenarios (single line, full rewrite, rename, delete)
- Embedder identity check: HF id same + SHA256 different = re-embed required
- Port walking: occupy 8765 → daemon binds 8766 → server.info reflects actual

### 15.2 Integration

- Full daemon lifecycle: start, index synthetic corpus, query via MCP, edit files, reload config, shutdown
- Multi-client concurrency: 10 simultaneous MCP clients searching the same index
- Cross-session isolation: session A sets `set_active_project("foo")`, session B's search uses session B's own scope
- Service installation per platform (mocked launchd/systemd/Windows interactions)
- Stale `server.info` reclaim after simulated crash

### 15.3 End-to-end

- Real corpus: index `~/dev/longctx`, connect via Claude Code, verify tool discovery + search
- Watcher under realistic load: simulate `git pull` event storm; no dropped events, no crashes
- Restart recovery: kill daemon mid-index; restart; verify clean recovery
- **Headline invariant test**: agent writes a file → immediately calls `search_codebase` (default `wait_for_quiescence_ms`); verify the new file appears in results
- **20-file refactor test**: simulate a refactor that creates 5, deletes 3, renames 2, modifies 10 files; verify index fully fresh in <500ms after the last write
- **Cross-project search**: query `"in mlx-swift handle centroid"` while session cwd is in `longctx`; verify mlx-swift-lm chunks rank above longctx chunks

---

## 16. Implementation phases

| Sub-phase | Scope | Effort |
|---|---|---|
| **2.0** | Foreground process, stdio MCP, single project (`--corpus-dir`), persistent SQLite + memmap, basic search_codebase tool | 2-3 days |
| **2.1** | Auto-discovery, multi-root, daemon mode, SSE + streamable-http, port discovery, `server.info`, singleton via flock | 1 week |
| **2.2** | Watcher (watchfiles, debounce, batched embed, periodic mtime sweep), staleness flags, `wait_for_quiescence` tool | 4 days |
| **2.3** | macOS launchd + first-time wizard + `longctx status` polish | 3 days |
| **2.4** | Linux systemd, Windows Service, hardening, full test matrix, secrets posture review | 1 week |

Total: **3-4 weeks for one engineer**, **~2 weeks with an agent**. Each sub-phase ships independently; 2.0 alone validates the MCP-tool-surface design with real agents and de-risks the rest.

---

## 17. Acceptance criteria

Phase 2 is shippable when:

1. Single daemon runs persistently and serves multiple concurrent MCP clients
2. `longctx init` auto-discovers projects under `~/dev` from sentinels
3. Per-session cwd hint scopes search to the right project; cross-project hits surface when relevant
4. Index survives daemon restart with no re-embedding required
5. Watcher detects file changes and updates index within: **500ms median, 1s p95** for typical edits (≤5 files); **<2s p95** for refactors up to 50 files; bounded by single-batch embedding throughput beyond that. Acceptance test asserts the median + p95 thresholds, not a hard ceiling.
6. **Agent writes a file → immediately searches → result includes the new file's chunks** (default-on auto-quiescence)
7. Watcher handles a 20-file refactor (mixed CREATE/DELETE/RENAME/MODIFY) atomically
8. Watcher handles `git pull` event storms without drops or crashes (queue overflow → rescan-pending fallback)
9. Config reload via SIGHUP works without dropping MCP connections; embedder change requires explicit `longctx reembed --confirm`
10. macOS launchd integration: `longctx service install && start` produces a daemon that survives logout/login
11. Memory RSS stays under 2GB for the 13.4M-token corpus
12. `longctx status` reports accurate runtime ports + state in all daemon states; ports auto-walked on conflict
13. First-time wizard runs cleanly on a fresh machine; honest about 5–15 min indexing time on first run
14. Privacy posture enforced: secret patterns always-excluded, forbidden directory names refused, sockets bound to localhost
15. Cleanup tiers behave correctly. Each is a separate testable criterion:
    - **15a.** Tier 1: project whose `root_path` is missing for >7 days auto-drops; project missing for <7 days survives.
    - **15b.** Tier 2: project not queried in >30 days has its watcher unsubscribed; first query post-rotation re-arms the watcher and returns fresh results within one mtime sweep cycle.
    - **15c.** Tier 3: when `disk_budget_gb > 0` and total cache exceeds it, projects evict in `last_query_at` LRU order until under budget; eviction logs at WARN.
    - **15d.** `longctx clean --orphans` runs implicitly on daemon startup; session-bound projects whose creating MCP session is no longer alive are GC'd.
16. All Phase 1 tests still pass; Phase 2 adds at least 100 new tests including the headline invariant from §6.5
17. Per-call MCP trace logging is wired and verified:
    - Every MCP tool call emits the §14.4 schema (trace_id, session_id, connection_id, client, tool, args, scope-with-source, latency_ms breakdown, result freshness flags)
    - Trace IDs propagate from the originating MCP call into watcher events, embedder forward passes, and DB transactions per §14.5
    - Per-harness identification via MCP `clientInfo` is recorded on connection and included in every per-call log per §14.6
    - `longctx watch` renders the structured stream live per §14.7
    - `interactions.jsonl` captures full payloads with the §14.8.1 retention policy in effect
    - Verified via grep-based assertions in integration tests (e.g. "every search_codebase call in the test session produces exactly one matching MCP trace line")

When 1–14, 15a–15d, 16, and 17 are met, longctx is shippable as a production tool.

---

## 18. Out of scope (Phase 3+)

- Real vector DB backend for >100M-token corpora (LanceDB / Qdrant; Phase 3 if demand)
- Per-project search permissions for security-sensitive contexts (Phase 4)
- Embedding model auto-upgrade (Phase 4)
- Distributed daemon (multiple machines sharing one index) — out of scope indefinitely
- Web UI for index browsing — out of scope; CLI is enough
- Tree-sitter chunking by default — Phase 1 ships with line-window; tree-sitter for code-aware splits stays optional via `LONGCTX_TS=1`

---

## 19. Headline invariants (the things that must always be true)

1. **The agent never has to know which port the daemon is on.** It reads `server.info` or uses `longctx mcp-stdio`.
2. **The inference server never has to know about retrieval.** Agent splices chunks itself.
3. **The user never has to manually configure the corpus.** Auto-discovery picks up everything sentinel-marked under `~/dev`.
4. **Between any two agent rounds, the index reflects every file change the agent made in the previous round.** Watcher + auto-quiescence on search.
5. **Every search response tells the agent about freshness.** `stale_files`, `pending_updates`, `indexed_through`. No silent staleness.
6. **No secret patterns are searchable by default.** Hard refusal, not just convention.
7. **Three concurrent terminal sessions in three projects don't interfere.** Per-session sticky context, isolated cwd-derived scope.
8. **`brew install longctx && longctx init && longctx service install` produces a working agent-discoverable daemon in under 20 minutes on a fresh machine.** Anything longer kills adoption.
