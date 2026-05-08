# PRD — 10M coherent context emulation, ≤32K active KV

## Goal
Demonstrate that a small-context model + longctx external memory can answer questions whose answers REQUIRE COHERENT REASONING across many distant evidence spans in a 10M-token corpus, on a single MI300X, with model active prompt ≤32K.

## Non-goals
- Native 10M attention. Model attention window stays ≤32K.
- Single-fact NIAH recall. The prior compact-mode 10M run (1/20) was a retrieval-recall test; this PRD is a reasoning test.
- Beating SubQ on every metric. The point is coherence-survival at scale, not state-of-the-art accuracy.

## Why this matters
The compact-mode 10M run on 2026-05-07 ingested 10M tokens and retrieved 1-2 chunks per question. That proved external memory plumbing, NOT coherent reasoning. SubQ-style claims require multi-evidence synthesis, contradiction resolution, and global aggregation. This PRD specifies that benchmark.

## Task families (in the synthetic corpus)
1. **Single-fact recall** — control. One exact answer in one span.
2. **Multi-hop join** — answer requires 2-5 evidence spans at distant positions, chained: `entity → codename → project → invoice`.
3. **Contradiction / latest-authority** — same entity has multiple values across the corpus; query asks for the *latest* or *highest-authority* value, not the first match.
4. **Global aggregation** — answer requires combining many dispersed facts: max / sum / count / rank by group. Cannot be answered from one chunk.
5. **Temporal state coherence** — a state changes across many updates. Query asks final state or state-at-checkpoint.

## Per-question metadata (recorded in haystack JSON)
- `task_family`: one of the 5 above
- `answer`: ground truth
- `required_evidence_ids`: ordered list of evidence span ids needed
- `evidence_spans`: list of {evidence_id, text, token_pos_start, token_pos_end}
- `min_evidence_count`: how many must be present for the answer to be derivable
- `corpus_token_positions`: where evidence sits in the haystack

## Scoring (4 mutually-exclusive outcomes per question)
- **exact** — answer matches truth, retrieval brought back ≥ min_evidence_count of the required spans
- **retrieval-miss** — retrieval did NOT surface enough required evidence; model couldn't have answered correctly
- **evidence-complete reasoning-fail** — retrieval brought back all required evidence, but model answered wrong (the model's reasoning failure)
- **coherent-wrong** — model produced a plausible-shaped but wrong answer; retrieval coverage indeterminate

## Headline metrics
- **retrieval_recall@K** — fraction of required evidence_ids present in the top-K retrieved chunks
- **answer_exact_rate** — exact / total questions
- **evidence_complete_wrong_rate** — model failures, evidence was present
- **evidence_missing_rate** — retrieval failures
- **active_prompt_tokens** — must stay ≤32K per request
- **longctx_coverage** — % of evidence spans actually in /evict/dump
- **per_question_latency_s**
- **total_runtime_estimate_s** for each rung

## Critical distinction
| evidence present? | answer correct? | classification |
|---|---|---|
| ≥ min | yes | exact |
| ≥ min | no | reasoning-fail |
| < min | yes | (reported, but suspicious — luck or memorization) |
| < min | no | retrieval-miss |

Reasoning-fail and retrieval-miss are reported separately. They have different fixes (model swap vs retrieval upgrade).

## Pass criteria

### 100K rung (smoke)
- retrieval_recall@K ≥ **90%**
- answer_exact ≥ **70%** on multi-hop / contradiction / state tasks (single-fact will be higher; reasoning families are the actual test)

### 1M rung
- retrieval_recall@K ≥ **80%**
- answer_exact ≥ **50%**

### 10M rung (publishable bar)
- retrieval_recall@K ≥ **65%**
- answer_exact ≥ **35-50%**, with clear separation of retrieval-failure vs reasoning-failure

## Stage gating (because credits are scarce)
1. Implement task generators + scorer + harness changes locally
2. Dry-run scoring without model (verify metadata + scoring path on a hand-built corpus)
3. Smoke at 100K with 5-10 questions on the AMD endpoint — only after explicit approval
4. 1M only after 100K passes its bar
5. 10M only after 1M passes its bar

## Active context constraint
- Model max-model-len ≤ 32K
- Per-request prompt = system + retrieved chunks + question
- Haystack chunks are NOT placed in the model's context — they live in longctx-svc

## What to call this
**Correct claim:** "10M-token coherent context emulation on one MI300X with ≤32K active KV"

**Wrong claim:** "10M native context"

## Model targets
- **Qwen2.5-7B-Instruct** (cheap, validated with K8V4 TQ+ + longctx on MI300X)
- Optional next: **Qwen2.5-7B-Instruct-1M** (already downloaded on droplet) or **Qwen2.5-14B-Instruct-1M**

## Deliverables before any cloud run
- This PRD
- `harness/tasks.py` — 5 task family generators
- `synthetic_haystack.py` extension — `build_haystack_with_tasks(...)`
- `scorer.py` extension — task-family-aware classification
- `streaming_driver.py` compact mode extension — per-question evidence tracking
- Example 100K corpus JSON
- Example answer-key JSON
- Smoke test (mocked endpoint, dry-run scoring)
- Estimated MI300X runtime per rung
- Exact command for 100K AMD run, awaiting approval
