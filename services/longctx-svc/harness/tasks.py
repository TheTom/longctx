"""Task generators for the 10M coherence PRD.

Each task is one question whose answer requires reasoning across one or
more evidence spans embedded in the haystack. The haystack builder
embeds the spans at known positions; the harness retrieves at question
time and checks which required spans were surfaced.

Five families:
  single_fact         — 1 span, control. Retrieval recall = 100% by design.
  multi_hop           — 3 spans chained: entity → codename → invoice → amount.
  contradiction       — 2 spans, latest-authority answer.
  aggregation         — 5 spans, reasoning-by-extremum (max/min/count).
  temporal            — 3+ spans, state-over-time, asks final state.

Each generator returns a `TaskInstance` carrying:
  * task_id (globally unique)
  * task_family
  * question text
  * ground-truth answer
  * list of EvidenceSpan(evidence_id, text, target_token_pos)
  * required_evidence_ids (subset of span ids that the answer DEPENDS on)
  * min_evidence_count (how many of `required` must be present for the
    answer to be derivable; for multi_hop = all of them; for aggregation
    = all; for contradiction = 1 (the LATEST one carries the answer);
    for temporal = 1 (the LAST update); for single_fact = 1)

The builder is responsible for distributing spans across the haystack
and pinning their final token positions; this module only generates
instances.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceSpan:
    evidence_id: str        # e.g. "task_3.span_1"
    text: str               # the sentence to embed in the haystack
    marker: str = ""        # unique-in-corpus substring used to detect presence
                            # of this span in a retrieved chunk
    target_token_pos: int = 0  # filled by builder
    actual_token_pos: int = 0  # filled by builder after offset_mapping pin


@dataclass
class TaskInstance:
    task_id: str
    task_family: str
    question: str
    answer: str
    answer_kind: str            # "6digit" / "5digit" / "iso_date" / "string"
    evidence_spans: list[EvidenceSpan]
    required_evidence_ids: list[str]
    min_evidence_count: int
    rationale: str = ""  # human-readable derivation, for debug


# ---------------------------------------------------------------------------
# Entity pools (kept distinct so tasks don't accidentally cross-contaminate)
# ---------------------------------------------------------------------------
PROJECTS_A = ["NOVA", "HELIOS", "ORION", "VEGA", "CYGNUS", "LYRA",
              "DRACO", "PERSEUS", "AQUILA", "SAGITTA"]
PROJECTS_B = ["TITAN", "ATLAS", "PROMETHEUS", "EPIMETHEUS", "HYPERION",
              "OCEANUS", "TETHYS", "RHEA", "CRONUS", "MNEMOSYNE"]
CODENAMES = ["Falcon-7", "Eagle-3", "Hawk-9", "Raven-12", "Osprey-4",
             "Kestrel-8", "Harrier-5", "Merlin-2", "Goshawk-6", "Buzzard-1"]
INVOICES = [f"INV-{n}" for n in (2845, 3021, 4156, 5023, 6710,
                                 7392, 8014, 9105, 1247, 2860)]
REGIONS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
           "Zeta", "Theta", "Lambda", "Sigma", "Omega"]
CONTRACTS = ["DELTA", "OMEGA", "SIGMA", "KAPPA", "TAU",
             "ALPHA", "BETA", "GAMMA", "EPSILON", "PHI"]


# ---------------------------------------------------------------------------
# Without-replacement sampling helper
# ---------------------------------------------------------------------------

def _take(rng: random.Random, pool: list[str],
          used: dict[str, set[str]], pool_name: str) -> str:
    """Sample one entry from `pool` that hasn't been taken yet for that
    pool. Records the take in `used` so subsequent calls won't collide.
    Raises if the pool is exhausted — generators should keep n_per_family
    within pool capacity."""
    seen = used.setdefault(pool_name, set())
    available = [x for x in pool if x not in seen]
    if not available:
        raise ValueError(
            f"pool {pool_name!r} exhausted (size {len(pool)}); "
            f"reduce n_per_family or expand the pool"
        )
    pick = rng.choice(available)
    seen.add(pick)
    return pick


# ---------------------------------------------------------------------------
# Family 1 — single_fact (control)
# ---------------------------------------------------------------------------

def gen_single_fact(
    rng: random.Random, task_idx: int,
    used: dict[str, set[str]],
) -> TaskInstance:
    """One sentence, one fact. Direct lookup baseline."""
    project = _take(rng, PROJECTS_A, used, "projects_a")
    code = f"{rng.randint(100_000, 999_999)}"
    span = EvidenceSpan(
        evidence_id=f"task_{task_idx}.fact",
        text=(
            f"Project {project} was provisioned with access code {code}. "
            f"All technicians must memorize this code prior to deployment."
        ),
        marker=code,
    )
    return TaskInstance(
        task_id=f"task_{task_idx}",
        task_family="single_fact",
        question=f"What access code does Project {project} use?",
        answer=code,
        answer_kind="6digit",
        evidence_spans=[span],
        required_evidence_ids=[span.evidence_id],
        min_evidence_count=1,
        rationale=f"single span at evidence_id={span.evidence_id}",
    )


# ---------------------------------------------------------------------------
# Family 2 — multi_hop join (entity → codename → invoice → amount)
# ---------------------------------------------------------------------------

def gen_multi_hop(
    rng: random.Random, task_idx: int,
    used: dict[str, set[str]],
) -> TaskInstance:
    """Three-hop chain: entity → codename → invoice → amount.

    Asking for the amount given the entity requires all three spans.
    Spans are placed at distant positions by the builder.
    """
    project = _take(rng, PROJECTS_A, used, "projects_a")
    codename = _take(rng, CODENAMES, used, "codenames")
    invoice = _take(rng, INVOICES, used, "invoices")
    amount = rng.randint(10_000, 999_999)
    amount_str = f"${amount:,}"
    spans = [
        EvidenceSpan(
            evidence_id=f"task_{task_idx}.hop1",
            text=(
                f"Project {project} was assigned codename {codename} "
                f"during the initial provisioning review."
            ),
            marker=f"{project} was assigned codename {codename}",
        ),
        EvidenceSpan(
            evidence_id=f"task_{task_idx}.hop2",
            text=(
                f"Codename {codename} corresponds to invoice line {invoice} "
                f"on the corporate ledger."
            ),
            marker=f"{codename} corresponds to invoice line {invoice}",
        ),
        EvidenceSpan(
            evidence_id=f"task_{task_idx}.hop3",
            text=(
                f"Invoice line {invoice} was billed in the amount of "
                f"{amount_str}, payable in net thirty terms."
            ),
            marker=f"{invoice} was billed in the amount of {amount_str}",
        ),
    ]
    return TaskInstance(
        task_id=f"task_{task_idx}",
        task_family="multi_hop",
        question=(
            f"What was the billed amount for Project {project}? "
            f"Answer with the dollar amount only."
        ),
        answer=amount_str,
        answer_kind="dollar",
        evidence_spans=spans,
        required_evidence_ids=[s.evidence_id for s in spans],
        min_evidence_count=3,
        rationale=(
            f"chain: {project} → {codename} → {invoice} → {amount_str}"
        ),
    )


# ---------------------------------------------------------------------------
# Family 3 — contradiction / latest-authority resolution
# ---------------------------------------------------------------------------

def gen_contradiction(
    rng: random.Random, task_idx: int,
    used: dict[str, set[str]],
) -> TaskInstance:
    """Same entity, two values. The LATER (effective-date authoritative)
    value is the right answer.

    Both spans are needed to resolve the contradiction; the latest-dated
    one carries the actual answer.
    """
    project = _take(rng, PROJECTS_B, used, "projects_b")
    old_code = f"{rng.randint(100_000, 999_999)}"
    new_code = f"{rng.randint(100_000, 999_999)}"
    while new_code == old_code:
        new_code = f"{rng.randint(100_000, 999_999)}"
    spans = [
        EvidenceSpan(
            evidence_id=f"task_{task_idx}.original",
            text=(
                f"Project {project} access code: {old_code}, "
                f"provisioned during the initial 2024 rollout."
            ),
            marker=f"{project} access code: {old_code}",
        ),
        EvidenceSpan(
            evidence_id=f"task_{task_idx}.update",
            text=(
                f"Project {project} access code was updated to {new_code}, "
                f"effective Q3 2026 per the latest security review. "
                f"The original code is now deprecated and must not be used."
            ),
            marker=f"{project} access code was updated to {new_code}",
        ),
    ]
    return TaskInstance(
        task_id=f"task_{task_idx}",
        task_family="contradiction",
        question=(
            f"What is the CURRENT access code for Project {project}? "
            f"Use the latest authoritative value."
        ),
        answer=new_code,
        answer_kind="6digit",
        evidence_spans=spans,
        # The .update span carries the answer; .original is the foil.
        # min_evidence_count=2 because the model needs to SEE both to
        # know there's been an update — one alone could be wrong.
        required_evidence_ids=[s.evidence_id for s in spans],
        min_evidence_count=2,
        rationale=(
            f"old={old_code}, new={new_code}; latest-authority is new"
        ),
    )


# ---------------------------------------------------------------------------
# Family 4 — global aggregation (max / sum over many spans)
# ---------------------------------------------------------------------------

def gen_aggregation(
    rng: random.Random, task_idx: int,
    used: dict[str, set[str]],
) -> TaskInstance:
    """Five regions, each with a unique incident count. Question asks
    which region had the maximum. Cannot be answered from one chunk.

    Aggregation reuses regions across tasks intentionally — different
    tasks ask about different region-sets, but a region can appear in
    multiple tasks since each task's question is self-contained.
    """
    n_regions = 5
    regions = rng.sample(REGIONS, n_regions)
    counts = rng.sample(range(1_000, 9_999), n_regions)
    max_idx = counts.index(max(counts))
    max_region = regions[max_idx]
    spans = []
    for i, (region, count) in enumerate(zip(regions, counts)):
        spans.append(EvidenceSpan(
            evidence_id=f"task_{task_idx}.region_{i}",
            text=(
                f"Region {region} reported {count:,} incidents during the "
                f"most recent quarterly audit cycle."
            ),
            marker=f"Region {region} reported {count:,}",
        ))
    return TaskInstance(
        task_id=f"task_{task_idx}",
        task_family="aggregation",
        question=(
            f"Across regions {', '.join(regions)}, which region reported "
            f"the highest incident count? Answer with just the region name."
        ),
        answer=max_region,
        answer_kind="string",
        evidence_spans=spans,
        required_evidence_ids=[s.evidence_id for s in spans],
        # All 5 needed to verify max; missing any could change the answer.
        min_evidence_count=n_regions,
        rationale=(
            f"counts={dict(zip(regions, counts))}, max={max_region}"
        ),
    )


# ---------------------------------------------------------------------------
# Family 5 — temporal state coherence (status updates over time)
# ---------------------------------------------------------------------------

def gen_temporal(
    rng: random.Random, task_idx: int,
    used: dict[str, set[str]],
) -> TaskInstance:
    """Status updates over time. Question asks for the LATEST state.
    Three updates; later ones overwrite earlier ones.
    """
    contract = _take(rng, CONTRACTS, used, "contracts")
    statuses = ["PENDING", "UNDER_REVIEW", "APPROVED", "ACTIVE"]
    # Pick 3 ordered statuses with ascending dates; final = answer.
    n = 3
    chosen = rng.sample(statuses, n)
    # Dates in ascending order
    base_year = 2026
    dates = sorted([
        f"{base_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        for _ in range(n)
    ])
    spans = []
    for i, (status, date) in enumerate(zip(chosen, dates)):
        spans.append(EvidenceSpan(
            evidence_id=f"task_{task_idx}.update_{i}",
            text=(
                f"On {date}, Contract {contract} transitioned to status: "
                f"{status}. The change was logged by the legal review team."
            ),
            marker=f"On {date}, Contract {contract} transitioned to status: {status}",
        ))
    final_status = chosen[-1]
    return TaskInstance(
        task_id=f"task_{task_idx}",
        task_family="temporal",
        question=(
            f"What is Contract {contract}'s most recent status? "
            f"Answer with just the status name."
        ),
        answer=final_status,
        answer_kind="string",
        evidence_spans=spans,
        # The model needs to see ALL updates to know which is most recent.
        # If retrieval surfaces only the first, the answer is wrong.
        required_evidence_ids=[s.evidence_id for s in spans],
        min_evidence_count=n,
        rationale=(
            f"updates: {list(zip(dates, chosen))}, final={final_status}"
        ),
    )


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "single_fact": gen_single_fact,
    "multi_hop": gen_multi_hop,
    "contradiction": gen_contradiction,
    "aggregation": gen_aggregation,
    "temporal": gen_temporal,
}


def generate_tasks(
    rng: random.Random,
    n_per_family: dict[str, int],
) -> list[TaskInstance]:
    """Instantiate tasks given a per-family count dict.

    Example: {"single_fact": 2, "multi_hop": 5, "contradiction": 3,
              "aggregation": 3, "temporal": 2} → 15 tasks total.
    """
    tasks: list[TaskInstance] = []
    used: dict[str, set[str]] = {}
    task_idx = 0
    for family, n in n_per_family.items():
        if family not in GENERATORS:
            raise ValueError(f"unknown family: {family}")
        gen = GENERATORS[family]
        for _ in range(n):
            tasks.append(gen(rng, task_idx, used))
            task_idx += 1
    return tasks


# ---------------------------------------------------------------------------
# Self-test — dry-run scoring without a model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from dataclasses import asdict
    rng = random.Random(0)
    counts = {
        "single_fact": 2, "multi_hop": 2, "contradiction": 2,
        "aggregation": 2, "temporal": 2,
    }
    tasks = generate_tasks(rng, counts)
    print(f"Generated {len(tasks)} tasks across {len(counts)} families")
    for t in tasks:
        print()
        print(f"[{t.task_family:>14}] {t.task_id}")
        print(f"  Q: {t.question}")
        print(f"  A: {t.answer}")
        print(f"  spans: {len(t.evidence_spans)}, "
              f"required: {len(t.required_evidence_ids)}, "
              f"min: {t.min_evidence_count}")
        for s in t.evidence_spans:
            print(f"    {s.evidence_id}: {s.text[:80]}...")
