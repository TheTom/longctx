"""Synthetic haystack builder for the 10M-effective-context PRD.

Builds a long document filled with low-information filler text and N
planted "fact" sentences uniformly distributed across the document.
Designed for evict-to-RAG testing where:

  * The model never sees the whole document in active KV.
  * V3 evicts low-attention positions during streaming prefill.
  * Tier 2 captures evicted spans; Tier 3 retrieves them on the
    final question turn.

Scales from 100K to 100M+ tokens. Tokenization-aware (uses an HF
tokenizer to land facts at known token positions, not character
positions). Output: a single JSON file with the full document plus
per-fact metadata (token position, entity, answer, question).

Usage:
    python synthetic_haystack.py \\
        --tokens 100000 --facts 20 \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --out haystack_100k.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# Filler sentence templates — boring English that won't accidentally
# pattern-match the fact templates. No numerics, no proper nouns
# resembling the fact entities. Each rendered sentence is roughly
# 10-20 tokens.
FILLER_TEMPLATES = [
    "The {adj} {noun} continued without incident throughout the morning.",
    "Researchers noted that the {adj} {noun} required further analysis.",
    "Standard procedure dictated a careful review of the {adj} {noun}.",
    "Operators observed the {adj} {noun} behaving within expected bounds.",
    "Documentation referenced the {adj} {noun} multiple times in the appendix.",
    "Each subsequent {noun} was {adj} but otherwise unremarkable.",
    "The committee discussed the {adj} {noun} at length but reached no decision.",
    "Reports indicated the {adj} {noun} had been categorized last quarter.",
    "Staff members confirmed that the {adj} {noun} required no further action.",
    "Working groups had previously cataloged the {adj} {noun} for archival.",
]
ADJECTIVES = [
    "routine", "preliminary", "ongoing", "scheduled", "informal",
    "tentative", "quarterly", "annual", "regional", "departmental",
    "auxiliary", "transitional", "institutional", "operational", "advisory",
    "regulatory", "interim", "supplementary", "provisional", "general",
]
NOUNS = [
    "review", "assessment", "discussion", "report", "summary",
    "memo", "briefing", "consultation", "analysis", "evaluation",
    "inspection", "checklist", "outline", "framework", "directive",
    "procedure", "guideline", "protocol", "schedule", "agenda",
]


# Fact templates — each defines:
#   * how to render the fact sentence (with an entity name + answer)
#   * the question that should retrieve this fact
#   * a regex-ish pattern to detect the answer in model output
#
# Entities are crafted to be SEMANTICALLY DISTINCT from each other AND
# from filler nouns, so retrieval has a fighting chance via cosine
# similarity at the embedder level.
FACT_TEMPLATES = [
    {
        "kind": "access_code",
        "entity_pool": [
            "NOVA", "HELIOS", "ORION", "VEGA", "CYGNUS",
            "LYRA", "DRACO", "PERSEUS", "AQUILA", "SAGITTA",
            "PHOENIX", "TUCANA", "CETUS", "HYDRA", "PEGASUS",
            "ANDROMEDA", "CASSIOPEIA", "BOOTES", "AURIGA", "ERIDANUS",
        ],
        "render": (
            "Project {entity} was provisioned with access code {answer}. "
            "All technicians must memorize this code prior to deployment."
        ),
        "question": "What access code does Project {entity} use?",
        "answer_kind": "6digit",
    },
    {
        "kind": "renewal_date",
        "entity_pool": [
            "TITAN", "ATLAS", "PROMETHEUS", "EPIMETHEUS", "HYPERION",
            "OCEANUS", "TETHYS", "RHEA", "CRONUS", "MNEMOSYNE",
            "PHOEBE", "THEIA", "IAPETUS", "COEUS", "CRIUS",
            "DIONE", "EURYBIA", "METIS", "PALLAS", "STYX",
        ],
        "render": (
            "Contract {entity} entered the active phase recently and "
            "is scheduled to renew on {answer}. The legal team has "
            "approved the terms."
        ),
        "question": "When does Contract {entity} renew?",
        "answer_kind": "iso_date",
    },
    {
        "kind": "record_count",
        "entity_pool": [
            "ORCHID", "TULIP", "DAFFODIL", "MAGNOLIA", "JASMINE",
            "PRIMROSE", "AZALEA", "GARDENIA", "VIOLET", "IRIS",
            "DAHLIA", "POPPY", "FOXGLOVE", "HIBISCUS", "OLEANDER",
            "CAMELLIA", "LAVENDER", "HYACINTH", "GERANIUM", "PEONY",
        ],
        "render": (
            "Dataset {entity} was finalized last week and contains "
            "exactly {answer} records. Auditors have signed off on "
            "the count."
        ),
        "question": "How many records are in Dataset {entity}?",
        "answer_kind": "5digit",
    },
]


@dataclass
class PlantedFact:
    """One fact placed at a known token position in the haystack."""
    fact_idx: int            # 0..N-1
    kind: str                # "access_code" / "renewal_date" / "record_count"
    entity: str              # "NOVA"
    answer: str              # "481729"
    question: str            # "What access code does Project NOVA use?"
    sentence: str            # the rendered fact sentence
    token_pos: int           # absolute token offset where the fact sentence starts
    char_pos: int            # absolute char offset (for sanity / fallback)


def _gen_answer(rng: random.Random, kind: str) -> str:
    """Generate a plausible answer for a fact-kind."""
    if kind == "6digit":
        return f"{rng.randint(100_000, 999_999)}"
    if kind == "5digit":
        return f"{rng.randint(10_000, 99_999):,}"  # commas for readability
    if kind == "iso_date":
        year = rng.randint(2026, 2032)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        return f"{year:04d}-{month:02d}-{day:02d}"
    raise ValueError(f"unknown answer_kind: {kind}")


def _filler_sentence(rng: random.Random) -> str:
    tpl = rng.choice(FILLER_TEMPLATES)
    return tpl.format(
        adj=rng.choice(ADJECTIVES), noun=rng.choice(NOUNS)
    )


def build_haystack(
    target_tokens: int,
    n_facts: int,
    tokenizer: Any,
    seed: int = 42,
) -> tuple[str, list[PlantedFact]]:
    """Build a haystack of approximately `target_tokens` tokens with
    `n_facts` facts uniformly distributed across the document.

    Strategy:
      1. Generate filler in fixed-size chunks (~256 tokens each).
      2. After each chunk, check current token count. When we cross
         the next planted-fact threshold, emit the fact sentence.
      3. Continue until we hit target_tokens.

    Returns (full_text, planted_facts).
    """
    rng = random.Random(seed)

    # Pick N fact slots uniformly across the document.
    if n_facts > sum(len(t["entity_pool"]) for t in FACT_TEMPLATES):
        raise ValueError(
            f"n_facts={n_facts} exceeds total entity pool size"
        )
    fact_slots = sorted(
        int((i + 0.5) / n_facts * target_tokens) for i in range(n_facts)
    )

    # Pick facts: cycle through templates, sample entities without replacement.
    used_entities: dict[str, set[str]] = {
        t["kind"]: set() for t in FACT_TEMPLATES
    }
    facts_to_plant: list[dict] = []
    for i in range(n_facts):
        tpl = FACT_TEMPLATES[i % len(FACT_TEMPLATES)]
        available = [
            e for e in tpl["entity_pool"]
            if e not in used_entities[tpl["kind"]]
        ]
        entity = rng.choice(available)
        used_entities[tpl["kind"]].add(entity)
        answer = _gen_answer(rng, tpl["answer_kind"])
        facts_to_plant.append({
            "fact_idx": i,
            "kind": tpl["kind"],
            "entity": entity,
            "answer": answer,
            "sentence": tpl["render"].format(entity=entity, answer=answer),
            "question": tpl["question"].format(entity=entity),
            "target_token_pos": fact_slots[i],
        })

    parts: list[str] = []
    planted: list[PlantedFact] = []
    cur_tokens = 0  # incremental count
    cur_chars = 0
    next_fact = 0

    # Pre-tokenize fact sentences once (cheap, fixed set).
    fact_token_lens = [
        len(tokenizer.encode(f["sentence"], add_special_tokens=False))
        for f in facts_to_plant
    ]

    # Filler chunks: pre-build and pre-count, since most filler is
    # repeating templates the tokenizer caches well. Each chunk is
    # ~16 sentences ≈ 200 tokens. Encoded once per chunk, NOT
    # re-tokenizing the cumulative text. O(n) overall.
    CHUNK_SENTENCES = 16

    def _new_chunk() -> tuple[str, int]:
        sents = [_filler_sentence(rng) for _ in range(CHUNK_SENTENCES)]
        text = " ".join(sents) + " "
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        return text, n_tok

    while cur_tokens < target_tokens:
        # Plant facts when we cross their target token position.
        if (next_fact < len(facts_to_plant)
                and cur_tokens >= facts_to_plant[next_fact]["target_token_pos"]):
            f = facts_to_plant[next_fact]
            sentence = f["sentence"] + " "
            parts.append(sentence)
            fact_token_pos = cur_tokens
            char_pos = cur_chars
            cur_chars += len(sentence)
            cur_tokens += fact_token_lens[next_fact]
            planted.append(PlantedFact(
                fact_idx=f["fact_idx"],
                kind=f["kind"],
                entity=f["entity"],
                answer=f["answer"],
                question=f["question"],
                sentence=f["sentence"],
                token_pos=fact_token_pos,
                char_pos=char_pos,
            ))
            next_fact += 1
            continue

        # Add a filler chunk. Incremental token count — no full
        # re-tokenization. Roughly accurate (BPE has minor drift
        # at boundaries; close enough for fact-position placement).
        chunk_text, chunk_tokens = _new_chunk()
        parts.append(chunk_text)
        cur_chars += len(chunk_text)
        cur_tokens += chunk_tokens

    full_text = "".join(parts)

    # Reconcile token positions: incremental count drifts because BPE
    # has token-boundary effects when concatenating chunks. Re-tokenize
    # once with offset_mapping to get precise char→token mapping, then
    # update each PlantedFact's token_pos. char_pos is exact (we tracked
    # it byte-precisely above).
    encoded = tokenizer(
        full_text, return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded["offset_mapping"]  # list of (char_start, char_end)
    # Build char_start → token_idx map. We only need it at the planted
    # fact char_pos values (small), so binary search over offsets.
    import bisect
    char_starts = [s for (s, _) in offsets]

    for f in planted:
        # Find the token whose start is >= f.char_pos.
        idx = bisect.bisect_left(char_starts, f.char_pos)
        f.token_pos = idx
    return full_text, planted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True,
                    help="Target token count of the haystack")
    ap.add_argument("--facts", type=int, default=20,
                    help="Number of planted facts")
    ap.add_argument("--tokenizer", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF tokenizer to use")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    print(f"Building haystack: {args.tokens:,} tokens, "
          f"{args.facts} facts...", file=sys.stderr)
    text, planted = build_haystack(
        args.tokens, args.facts, tok, seed=args.seed
    )

    final_token_count = len(tok.encode(text))
    print(f"Final: {final_token_count:,} tokens, "
          f"{len(planted)} facts placed", file=sys.stderr)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "tokens_target": args.tokens,
        "tokens_actual": final_token_count,
        "n_facts": args.facts,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "haystack": text,
        "facts": [asdict(f) for f in planted],
    }))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
