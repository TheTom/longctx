# messy MRCR: robustness x context-length retrieval matrix

Generated: 2026-05-09T16:32:03.528570+00:00  
Embedder: `BAAI/bge-small-en-v1.5`  
Needles: 8  
Samples per cell: 5  
Top-K: 5  
Seed: 0

## what this measures

long-context retrieval has two axes that the literature usually conflates:

1. **context length**: how does recall degrade as the haystack grows?
2. **query messiness**: how does recall degrade when the user's
   question is malformed (typos, run-on, multi-question, off-corpus)?

messy MRCR crosses them. each cell of the matrix is recall@K for one
(messiness, context-bin) pair. that's enough to see if the messiness
penalty scales linearly with context, super-linearly, or saturates.

## why it's novel

MRCR measures retrieval recall at long context but with clean queries.
messy-query work measures messiness penalty but on small corpora.
nobody has crossed the axes. this matrix is the crossing.

the off-corpus row also gives a calibration line. if my abstention
path is honest, those cells should be near-zero across all bins.

## method

* baseline: openai/mrcr 8-needle samples streamed
  from huggingface, filtered into char-range bins.
* corpus: each conversation message is written as its own file,
  indexed via the longctx daemon's BM25 + dense + RRF stack.
* query: each MRCR sample's final user message is fed through one
  of the messiness transforms.
* score: recall@5. did any chunk in the top-5 retrieved
  results contain the gold answer text (or its random-string prefix)?
* embedder: BAAI/bge-small-en-v1.5, cosine on L2-normalized vectors.

## algorithms

* **clean**: control: identity transform; the unmodified query.
* **typo**: adjacent-letter swaps in 3-5% of words; drops trailing ?
* **multi-question**: prepend 2-3 unrelated questions before the real query.
* **run-on**: strip punctuation, lowercase, collapse whitespace.
* **fragments**: content tokens only: drop stopwords, drop syntax.
* **blob+question**: prepend a fake stack-trace blob before the real query.
* **off-corpus**: replace with an abstention query (sanity-check row).

see `longctx_daemon/eval/messy_mrcr.py` for the exact algorithm of
each transform with line-by-line pseudocode.

## results

| query_type     |    32K |    64K |
|----------------|--------|--------|
| clean          |  0.800 |  1.000 |
| typo           |  0.800 |  1.000 |
| multi-question |  0.800 |  1.000 |
| run-on         |  0.800 |  1.000 |
| fragments      |  0.800 |  1.000 |
| blob+question  |  0.800 |  1.000 |
| off-corpus     |  0.000 |  0.000 |

## reading guide

* read the **clean** row left-to-right to see how recall degrades
  with context length on perfectly-formed queries.
* read each **messy** row against the clean row at the same column
  to see how much that specific messiness costs at that context size.
* read the **off-corpus** row across the board to confirm the
  abstention path holds. those cells should be near-zero. if they're
  not, the relevance floor is mis-calibrated for that bin.
* compare deltas diagonally to spot compounding effects: does the
  typo penalty at 256K eat more recall than at 8K?

## caveats

* embedder-specific. all numbers are BAAI/bge-small-en-v1.5 on cosine.
  bigger embedders (bge-large, e5-mistral) will move the absolute
  numbers but the SHAPE of the matrix should hold.
* retrieval-only. the generator is not in the loop. these are pure
  retrieval recall numbers, not MRCR's published prefix-gated
  SequenceMatcher score (which mixes retrieval and generation).
* sample size: 5 per cell. fast-iteration
  default. 100/cell is the headline-numbers setting.
* the 1M bin is opt-in via the CLI. each cell at 1M can take 30+
  minutes; default bins stop at 256K.
* MRCR's 8-needle split has no samples below ~32K char context, so
  the 8K column is genuinely empty for the 8-needle matrix. use 2-
  or 4-needle if you need that column populated.

## reproduce

```bash
python3 -m longctx_daemon.eval.messy_mrcr \
    --needles 8 \
    --samples 5 \
    --bins 32K,64K \
    --transforms clean,typo,multi-question,run-on,fragments,blob+question,off-corpus \
    --seed 0
```

raw per-sample records live alongside this file in
`messy_mrcr_<date>.json` for replay or alternate scoring.
