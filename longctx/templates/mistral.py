"""Mistral family chat templates.

Mistral-7B-Instruct-v0.3 with the default Qwen2.5-style template scored
0/16 on MRCR 8K bin in 2026-05-06 testing (the model doesn't echo the
prefix string as the first part of the response).

This template is tighter on prefix expectations and uses Mistral's
[INST]-style cue strength.
"""

MISTRAL_VERBATIM_TEMPLATE = (
    "You will be given a small set of candidate text chunks and a user "
    "question. Find the single chunk that answers the question. "
    "Your response must begin EXACTLY with the prefix string the user "
    "provides, followed immediately by the verbatim content of the "
    "matching chunk. No introduction, no commentary, no markdown. "
    "If you cannot identify a match, still begin with the prefix and "
    "respond with the closest candidate."
)
