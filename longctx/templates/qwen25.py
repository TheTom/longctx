"""Qwen2.5 family chat templates.

Validated against Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct-1M,
Qwen2.5-32B-Instruct.
"""

QWEN25_TEMPLATE = (
    "You are given a small set of candidate prior assistant messages "
    "retrieved from a longer conversation, plus the user's final question. "
    "The user asks for one specific message (e.g. 'the 2nd play about the "
    "fugitive'). Identify which retrieved candidate matches and reproduce "
    "it verbatim, prepending the prefix string the user provides. "
    "Output ONLY the requested message: prefix + verbatim content. "
    "No commentary, no analysis."
)

# Stricter variant: emphasizes "no thinking, no preamble" for cases where
# the user wants the reproduction even if the model would normally add a
# brief acknowledgement.
QWEN25_VERBATIM_TEMPLATE = (
    "Task: identify the requested item from the candidates and reproduce "
    "it verbatim with the user's prefix prepended. "
    "Do not write 'sure', 'here', or any acknowledgement. "
    "Do not summarize or explain. Output only: prefix + verbatim content."
)
