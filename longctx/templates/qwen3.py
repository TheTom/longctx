"""Qwen3 family chat templates.

Qwen3 has a 'thinking' mode that produces reasoning preamble before the
final answer. For verbatim-prefix tasks, the preamble breaks the
"first token must be the prefix" expectation. The no-think template
suppresses this.

Empirical: 2026-05-06 testing of Qwen3-8B with the default Qwen2.5-style
template scored 0/30 on MRCR 8K bin (prefix_pass=False on every sample).
The no-think variant below addresses this.
"""

QWEN3_NO_THINK_TEMPLATE = (
    "/no_think\n"
    "Task: identify the requested item from the candidates and reproduce "
    "it verbatim with the user's prefix prepended. "
    "Do not produce any reasoning or thinking output. "
    "Do not write 'sure', 'here', or any acknowledgement. "
    "Output only: prefix + verbatim content. The very first token of your "
    "response must be the prefix character."
)
