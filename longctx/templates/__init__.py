"""Chat templates per model family.

Different model families need different system prompts and message
formatting to do well on retrieval-style verbatim-prefix tasks. This
module provides tested templates.

Validated 2026-05-06:
- Qwen2.5 family: default template works well, scored 0.760 on MRCR 8K
- Qwen2.5-7B vanilla: default template, scored 0.567 on MRCR 8K
- Mistral-7B-Instruct-v0.3: default template fails (model doesn't echo prefix)
- Qwen3-8B: default template fails (reasoning preamble breaks prefix-first)

Both negative cases are fixable with model-specific templates that
suppress reasoning and emphasize raw-text-output mode.
"""
from longctx.templates.qwen25 import QWEN25_TEMPLATE, QWEN25_VERBATIM_TEMPLATE
from longctx.templates.qwen3 import QWEN3_NO_THINK_TEMPLATE
from longctx.templates.mistral import MISTRAL_VERBATIM_TEMPLATE


TEMPLATES = {
    "qwen2.5": QWEN25_TEMPLATE,
    "qwen2.5-verbatim": QWEN25_VERBATIM_TEMPLATE,
    "qwen3-no-think": QWEN3_NO_THINK_TEMPLATE,
    "mistral-verbatim": MISTRAL_VERBATIM_TEMPLATE,
}


__all__ = [
    "TEMPLATES",
    "QWEN25_TEMPLATE",
    "QWEN25_VERBATIM_TEMPLATE",
    "QWEN3_NO_THINK_TEMPLATE",
    "MISTRAL_VERBATIM_TEMPLATE",
]
