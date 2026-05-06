"""Quickstart: 'find the right chunk in a long document' workflow.

The simplest possible longctx example. Drop in a list of candidate
text chunks, ask a natural-language question, get the verbatim
answer back.
"""
from longctx import LongCtxClient


def main():
    # Imagine you have a long document split into N chunks. Could be
    # paragraphs of a contract, sections of a codebase, prior chat
    # messages, search results, anything.
    candidates = [
        "Q1 2026 revenue: $12.4M, up 23% YoY. Net loss $2.1M.",
        "Q2 2026 revenue: $14.8M, up 19% YoY. Net loss $1.6M.",
        "Q3 2026 revenue: $18.0M, up 21% YoY. Net loss $0.4M.",
        "Q4 2026 revenue: $22.7M, up 26% YoY. Net income $1.9M.",
        "FY 2026 cash position: $84M, down from $98M FY 2025.",
        "Headcount end of 2026: 187, up from 142 (Dec 2025).",
        # ... could be thousands of chunks
    ]

    client = LongCtxClient(model="qwen25-14b")

    result = client.ask(
        query="What was Q3 2026 revenue and what was the YoY growth?",
        candidates=candidates,
        top_k=4,
    )

    print(f"Retrieved indices: {result.retrieved_indices}")
    print(f"Answer: {result.content}")
    print(
        f"Prompt tokens: {result.prompt_tokens}, "
        f"latency: {result.latency_s:.2f}s"
    )


if __name__ == "__main__":
    main()
