"""LongCtxClient: end-to-end retrieval + generation client.

Wires RetrievalPipeline to an OpenAI-compatible chat completions endpoint
(vLLM, SGLang, llama.cpp server, etc.). Default config matches the stack
that scored 0.760 on MRCR v2 8K bin: sentence-transformers/all-MiniLM-L6-v2
+ Qwen2.5-14B-Instruct-1M.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import requests

from longctx.rag.pipeline import RetrievalPipeline


@dataclass
class LongCtxResponse:
    content: str
    retrieved_indices: list[int]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


DEFAULT_SYSTEM_PROMPT = (
    "You are given a small set of candidate prior assistant messages "
    "retrieved from a longer conversation, plus the user's final question. "
    "The user asks for one specific message (e.g. 'the 2nd play about the "
    "fugitive'). Identify which retrieved candidate matches and reproduce "
    "it verbatim. If the user provides a prefix string to prepend, prepend "
    "it. Output ONLY the requested message: prefix + verbatim content. "
    "No commentary, no analysis."
)


class LongCtxClient:
    """End-to-end retrieval + generation client.

    Usage:
        client = LongCtxClient()
        result = client.ask(
            query="What was the 2nd play about the fugitive?",
            candidates=[msg1, msg2, ..., msgN],
            top_k=8,
        )
        print(result.content)

    Defaults to a local vLLM server at http://localhost:5050 serving
    Qwen2.5-14B-Instruct-1M. Pass model name + server URL to override.
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline | None = None,
        server: str = "http://localhost:5050/v1/chat/completions",
        model: str = "qwen25-14b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 600.0,
    ) -> None:
        self.pipeline = pipeline or RetrievalPipeline()
        self.server = server
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout

    def ask(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LongCtxResponse:
        import time

        retrieval = self.pipeline.retrieve(query, candidates, top_k=top_k)

        body = "\n\n".join(
            f"=== CANDIDATE {pos + 1} (originally item {idx + 1}) ===\n{m}"
            for pos, (idx, m) in enumerate(
                zip(retrieval.indices, retrieval.candidates)
            )
        )
        user = (
            f"{body}\n\n=== USER FINAL QUESTION ===\n{query}\n\n"
            "Output ONLY the requested message."
        )

        t0 = time.time()
        response = requests.post(
            self.server,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        elapsed = time.time() - t0
        response.raise_for_status()
        data = response.json()

        return LongCtxResponse(
            content=data["choices"][0]["message"]["content"],
            retrieved_indices=retrieval.indices,
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            latency_s=elapsed,
        )
