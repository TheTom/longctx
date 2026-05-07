"""Engine-agnostic client for longctx-svc.

Any inference engine — vllm-swift, llama.cpp (TheTom/llama-cpp-turboquant),
vLLM CUDA, vLLM AMD (TheTom/vllm feature/turboquant-amd-noautotune), or
plain Python — can import this to call /retrieve over HTTP. The tool
remains *optional*: the importing engine should disable retrieval if the
endpoint is unset or unreachable.

Usage from an engine adapter:

    from longctx_svc.client import LongctxClient
    cli = LongctxClient.from_env()    # honors LONGCTX_ENDPOINT
    if cli is not None:               # tool is optional
        result = cli.retrieve(
            prefill_text=full_prompt,
            query=user_message,
            session_id=session_id,
            top_k=8,
        )
        full_prompt = cli.splice(full_prompt, result)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
    file_type: str
    score: float

    def header(self) -> str:
        return f"// {self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class RetrieveResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    scope_path: str | None = None
    scope_status: str = "no-scope"
    scope_sentinel: str | None = None
    scope_hash: str | None = None
    session_id: str | None = None
    used_rerank: bool = False
    paraphrases_count: int = 0


class LongctxClient:
    """Thin HTTP client. Sync — pick `aretrieve` for async hot paths."""

    DEFAULT_TIMEOUT = 30.0

    def __init__(self, endpoint: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls, var: str = "LONGCTX_ENDPOINT") -> "LongctxClient | None":
        """Construct from env. Returns None if the var is unset → engine
        falls back to non-retrieval path.
        """
        url = os.environ.get(var)
        if not url:
            return None
        return cls(url)

    def healthz(self) -> bool:
        try:
            r = httpx.get(f"{self.endpoint}/healthz", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def retrieve(self, prefill_text: str, query: str,
                 session_id: str | None = None,
                 top_k: int = 8,
                 explicit_scope: str | None = None,
                 ) -> RetrieveResult:
        headers: dict[str, str] = {}
        if session_id is not None:
            headers["x-session-affinity"] = session_id
        body: dict[str, Any] = {
            "prefill_text": prefill_text,
            "query": query,
            "top_k": top_k,
        }
        if explicit_scope is not None:
            body["explicit_scope"] = explicit_scope
        try:
            r = httpx.post(
                f"{self.endpoint}/retrieve",
                json=body, headers=headers, timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError:
            # Optional tool: degrade silently to no chunks.
            return RetrieveResult(session_id=session_id)
        return self._parse(data)

    async def aretrieve(self, prefill_text: str, query: str,
                        session_id: str | None = None,
                        top_k: int = 8,
                        explicit_scope: str | None = None,
                        ) -> RetrieveResult:
        headers: dict[str, str] = {}
        if session_id is not None:
            headers["x-session-affinity"] = session_id
        body: dict[str, Any] = {
            "prefill_text": prefill_text,
            "query": query,
            "top_k": top_k,
        }
        if explicit_scope is not None:
            body["explicit_scope"] = explicit_scope
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as ac:
                r = await ac.post(
                    f"{self.endpoint}/retrieve",
                    json=body, headers=headers,
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError:
            return RetrieveResult(session_id=session_id)
        return self._parse(data)

    @staticmethod
    def _parse(data: dict[str, Any]) -> RetrieveResult:
        chunks = [RetrievedChunk(
            text=c["text"], file_path=c["file_path"],
            start_line=c["start_line"], end_line=c["end_line"],
            file_type=c.get("file_type", "other"),
            score=float(c.get("score", 0.0)),
        ) for c in data.get("chunks", [])]
        return RetrieveResult(
            chunks=chunks,
            scope_path=data.get("scope_path"),
            scope_status=data.get("scope_status", "no-scope"),
            scope_sentinel=data.get("scope_sentinel"),
            scope_hash=data.get("scope_hash"),
            session_id=data.get("session_id"),
            used_rerank=bool(data.get("used_rerank", False)),
            paraphrases_count=int(data.get("paraphrases_count", 0)),
        )

    @staticmethod
    def splice(prefill: str, result: RetrieveResult,
               header: str = "## Retrieved code context") -> str:
        """Splice retrieved chunks into the prompt as a fenced block.

        Engine-agnostic format: a header section the model can see,
        with each chunk preceded by its `// path:start-end` line so the
        model can cite the source.
        """
        if not result.chunks:
            return prefill
        parts = [header, ""]
        for c in result.chunks:
            parts.append(c.header())
            parts.append("```")
            parts.append(c.text.rstrip())
            parts.append("```")
            parts.append("")
        block = "\n".join(parts).rstrip() + "\n\n"
        return block + prefill
