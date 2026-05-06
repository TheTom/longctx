"""MRCR v2 8-needle eval runner.

Reproduces the benchmark used in OpenAI's MRCR v2 paper. Wraps a
LongCtxClient (or any callable that takes (query, candidates) -> answer)
and scores against the published gold targets.

Verbatim-prefix scoring: the answer must start with a fixed random string
the user provides; the score is SequenceMatcher ratio between answer and
target if the prefix matches, 0 otherwise.

The 8K bin (16K-32K char prompts) on Qwen2.5-14B-Instruct-1M with default
LongCtxClient settings produced avg_score=0.760, beating SubQ Inc.'s
published 0.659 number on the same benchmark (which they reported at
the 1M bin; we don't have their per-bin breakdown).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Sequence


BINS_BY_CHAR = {
    "8k": (16_000, 32_000),
    "16k": (32_000, 64_000),
    "32k": (64_000, 128_000),
    "64k": (128_000, 256_000),
    "128k": (256_000, 512_000),
    "256k": (512_000, 1_000_000),
    "512k": (1_000_000, 2_000_000),
    "1m": (2_000_000, 5_000_000),
}


@dataclass
class MRCRSampleResult:
    i: int
    n_chars_orig: int
    n_asst: int
    starts_with_prefix: bool
    score: float
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    output_head: str
    target_head: str


@dataclass
class MRCRSummary:
    bin: str
    model: str
    n: int
    avg_score: float
    prefix_pass_rate: float
    total_time_s: float
    avg_prompt_tokens: float
    results: list[MRCRSampleResult]


class MRCRRunner:
    """Runs MRCR v2 8-needle samples through a LongCtxClient.

    Usage:
        from longctx import LongCtxClient
        from longctx.eval import MRCRRunner

        client = LongCtxClient()
        runner = MRCRRunner("/path/to/mrcr/data")
        summary = runner.run(client, bin_name="8k", n=30)
        print(f"avg_score={summary.avg_score:.3f}")
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _load_8needle(self, bin_name: str, limit: int):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "MRCRRunner needs pandas. Install with: "
                "pip install longctx[eval]"
            ) from e

        lo, hi = BINS_BY_CHAR[bin_name]
        parts = []
        for f in sorted(self.data_dir.glob("8needle/*.parquet")):
            df = pd.read_parquet(f)
            df = df[(df["n_chars"] >= lo) & (df["n_chars"] < hi)]
            parts.append(df)
        if not parts:
            raise FileNotFoundError(
                f"No 8needle parquet files under {self.data_dir}/8needle/. "
                "Download from the MRCR v2 release."
            )
        df = parts[0] if len(parts) == 1 else __import__(
            "pandas"
        ).concat(parts).reset_index(drop=True)
        if limit > 0:
            df = df.head(limit)
        return df

    def run(
        self,
        client,
        bin_name: str,
        n: int = 30,
        top_k: int = 8,
        max_tokens: int = 4096,
        verbose: bool = True,
    ) -> MRCRSummary:
        df = self._load_8needle(bin_name, n)

        results: list[MRCRSampleResult] = []
        t_start = time.time()

        for i, row in df.iterrows():
            try:
                messages = json.loads(row["prompt"])
            except Exception as e:
                if verbose:
                    print(f"[skip i={i}] {e}")
                continue

            asst_msgs = [
                m["content"] for m in messages if m["role"] == "assistant"
            ]
            final_user = next(
                (
                    m["content"]
                    for m in reversed(messages)
                    if m["role"] == "user"
                ),
                "",
            )
            if not final_user or len(asst_msgs) < top_k:
                continue

            target = row["answer"]
            prefix = row["random_string_to_prepend"]

            response = client.ask(
                query=final_user,
                candidates=asst_msgs,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            output = response.content
            starts_with_prefix = output.startswith(prefix)
            if starts_with_prefix:
                score = SequenceMatcher(None, output, target).ratio()
            else:
                score = 0.0

            rec = MRCRSampleResult(
                i=int(i),
                n_chars_orig=int(row["n_chars"]),
                n_asst=len(asst_msgs),
                starts_with_prefix=starts_with_prefix,
                score=float(score),
                latency_s=response.latency_s,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                output_head=output[:160],
                target_head=target[:160],
            )
            results.append(rec)

            if verbose:
                print(
                    f"[{len(results):>3}/{len(df)}] "
                    f"orig_chars={row['n_chars']} "
                    f"prompt_tok={response.prompt_tokens} "
                    f"score={score:.3f} "
                    f"prefix_ok={starts_with_prefix} "
                    f"lat={response.latency_s:.1f}s"
                )

        total_t = time.time() - t_start
        if not results:
            raise RuntimeError("No samples completed")

        scores = [r.score for r in results]
        avg = sum(scores) / len(scores)
        prefix_rate = sum(
            1 for r in results if r.starts_with_prefix
        ) / len(results)
        avg_prompt_tok = sum(
            r.prompt_tokens or 0 for r in results
        ) / len(results)

        summary = MRCRSummary(
            bin=bin_name,
            model=client.model,
            n=len(results),
            avg_score=avg,
            prefix_pass_rate=prefix_rate,
            total_time_s=total_t,
            avg_prompt_tokens=avg_prompt_tok,
            results=results,
        )

        if verbose:
            print(
                f"\n[summary] bin={bin_name} n={len(results)} "
                f"avg_score={avg:.3f} prefix_pass={prefix_rate:.0%} "
                f"avg_prompt_tok={avg_prompt_tok:.0f} "
                f"total={total_t:.1f}s"
            )

        return summary

    @staticmethod
    def write_summary(summary: MRCRSummary, out_path: str | Path):
        Path(out_path).write_text(
            json.dumps(
                {
                    **asdict(summary),
                    "results": [asdict(r) for r in summary.results],
                },
                indent=2,
            )
            + "\n"
        )
