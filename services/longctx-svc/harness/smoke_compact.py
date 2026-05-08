"""End-to-end smoke for `compact` mode.

Phase 1: write 10 chunks to longctx-svc directly (via TestClient).
Phase 2: mock _post_chat to return rescued-text answers.
Verifies:
  * chunks reach /evict/dump under the harness session_id
  * coverage = 100% for facts whose token_pos lies in any chunk range
  * questions produce one classification per fact
  * native_hit is suppressed (compact mode forces zero active window)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

os.environ.setdefault("LONGCTX_NO_JANITOR", "1")

from harness import streaming_driver
from harness.synthetic_haystack import build_haystack


def main() -> int:
    from fastapi.testclient import TestClient
    from longctx_svc.app import app
    client = TestClient(app)
    print(f"[smoke_compact] longctx-svc up: {client.get('/healthz').json()}",
          file=sys.stderr)

    # Build a tiny haystack
    from transformers import AutoTokenizer
    tok_name = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    text, planted = build_haystack(
        target_tokens=10_000, n_facts=5, tokenizer=tok, seed=42,
    )

    tmp = Path(tempfile.mkdtemp(prefix="harness_compact_"))
    haystack_path = tmp / "haystack.json"
    out_path = tmp / "run.json"
    haystack_path.write_text(json.dumps({
        "tokens_target": 10_000,
        "tokens_actual": len(tok.encode(text)),
        "n_facts": 5,
        "tokenizer": tok_name,
        "seed": 42,
        "haystack": text,
        "facts": [asdict(f) for f in planted],
    }))

    session_id = "smoke-compact-session"

    # Patch _post_chat — mock vLLM Tier 3 by returning the truth.
    facts = json.loads(haystack_path.read_text())["facts"]
    truths = {f["question"]: f["answer"] for f in facts}

    def stub(endpoint, model, messages, max_tokens=64, temperature=0.0,
             timeout=600.0, session_id=None):
        last_user = next(
            (m for m in reversed(messages) if m["role"] == "user"), None
        )
        text = last_user["content"] if last_user else ""
        # Mirror the X-Longctx-Session check: in real flow, vLLM's hook
        # would have prepended rescued chunks; here we just look up the
        # truth based on the question marker.
        q = text.split("QUESTION:", 1)[1].strip() if "QUESTION:" in text else ""
        ans = f"The answer is {truths.get(q, 'unknown')}."
        return {
            "choices": [{"message": {"content": ans}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 12},
        }
    streaming_driver._post_chat = stub

    # Patch requests.post (used by Phase 1 ingest) onto TestClient.
    import requests as _req
    real_post = _req.post

    def post_stub(url, json=None, timeout=None, **kw):
        from urllib.parse import urlparse
        path = urlparse(url).path
        r = client.post(path, json=json)

        class _R:
            status_code = r.status_code

            def raise_for_status(self):
                if r.status_code >= 400:
                    raise Exception(f"HTTP {r.status_code}: {r.text}")

            def json(self):
                return r.json()
        return _R()
    real_get = _req.get

    def get_stub(url, params=None, timeout=None, **kw):
        from urllib.parse import urlparse
        path = urlparse(url).path
        r = client.get(path, params=params)

        class _R:
            status_code = r.status_code

            def raise_for_status(self):
                if r.status_code >= 400:
                    raise Exception(f"HTTP {r.status_code}: {r.text}")

            def json(self):
                return r.json()
        return _R()
    _req.post = post_stub
    _req.get = get_stub
    try:
        streaming_driver.run(
            haystack_path=haystack_path,
            endpoint="http://mock",
            model="mock",
            turn_tokens=2048,
            out_path=out_path,
            mode="compact",
            longctx_endpoint="http://test",
            session_id=session_id,
        )
    finally:
        _req.post = real_post
        _req.get = real_get

    out = json.loads(out_path.read_text())
    summary = out["summary"]
    cov = out.get("coverage") or {}
    print(f"[smoke_compact] summary={summary}", file=sys.stderr)
    print(f"[smoke_compact] coverage={cov.get('coverage_pct')}% "
          f"({cov.get('n_covered')}/{cov.get('n_facts')}, "
          f"{cov.get('n_chunks')} chunks)", file=sys.stderr)

    fails: list[str] = []
    if summary.get("native_hit", 0) != 0:
        fails.append(f"native_hit must be 0 in compact mode, "
                     f"got {summary['native_hit']}")
    if summary.get("exact", 0) != len(facts):
        fails.append(f"exact={summary.get('exact', 0)} expected {len(facts)}")
    if cov.get("coverage_pct", 0) != 100.0:
        fails.append(f"coverage_pct={cov.get('coverage_pct')}% expected 100%")
    if cov.get("n_chunks", 0) <= 0:
        fails.append("dump returned zero chunks — session id didn't propagate")

    if fails:
        for f in fails:
            print(f"[smoke_compact] FAIL: {f}", file=sys.stderr)
        return 1
    print("[smoke_compact] PASS — compact mode end-to-end clean",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
