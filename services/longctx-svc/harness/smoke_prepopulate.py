"""Smoke test the turn-1 prepopulate path end-to-end (no GPU).

Brings up an in-process longctx-svc (TestClient), monkey-patches
backend_helpers env + globals, runs prepopulate_rescue_store, then
validates /evict/dump shows synthetic chunks (layer=-2) and
/evict/retrieve surfaces them by query.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    # Monkey LONGCTX env BEFORE importing backend_helpers so the env-gated
    # _PREPOPULATE_ENABLED constant captures the right value.
    os.environ["VLLM_TRIATT_PREPOPULATE_TURN1"] = "1"
    os.environ["VLLM_TRIATT_PREPOPULATE_SPAN"] = "32"
    os.environ["LONGCTX_ENDPOINT"] = "http://test"

    # Import and patch the vllm-turboquant module from disk.
    import importlib.util
    bh_path = (
        "/Users/tom/dev/vllm-turboquant/vllm/v1/attention/triattention/"
        "backend_helpers.py"
    )
    spec = importlib.util.spec_from_file_location("bh", bh_path)
    bh = importlib.util.module_from_spec(spec)
    # Stub `vllm.logger` + `vllm.model_executor.models.utils` to avoid a
    # full vLLM import.
    import types
    fake_logger_mod = types.ModuleType("vllm.logger")
    fake_logger_mod.init_logger = lambda name: __import__(
        "logging"
    ).getLogger(name)
    sys.modules["vllm.logger"] = fake_logger_mod
    fake_utils = types.ModuleType("vllm.model_executor.models.utils")
    fake_utils.extract_layer_index = lambda name: 0
    sys.modules["vllm.model_executor.models.utils"] = fake_utils
    fake_hooks = types.ModuleType("vllm.v1.attention.triattention.hooks")
    fake_hooks.get_engine = lambda: None
    sys.modules["vllm.v1.attention.triattention.hooks"] = fake_hooks
    spec.loader.exec_module(bh)

    # Bring up longctx-svc as TestClient
    from fastapi.testclient import TestClient
    from longctx_svc.app import app
    client = TestClient(app)
    health = client.get("/healthz").json()
    print(f"[smoke] longctx-svc health: {health}")

    # Stub a tiny tokenizer
    class _StubTok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"tok{i}" for i in ids)

    bh.set_tokenizer(_StubTok())
    bh.set_longctx_session_id("smoke-prepop")
    bh._PROMPT_TOKEN_IDS[0] = list(range(200))

    # Replace _get_http_session with one that targets TestClient
    class _Sess:
        def post(self, url, json=None, timeout=None):
            from urllib.parse import urlparse
            path = urlparse(url).path
            r = client.post(path, json=json)
            class _R:
                status_code = r.status_code
                def json(self):
                    return r.json()
            return _R()

    bh._LONGCTX_HTTP = _Sess()
    bh._LONGCTX_BASE_URL = "http://test"

    n = bh.prepopulate_rescue_store(0)
    print(f"[smoke] prepopulate posted {n} chunks")
    if n == 0:
        print("[smoke] FAIL — prepopulate posted no chunks")
        return 1

    dump = client.get("/evict/dump",
                      params={"session_id": "smoke-prepop"}).json()
    print(f"[smoke] dump session_total={dump['session_total']} "
          f"layers={set(dump['layers'])}")

    # All synthetic chunks should carry layer=-2.
    if not dump["session_total"] == n:
        print(f"[smoke] FAIL: dump count {dump['session_total']} != {n}")
        return 1
    if set(dump["layers"]) != {-2}:
        print(f"[smoke] FAIL: layers {set(dump['layers'])} != {{-2}}")
        return 1

    # Retrieve by query should return >0 chunks.
    r = client.post("/evict/retrieve", json={
        "session_id": "smoke-prepop",
        "query": "tok42 tok43 tok44",
        "top_k": 4,
        "score_floor": 0.0,
    }).json()
    print(f"[smoke] retrieve returned {len(r['chunks'])} chunks "
          f"(session_total={r['session_total']})")
    if len(r["chunks"]) == 0:
        print("[smoke] FAIL — retrieve returned 0 chunks")
        return 1

    # Idempotence: second prepopulate call must be a no-op for same session.
    n2 = bh.prepopulate_rescue_store(0)
    if n2 != 0:
        print(f"[smoke] FAIL — second prepopulate posted {n2}, expected 0")
        return 1

    print("[smoke] PASS — prepopulate path armed and idempotent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
