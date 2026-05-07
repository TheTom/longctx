"""Setup / installation / boot-time tests.

These guard the path Tom's testers will hit first: a fresh pip install,
CLI --help, healthz on a new process, env-var driven configuration,
no surprise side effects from importing the package. If any of these
break, the project is unshippable regardless of feature work.
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_top_level_version_string():
    import longctx_svc
    assert isinstance(longctx_svc.__version__, str)
    assert "." in longctx_svc.__version__


def test_import_does_not_load_models():
    """Importing the package must NOT instantiate a SentenceTransformer
    or download anything. First-call latency lives in lazy code paths
    (RetrievePipeline._ensure_embedder), not at import time."""
    src = Path(__file__).resolve().parent.parent
    code = (
        "import sys, importlib.util, json\n"
        "before = set(sys.modules)\n"
        "import longctx_svc\n"
        "import longctx_svc.app\n"
        "import longctx_svc.cli\n"
        "import longctx_svc.client\n"
        "import longctx_svc.proxy\n"
        "after = set(sys.modules) - before\n"
        "loaded = {m for m in after if m.split('.')[0] in "
        "  ('sentence_transformers', 'torch', 'transformers')}\n"
        "print(json.dumps(sorted(loaded)))\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code], text=True,
        cwd=str(src), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    loaded = json.loads(out.strip())
    assert loaded == [], (
        f"package import pulled in heavy deps: {loaded}. Embedder/reranker "
        "must stay lazy or fresh installs will pay the cost on `import`."
    )


def test_all_subpackages_importable():
    for name in [
        "longctx_svc",
        "longctx_svc.app",
        "longctx_svc.cli",
        "longctx_svc.client",
        "longctx_svc.config",
        "longctx_svc.proxy",
        "longctx_svc.cache",
        "longctx_svc.cache.disk",
        "longctx_svc.indexer",
        "longctx_svc.scope.detect",
        "longctx_svc.scope.walk",
        "longctx_svc.session.manager",
        "longctx_svc.state",
        "longctx_svc.watcher",
    ]:
        importlib.import_module(name)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_cli_version():
    out = subprocess.check_output(
        [sys.executable, "-m", "longctx_svc.cli", "version"],
        text=True,
    ).strip()
    import longctx_svc
    assert out == longctx_svc.__version__


def test_cli_help():
    out = subprocess.check_output(
        [sys.executable, "-m", "longctx_svc.cli", "serve", "--help"],
        text=True,
    )
    assert "--upstream" in out
    assert "--port" in out
    assert "--host" in out


def test_cli_clean_when_cache_empty(tmp_path):
    env = {**os.environ, "LONGCTX_CACHE_DIR": str(tmp_path / "doesnotexist")}
    out = subprocess.check_output(
        [sys.executable, "-m", "longctx_svc.cli",
         "clean", "--older-than", "0"],
        text=True, env=env,
    )
    assert "removed 0" in out


# ---------------------------------------------------------------------------
# Boot-time wiring (real subprocess, real HTTP)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    return False


@pytest.fixture
def boot_svc(tmp_path):
    """Boot the real longctx-svc subprocess on a free port. Yields the
    base URL. Janitor disabled to keep test fast."""
    port = _free_port()
    cache = tmp_path / "longctx-cache"
    env = {
        **os.environ,
        "LONGCTX_NO_JANITOR": "1",
        "LONGCTX_CACHE_DIR": str(cache),
    }
    p = subprocess.Popen(
        [sys.executable, "-m", "longctx_svc.cli",
         "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    if not _wait(f"{base}/healthz"):
        p.terminate()
        p.wait(timeout=5)
        pytest.fail("longctx-svc did not boot")
    try:
        yield base, cache
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def test_real_subprocess_healthz(boot_svc):
    base, _ = boot_svc
    with urllib.request.urlopen(f"{base}/healthz") as r:
        data = json.loads(r.read())
    assert data["status"] == "ok"
    assert "version" in data


def test_real_subprocess_status_json(boot_svc):
    base, _ = boot_svc
    with urllib.request.urlopen(f"{base}/longctx/status") as r:
        data = json.loads(r.read())
    assert data["mode"] == "local-only"
    assert "scopes" in data and isinstance(data["scopes"], list)
    assert "memory_mb" in data


def test_real_subprocess_status_text(boot_svc):
    base, _ = boot_svc
    req = urllib.request.Request(
        f"{base}/longctx/status",
        headers={"accept": "text/plain"},
    )
    with urllib.request.urlopen(req) as r:
        text = r.read().decode()
    assert "[longctx] mode: local-only" in text
    assert "[longctx] memory:" in text
    assert "[longctx] disk cache:" in text


def test_real_subprocess_proxy_503_without_upstream(boot_svc):
    """Optional-tool guarantee: proxy endpoints return 503 when upstream
    is unset, never silently 200 with a non-proxied response."""
    base, _ = boot_svc
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body.encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        assert e.code == 503
        return
    pytest.fail("expected 503")


def test_real_subprocess_cache_dir_respected(boot_svc, tmp_path):
    base, cache = boot_svc
    # cache dir created lazily on first save; just verify env override
    # took effect by checking the status text shows the right path.
    req = urllib.request.Request(
        f"{base}/longctx/status",
        headers={"accept": "text/plain"},
    )
    with urllib.request.urlopen(req) as r:
        text = r.read().decode()
    assert str(cache) in text


# ---------------------------------------------------------------------------
# Optional-tool guarantee (regression shield)
# ---------------------------------------------------------------------------

def test_client_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    from longctx_svc.client import LongctxClient
    assert LongctxClient.from_env() is None


def test_client_optional_pattern_engine_can_skip_when_none(monkeypatch):
    """The literal idiom every engine integration uses: get-or-None,
    skip when None. Lock it down so future refactors can't break the
    shape that vllm-swift / llama.cpp / vllm-amd adapters depend on."""
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    from longctx_svc.client import LongctxClient

    cli = LongctxClient.from_env()
    if cli is None:
        # this is the no-retrieval path the engine takes
        retrieved_chunks = []
    else:
        retrieved_chunks = ["never reached"]
    assert retrieved_chunks == []


# ---------------------------------------------------------------------------
# Debug-dump env var (used by integration harness + ad-hoc debugging)
# ---------------------------------------------------------------------------

def test_debug_dump_env_writes_rewritten_body(boot_svc, tmp_path,
                                                 monkeypatch):
    """When LONGCTX_DEBUG_DUMP is set, every rewritten upstream body is
    persisted as <ts>-<route>.json. Critical for reproducing tester
    bug reports later."""
    base, _ = boot_svc
    # boot_svc didn't set LONGCTX_DEBUG_DUMP; spawn a second svc that does.
    dump_dir = tmp_path / "dumps"
    port = _free_port()
    env = {
        **os.environ,
        "LONGCTX_NO_JANITOR": "1",
        "LONGCTX_DEBUG_DUMP": str(dump_dir),
        "LONGCTX_UPSTREAM": "http://127.0.0.1:1",  # unreachable on purpose
    }
    p = subprocess.Popen(
        [sys.executable, "-m", "longctx_svc.cli",
         "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        base2 = f"http://127.0.0.1:{port}"
        assert _wait(f"{base2}/healthz")
        body = json.dumps({
            "model": "test", "messages": [{"role": "user", "content": "hi"}],
        })
        req = urllib.request.Request(
            f"{base2}/v1/chat/completions",
            data=body.encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        # We expect a 5xx because upstream is unreachable; the dump is
        # what we actually care about.
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError:
            pass
        except urllib.error.URLError:
            pass
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    # Dump should have been written before the upstream call failed.
    files = list(dump_dir.glob("*.json")) if dump_dir.exists() else []
    assert files, f"expected dump in {dump_dir}, got none"
    body = json.loads(files[0].read_text())
    assert "messages" in body
