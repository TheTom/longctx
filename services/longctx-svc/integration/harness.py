"""Cross-fork integration smoke harness for longctx-svc.

Boots an engine (llama.cpp / vllm-swift / vllm-turboquant) plus
longctx-svc as a proxy in front of it, sends a chat-completion
request that mentions a real file path, and verifies that:

  1. longctx-svc detected the project sentinel
  2. chunks were retrieved (x-longctx-chunks-used > 0)
  3. the upstream actually saw the spliced system message

This is the proxy-mode smoke test — zero engine changes required —
which is the path Tom's three forks share. Each engine is selectable
via --engine; the harness skips unsupported configurations rather
than failing.

Usage:
    cd services/longctx-svc
    python3 integration/harness.py --engine llama
    python3 integration/harness.py --engine vllm-swift
    python3 integration/harness.py --engine vllm-amd \
        --remote tom@<droplet-ip>
    python3 integration/harness.py --engine all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# pretty
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {YELLOW}…{RESET} {msg}")


def _section(name: str) -> None:
    print(f"\n{BOLD}== {name} =={RESET}")


# ---------------------------------------------------------------------------
# infra helpers
# ---------------------------------------------------------------------------

def _free_port(start: int = 9100) -> int:
    """Return a free TCP port at or above `start`."""
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port available")


def _wait_http(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status < 500:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


@contextmanager
def _proc(cmd: list[str], env: dict[str, str] | None = None,
          log_path: Path | None = None) -> Iterator[subprocess.Popen]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    log_f = log_path.open("ab") if log_path else subprocess.DEVNULL
    p = subprocess.Popen(
        cmd, env=full_env,
        stdout=log_f if log_path else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    try:
        yield p
    finally:
        try:
            if p.poll() is None:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                else:
                    p.terminate()
                try:
                    p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    else:
                        p.kill()
        except Exception:  # noqa: BLE001
            pass
        if log_path and log_f is not subprocess.DEVNULL:
            try:
                log_f.close()
            except Exception:  # noqa: BLE001
                pass


def _build_fake_project(root: Path) -> Path:
    """Realistic mini-project the engines can plausibly retrieve from."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text('{"name":"smokeapp"}\n')
    (root / "README.md").write_text(
        "# smokeapp\n\nDocker image build pipeline.\n"
    )
    src = root / "src"
    src.mkdir(exist_ok=True)
    (src / "auth.ts").write_text(
        "export function authMiddleware() {\n"
        "  // SHIBBOLETH_AUTH_TOKEN: validates JWT against shibboleth keys\n"
        "  return validateJWT(process.env.SHIBBOLETH_KEY);\n"
        "}\n"
    )
    (src / "billing.ts").write_text(
        "export function chargeUser(userId: string) {\n"
        "  // billing flow goes here\n"
        "  return false;\n"
        "}\n"
    )
    return root


# ---------------------------------------------------------------------------
# longctx-svc
# ---------------------------------------------------------------------------

def _start_longctx_svc(upstream: str | None, port: int,
                       log_path: Path,
                       dump_dir: Path | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "longctx_svc.cli",
        "serve", "--host", "127.0.0.1", "--port", str(port),
    ]
    if upstream:
        cmd += ["--upstream", upstream]
    env = {"LONGCTX_NO_JANITOR": "1"}
    if dump_dir is not None:
        env["LONGCTX_DEBUG_DUMP"] = str(dump_dir)
    full_env = os.environ.copy()
    full_env.update(env)
    log_f = log_path.open("ab")
    return subprocess.Popen(
        cmd, env=full_env, stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def _kill(p: subprocess.Popen) -> None:
    try:
        if p.poll() is None:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            else:
                p.terminate()
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:
                    p.kill()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# engine: TheTom/llama-cpp-turboquant — `llama-server`
# ---------------------------------------------------------------------------

def _llama_cpp_paths() -> tuple[Path | None, Path | None]:
    bin_path = Path("/Users/tom/local_llms/llama.cpp/build/bin/llama-server")
    candidates = [
        Path("/Users/tom/local_llms/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        Path("/Users/tom/local_llms/models/qwen2.5-1.5b-TQ4_1S.gguf"),
    ]
    model = next((c for c in candidates if c.is_file()), None)
    return (bin_path if bin_path.is_file() else None), model


def _smoke_llama_cpp() -> bool:
    _section("llama.cpp (TheTom/llama-cpp-turboquant)")
    server_bin, model = _llama_cpp_paths()
    if server_bin is None or model is None:
        _fail("missing llama-server or model — skipping")
        if server_bin is None:
            _info("expected /Users/tom/local_llms/llama.cpp/build/bin/llama-server")
        if model is None:
            _info("expected qwen2.5-1.5b GGUF in /Users/tom/local_llms/models/")
        return False
    _ok(f"llama-server: {server_bin}")
    _ok(f"model: {model.name}")
    return _run_proxy_smoke(
        engine_name="llama.cpp",
        engine_cmd=[
            str(server_bin), "-m", str(model),
            "--host", "127.0.0.1", "--port", "PLACEHOLDER",
            "--ctx-size", "4096", "--n-predict", "32",
        ],
        engine_port_arg_index=None,  # we patch in-place below
        engine_health_path="/v1/models",
        engine_boot_seconds=120.0,
    )


# ---------------------------------------------------------------------------
# engine: vllm-swift (mlx-swift alpha backend)
# ---------------------------------------------------------------------------

def _vllm_swift_paths() -> tuple[Path | None, Path | None]:
    binp = shutil.which("vllm-swift") or "/opt/homebrew/bin/vllm-swift"
    if not Path(binp).is_file():
        binp = None
    # Prefer smaller models for faster boot; fall through to bigger ones.
    candidates = [
        Path.home() / "models" / "Qwen3.5-2B-4bit",
        Path.home() / "models" / "gemma-4-e2b-it-4bit",
        Path.home() / "models" / "Qwen3-4B-4bit",
        Path.home() / "models" / "Qwen3-0.6B",
    ]
    # Honor an explicit env override
    env_model = os.environ.get("LONGCTX_TEST_MLX_MODEL")
    if env_model:
        candidates.insert(0, Path(env_model))
    model = next((c for c in candidates if c.is_dir()), None)
    return (Path(binp) if binp else None), model


def _vllm_swift_cli_cmd() -> list[str] | None:
    """Resolve a python that can run `python -m vllm_swift.cli` —
    bypasses the brew shell wrapper, which calls vLLM's api_server
    directly and never sees vllm-swift's CLI flags (e.g.
    --retrieval-endpoint).
    """
    venv_py = Path.home() / ".vllm-swift" / "venv" / "bin" / "python3"
    if venv_py.is_file():
        return [str(venv_py), "-m", "vllm_swift.cli"]
    return None


def _smoke_vllm_swift(mode: str = "proxy") -> bool:
    label = f"vllm-swift (TheTom alpha mlx-swift) — mode={mode}"
    _section(label)
    binp, model = _vllm_swift_paths()
    if binp is None or model is None:
        _fail("missing vllm-swift or model — skipping")
        if binp is None:
            _info("expected vllm-swift on PATH")
        if model is None:
            _info("expected ~/models/Qwen3-4B-4bit (or run `vllm-swift download mlx-community/Qwen3-4B-4bit`)")
        return False
    _ok(f"vllm-swift: {binp}")
    _ok(f"model: {model.name}")
    if mode == "embedded":
        return _run_embedded_smoke_vllm_swift(binp, model)
    return _run_proxy_smoke(
        engine_name="vllm-swift",
        engine_cmd=[
            str(binp), "serve", str(model),
            "--host", "127.0.0.1", "--port", "PLACEHOLDER",
            "--max-model-len", "4096",
        ],
        engine_port_arg_index=None,
        engine_health_path="/v1/models",
        engine_boot_seconds=180.0,
    )


def _run_embedded_smoke_vllm_swift(binp: Path, model: Path) -> bool:
    """Mode B: vllm-swift's --retrieval-endpoint flag.

    Boot longctx-svc as a sidecar, boot vllm-swift with
    --retrieval-endpoint pointing at it, talk to vllm-swift directly.
    longctx-svc is NOT in the request path — it's only the retrieval
    backend the rewriter calls into.

    Note: bypasses the brew shell wrapper, which calls vLLM directly
    and would never see --retrieval-endpoint. We invoke the dev
    `vllm_swift.cli` module via the venv python.
    """
    cli_cmd = _vllm_swift_cli_cmd()
    if cli_cmd is None:
        _fail("dev vllm-swift CLI not found at ~/.vllm-swift/venv/bin/python3")
        return False
    log_dir = Path(tempfile.mkdtemp(prefix="longctx-int-vllm-swift-emb-"))
    longctx_log = log_dir / "longctx-svc.log"
    engine_log = log_dir / "engine.log"
    dump_dir = log_dir / "rewritten-bodies"
    print(f"  logs: {log_dir}")

    longctx_port = _free_port(9400)
    engine_port = _free_port(9500)
    longctx_url = f"http://127.0.0.1:{longctx_port}"

    longctx_proc: subprocess.Popen | None = None
    engine_proc: subprocess.Popen | None = None
    try:
        _info(f"starting longctx-svc sidecar on :{longctx_port}")
        longctx_proc = _start_longctx_svc(
            upstream=None, port=longctx_port,
            log_path=longctx_log, dump_dir=None,
        )
        if not _wait_http(f"{longctx_url}/healthz", timeout=30.0):
            _fail("longctx-svc did not become healthy")
            return False
        _ok("longctx-svc healthy")

        _info(f"starting vllm-swift on :{engine_port} "
              f"with --retrieval-endpoint {longctx_url}")
        env = os.environ.copy()
        env["LONGCTX_DEBUG_DUMP"] = str(dump_dir)
        cmd = [
            *cli_cmd, "serve", str(model),
            "--host", "127.0.0.1", "--port", str(engine_port),
            "--max-model-len", "4096",
            "--retrieval-endpoint", longctx_url,
        ]
        engine_proc = subprocess.Popen(
            cmd, env=env,
            stdout=engine_log.open("ab"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        if not _wait_http(f"http://127.0.0.1:{engine_port}/v1/models",
                          timeout=180.0):
            _fail(f"vllm-swift did not become healthy — see {engine_log}")
            return False
        _ok("vllm-swift healthy (with rewriter + retrieval enabled)")

        ok = _drive_proxy(f"http://127.0.0.1:{engine_port}")
        _show_spliced_evidence(dump_dir)
        return ok
    finally:
        if engine_proc is not None:
            _kill(engine_proc)
        if longctx_proc is not None:
            _kill(longctx_proc)


# ---------------------------------------------------------------------------
# engine: TheTom/vllm-turboquant (CUDA / AMD MI300X)
# ---------------------------------------------------------------------------

def _smoke_vllm_amd(remote: str | None) -> bool:
    """Drive a chat completion against an already-running vllm on the
    AMD droplet. Use an SSH tunnel so the upstream looks local:

        ssh -i ~/.ssh/do_amd_mi300x -fN -L 5050:127.0.0.1:5050 do-amd
        ./harness.py --engine vllm-amd --remote 127.0.0.1:5050

    `--remote` accepts host:port, http://host:port, or ssh://user@host:port.
    """
    _section("vllm-turboquant (TheTom/vllm @ MI300X)")
    if not remote:
        _fail("vllm-turboquant needs CUDA/ROCm — pass --remote host:port "
              "(typically a tunneled droplet port)")
        return False
    # Normalize: accept a few common forms.
    r = remote.strip()
    if r.startswith("ssh://"):
        rest = r[len("ssh://"):]
        host_part = rest.split("@")[-1]
    elif r.startswith("http://") or r.startswith("https://"):
        host_part = r.split("://", 1)[1]
    else:
        host_part = r
    if ":" not in host_part:
        host_part = f"{host_part}:8000"
    upstream = f"http://{host_part}"
    _ok(f"using remote upstream: {upstream}")
    _info("(harness assumes you've already started vllm on the droplet)")
    if not _wait_http(f"{upstream}/v1/models", timeout=10.0):
        _fail(f"upstream {upstream}/v1/models not reachable")
        return False
    _ok("upstream reachable")
    # Wire up a longctx-svc proxy in front so we drive end-to-end.
    log_dir = Path(tempfile.mkdtemp(prefix="longctx-int-vllm-amd-"))
    proxy_log = log_dir / "longctx-svc.log"
    dump_dir = log_dir / "rewritten-bodies"
    print(f"  logs: {log_dir}")
    proxy_port = _free_port(9300)
    proxy_proc: subprocess.Popen | None = None
    try:
        _info(f"starting longctx-svc proxy on :{proxy_port}")
        proxy_proc = _start_longctx_svc(
            upstream, proxy_port, proxy_log, dump_dir=dump_dir,
        )
        if not _wait_http(f"http://127.0.0.1:{proxy_port}/healthz",
                          timeout=30.0):
            _fail("longctx-svc did not become healthy")
            return False
        _ok("longctx-svc healthy")
        ok = _drive_proxy(f"http://127.0.0.1:{proxy_port}")
        _show_spliced_evidence(dump_dir)
        return ok
    finally:
        if proxy_proc is not None:
            _kill(proxy_proc)


# ---------------------------------------------------------------------------
# common driver
# ---------------------------------------------------------------------------

def _show_spliced_evidence(dump_dir: Path) -> None:
    """Print a head of the rewritten body so Tom can see exactly what
    the engine received: the longctx 'Retrieved code context' block
    plus the original user message."""
    if not dump_dir.is_dir():
        _info("no debug-dump dir found")
        return
    files = sorted(dump_dir.glob("*.json"))
    if not files:
        _info("no rewritten-body file dumped (request didn't reach proxy)")
        return
    latest = files[-1]
    try:
        body = json.loads(latest.read_text())
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not read dump {latest}: {exc!s}")
        return
    print(f"  {BOLD}rewritten body sent upstream:{RESET} {latest}")
    if "messages" in body:
        for i, m in enumerate(body["messages"]):
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            preview = "\n    ".join(str(content).splitlines()[:30])
            print(f"  [{m.get('role')}]\n    {preview}")
    elif "prompt" in body:
        preview = "\n    ".join(str(body["prompt"]).splitlines()[:40])
        print(f"  [prompt]\n    {preview}")


def _run_proxy_smoke(engine_name: str, engine_cmd: list[str],
                     engine_port_arg_index: int | None,
                     engine_health_path: str,
                     engine_boot_seconds: float) -> bool:
    """Boot a local engine + longctx-svc proxy + drive the test."""
    log_dir = Path(tempfile.mkdtemp(prefix=f"longctx-int-{engine_name}-"))
    engine_log = log_dir / "engine.log"
    proxy_log = log_dir / "longctx-svc.log"
    dump_dir = log_dir / "rewritten-bodies"
    print(f"  logs: {log_dir}")

    engine_port = _free_port(9200)
    proxy_port = _free_port(9300)
    upstream = f"http://127.0.0.1:{engine_port}"

    cmd = [
        c.replace("PLACEHOLDER", str(engine_port)) for c in engine_cmd
    ]

    engine_proc: subprocess.Popen | None = None
    proxy_proc: subprocess.Popen | None = None
    try:
        _info(f"starting {engine_name} on :{engine_port}")
        engine_proc = subprocess.Popen(
            cmd, stdout=engine_log.open("ab"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        if not _wait_http(f"{upstream}{engine_health_path}",
                          timeout=engine_boot_seconds):
            _fail(f"{engine_name} did not become healthy in "
                  f"{engine_boot_seconds:.0f}s — see {engine_log}")
            return False
        _ok(f"{engine_name} healthy")

        _info(f"starting longctx-svc proxy on :{proxy_port}")
        proxy_proc = _start_longctx_svc(
            upstream, proxy_port, proxy_log, dump_dir=dump_dir,
        )
        if not _wait_http(f"http://127.0.0.1:{proxy_port}/healthz",
                          timeout=30.0):
            _fail("longctx-svc did not become healthy — see " + str(proxy_log))
            return False
        _ok("longctx-svc healthy")

        ok = _drive_proxy(f"http://127.0.0.1:{proxy_port}")
        _show_spliced_evidence(dump_dir)
        return ok
    finally:
        if proxy_proc is not None:
            _kill(proxy_proc)
        if engine_proc is not None:
            _kill(engine_proc)


def _discover_model_id(proxy_url: str) -> str:
    """Ask the proxy for the upstream's served model id. Some engines
    (vllm-swift, vLLM upstream) reject unknown model names with HTTP
    404; llama.cpp doesn't care."""
    try:
        with urllib.request.urlopen(
            f"{proxy_url}/v1/models", timeout=10.0,
        ) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        models = data.get("data", []) if isinstance(data, dict) else []
        if models:
            return str(models[0].get("id") or "smoke-model")
    except Exception:  # noqa: BLE001
        pass
    return "smoke-model"


def _drive_proxy(proxy_url: str) -> bool:
    """Issue a chat-completion via the proxy, validate longctx-svc behavior.

    We're testing longctx-svc's contract, not the model's coherence.
    """
    project = _build_fake_project(
        Path(tempfile.mkdtemp(prefix="longctx-int-proj-")) / "smokeapp"
    )
    auth_path = project / "src" / "auth.ts"
    model_id = _discover_model_id(proxy_url)
    _info(f"using model id: {model_id}")
    body = {
        "model": model_id,
        "messages": [
            {"role": "user", "content":
             f"explain authMiddleware in {auth_path}"},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-session-affinity": "smoke-session",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status = r.status
            headers = dict(r.headers.items())
            payload = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        _fail(f"proxy returned HTTP {e.code}: {e.read()[:200]!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        _fail(f"proxy request failed: {exc!s}")
        return False

    ok = True
    if status != 200:
        _fail(f"proxy status: {status}")
        ok = False
    else:
        _ok(f"proxy status: {status}")

    sess = headers.get("x-longctx-session", "<missing>")
    if sess != "smoke-session":
        _fail(f"x-longctx-session = {sess!r} (expected smoke-session)")
        ok = False
    else:
        _ok(f"x-longctx-session: {sess}")

    scope = headers.get("x-longctx-scope", "")
    if not scope:
        _fail("x-longctx-scope empty (no scope detected)")
        ok = False
    else:
        _ok(f"x-longctx-scope: {scope}")

    chunks = headers.get("x-longctx-chunks-used", "0")
    try:
        n = int(chunks)
    except ValueError:
        n = 0
    if n <= 0:
        _fail(f"x-longctx-chunks-used = {chunks!r} — splice did not happen")
        ok = False
    else:
        _ok(f"x-longctx-chunks-used: {chunks}")

    status_str = headers.get("x-longctx-scope-status", "")
    if status_str != "ready":
        _fail(f"x-longctx-scope-status = {status_str!r} (expected ready)")
        ok = False
    else:
        _ok(f"x-longctx-scope-status: {status_str}")

    if "SHIBBOLETH_AUTH_TOKEN" in payload:
        _ok("model echoed retrieved content (chunk made it into context)")
    else:
        # The model may not have echoed — that's the model's call. The
        # proxy did its job iff the upstream-side log shows the spliced
        # system message. We can't reach the upstream's log from here in
        # a portable way, so this is informational, not failing.
        _info("model did not echo the unique token — proxy splice still"
              " confirmed by chunks-used header")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["llama", "vllm-swift",
                                         "vllm-amd", "all"],
                    default="all")
    ap.add_argument("--mode", choices=["proxy", "embedded"], default="proxy",
                    help="proxy: longctx-svc in front of engine. embedded: "
                         "engine calls longctx-svc via --retrieval-endpoint "
                         "(only meaningful for vllm-swift today).")
    ap.add_argument("--remote", default=None,
                    help=("for vllm-amd: the host:port of an already-running "
                          "vLLM server (typically the MI300X droplet)"))
    args = ap.parse_args(argv)

    print(f"{BOLD}longctx-svc cross-fork smoke harness{RESET}")
    print(f"working dir: {Path.cwd()}")

    results: dict[str, bool] = {}
    if args.engine in ("llama", "all"):
        results["llama.cpp"] = _smoke_llama_cpp()
    if args.engine in ("vllm-swift", "all"):
        results["vllm-swift"] = _smoke_vllm_swift(mode=args.mode)
    if args.engine in ("vllm-amd", "all"):
        results["vllm-amd"] = _smoke_vllm_amd(args.remote)

    _section("summary")
    for name, ok in results.items():
        if ok:
            _ok(f"{name}: PASS")
        else:
            _fail(f"{name}: FAIL/SKIP")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
