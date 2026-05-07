"""Prove the cross-model claim: longctx-svc plumbing is identical, but
answer quality varies by model family/size.

Plants a unique magic string inside auth.ts, asks the same question
across N models with the same longctx-svc front, scores each by:

  retrieval_ok : did the spliced body contain the magic string?
                 (this is a longctx-svc property — should be uniform)
  recall_ok    : does the model's reply mention the magic string or
                 the function it's in? (model property)
  cited_path   : does the reply cite the file path?
  cited_line   : does the reply mention a line number from the chunk?
  word_count   : reply length

Run:
    python3 integration/cross_model_bakeoff.py [--models all|local|mlx|gguf|droplet]

Outputs a markdown table to stdout + JSON to ./bakeoff_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

MAGIC_TOKEN = "SHIBBOLETH_42_VAULT_BUNNY_X92"
LLAMA_BIN = Path("/Users/tom/local_llms/llama.cpp/build/bin/llama-server")
LLAMA_MODELS_DIR = Path("/Users/tom/local_llms/models")
MLX_MODELS_DIR = Path.home() / "models"
VLLM_SWIFT_VENV_PY = Path.home() / ".vllm-swift" / "venv" / "bin" / "python3"


@dataclass
class ModelEntry:
    name: str
    family: str
    size_b: float
    engine: str          # "llama.cpp" | "vllm-swift"
    model_path: Path
    boot_seconds: float = 90.0
    extra_args: list[str] = field(default_factory=list)


@dataclass
class Result:
    name: str
    family: str
    size_b: float
    engine: str
    booted: bool = False
    chat_ok: bool = False
    retrieval_ok: bool = False
    recall_ok: bool = False
    cited_path: bool = False
    cited_line: bool = False
    answer_chars: int = 0
    answer_excerpt: str = ""
    error: str = ""


def _free_port(start: int = 19000) -> int:
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


def _wait_http(url: str, timeout: float) -> bool:
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


def _kill_pg(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        else:
            p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _build_project(root: Path) -> Path:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text('{"name":"bake"}\n')
    auth = src / "auth.ts"
    auth.write_text(
        "// constant secret — only revealed when retrieval works\n"
        f"export const SECRET = '{MAGIC_TOKEN}';\n"
        "\n"
        "export function authMiddleware() {\n"
        f"  // The secret token is {MAGIC_TOKEN}\n"
        "  // It is required by the validateJWT call below.\n"
        "  return validateJWT(SECRET);\n"
        "}\n"
    )
    (src / "billing.ts").write_text(
        "export function chargeUser() { return 0; }\n"
    )
    return auth


def _start_engine(m: ModelEntry, port: int, log: Path) -> subprocess.Popen:
    if m.engine == "llama.cpp":
        cmd = [
            str(LLAMA_BIN), "-m", str(m.model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", "8192", "--n-predict", "200",
            *m.extra_args,
        ]
    elif m.engine == "vllm-swift":
        if not VLLM_SWIFT_VENV_PY.is_file():
            raise RuntimeError("vllm-swift venv python missing")
        cmd = [
            str(VLLM_SWIFT_VENV_PY), "-m", "vllm_swift.cli", "serve",
            str(m.model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", "8192",
            *m.extra_args,
        ]
    else:
        raise RuntimeError(f"unknown engine {m.engine}")
    return subprocess.Popen(
        cmd,
        stdout=log.open("ab"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def _start_longctx_proxy(upstream: str, port: int, log: Path,
                          dump_dir: Path) -> subprocess.Popen:
    env = {
        **os.environ,
        "LONGCTX_NO_JANITOR": "1",
        "LONGCTX_DEBUG_DUMP": str(dump_dir),
    }
    cmd = [
        sys.executable, "-m", "longctx_svc.cli",
        "serve", "--host", "127.0.0.1", "--port", str(port),
        "--upstream", upstream,
    ]
    return subprocess.Popen(
        cmd, env=env,
        stdout=log.open("ab"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def _post_chat(proxy_url: str, model_id: str, prompt: str,
               timeout: float = 120.0) -> tuple[int, dict, dict]:
    body = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system",
             "content": "Be concise. Answer in 2-4 sentences."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=body,
        headers={
            "content-type": "application/json",
            "x-session-affinity": "bake-off",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        headers = dict(r.headers.items())
        data = json.loads(r.read())
    return status, headers, data


def _discover_model_id(proxy_url: str) -> str:
    try:
        with urllib.request.urlopen(
            f"{proxy_url}/v1/models", timeout=10.0,
        ) as r:
            payload = json.loads(r.read())
        return payload["data"][0]["id"]
    except Exception:
        return "default"


def _score(reply_text: str, auth_path: Path) -> dict:
    rt = reply_text or ""
    return {
        "recall_ok": MAGIC_TOKEN in rt,
        "cited_path": "auth.ts" in rt,
        "cited_line": bool(re.search(r"line\s*\d|:\d+", rt, re.I)),
        "answer_chars": len(rt),
    }


def run_one(m: ModelEntry, project_root: Path,
            auth_path: Path, log_dir: Path) -> Result:
    res = Result(name=m.name, family=m.family, size_b=m.size_b,
                 engine=m.engine)
    engine_port = _free_port(19100)
    proxy_port = _free_port(19200)
    engine_log = log_dir / f"{m.name}.engine.log"
    proxy_log = log_dir / f"{m.name}.proxy.log"
    dump_dir = log_dir / f"{m.name}.dumps"
    upstream = f"http://127.0.0.1:{engine_port}"
    proxy_url = f"http://127.0.0.1:{proxy_port}"

    eng_p = None
    px_p = None
    try:
        print(f"  {YELLOW}→{RESET} booting {m.engine}: {m.model_path.name} on :{engine_port}")
        eng_p = _start_engine(m, engine_port, engine_log)
        if not _wait_http(f"{upstream}/v1/models", timeout=m.boot_seconds):
            res.error = f"engine boot timeout (see {engine_log})"
            return res
        res.booted = True
        print(f"  {GREEN}✓{RESET} engine healthy")

        px_p = _start_longctx_proxy(upstream, proxy_port,
                                     proxy_log, dump_dir)
        if not _wait_http(f"{proxy_url}/healthz", timeout=30.0):
            res.error = "proxy boot timeout"
            return res

        model_id = _discover_model_id(proxy_url)
        prompt = (
            f"What is the secret token defined in {auth_path}? "
            f"Quote the value verbatim if you can find it."
        )
        try:
            status, hdrs, data = _post_chat(proxy_url, model_id, prompt)
        except Exception as exc:
            res.error = f"chat failed: {type(exc).__name__}: {exc}"
            return res
        res.chat_ok = (status == 200)
        if not res.chat_ok:
            res.error = f"chat status={status}"
            return res

        # Retrieval should have happened — confirm via header
        chunks = int(hdrs.get("x-longctx-chunks-used", "0") or "0")
        res.retrieval_ok = chunks > 0
        # Scrape the magic from the latest dump for sanity
        if dump_dir.is_dir():
            files = sorted(dump_dir.glob("*.json"))
            if files:
                spliced = files[-1].read_text()
                if MAGIC_TOKEN not in spliced:
                    # retrieval header said yes but the magic isn't in the
                    # spliced body — surface as a plumbing fail
                    res.retrieval_ok = False

        # Get the model's reply
        try:
            reply = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            reply = ""
        score = _score(reply, auth_path)
        res.recall_ok = score["recall_ok"]
        res.cited_path = score["cited_path"]
        res.cited_line = score["cited_line"]
        res.answer_chars = score["answer_chars"]
        res.answer_excerpt = reply.strip().replace("\n", " ")[:200]
        return res
    finally:
        if px_p is not None:
            _kill_pg(px_p)
        if eng_p is not None:
            _kill_pg(eng_p)


def discover_models(filter_kind: str) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    # llama.cpp GGUF models
    if filter_kind in ("all", "gguf", "local"):
        candidates = [
            ("Qwen2.5-1.5B-Q4", "Qwen", 1.5,
             LLAMA_MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
             []),
            ("Mistral-Small-24B-Q4", "Mistral", 24,
             LLAMA_MODELS_DIR / "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
             []),
            ("Gemma-4-E2B-Q4", "Gemma", 2,
             LLAMA_MODELS_DIR / "google_gemma-4-E2B-it-Q4_K_L.gguf",
             []),
        ]
        if LLAMA_BIN.is_file():
            for name, family, size, path, extra in candidates:
                if path.is_file():
                    out.append(ModelEntry(
                        name=name, family=family, size_b=size,
                        engine="llama.cpp", model_path=path,
                        extra_args=extra,
                        boot_seconds=300.0 if size > 20 else 90.0,
                    ))

    # vllm-swift MLX models
    if filter_kind in ("all", "mlx", "local"):
        candidates = [
            ("Qwen3-4B", "Qwen", 4, MLX_MODELS_DIR / "Qwen3-4B-4bit"),
            ("Gemma-3-4B", "Gemma", 4, MLX_MODELS_DIR / "gemma-3-4b-it-4bit"),
            ("Llama-3.2-1B", "Llama", 1,
             MLX_MODELS_DIR / "Llama-3.2-1B-Instruct-hf"),
        ]
        if VLLM_SWIFT_VENV_PY.is_file():
            for name, family, size, path in candidates:
                if path.is_dir():
                    out.append(ModelEntry(
                        name=name, family=family, size_b=size,
                        engine="vllm-swift", model_path=path,
                        boot_seconds=180.0,
                    ))
    return out


def render_table(results: list[Result]) -> str:
    headers = ["model", "family", "size", "engine", "retrieval",
               "recall", "cited path", "ans chars"]
    rows = []
    for r in results:
        rows.append([
            r.name,
            r.family,
            f"{r.size_b}B",
            r.engine,
            "✓" if r.retrieval_ok else "✗",
            "✓" if r.recall_ok else "✗",
            "✓" if r.cited_path else "✗",
            str(r.answer_chars),
        ])
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]

    def fmt(row):
        return "| " + " | ".join(
            cell.ljust(widths[i]) for i, cell in enumerate(row)
        ) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(headers), sep]
    for row in rows:
        out.append(fmt(row))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    choices=["all", "local", "mlx", "gguf"])
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of models (0 = no cap)")
    args = ap.parse_args(argv)

    models = discover_models(args.models)
    if args.limit > 0:
        models = models[:args.limit]
    if not models:
        print("no models found")
        return 1

    print(f"{BOLD}cross-model longctx bake-off{RESET}")
    print(f"  {len(models)} model(s):", ", ".join(m.name for m in models))

    log_dir = Path(tempfile.mkdtemp(prefix="longctx-bakeoff-"))
    print(f"  logs: {log_dir}")

    project_root = log_dir / "project"
    auth_path = _build_project(project_root)
    print(f"  project: {project_root} (magic token: {MAGIC_TOKEN})")

    results: list[Result] = []
    for m in models:
        print(f"\n{BOLD}{m.name}{RESET} ({m.family}, {m.size_b}B, {m.engine})")
        r = run_one(m, project_root, auth_path, log_dir)
        results.append(r)
        if r.error:
            print(f"  {RED}✗{RESET} {r.error}")
        elif r.recall_ok:
            print(f"  {GREEN}✓{RESET} recalled magic token; "
                  f"answer: {DIM}{r.answer_excerpt[:120]}{RESET}")
        else:
            print(f"  {YELLOW}~{RESET} retrieval={'ok' if r.retrieval_ok else 'no'} "
                  f"recall=miss; answer: {DIM}{r.answer_excerpt[:120]}{RESET}")

    print(f"\n{BOLD}== summary table =={RESET}\n")
    print(render_table(results))
    out = Path("integration") / "bakeoff_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2,
                               default=str))
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
