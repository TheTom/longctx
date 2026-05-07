# Cross-fork integration smoke harness

End-to-end test that boots an inference engine, puts longctx-svc in
front of it as an OpenAI-compat proxy, sends a chat-completion request
that mentions a real file path, and verifies that retrieval happened —
all in one Python script.

## What it tests

For each engine it boots:

1. The engine itself comes up healthy on `/v1/models`
2. longctx-svc proxy on a free port, `--upstream <engine>`
3. A chat-completion goes through the proxy and returns 200
4. Response carries the longctx debug headers:
   - `x-longctx-session: smoke-session`
   - `x-longctx-scope: <project root>` (non-empty)
   - `x-longctx-chunks-used: > 0`
   - `x-longctx-scope-status: ready`
5. Optional: the model echoes a unique token from the spliced chunk
   (informational — model may not echo, doesn't fail the test)

## Run

```bash
cd services/longctx-svc

# all three (skips ones whose binaries / models aren't available)
python3 integration/harness.py --engine all

# single engine
python3 integration/harness.py --engine llama
python3 integration/harness.py --engine vllm-swift
python3 integration/harness.py --engine vllm-amd \
    --remote tom@<droplet-ip>:8000
```

## Expected setup per engine

### `llama` — TheTom/llama-cpp-turboquant

- Build: `/Users/tom/local_llms/llama.cpp/build/bin/llama-server`
- Model: any small GGUF in `/Users/tom/local_llms/models/`
  (qwen2.5-1.5b-instruct-q4_k_m.gguf or qwen2.5-1.5b-TQ4_1S.gguf)

### `vllm-swift` — TheTom alpha mlx-swift via vllm-swift

- CLI: `vllm-swift` on PATH (`brew install vllm-swift`)
- Model: `~/models/Qwen3-4B-4bit`
  (`vllm-swift download mlx-community/Qwen3-4B-4bit`)

### `vllm-amd` — TheTom/vllm `feature/turboquant-amd-noautotune` on MI300X

The local Mac can't run CUDA/ROCm, so this mode targets an already-
running vLLM on the droplet. Boot it manually first:

```bash
ssh -i ~/.ssh/do_amd_mi300x tom@<droplet-ip>
docker run --device=/dev/kfd --device=/dev/dri \
    -p 8000:8000 \
    rocm/vllm-dev:base \
    --model Qwen/Qwen2.5-7B-Instruct \
    --kv-cache-dtype turboquant_k8v4 \
    --port 8000
```

Then on the Mac:

```bash
python3 integration/harness.py --engine vllm-amd \
    --remote tom@<droplet-ip>:8000
```

## What "PASS" means

Proxy mode is engine-agnostic by construction: longctx-svc rewrites the
prompt before forwarding. Pass = the rewrite happened end-to-end against
a real engine. It does not certify that the engine answered the question
well — that's a separate eval.

## Troubleshooting

- `engine did not become healthy in N seconds`: model is loading.
  Re-run with a smaller model, or bump the boot timeout in
  `_smoke_<engine>` if your hardware is slow.
- `x-longctx-chunks-used: 0`: scope detect missed the path. Verify the
  file in the user message is an absolute path that exists on the box
  running longctx-svc, and that there is a sentinel (`package.json`,
  `.git`, `pyproject.toml`, …) at or above the file.
- `proxy status: 503`: longctx-svc started without `--upstream`.
  The harness sets it; if you're running longctx-svc by hand, add it.

## Logs

Each run prints the temp dir holding both `engine.log` and
`longctx-svc.log`. They're not auto-deleted — useful for postmortems.
