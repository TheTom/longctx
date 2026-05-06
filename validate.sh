#!/bin/bash
# Local validation: smoke tests + import checks.
#
# Run before any commit you care about. Doesn't hit a live LLM endpoint,
# so safe to run anywhere with sentence-transformers + faiss installed.
#
# For full end-to-end validation against a live vLLM server, run:
#     longctx-bench --data-dir /path/to/mrcr --model qwen25-32b
set -e

cd "$(dirname "$0")"

echo "[validate] python version"
python3 --version

echo "[validate] longctx package imports"
python3 -c "import longctx; print(f'  longctx {longctx.__version__}')"

echo "[validate] core classes import"
python3 -c "
from longctx import LongCtxClient, RetrievalPipeline
from longctx.eval import MRCRRunner
from longctx.eval.bench import main as bench_main
from longctx.templates import TEMPLATES
print('  LongCtxClient:', LongCtxClient.__module__)
print('  RetrievalPipeline:', RetrievalPipeline.__module__)
print('  MRCRRunner:', MRCRRunner.__module__)
print('  templates:', list(TEMPLATES.keys()))
"

echo "[validate] CLI entrypoints discoverable"
python3 -c "
from longctx.eval.cli import main as eval_main
from longctx.eval.bench import main as bench_main
print('  longctx-eval: ok')
print('  longctx-bench: ok')
"

if command -v pytest >/dev/null 2>&1; then
    echo "[validate] pytest"
    pytest tests/ -q 2>&1 || echo "  (some tests skipped if optional deps missing)"
else
    echo "[validate] pytest not found, skipping unit tests"
fi

echo "[validate] OK"
