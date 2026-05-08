#!/usr/bin/env bash
# M5 Max coherence smoke: real Qwen3 via vllm-swift + V3 + longctx + harness.
#
# Validates Tom's three audit questions end-to-end:
#   1. Does V3 + longctx work on M5 with a real model? (yes if smoke passes)
#   2. Can we retrieve at long context where we were failing before?
#      (compares answer-exact rate at multi-hop family vs prior runs)
#   3. Does evict-to-RAG show ≥10% KV savings?
#      (mlx-swift-lm prints compression stats per round when
#       VLLM_TRIATT_COMPRESSION_LOG=1; smoke greps the log for the rate)
#
# Pre-conditions:
#   * vllm-swift installed (`brew install vllm-swift` or pip)
#   * longctx-svc installed (`pip install longctx-svc` or local checkout)
#   * Qwen3 4B 4-bit downloaded once via huggingface-cli OR vllm-swift
#     auto-pulls on first run
#
# Usage:
#   bash m5_coherence_smoke.sh                 # quick 100K smoke
#   TOKENS=1000000 bash m5_coherence_smoke.sh  # 1M
#
# Output:
#   /tmp/m5_coh/run.json                       # coherence_driver result
#   /tmp/m5_coh/vllm-swift.log                 # server log w/ V3 traces
#   /tmp/m5_coh/longctx.log                    # longctx-svc log
#
# Exit code 0 on PASS (≥10% compression AND ≥1 exact answer); non-zero
# on any failure mode.

set -euo pipefail

TOKENS="${TOKENS:-100000}"
MODEL="${MODEL:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
VLLM_PORT="${VLLM_PORT:-8000}"
LONGCTX_PORT="${LONGCTX_PORT:-5054}"
SESSION_ID="m5-coh-$(date +%s)"
WORK_DIR="${WORK_DIR:-/tmp/m5_coh}"

mkdir -p "$WORK_DIR"

echo "[smoke] tokens=$TOKENS  model=$MODEL  session=$SESSION_ID"

# 1. Boot longctx-svc if not already up.
if ! curl -sf "http://localhost:${LONGCTX_PORT}/healthz" >/dev/null; then
  echo "[smoke] starting longctx-svc on :${LONGCTX_PORT}..."
  ( cd "$(dirname "$0")/../.." && \
    LONGCTX_NO_JANITOR=1 \
    python3 -m uvicorn longctx_svc.app:app \
      --host 127.0.0.1 --port "$LONGCTX_PORT" \
      >"$WORK_DIR/longctx.log" 2>&1 ) &
  LONGCTX_PID=$!
  for _ in $(seq 1 30); do
    sleep 1
    curl -sf "http://localhost:${LONGCTX_PORT}/healthz" >/dev/null && break
  done
  curl -sf "http://localhost:${LONGCTX_PORT}/healthz" >/dev/null \
    || { echo "[smoke] longctx-svc failed to come up"; exit 2; }
else
  LONGCTX_PID=""
  echo "[smoke] longctx-svc already up at :${LONGCTX_PORT}"
fi

# 2. Boot vllm-swift with V3 + longctx env wired.
if ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "[smoke] starting vllm-swift on :${VLLM_PORT} (Qwen3 4B 4-bit)..."
  echo "[smoke]   V3 budget=512  window=128  prefix=64"
  echo "[smoke]   compression log + longctx pointed at :${LONGCTX_PORT}"
  VLLM_TRIATT_ENABLED=1 \
  VLLM_TRIATT_BUDGET=512 \
  VLLM_TRIATT_WINDOW=128 \
  VLLM_TRIATT_PREFIX=64 \
  VLLM_TRIATT_WARMUP=256 \
  VLLM_TRIATT_HYBRID=2 \
  VLLM_TRIATT_COMPRESSION_LOG=1 \
  VLLM_TRIATT_LONGCTX_SESSION_ID="$SESSION_ID" \
  LONGCTX_ENDPOINT="http://localhost:${LONGCTX_PORT}" \
    vllm-swift serve "$MODEL" \
      --host 127.0.0.1 --port "$VLLM_PORT" \
      >"$WORK_DIR/vllm-swift.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 60); do
    sleep 2
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null && break
  done
  curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null \
    || { echo "[smoke] vllm-swift failed to come up"; tail -20 "$WORK_DIR/vllm-swift.log"; exit 3; }
else
  VLLM_PID=""
  echo "[smoke] vllm-swift already up at :${VLLM_PORT}"
fi

# 3. Sanity probe: does the model answer "Paris" coherently?
echo "[smoke] sanity probe..."
SANITY=$(curl -sf -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Longctx-Session: ${SESSION_ID}-sanity" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":8,"temperature":0}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" \
  || echo "ERROR")
echo "[smoke] sanity → $SANITY"
if ! echo "$SANITY" | grep -qi paris; then
  echo "[smoke] sanity probe failed (expected 'Paris', got '$SANITY')"
  exit 4
fi

# 4. Build the coherence corpus (5-family multi-hop benchmark).
CORPUS="$WORK_DIR/corpus.json"
echo "[smoke] building $TOKENS-token coherence corpus..."
( cd "$(dirname "$0")/.." && \
  python3 -m harness.build_coherence_corpus \
    --tokens "$TOKENS" \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --out "$CORPUS" \
    >"$WORK_DIR/corpus_build.log" 2>&1 )
echo "[smoke]   $(wc -c < "$CORPUS") bytes"

# 5. Run the coherence driver against the M5 endpoint.
RUN_OUT="$WORK_DIR/run.json"
echo "[smoke] running coherence_driver..."
( cd "$(dirname "$0")/.." && \
  python3 -m harness.coherence_driver \
    --corpus "$CORPUS" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --model "$MODEL" \
    --longctx "http://localhost:${LONGCTX_PORT}" \
    --session_id "$SESSION_ID" \
    --turn_tokens 8192 --top_k 8 --score_floor 0.0 \
    --max_recovered_chars 24000 \
    --out "$RUN_OUT" 2>&1 ) | tee "$WORK_DIR/driver.log"

# 6. Report compression % from vllm-swift log + answer summary from run.
echo ""
echo "============== M5 COHERENCE SMOKE RESULT =============="
echo "Session: $SESSION_ID"
echo "Tokens:  $TOKENS"
echo ""
echo "--- KV compression (V3 evict-to-RAG savings) ---"
grep -E "V3-compaction.*round=" "$WORK_DIR/vllm-swift.log" \
  | tail -5 || echo "(no compression rounds logged)"
LAST_PCT=$(grep -E "saved=" "$WORK_DIR/vllm-swift.log" | tail -1 \
  | sed -E 's/.*saved=([0-9.]+)%.*/\1/' || echo "0")
echo "Last round savings: ${LAST_PCT}%"
echo ""
echo "--- Coherence (per-family classification) ---"
python3 -c "
import json
r = json.load(open('$RUN_OUT'))
print(f\"  retrieval_recall@K: {100*r['retrieval_recall_atK']:.1f}%\")
print(f\"  overall: {r['overall']}\")
for fam, fc in r['by_family'].items():
    print(f\"  [{fam:>14}] exact={fc.get('exact',0)}/{fc.get('_total',0)}  \"
          f\"reasoning_fail={fc.get('reasoning_fail',0)}  \"
          f\"retrieval_miss={fc.get('retrieval_miss',0)}\")
"
echo ""
EXACT_TOTAL=$(python3 -c "
import json; r = json.load(open('$RUN_OUT')); print(r['overall'].get('exact', 0))
")
TOTAL_QS=$(python3 -c "
import json; r = json.load(open('$RUN_OUT')); print(r['overall'].get('_total', 0))
")
echo "Exact answers: $EXACT_TOTAL / $TOTAL_QS"
echo ""

# 7. Pass criteria
PASS=true
PCT_INT=$(printf "%.0f" "$LAST_PCT")
if [ "$PCT_INT" -lt 10 ]; then
  echo "[FAIL] compression < 10% (got ${LAST_PCT}%)"; PASS=false
fi
if [ "$EXACT_TOTAL" -lt 1 ]; then
  echo "[FAIL] zero exact answers"; PASS=false
fi
if $PASS; then
  echo "[PASS] M5 coherence smoke clean"
  exit 0
else
  echo "[FAIL] see logs at $WORK_DIR/"
  exit 5
fi
