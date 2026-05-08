#!/usr/bin/env bash
# M5 V3+longctx validation matrix runner.
#
# Sweeps (model × context × eviction%) cells against vllm-swift, captures
# correctness + compression % + latency. Output is one CSV row per cell
# so a python analyzer can derive per-family "default-on" recommendations.
#
# THE STRATEGIC GOAL (Tom, 2026-05-07 night):
#   "I want to be comfortable to always have triattention on as long as
#    longctx is installed and at which percentage by data driven decisions"
#
# So this is the data collector. Pair with `m5_recommend_default.py`
# (next file) which reads the CSV and emits per-family rules.
#
# Pre-conditions:
#   * vllm-swift installed
#   * longctx-svc reachable (5054 default; auto-started below if down)
#   * The models in MODELS array cached locally via huggingface-cli
#
# Usage:
#   bash m5_validation_matrix.sh                    # default ladder
#   MAX_TOKENS=16384 bash m5_validation_matrix.sh   # tighter ctx ceiling
#   MODELS=qwen3-4b bash m5_validation_matrix.sh    # one model
#
# Output: $WORK_DIR/results.csv

set -euo pipefail

WORK_DIR="${WORK_DIR:-/tmp/m5_matrix}"
mkdir -p "$WORK_DIR"
RESULT_CSV="$WORK_DIR/results.csv"

# Tom's Tier-1 model list — the V3-ported families. Other families need
# the per-attention captureV3PreRopeQuery hook before they can join.
DEFAULT_MODELS=(
  "mlx-community/Qwen3-0.6B-4bit"
  "mlx-community/Qwen3-4B-4bit"
  "mlx-community/Qwen3.5-2B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
)
MODELS_INPUT="${MODELS:-${DEFAULT_MODELS[*]}}"
read -r -a MODELS <<< "$MODELS_INPUT"

# Context ladder — start small (correctness gate), grow only if all
# previous rungs land coherent. Tom's mandate: correctness first, push
# context only after.
DEFAULT_CTX=(2048 4096 8192 16384 32768)
CTX_INPUT="${CTX_LADDER:-${DEFAULT_CTX[*]}}"
read -r -a CTX_LADDER <<< "$CTX_INPUT"
MAX_TOKENS="${MAX_TOKENS:-32768}"

# Eviction rate ladder. V3 budget = ctx * (1 - rate). 0.0 = no eviction
# (V3 disabled, baseline cell). Higher rates reduce live KV more.
DEFAULT_RATES=(0.0 0.10 0.20 0.30 0.40)
RATES_INPUT="${EVICTION_RATES:-${DEFAULT_RATES[*]}}"
read -r -a RATES <<< "$RATES_INPUT"

# Modes (stack toggles)
DEFAULT_MODES=(baseline v3_only v3_plus_longctx)
MODES_INPUT="${MODES:-${DEFAULT_MODES[*]}}"
read -r -a MODES <<< "$MODES_INPUT"

VLLM_PORT="${VLLM_PORT:-8000}"
LONGCTX_PORT="${LONGCTX_PORT:-5054}"

# CSV header
if [ ! -f "$RESULT_CSV" ]; then
  echo "timestamp,model,ctx,eviction_rate,mode,boot_s,sanity_ok,exact_n,total_n,compression_pct,session_total,wall_s,verdict" \
    > "$RESULT_CSV"
fi

echo "[matrix] models=${#MODELS[@]} ctx=${#CTX_LADDER[@]} rates=${#RATES[@]} modes=${#MODES[@]}"
TOTAL_CELLS=$(( ${#MODELS[@]} * ${#CTX_LADDER[@]} * ${#RATES[@]} * ${#MODES[@]} ))
echo "[matrix] total cells: $TOTAL_CELLS"

# Boot longctx-svc if needed
if ! curl -sf "http://localhost:${LONGCTX_PORT}/healthz" >/dev/null; then
  echo "[matrix] starting longctx-svc..."
  ( cd "$(dirname "$0")/../.." && \
    LONGCTX_NO_JANITOR=1 \
    python3 -m uvicorn longctx_svc.app:app \
      --host 127.0.0.1 --port "$LONGCTX_PORT" \
      >"$WORK_DIR/longctx.log" 2>&1 ) &
  for _ in $(seq 1 30); do
    sleep 1
    curl -sf "http://localhost:${LONGCTX_PORT}/healthz" >/dev/null && break
  done
fi

CELL_IDX=0
for model in "${MODELS[@]}"; do
  for ctx in "${CTX_LADDER[@]}"; do
    if [ "$ctx" -gt "$MAX_TOKENS" ]; then continue; fi
    for rate in "${RATES[@]}"; do
      for mode in "${MODES[@]}"; do
        CELL_IDX=$((CELL_IDX + 1))
        echo ""
        echo "[$CELL_IDX/$TOTAL_CELLS] model=$model ctx=$ctx rate=$rate mode=$mode"

        # Compute V3 budget = ctx * (1 - rate). Round to int.
        BUDGET=$(python3 -c "print(int($ctx * (1 - $rate)))")

        # Mode env mapping
        TRIATT_ENABLED=0
        LONGCTX_ENV=""
        if [ "$mode" = "v3_only" ] || [ "$mode" = "v3_plus_longctx" ]; then
          TRIATT_ENABLED=1
        fi
        if [ "$mode" = "v3_plus_longctx" ]; then
          LONGCTX_ENV="LONGCTX_ENDPOINT=http://localhost:${LONGCTX_PORT}"
        fi

        SESSION_ID="m5-mat-${model//\//_}-${ctx}-${rate}-${mode}-$(date +%s)"

        # Boot vllm-swift fresh per cell (model load is slow but
        # required for env vars to take effect)
        # Kill any prior vllm-swift on the port
        pkill -f "vllm-swift serve" 2>/dev/null || true
        sleep 2

        BOOT_T=$(date +%s)
        VLLM_TRIATT_ENABLED=$TRIATT_ENABLED \
        VLLM_TRIATT_BUDGET=$BUDGET \
        VLLM_TRIATT_WINDOW=128 \
        VLLM_TRIATT_PREFIX=64 \
        VLLM_TRIATT_WARMUP=256 \
        VLLM_TRIATT_HYBRID=2 \
        VLLM_TRIATT_COMPRESSION_LOG=1 \
        VLLM_TRIATT_LONGCTX_SESSION_ID="$SESSION_ID" \
        $LONGCTX_ENV \
        vllm-swift serve "$model" \
          --host 127.0.0.1 --port "$VLLM_PORT" \
          >"$WORK_DIR/cell_${CELL_IDX}.log" 2>&1 &
        VLLM_PID=$!

        for _ in $(seq 1 60); do
          sleep 2
          curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null && break
        done
        BOOT_S=$(( $(date +%s) - BOOT_T ))

        # Sanity probe
        SANITY=$(curl -sf -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -H "X-Longctx-Session: ${SESSION_ID}" \
          -d '{"model":"'"$model"'","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":4,"temperature":0}' \
          | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'].strip()[:20])" \
          2>/dev/null || echo "ERROR")
        SANITY_OK=0
        echo "$SANITY" | grep -qE "[Ff]our|^4" && SANITY_OK=1

        # Build corpus + run coherence_driver if sanity passed
        EXACT_N=0; TOTAL_N=0; COMP_PCT=0; SESS_TOTAL=0; WALL_S=0; VERDICT="skip_no_sanity"
        if [ "$SANITY_OK" = "1" ]; then
          CORPUS="$WORK_DIR/corpus_${CELL_IDX}.json"
          ( cd "$(dirname "$0")/.." && \
            python3 -m harness.build_coherence_corpus \
              --tokens "$ctx" \
              --tokenizer Qwen/Qwen2.5-7B-Instruct \
              --out "$CORPUS" >/dev/null 2>&1 )
          if [ -f "$CORPUS" ]; then
            T0=$(date +%s)
            RUN_OUT="$WORK_DIR/run_${CELL_IDX}.json"
            ( cd "$(dirname "$0")/.." && \
              timeout 300 python3 -m harness.coherence_driver \
                --corpus "$CORPUS" \
                --endpoint "http://localhost:${VLLM_PORT}/v1" \
                --model "$model" \
                --longctx "http://localhost:${LONGCTX_PORT}" \
                --session_id "$SESSION_ID" \
                --turn_tokens 8192 --top_k 8 --score_floor 0.0 \
                --max_recovered_chars 24000 \
                --out "$RUN_OUT" >>"$WORK_DIR/cell_${CELL_IDX}.log" 2>&1 ) || true
            WALL_S=$(( $(date +%s) - T0 ))
            if [ -f "$RUN_OUT" ]; then
              EXACT_N=$(python3 -c "import json; r=json.load(open('$RUN_OUT')); print(r['overall'].get('exact', 0))")
              TOTAL_N=$(python3 -c "import json; r=json.load(open('$RUN_OUT')); print(r['overall'].get('_total', 0))")
            fi
          fi

          # Compression % from vllm-swift log (last [V3-compaction] line)
          COMP_PCT=$(grep -E "V3-compaction.*saved=" "$WORK_DIR/cell_${CELL_IDX}.log" \
            | tail -1 | sed -E 's/.*saved=([0-9.]+)%.*/\1/' || echo "0")
          [ -z "$COMP_PCT" ] && COMP_PCT=0

          # session_total from longctx-svc /evict/dump
          SESS_TOTAL=$(curl -sf "http://localhost:${LONGCTX_PORT}/evict/dump?session_id=${SESSION_ID}" \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_total', 0))" \
            2>/dev/null || echo "0")

          VERDICT="ok"
        fi

        # Stop vllm-swift cleanly
        kill $VLLM_PID 2>/dev/null || true
        sleep 1

        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$model,$ctx,$rate,$mode,$BOOT_S,$SANITY_OK,$EXACT_N,$TOTAL_N,$COMP_PCT,$SESS_TOTAL,$WALL_S,$VERDICT" \
          >> "$RESULT_CSV"
        echo "  → exact=$EXACT_N/$TOTAL_N comp=${COMP_PCT}% sess_total=$SESS_TOTAL wall=${WALL_S}s ($VERDICT)"
      done
    done
  done
done

echo ""
echo "[matrix] complete. Results: $RESULT_CSV"
echo "[matrix] run analyzer: python3 m5_recommend_default.py --csv $RESULT_CSV"
