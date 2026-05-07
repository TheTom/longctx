#!/usr/bin/env bash
# Drives the 10M-PRD arms A..F end-to-end against a vLLM-served
# Qwen2.5-32B-Instruct + a co-located longctx-svc.
#
# Pre-conditions:
#   * vLLM up on $VLLM_ENDPOINT with --max-model-len ≥10485760
#   * longctx-svc up on $LONGCTX_ENDPOINT (with /evict/dump)
#   * Haystack JSON pre-built at $HAYSTACK_PATH
#   * V3 wired: VLLM_TRIATT_ENABLED=1, LONGCTX_ENDPOINT set in vLLM env
#   * Each arm's vLLM is restarted with the right env beforehand
#     (this script doesn't manage server lifecycle)
#
# Each arm produces:
#   results/arm-X/run.json       (driver output: per-q + summary + coverage)
#   results/arm-X/admission.txt  (admission ladder pre-flight)
#
# Arms in PRD project_10m_context_prd.md:
#   A: fp16 / native context — controls
#   B: TQ+ K8V4, no V3
#   C: K8V4 + V3, no rescue (LONGCTX_ENDPOINT unset on vLLM)
#   D: K8V4 + V3 + longctx     ← headline
#   E: K8V3 + V3, no rescue
#   F: K8V3 + V3 + longctx     ← stretch headline
#
# This script JUST drives the harness; the operator owns server config.
set -euo pipefail

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
LONGCTX_ENDPOINT="${LONGCTX_ENDPOINT:-http://localhost:5054}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
HAYSTACK_PATH="${HAYSTACK_PATH:-/tmp/haystack_10m.json}"
TURN_TOKENS="${TURN_TOKENS:-8192}"
ARM="${1:?usage: run_10m_arms.sh <A|B|C|D|E|F>}"
RESULTS_DIR="${RESULTS_DIR:-./results}/arm-${ARM}"
mkdir -p "$RESULTS_DIR"

SESSION_ID="prd10m-arm${ARM}-$(date +%s)"
echo "[run] arm=${ARM} session=${SESSION_ID} model=${MODEL}"
echo "[run] endpoint=${VLLM_ENDPOINT} longctx=${LONGCTX_ENDPOINT}"
echo "[run] haystack=${HAYSTACK_PATH} turn_tokens=${TURN_TOKENS}"

# 1) Admission ladder pre-flight: confirm the engine accepts prompts up
#    to the rung the arm needs. If a rung rejects, the run aborts.
echo "[run] admission ladder..."
python3 -m harness.admission_ladder \
  --endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --rungs "32000,256000,1000000,10000000" \
  --max_tokens 8 \
  | tee "${RESULTS_DIR}/admission.txt"

# 2) Streaming driver: full 10M chunked-turn ingestion + question phase.
echo "[run] streaming driver..."
python3 -m harness.streaming_driver \
  --haystack "${HAYSTACK_PATH}" \
  --endpoint "${VLLM_ENDPOINT}" \
  --model "${MODEL}" \
  --turn_tokens "${TURN_TOKENS}" \
  --mode streaming \
  --longctx "${LONGCTX_ENDPOINT}" \
  --session_id "${SESSION_ID}" \
  --out "${RESULTS_DIR}/run.json"

# 3) Coverage scoring is already inside run.json (driver pulls it).
#    Print a one-line summary for ops visibility.
python3 -c "
import json, sys
r = json.load(open('${RESULTS_DIR}/run.json'))
s = r['summary']
c = r.get('coverage') or {}
exact = s.get('exact', 0); nh = s.get('native_hit', 0)
n = r['n_facts']
print(f\"arm=${ARM}  exact={exact}/{n}  native_hit={nh}  \"
      f\"cw={s.get('coherent_wrong',0)} miss={s.get('miss',0)} \"
      f\"deg={s.get('degenerate',0)}  cov=\"
      f\"{c.get('coverage_pct','-')}%  ({c.get('n_chunks','-')} chunks)\")
"

echo "[run] arm ${ARM} complete — see ${RESULTS_DIR}/run.json"
