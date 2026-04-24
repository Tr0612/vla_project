#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CKPT="${CKPT:-checkpoints/checkpoints_stage2_moe_text_holdout/best.pt}"
CONFIG="${CONFIG:-experiments/moe_text.yaml}"
TASK="${TASK:-peg-insert-side-v3}"
EPISODES="${EPISODES:-20}"
OUT_DIR="${OUT_DIR:-Output/presentation/tokenization_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

echo "mode,quant_step,quant_bins,success_rate,best_camera,log" > "$CSV"

run_case() {
  local mode="$1"
  local qstep="$2"
  local qbins="$3"
  local log_file="$OUT_DIR/${mode}.log"

  "$PYTHON_BIN" eval_rollout.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --task "$TASK" \
    --episodes "$EPISODES" \
    --camera-sweep \
    --action-quant-step "$qstep" \
    --action-quant-bins "$qbins" \
    > "$log_file" 2>&1

  local parsed
  parsed="$($PYTHON_BIN - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
m = re.findall(r"best_camera=([^\s]+)\s+task=([^\s]+)\s+success_rate=([0-9.]+)", text)
if not m:
    print("nan,na")
else:
    cam, _task, sr = m[-1]
    print(f"{sr},{cam}")
PY
)"
  local sr="${parsed%%,*}"
  local cam="${parsed##*,}"
  echo "$mode,$qstep,$qbins,$sr,$cam,$log_file" >> "$CSV"
  echo "[$mode] success_rate=$sr best_camera=$cam"
}

# Baseline continuous action output.
run_case "continuous" "0.0" "0"

# Step-size coarsening (tokenization proxy).
run_case "step_0.10" "0.10" "0"
run_case "step_0.05" "0.05" "0"
run_case "step_0.02" "0.02" "0"

# Fixed-bin coarsening over [-1,1].
run_case "bins_9" "0.0" "9"
run_case "bins_17" "0.0" "17"
run_case "bins_33" "0.0" "33"

echo "Saved: $CSV"
