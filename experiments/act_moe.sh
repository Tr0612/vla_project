#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-Output/presentation/act_moe_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
VID_DIR="${RUN_DIR}/videos"
ARCH_DIR="${RUN_DIR}/architectures"
mkdir -p "$LOG_DIR" "$VID_DIR" "$ARCH_DIR"

CFG="${CFG:-experiments/act_moe_unfreeze.yaml}"
CORE_OUT="${CORE_OUT:-checkpoints/checkpoints_stage2_act_moe_unfreeze}"
HOLDOUT_OUT="${HOLDOUT_OUT:-checkpoints/checkpoints_stage2_act_moe_unfreeze_holdout}"
HOLDOUT_TASKS_CSV="${HOLDOUT_TASKS_CSV:-door-open-v3,peg-insert-side-v3}"

RUN_CORE_TRAIN="${RUN_CORE_TRAIN:-1}"
RUN_HOLDOUT_TRAIN="${RUN_HOLDOUT_TRAIN:-1}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-1}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_CAMERA="${EVAL_CAMERA:-corner2}"

METRICS_CSV="${RUN_DIR}/summary_models.csv"
ROLLOUT_CSV="${RUN_DIR}/summary_rollouts.csv"
SUMMARY_MD="${RUN_DIR}/summary.md"

echo "setting,config,out_dir,best_ckpt,val_loss,epoch,router_mean_gap_last_epoch,router_mean_entropy_last_epoch" > "$METRICS_CSV"
echo "setting,task,success_rate,best_camera,safety_aborts,avg_ep_max_success,avg_ep_max_reward,avg_ep_min_obj_to_target,video_path,log_path" > "$ROLLOUT_CSV"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_cmd() {
  local log_file="$1"
  shift
  log "Running: $*"
  "$@" 2>&1 | tee "$log_file"
}

write_architecture_note() {
  local out_dir="$1"
  local setting="$2"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" - "$out_dir" "$setting" "$CFG" <<'PY'
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
setting = sys.argv[2]
config_path = Path(sys.argv[3])
vals = {}
for raw in config_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or ":" not in line:
        continue
    k, v = line.split(":", 1)
    vals[k.strip()] = v.strip().split("#", 1)[0].strip()

lines = [
    f"model_tag: act_moe",
    f"setting: {setting}",
    f"config: {config_path}",
    f"action_head_type: {vals.get('action_head_type', 'unknown')}",
    f"router_condition: {vals.get('router_condition', 'n/a')}",
    f"moe_num_experts: {vals.get('moe_num_experts', 'n/a')}",
    f"act_chunk_size: {vals.get('act_chunk_size', 'n/a')}",
    f"act_moe_context_dim: {vals.get('act_moe_context_dim', 'n/a')}",
]
(out_dir / "architecture.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  cp "${out_dir}/architecture.txt" "${ARCH_DIR}/${setting}.txt"
}

extract_ckpt_metrics() {
  local ckpt_path="$1"
  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
import math
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
val = float(ckpt.get("val_loss", float("nan")))
ep = float(ckpt.get("epoch", float("nan")))
def fmt(x):
    return "nan" if not math.isfinite(x) else f"{x:.6f}"
print(f"{fmt(val)},{fmt(ep)}")
PY
}

extract_router_metrics() {
  local out_dir="$1"
  "$PYTHON_BIN" - "$out_dir" <<'PY'
import csv
import math
import os
import sys
from collections import defaultdict

out_dir = sys.argv[1]
weights_path = os.path.join(out_dir, "moe_router_weights.csv")
entropy_path = os.path.join(out_dir, "moe_router_entropy.csv")
if not (os.path.exists(weights_path) and os.path.exists(entropy_path)):
    print("nan,nan")
    raise SystemExit(0)

rows = []
last_epoch = None
with open(weights_path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        e = int(r["epoch"])
        rows.append(r)
        if last_epoch is None or e > last_epoch:
            last_epoch = e
if last_epoch is None:
    print("nan,nan")
    raise SystemExit(0)

by_task = defaultdict(list)
for r in rows:
    if int(r["epoch"]) == last_epoch:
        by_task[r["task_name"]].append(float(r["mean_router_weight"]))
gaps = [(max(w) - min(w)) for w in by_task.values() if len(w) > 1]
gap = float("nan") if not gaps else sum(gaps) / len(gaps)

ent_vals = []
with open(entropy_path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if int(r["epoch"]) == last_epoch:
            ent_vals.append(float(r["mean_router_entropy"]))
ent = float("nan") if not ent_vals else sum(ent_vals) / len(ent_vals)

def fmt(x):
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}"
print(f"{fmt(gap)},{fmt(ent)}")
PY
}

extract_eval_metrics() {
  local log_file="$1"
  "$PYTHON_BIN" - "$log_file" <<'PY'
import re
import sys
txt = open(sys.argv[1], encoding="utf-8").read()
m = re.findall(
    r"best_camera=([^\s]+)\s+task=([^\s]+)\s+success_rate=([0-9.]+)\s+"
    r"safety_aborts=([0-9]+)\s+avg_ep_max_success=([0-9.]+)\s+"
    r"avg_ep_max_reward=([0-9.]+)\s+avg_ep_min_obj_to_target=([0-9.]+)",
    txt,
)
if m:
    cam, _task, sr, sab, ams, amr, amin = m[-1]
    print(f"{sr},{cam},{sab},{ams},{amr},{amin}")
else:
    print("nan,na,nan,nan,nan,nan")
PY
}

append_metrics_row() {
  local setting="$1"
  local out_dir="$2"
  local ckpt_path="${out_dir}/best.pt"
  local val_and_epoch
  local router_metrics
  val_and_epoch="$(extract_ckpt_metrics "$ckpt_path")"
  router_metrics="$(extract_router_metrics "$out_dir")"
  echo "${setting},${CFG},${out_dir},${ckpt_path},${val_and_epoch},${router_metrics}" >> "$METRICS_CSV"
}

write_summary() {
  "$PYTHON_BIN" - "$METRICS_CSV" "$ROLLOUT_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
models = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
rolls = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    f.write("# ACT+MoE Run Summary\n\n")
    f.write("## Training\n\n")
    f.write("| Setting | Val Loss | Epoch | Router Gap | Router Entropy |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for r in models:
        f.write(
            f"| {r['setting']} | {r['val_loss']} | {r['epoch']} | "
            f"{r['router_mean_gap_last_epoch']} | {r['router_mean_entropy_last_epoch']} |\n"
        )
    f.write("\n## Holdout Rollouts\n\n")
    f.write("| Setting | Task | Success | Avg Max Reward | Avg Min Obj Dist |\n")
    f.write("|---|---|---:|---:|---:|\n")
    for r in rolls:
        f.write(
            f"| {r['setting']} | {r['task']} | {r['success_rate']} | "
            f"{r['avg_ep_max_reward']} | {r['avg_ep_min_obj_to_target']} |\n"
        )
PY
}

log "Run directory: ${RUN_DIR}"
log "Config: ${CFG}"
log "Holdout tasks: ${HOLDOUT_TASKS_CSV}"
log "Eval: episodes=${EVAL_EPISODES} max_steps=${EVAL_MAX_STEPS} fps=${EVAL_FPS} camera=${EVAL_CAMERA}"

if [[ "$RUN_CORE_TRAIN" == "1" ]]; then
  core_ckpt="${CORE_OUT}/best.pt"
  if [[ "$FORCE_RETRAIN" != "1" && "$SKIP_EXISTING_TRAIN" == "1" && -f "$core_ckpt" ]]; then
    log "Skipping core train (found ${core_ckpt})"
  else
    run_cmd "${LOG_DIR}/train_core.log" "$PYTHON_BIN" train.py --config "$CFG" --out-dir "$CORE_OUT"
  fi
  write_architecture_note "$CORE_OUT" "core"
  append_metrics_row "core" "$CORE_OUT"
fi

if [[ "$RUN_HOLDOUT_TRAIN" == "1" ]]; then
  hold_ckpt="${HOLDOUT_OUT}/best.pt"
  if [[ "$FORCE_RETRAIN" != "1" && "$SKIP_EXISTING_TRAIN" == "1" && -f "$hold_ckpt" ]]; then
    log "Skipping holdout train (found ${hold_ckpt})"
  else
    run_cmd "${LOG_DIR}/train_holdout.log" "$PYTHON_BIN" train.py --config "$CFG" --out-dir "$HOLDOUT_OUT" --exclude-tasks "$HOLDOUT_TASKS_CSV"
  fi
  write_architecture_note "$HOLDOUT_OUT" "holdout_train"
  append_metrics_row "holdout_train" "$HOLDOUT_OUT"
fi

if [[ "$RUN_HOLDOUT_EVAL" == "1" ]]; then
  IFS=',' read -r -a tasks <<< "$HOLDOUT_TASKS_CSV"
  for task in "${tasks[@]}"; do
    t="$(echo "$task" | xargs)"
    eval_log="${LOG_DIR}/eval_${t}.log"
    video_path="${VID_DIR}/act_moe_${t}.mp4"
    run_cmd "$eval_log" \
      "$PYTHON_BIN" eval_rollout.py \
      --config "$CFG" \
      --ckpt "${HOLDOUT_OUT}/best.pt" \
      --task "$t" \
      --episodes "$EVAL_EPISODES" \
      --max-steps "$EVAL_MAX_STEPS" \
      --fps "$EVAL_FPS" \
      --camera "$EVAL_CAMERA" \
      --record-video "$video_path"

    recorded_video_path="$video_path"
    suffixed_video_path="${video_path%.mp4}_${EVAL_CAMERA}.mp4"
    if [[ -f "$suffixed_video_path" ]]; then
      recorded_video_path="$suffixed_video_path"
    fi

    sr_cam="$(extract_eval_metrics "$eval_log")"
    IFS=',' read -r sr cam sab ams amr amin <<< "$sr_cam"
    echo "holdout_eval,${t},${sr},${cam},${sab},${ams},${amr},${amin},${recorded_video_path},${eval_log}" >> "$ROLLOUT_CSV"
  done
fi

write_summary
log "Done."
log "Summary: ${SUMMARY_MD}"
log "Model CSV: ${METRICS_CSV}"
log "Rollout CSV: ${ROLLOUT_CSV}"
