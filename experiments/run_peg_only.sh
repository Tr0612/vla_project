#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-Output/presentation/peg_only_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
VID_DIR="${RUN_DIR}/videos"
ARCH_DIR="${RUN_DIR}/architectures"
mkdir -p "$LOG_DIR" "$VID_DIR" "$ARCH_DIR"

PEG_TASK="${PEG_TASK:-peg-insert-side-v3}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_CAMERA="${EVAL_CAMERA:-corner2}"

SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

MODEL_TAGS=("no_moe" "moe_text" "moe_full" "act" "act_moe")
MODEL_CONFIGS=("experiments/no_moe_unfreeze.yaml" "experiments/moe_text_unfreeze.yaml" "experiments/moe_full_unfreeze.yaml" "experiments/act_unfreeze.yaml" "experiments/act_moe_unfreeze.yaml")

METRICS_CSV="${RUN_DIR}/summary_models.csv"
ROLLOUT_CSV="${RUN_DIR}/summary_rollouts.csv"
SUMMARY_MD="${RUN_DIR}/summary.md"

echo "model_tag,task,config,out_dir,best_ckpt,val_loss,epoch,router_mean_gap_last_epoch,router_mean_entropy_last_epoch" > "$METRICS_CSV"
echo "model_tag,task,success_rate,best_camera,safety_aborts,avg_ep_max_success,avg_ep_max_reward,avg_ep_min_obj_to_target,video_path,log_path" > "$ROLLOUT_CSV"

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
  local model_tag="$2"
  local config_path="$3"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" - "$out_dir" "$model_tag" "$PEG_TASK" "$config_path" <<'PY'
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
model_tag = sys.argv[2]
task = sys.argv[3]
config_path = Path(sys.argv[4])

vals = {}
for raw in config_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or ":" not in line:
        continue
    k, v = line.split(":", 1)
    vals[k.strip()] = v.strip().split("#", 1)[0].strip()

lines = [
    f"model_tag: {model_tag}",
    f"task: {task}",
    f"config: {config_path}",
    f"action_head_type: {vals.get('action_head_type', 'unknown')}",
    f"router_condition: {vals.get('router_condition', 'n/a')}",
    f"freeze_vision: {vals.get('freeze_vision', 'unknown')}",
    f"freeze_text: {vals.get('freeze_text', 'unknown')}",
    f"unfreeze_vision_last_n_layers: {vals.get('unfreeze_vision_last_n_layers', '0')}",
    f"unfreeze_text_last_n_layers: {vals.get('unfreeze_text_last_n_layers', '0')}",
    f"act_chunk_size: {vals.get('act_chunk_size', 'n/a')}",
]
(out_dir / "architecture.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  cp "${out_dir}/architecture.txt" "${ARCH_DIR}/${model_tag}.txt"
}

extract_ckpt_metrics() {
  local ckpt_path="$1"
  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
import math
import sys
try:
    import torch
except Exception:
    print("nan,nan")
    raise SystemExit(0)

ckpt_path = sys.argv[1]
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
except Exception:
    print("nan,nan")
    raise SystemExit(0)

val = ckpt.get("val_loss", float("nan"))
ep = ckpt.get("epoch", float("nan"))

def fmt(x):
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}"

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

last_epoch = None
rows = []
with open(weights_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
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
gaps = []
for w in by_task.values():
    if len(w) > 1:
        gaps.append(max(w) - min(w))
gap = float("nan") if not gaps else sum(gaps) / len(gaps)

ent_vals = []
with open(entropy_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
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

write_summary() {
  "$PYTHON_BIN" - "$METRICS_CSV" "$ROLLOUT_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
model_csv, rollout_csv, summary_md = sys.argv[1:4]
models = list(csv.DictReader(open(model_csv, encoding="utf-8")))
rollouts = list(csv.DictReader(open(rollout_csv, encoding="utf-8")))
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Peg-Only Experiment Summary\n\n")
    f.write("## Training Metrics\n\n")
    f.write("| Model | Val Loss | Epoch | Router Gap | Router Entropy |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for r in models:
        f.write(
            f"| {r['model_tag']} | {r['val_loss']} | {r['epoch']} | "
            f"{r['router_mean_gap_last_epoch']} | {r['router_mean_entropy_last_epoch']} |\n"
        )
    f.write("\n## Rollout Metrics (peg-insert-side-v3)\n\n")
    f.write("| Model | Success | Camera | Avg Max Reward | Avg Min Obj Dist |\n")
    f.write("|---|---:|---|---:|---:|\n")
    for r in rollouts:
        f.write(
            f"| {r['model_tag']} | {r['success_rate']} | {r['best_camera']} | "
            f"{r['avg_ep_max_reward']} | {r['avg_ep_min_obj_to_target']} |\n"
        )
    f.write("\n## Artifacts\n\n")
    f.write(f"- Models CSV: `{model_csv}`\n")
    f.write(f"- Rollouts CSV: `{rollout_csv}`\n")
    f.write("- Logs: `logs/`\n")
    f.write("- Videos: `videos/`\n")
    f.write("- Architecture notes: `architectures/`\n")
PY
}

log "Run directory: ${RUN_DIR}"
log "Task: ${PEG_TASK}"
log "Eval: episodes=${EVAL_EPISODES} max_steps=${EVAL_MAX_STEPS} fps=${EVAL_FPS} camera=${EVAL_CAMERA}"
log "Skip existing train: ${SKIP_EXISTING_TRAIN} (force retrain: ${FORCE_RETRAIN})"

for i in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$i]}"
  cfg="${MODEL_CONFIGS[$i]}"
  out_dir="checkpoints/checkpoints_stage2_${tag}_peg_only"
  best_ckpt="${out_dir}/best.pt"
  train_log="${LOG_DIR}/train_${tag}.log"
  eval_log="${LOG_DIR}/eval_${tag}.log"
  video_path="${VID_DIR}/${tag}_${PEG_TASK}.mp4"

  if [[ "$FORCE_RETRAIN" != "1" && "$SKIP_EXISTING_TRAIN" == "1" && -f "$best_ckpt" ]]; then
    log "Skipping train for ${tag} (found ${best_ckpt})"
  else
    run_cmd "$train_log" "$PYTHON_BIN" train.py --config "$cfg" --out-dir "$out_dir" --task "$PEG_TASK"
  fi

  write_architecture_note "$out_dir" "$tag" "$cfg"

  val_and_epoch="$(extract_ckpt_metrics "$best_ckpt")"
  router_metrics="$(extract_router_metrics "$out_dir")"
  echo "${tag},${PEG_TASK},${cfg},${out_dir},${best_ckpt},${val_and_epoch},${router_metrics}" >> "$METRICS_CSV"

  run_cmd "$eval_log" \
    "$PYTHON_BIN" eval_rollout.py \
    --config "$cfg" \
    --ckpt "$best_ckpt" \
    --task "$PEG_TASK" \
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
  echo "${tag},${PEG_TASK},${sr},${cam},${sab},${ams},${amr},${amin},${recorded_video_path},${eval_log}" >> "$ROLLOUT_CSV"
done

write_summary
log "Done."
log "Summary: ${SUMMARY_MD}"
log "Models CSV: ${METRICS_CSV}"
log "Rollouts CSV: ${ROLLOUT_CSV}"
