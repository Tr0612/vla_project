#!/usr/bin/env bash
set -euo pipefail

# End-to-end presentation pipeline:
# 1) Train core 3-model comparison
# 2) Train held-out-task variants
# 3) Run rollout evals on held-out tasks
# 4) Write CSV + Markdown summaries

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-Output/presentation/run_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
ARCH_DIR="${RUN_DIR}/architectures"
mkdir -p "$LOG_DIR"
mkdir -p "$ARCH_DIR"

EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_CAMERA="${EVAL_CAMERA:-corner2}"
HOLDOUT_TASKS_CSV="${HOLDOUT_TASKS_CSV:-door-open-v3,peg-insert-side-v3}"
ENABLE_SAFETY_CHECK="${ENABLE_SAFETY_CHECK:-0}"
SAFETY_THRESHOLD="${SAFETY_THRESHOLD:-0.20}"
SAFETY_OBJECT="${SAFETY_OBJECT:-}"

RUN_CORE_TRAIN="${RUN_CORE_TRAIN:-1}"
RUN_HOLDOUT_TRAIN="${RUN_HOLDOUT_TRAIN:-1}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-1}"
SKIP_EXISTING_TRAIN="${SKIP_EXISTING_TRAIN:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

MODEL_TAGS=("no_moe" "moe_text" "moe_full" "act" "act_moe")
MODEL_CONFIGS=("experiments/no_moe_unfreeze.yaml" "experiments/moe_text_unfreeze.yaml" "experiments/moe_full_unfreeze.yaml" "experiments/act_unfreeze.yaml" "experiments/act_moe_unfreeze.yaml")
CORE_OUTS=("checkpoints/checkpoints_stage2_no_moe_unfreeze" "checkpoints/checkpoints_stage2_moe_text_unfreeze" "checkpoints/checkpoints_stage2_moe_full_unfreeze" "checkpoints/checkpoints_stage2_act_unfreeze" "checkpoints/checkpoints_stage2_act_moe_unfreeze")
HOLDOUT_OUTS=("checkpoints/checkpoints_stage2_no_moe_unfreeze_holdout" "checkpoints/checkpoints_stage2_moe_text_unfreeze_holdout" "checkpoints/checkpoints_stage2_moe_full_unfreeze_holdout" "checkpoints/checkpoints_stage2_act_unfreeze_holdout" "checkpoints/checkpoints_stage2_act_moe_unfreeze_holdout")

METRICS_CSV="${RUN_DIR}/summary_models.csv"
ROLLOUT_CSV="${RUN_DIR}/summary_rollouts.csv"
SUMMARY_MD="${RUN_DIR}/summary.md"

echo "model_tag,setting,config,out_dir,best_ckpt,val_loss,epoch,router_mean_gap_last_epoch,router_mean_entropy_last_epoch" > "$METRICS_CSV"
echo "model_tag,setting,task,success_rate,best_camera,safety_aborts,avg_ep_max_success,avg_ep_max_reward,avg_ep_min_obj_to_target,video_path,log_path" > "$ROLLOUT_CSV"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

write_architecture_note() {
  local out_dir="$1"
  local model_tag="$2"
  local setting="$3"
  local config_path="$4"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" - "$out_dir" "$model_tag" "$setting" "$config_path" <<'PY'
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
model_tag = sys.argv[2]
setting = sys.argv[3]
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
    f"setting: {setting}",
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

  local run_arch_file="${ARCH_DIR}/${setting}_${model_tag}.txt"
  cp "${out_dir}/architecture.txt" "$run_arch_file"
}

write_run_manifest() {
  local manifest="${RUN_DIR}/architecture_manifest.txt"
  : > "$manifest"
  {
    echo "run_dir: ${RUN_DIR}"
    echo "models:"
    for i in "${!MODEL_TAGS[@]}"; do
      echo "- tag=${MODEL_TAGS[$i]} config=${MODEL_CONFIGS[$i]} core_out=${CORE_OUTS[$i]} holdout_out=${HOLDOUT_OUTS[$i]}"
    done
  } >> "$manifest"
}

run_cmd() {
  local log_file="$1"
  shift
  log "Running: $*"
  "$@" 2>&1 | tee "$log_file"
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
if not isinstance(val, (int, float)):
    val = float("nan")
if not isinstance(ep, (int, float)):
    ep = float("nan")

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
for _, weights in by_task.items():
    if len(weights) > 1:
        gaps.append(max(weights) - min(weights))
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

append_model_metrics_row() {
  local model_tag="$1"
  local setting="$2"
  local config_path="$3"
  local out_dir="$4"
  local ckpt_path="${out_dir}/best.pt"
  local val_and_epoch router_metrics
  val_and_epoch="$(extract_ckpt_metrics "$ckpt_path")"
  router_metrics="$(extract_router_metrics "$out_dir")"
  echo "${model_tag},${setting},${config_path},${out_dir},${ckpt_path},${val_and_epoch},${router_metrics}" >> "$METRICS_CSV"
}

train_core_models() {
  for i in "${!MODEL_TAGS[@]}"; do
    local tag="${MODEL_TAGS[$i]}"
    local cfg="${MODEL_CONFIGS[$i]}"
    local out="${CORE_OUTS[$i]}"
    local log_file="${LOG_DIR}/train_core_${tag}.log"
    local best_ckpt="${out}/best.pt"
    if [[ "$FORCE_RETRAIN" != "1" && "$SKIP_EXISTING_TRAIN" == "1" && -f "$best_ckpt" ]]; then
      log "Skipping core train for ${tag} (found ${best_ckpt})"
    else
      run_cmd "$log_file" "$PYTHON_BIN" train.py --config "$cfg" --out-dir "$out"
    fi
    write_architecture_note "$out" "$tag" "core" "$cfg"
    append_model_metrics_row "$tag" "core" "$cfg" "$out"
  done
}

train_holdout_models() {
  for i in "${!MODEL_TAGS[@]}"; do
    local tag="${MODEL_TAGS[$i]}"
    local cfg="${MODEL_CONFIGS[$i]}"
    local out="${HOLDOUT_OUTS[$i]}"
    local log_file="${LOG_DIR}/train_holdout_${tag}.log"
    local best_ckpt="${out}/best.pt"
    if [[ "$FORCE_RETRAIN" != "1" && "$SKIP_EXISTING_TRAIN" == "1" && -f "$best_ckpt" ]]; then
      log "Skipping holdout train for ${tag} (found ${best_ckpt})"
    else
      run_cmd "$log_file" "$PYTHON_BIN" train.py --config "$cfg" --out-dir "$out" --exclude-tasks "$HOLDOUT_TASKS_CSV"
    fi
    write_architecture_note "$out" "$tag" "holdout_train" "$cfg"
    append_model_metrics_row "$tag" "holdout_train" "$cfg" "$out"
  done
}

eval_holdout_models() {
  IFS=',' read -r -a holdout_tasks <<< "$HOLDOUT_TASKS_CSV"
  mkdir -p "${RUN_DIR}/videos"

  local safety_args=()
  local camera_args=()
  if [[ "$ENABLE_SAFETY_CHECK" == "1" ]]; then
    safety_args+=(--enable-safety-check --safety-threshold "$SAFETY_THRESHOLD")
    if [[ -n "$SAFETY_OBJECT" ]]; then
      safety_args+=(--safety-object "$SAFETY_OBJECT")
    fi
  fi
  # Camera sweep is intentionally disabled for faster/shorter presentation runs.
  camera_args+=(--camera "$EVAL_CAMERA")

  for i in "${!MODEL_TAGS[@]}"; do
    local tag="${MODEL_TAGS[$i]}"
    local cfg="${MODEL_CONFIGS[$i]}"
    local out="${HOLDOUT_OUTS[$i]}"
    local ckpt="${out}/best.pt"
    for task in "${holdout_tasks[@]}"; do
      local t_trimmed
      t_trimmed="$(echo "$task" | xargs)"
      local log_file="${LOG_DIR}/eval_holdout_${tag}_${t_trimmed}.log"
      local vid_path="${RUN_DIR}/videos/${tag}_${t_trimmed}.mp4"
      run_cmd "$log_file" \
        "$PYTHON_BIN" eval_rollout.py \
        --config "$cfg" \
        --ckpt "$ckpt" \
        --task "$t_trimmed" \
        --episodes "$EVAL_EPISODES" \
        --max-steps "$EVAL_MAX_STEPS" \
        --fps "$EVAL_FPS" \
        "${camera_args[@]}" \
        --record-video "$vid_path" \
        "${safety_args[@]}"
      local sr_cam
      sr_cam="$(extract_eval_metrics "$log_file")"
      IFS=',' read -r sr cam sab ams amr amin <<< "$sr_cam"
      echo "${tag},holdout_eval,${t_trimmed},${sr},${cam},${sab},${ams},${amr},${amin},${vid_path},${log_file}" >> "$ROLLOUT_CSV"
    done
  done
}

write_markdown_summary() {
  "$PYTHON_BIN" - "$METRICS_CSV" "$ROLLOUT_CSV" "$SUMMARY_MD" <<'PY'
import csv
import math
import sys
from collections import defaultdict

metrics_csv, rollout_csv, summary_md = sys.argv[1:4]

metrics_rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
rollout_rows = list(csv.DictReader(open(rollout_csv, encoding="utf-8")))

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Presentation Summary\n\n")

    f.write("## Model Metrics\n\n")
    f.write("| Model | Setting | Val Loss | Epoch | Router Gap | Router Entropy |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in metrics_rows:
        f.write(
            f"| {r['model_tag']} | {r['setting']} | {r['val_loss']} | {r['epoch']} | "
            f"{r['router_mean_gap_last_epoch']} | {r['router_mean_entropy_last_epoch']} |\n"
        )

    f.write("\n## Held-out Rollout Results\n\n")
    f.write("| Model | Task | Success Rate | Best Camera | Avg Max Reward | Avg Min Obj Dist |\n")
    f.write("|---|---|---:|---|---:|---:|\n")
    for r in rollout_rows:
        f.write(
            f"| {r['model_tag']} | {r['task']} | {r['success_rate']} | {r['best_camera']} | "
            f"{r.get('avg_ep_max_reward', 'nan')} | {r.get('avg_ep_min_obj_to_target', 'nan')} |\n"
        )

    grouped = defaultdict(list)
    for r in rollout_rows:
        sr = to_float(r["success_rate"])
        if math.isfinite(sr):
            grouped[r["model_tag"]].append(sr)

    f.write("\n## Avg Held-out Success\n\n")
    f.write("| Model | Avg Success Rate |\n")
    f.write("|---|---:|\n")
    for model, vals in sorted(grouped.items()):
        avg = sum(vals) / max(len(vals), 1)
        f.write(f"| {model} | {avg:.4f} |\n")

    f.write("\n## Artifacts\n\n")
    f.write(f"- Model CSV: `{metrics_csv}`\n")
    f.write(f"- Rollout CSV: `{rollout_csv}`\n")
    f.write("- Videos are under `videos/` in this run directory.\n")
PY
}

log "Run directory: ${RUN_DIR}"
log "Architecture notes: ${ARCH_DIR}"
log "Holdout tasks: ${HOLDOUT_TASKS_CSV}"
log "Eval: episodes=${EVAL_EPISODES} max_steps=${EVAL_MAX_STEPS} fps=${EVAL_FPS} camera_sweep=0 camera=${EVAL_CAMERA}"
log "Skip existing train: ${SKIP_EXISTING_TRAIN} (force retrain: ${FORCE_RETRAIN})"
write_run_manifest
if [[ "$ENABLE_SAFETY_CHECK" == "1" ]]; then
  log "Safety check: enabled (threshold=${SAFETY_THRESHOLD}, object=${SAFETY_OBJECT:-auto})"
else
  log "Safety check: disabled"
fi

if [[ "$RUN_CORE_TRAIN" == "1" ]]; then
  log "Phase 1/4: Core training"
  train_core_models
else
  log "Skipping core training"
fi

if [[ "$RUN_HOLDOUT_TRAIN" == "1" ]]; then
  log "Phase 2/4: Holdout training"
  train_holdout_models
else
  log "Skipping holdout training"
fi

if [[ "$RUN_HOLDOUT_EVAL" == "1" ]]; then
  log "Phase 3/4: Holdout rollout eval"
  eval_holdout_models
else
  log "Skipping holdout rollout eval"
fi

log "Phase 4/4: Writing summary"
write_markdown_summary

log "Done."
log "Summary markdown: ${SUMMARY_MD}"
log "Model CSV: ${METRICS_CSV}"
log "Rollout CSV: ${ROLLOUT_CSV}"
