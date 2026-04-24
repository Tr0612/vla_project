#!/usr/bin/env bash
set -euo pipefail

# Seen-task rollout evaluation for core checkpoints.
# This script mirrors holdout-eval reporting but runs on tasks that were included in training.

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-Output/presentation/seen_eval_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
VID_DIR="${RUN_DIR}/videos"
mkdir -p "$LOG_DIR" "$VID_DIR"

EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-250}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_CAMERA="${EVAL_CAMERA:-corner2}"
SEEN_TASKS_CSV="${SEEN_TASKS_CSV:-door-open-v3,peg-insert-side-v3,push-v3,reach-v3,sweep-v3,pick-place-v3,button-press-topdown-v3}"
ENABLE_SAFETY_CHECK="${ENABLE_SAFETY_CHECK:-0}"
SAFETY_THRESHOLD="${SAFETY_THRESHOLD:-0.20}"
SAFETY_OBJECT="${SAFETY_OBJECT:-}"

MODEL_TAGS=("no_moe" "moe_text" "moe_full" "act")
MODEL_CONFIGS=("experiments/no_moe_unfreeze.yaml" "experiments/moe_text_unfreeze.yaml" "experiments/moe_full_unfreeze.yaml" "experiments/act_unfreeze.yaml")
CORE_OUTS=("checkpoints/checkpoints_stage2_no_moe_unfreeze" "checkpoints/checkpoints_stage2_moe_text_unfreeze" "checkpoints/checkpoints_stage2_moe_full_unfreeze" "checkpoints/checkpoints_stage2_act_unfreeze")

ROLLOUT_CSV="${RUN_DIR}/summary_seen_rollouts.csv"
SUMMARY_MD="${RUN_DIR}/summary_seen.md"

echo "model_tag,setting,task,success_rate,best_camera,safety_aborts,avg_ep_max_success,avg_ep_max_reward,avg_ep_min_obj_to_target,video_path,log_path" > "$ROLLOUT_CSV"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
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

write_markdown_summary() {
  "$PYTHON_BIN" - "$ROLLOUT_CSV" "$SUMMARY_MD" <<'PY'
import csv
import math
import sys
from collections import defaultdict

rollout_csv, summary_md = sys.argv[1:3]
rows = list(csv.DictReader(open(rollout_csv, encoding="utf-8")))

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Seen-Task Rollout Summary\n\n")
    f.write("| Model | Task | Success Rate | Camera | Avg Max Reward | Avg Min Obj Dist |\n")
    f.write("|---|---|---:|---|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['model_tag']} | {r['task']} | {r['success_rate']} | {r['best_camera']} | "
            f"{r.get('avg_ep_max_reward', 'nan')} | {r.get('avg_ep_min_obj_to_target', 'nan')} |\n"
        )

    grouped = defaultdict(list)
    for r in rows:
        sr = to_float(r["success_rate"])
        if math.isfinite(sr):
            grouped[r["model_tag"]].append(sr)

    f.write("\n## Avg Seen-Task Success\n\n")
    f.write("| Model | Avg Success Rate |\n")
    f.write("|---|---:|\n")
    for model, vals in sorted(grouped.items()):
        avg = sum(vals) / max(len(vals), 1)
        f.write(f"| {model} | {avg:.4f} |\n")
PY
}

log "Run directory: ${RUN_DIR}"
log "Seen tasks: ${SEEN_TASKS_CSV}"
log "Eval: episodes=${EVAL_EPISODES} max_steps=${EVAL_MAX_STEPS} fps=${EVAL_FPS} camera=${EVAL_CAMERA}"

safety_args=()
if [[ "$ENABLE_SAFETY_CHECK" == "1" ]]; then
  safety_args+=(--enable-safety-check --safety-threshold "$SAFETY_THRESHOLD")
  if [[ -n "$SAFETY_OBJECT" ]]; then
    safety_args+=(--safety-object "$SAFETY_OBJECT")
  fi
fi

IFS=',' read -r -a seen_tasks <<< "$SEEN_TASKS_CSV"
for i in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$i]}"
  cfg="${MODEL_CONFIGS[$i]}"
  out="${CORE_OUTS[$i]}"
  ckpt="${out}/best.pt"

  if [[ ! -f "$ckpt" ]]; then
    log "Missing checkpoint for ${tag}: ${ckpt}. Skipping model."
    continue
  fi

  for task in "${seen_tasks[@]}"; do
    t_trimmed="$(echo "$task" | xargs)"
    log_file="${LOG_DIR}/eval_seen_${tag}_${t_trimmed}.log"
    vid_path="${VID_DIR}/${tag}_${t_trimmed}.mp4"

    log "Running seen eval: model=${tag} task=${t_trimmed}"
    "$PYTHON_BIN" eval_rollout.py \
      --config "$cfg" \
      --ckpt "$ckpt" \
      --task "$t_trimmed" \
      --episodes "$EVAL_EPISODES" \
      --max-steps "$EVAL_MAX_STEPS" \
      --fps "$EVAL_FPS" \
      --camera "$EVAL_CAMERA" \
      --record-video "$vid_path" \
      "${safety_args[@]}" \
      2>&1 | tee "$log_file"

    sr_cam="$(extract_eval_metrics "$log_file")"
    IFS=',' read -r sr cam sab ams amr amin <<< "$sr_cam"
    echo "${tag},seen_eval_core,${t_trimmed},${sr},${cam},${sab},${ams},${amr},${amin},${vid_path},${log_file}" >> "$ROLLOUT_CSV"
  done
done

write_markdown_summary
log "Done."
log "Seen rollout CSV: ${ROLLOUT_CSV}"
log "Seen summary markdown: ${SUMMARY_MD}"
