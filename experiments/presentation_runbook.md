# Presentation Runbook (Now -> Apr 22 midnight)

## One-command pipeline

Full pipeline (core train + held-out train + held-out eval + auto summary):
```bash
bash experiments/run_all.sh
```

Fast rerun of eval + summary only (reuse trained checkpoints):
```bash
RUN_CORE_TRAIN=0 RUN_HOLDOUT_TRAIN=0 RUN_HOLDOUT_EVAL=1 bash experiments/run_all.sh
```

## 0) One-time code status
- `router_condition` is now active in `model.py`.
- New train filters exist: `--include-tasks`, `--exclude-tasks`.

## 1) Core 3-model comparison (same split)

### A. No-MoE baseline
```bash
python3 train.py --config experiments/no_moe.yaml
```

### B. Language-conditioned MoE
```bash
python3 train.py --config experiments/moe_text.yaml
```

### C. Full-feature MoE (router sees full action input)
```bash
python3 train.py --config experiments/moe_full.yaml
```

## 2) Held-out task generalization (unseen task family)
Pick 2 held-out tasks. Example:
- `door-open-v3`
- `peg-insert-side-v3`

Train each model on seen tasks only:
```bash
python3 train.py --config experiments/no_moe.yaml --out-dir checkpoints/checkpoints_stage2_no_moe_holdout --exclude-tasks door-open-v3,peg-insert-side-v3
python3 train.py --config experiments/moe_text.yaml --out-dir checkpoints/checkpoints_stage2_moe_text_holdout --exclude-tasks door-open-v3,peg-insert-side-v3
python3 train.py --config experiments/moe_full.yaml --out-dir checkpoints/checkpoints_stage2_moe_full_holdout --exclude-tasks door-open-v3,peg-insert-side-v3
```

Evaluate in rollout on held-out tasks:
```bash
python3 eval_rollout.py --config experiments/no_moe.yaml --ckpt checkpoints/checkpoints_stage2_no_moe_holdout/best.pt --task door-open-v3 --episodes 20 --camera-sweep --record-video Output/presentation/no_moe_door.mp4
python3 eval_rollout.py --config experiments/moe_text.yaml --ckpt checkpoints/checkpoints_stage2_moe_text_holdout/best.pt --task door-open-v3 --episodes 20 --camera-sweep --record-video Output/presentation/moe_text_door.mp4
python3 eval_rollout.py --config experiments/moe_full.yaml --ckpt checkpoints/checkpoints_stage2_moe_full_holdout/best.pt --task door-open-v3 --episodes 20 --camera-sweep --record-video Output/presentation/moe_full_door.mp4
```

Repeat for `peg-insert-side-v3`.

## 3) Language robustness (same env task, paraphrased instruction)
Run one extra line per model with manual language override:
```bash
python3 eval_rollout.py --config experiments/moe_text.yaml --ckpt checkpoints/checkpoints_stage2_moe_text/best.pt --task door-open-v3 --instruction "Pull the door open by moving the gripper to the handle and dragging outward." --episodes 20 --camera corner2
```

## 4) Data mix (if time remains)
Fast option:
- Keep architecture fixed (`moe_text` only).
- Run another training with alternate root if prepared locally.

Example:
```bash
python3 train.py --config experiments/moe_text.yaml --data-root data/short-metaworld-v3 --out-dir checkpoints/checkpoints_stage2_moe_text_mixv3
```

## 5) Slide table template
For each model report:
- Validation loss (seen-task split)
- Held-out task success rate (door-open-v3, peg-insert-side-v3)
- Paraphrased-instruction success rate
- MoE router entropy / specialization gap (for MoE runs)
