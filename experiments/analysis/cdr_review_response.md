# TinyVLA Final Experiment Report

## 1) Decision (Best Model to Pick)
**Recommended model: `no_moe` (core unfreeze variant)**

**Checkpoint to use:**
- `checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt`

**Why this is the best current choice:**
1. It has the highest **seen-task closed-loop success** in the latest evaluation set.
2. It is the most reliable behaviorally across tasks that currently solve at non-zero rates.
3. Although `moe_full` has the lowest validation loss, that did not translate into better rollout success.

## 2) Evidence Sources
- Core + holdout training summary: `Output/presentation/run_20260419_161721/summary_models.csv`
- Holdout rollout summary: `Output/presentation/run_20260419_161721/summary_rollouts.csv`
- Seen-task rollout summary: `Output/presentation/seen_eval_20260420_201211/summary_seen_rollouts.csv`
- Architecture manifest: `Output/presentation/run_20260419_161721/architecture_manifest.txt`
- Per-model architecture snapshots: `Output/presentation/run_20260419_161721/architectures/*.txt`

## 3) Architecture Snapshot (Latest Controlled Run)
All models share the same encoder/backbone strategy in this run:
- `freeze_vision: true`, `freeze_text: true`
- `unfreeze_vision_last_n_layers: 2`
- `unfreeze_text_last_n_layers: 2`

Head differences:
- `no_moe`: MLP action head
- `moe_text`: MoE head, router conditioned on `text`
- `moe_full`: MoE head, router conditioned on `action_input`
- `act`: ACT chunking head (`act_chunk_size: 8`)

## 4) Quantitative Results

### 4.1 Training / Validation (run_20260419_161721)
| Model | Core Val Loss | Holdout-Train Val Loss |
|---|---:|---:|
| no_moe | 0.004511 | 0.005605 |
| moe_text | 0.004545 | 0.005117 |
| **moe_full** | **0.004336** | **0.004488** |
| act | 0.005625 | 0.006512 |

Observation: `moe_full` is best on validation loss.

### 4.2 Held-out Rollouts (door-open-v3, peg-insert-side-v3)
All models are currently at **0.0000 success rate** on held-out tasks.

Dense metrics (average max reward over the two tasks):
- `moe_full`: **0.9077**
- `no_moe`: 0.7381
- `moe_text`: 0.7344
- `act`: 0.3270

Observation: held-out binary completion remains unresolved for all contenders.

### 4.3 Seen-task Rollouts (7 tasks)
Average success rate across seen tasks:
- **`no_moe`: 0.6857**
- `moe_full`: 0.5714
- `act`: 0.4286
- `moe_text`: 0.4000

Observation: `no_moe` has the strongest realized control performance today.

## 5) Final Recommendation Rationale
There are two possible selection criteria:

1. **Pick by supervised fit (`val_loss`)** -> `moe_full`
2. **Pick by actual closed-loop success** -> `no_moe`

For deployment/readout where task completion matters most, criterion (2) should dominate. Therefore:

## Final pick: `no_moe` (`checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt`)

## 6) Caveat and Next Priority
Primary unresolved issue: held-out success is still zero for all models on `door-open-v3` and `peg-insert-side-v3` in the latest run.

Immediate next experiment priority:
1. Increase held-out robustness (data/task coverage + longer adaptation).
2. Keep `no_moe` as serving baseline.
3. Continue `moe_full` as research branch (best loss, stronger dense holdout reward signal).
