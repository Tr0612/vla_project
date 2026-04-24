# Failure Analysis Report

## Scope
This report summarizes why held-out evaluation currently reports `0.0000` success and what has been changed to improve diagnosis and recovery.

Primary run analyzed:
- `Output/presentation/run_20260418_061016`

Tasks analyzed:
- `door-open-v3`
- `peg-insert-side-v3`

Models analyzed:
- `no_moe`
- `moe_text`
- `moe_full`

## Observed Outcome
All model-task pairs on held-out evaluation reported zero binary success:
- `summary_rollouts.csv`: all `success_rate=0.0000`

## What Failed vs What Did Not Fail
### Not a safety-gate failure
- `safety_aborts=0` in evaluation logs.
- Safety logic did not suppress policy execution in these runs.

### Not a logging bug
- `info['success']` is present in MetaWorld env info keys.
- Evaluator reads `success` directly and consistently.

### Actual failure mode
- Policies produce nontrivial rewards but fail to cross strict task completion threshold.
- For `door-open-v3`, max rewards are often moderate/high, but binary success remains 0.
- For `peg-insert-side-v3`, `min_obj_to_target` is often far from insertion target (around `0.60` in many episodes), with occasional partial progress.

Interpretation:
- This is primarily a **generalization/control-precision failure on held-out tasks**, not an instrumentation bug.

## Contributing Factors
1. Hard held-out split (`--exclude-tasks`) on difficult manipulation tasks.
2. Frozen vision/text encoders limit adaptation capacity.
3. Short prior training horizon (previously 12 epochs).
4. Binary success alone is too coarse to reflect partial progress.

## Changes Implemented
### 1) Richer eval metrics (beyond binary success)
`eval_rollout.py` now reports:
- `avg_ep_max_success`
- `avg_ep_max_reward`
- `avg_ep_min_obj_to_target`
- `safety_aborts`

`run_all.sh` now parses and writes these to `summary_rollouts.csv`.

### 2) Higher max epochs + early stopping
Training now supports:
- `early_stopping_patience`
- `early_stopping_min_delta`

Config updates in:
- `experiments/no_moe.yaml`
- `experiments/moe_text.yaml`
- `experiments/moe_full.yaml`

Current defaults:
- `epochs: 80`
- `early_stopping_patience: 8`
- `early_stopping_min_delta: 0.0005`

This allows longer training when useful, while stopping automatically once validation plateaus.

## Immediate Recovery Plan
1. Re-run with updated configs + early stopping:
```bash
bash experiments/run_all.sh
```

2. Compare by both binary and dense metrics:
- `success_rate`
- `avg_ep_max_reward`
- `avg_ep_min_obj_to_target`

3. Run tokenization sensitivity deep-dive for peg insertion:
```bash
bash experiments/analysis/peg_tokenization_sweep.sh
```

4. Report both:
- strict success (headline)
- partial progress metrics (diagnostic evidence)

## Presentation Talking Point
"Held-out success is currently zero for the most difficult unseen tasks, but dense rollout diagnostics show non-random progress. We introduced richer evaluation metrics and early-stopped long-horizon training to separate optimization plateau from true generalization failure, and added a tokenization/coarsening ablation for peg insertion precision analysis."
