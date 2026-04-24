# TinyVLA Architecture Deep Dive

This document is the full architecture reference across all implemented variants in this repo.
It is intended to be the single place to understand model design, input/output definitions, freezing policy, and head variants (`no_moe`, `moe_text`, `moe_full`, `act`, `act_moe`).

## 1) Where the architecture docs are now

Run-local lightweight notes:
- `Output/presentation/run_20260419_161721/architectures/*.txt`
- `Output/presentation/peg_only_20260422_040829/architectures/*.txt`
- `Output/presentation/act_moe_20260422_192347/architectures/*.txt`

Comprehensive reference (this file):
- `experiments/analysis/architecture_deep_dive.md`

Core implementation sources:
- `config.py`
- `model.py`
- `dataset.py`
- `train.py`
- `eval_rollout.py`
- `infer.py`

## 2) Global pipeline (runtime vs model-forward)

There are two different flows:

### 2.1 Runtime execution flow (with safety gate)
This is what happens in `infer.py` / `eval_rollout.py` when safety is enabled.

1. Input image + instruction arrive.
2. Safety layer checks object presence from vision-text similarity.
3. If `presence_score < threshold`: `SAFETY_ABORT` and stop before action execution.
4. Else: call policy model forward and predict action.
5. Execute predicted action (or first chunk action for ACT-family).

Important:
- Safety is a pre-action runtime gate.
- It is outside `VLAFusionPolicy.forward` and runs before policy action execution.

### 2.2 Core model forward flow (`VLAFusionPolicy`)
This is the neural policy computation itself.

1. Image + text (+ optional geometry history) are encoded.
2. Vision/text embeddings are projected to shared `proj_dim`.
3. Features are fused (default experiments use `cross_attn`).
4. Geometry features are concatenated to fused feature (if enabled).
5. Action head maps to action output:
   - single-step action `[B, A]` for `mlp` and `moe`
   - chunked action `[B, K, A]` for `act` and `act_moe`
6. Final learnable per-dimension scale is applied:
   - `output = base_action * exp(action_log_scale)`

Key shapes:
- `B`: batch size
- `A`: action dimension (`action_dim`, default 4)
- `K`: ACT chunk size (`act_chunk_size`, default 8)
- `D_proj`: `proj_dim` (default 512)
- `D_geom`: `geometry_dim * temporal_context` (default `6 * 4 = 24` in experiment YAMLs)
- `D_action_in = D_fused + D_geom` when geometry is enabled

## 3) Inputs and data definition

### 3.1 Per-sample fields
Canonical sample keys used by training:
- `image` or `obs.image`
- `instruction` or `task.prompt`
- `action`: single action vector
- `action_chunk`: future action chunk for ACT-style supervision
- `geometry` or `obs.state`: geometry/history feature vector
- `task_name` / `meta.dataset` (for logging)

### 3.2 Collated model inputs
`train.py` builds:
- `pixel_values`: processed image tensor
- `input_ids`, `attention_mask`: tokenized instruction
- `geometry_features`: stacked geometry vectors
- `actions`: `[B, A]`
- `action_chunks`: `[B, K, A]`

### 3.3 Geometry feature definition
`dataset.py` computes a 6D geometry primitive from state:
- normalized `ee_to_obj` (3D)
- normalized `obj_to_goal` (3D)

With temporal context, geometry is concatenated over time:
- final geometry size is `geometry_dim * temporal_context`.

## 4) Backbones and freezing strategy

### 4.1 Backbone choices
From experiment YAMLs used in your runs:
- `vision_model_name: google/siglip2-base-patch16-224`
- `text_model_name: google/siglip2-base-patch16-224`
- `separate_backbones: false` (shared model object, extracted vision/text submodules)

### 4.2 Freeze / unfreeze policy
Supported config knobs:
- `freeze_vision`
- `freeze_text`
- `unfreeze_vision_last_n_layers`
- `unfreeze_text_last_n_layers`

In your main unfreeze runs:
- `freeze_vision: true`
- `freeze_text: true`
- `unfreeze_vision_last_n_layers: 2`
- `unfreeze_text_last_n_layers: 2`

Meaning:
- Start with frozen encoders.
- Re-enable gradients only for final `N` transformer blocks in each encoder.

## 5) Fusion module

Supported `fusion_type`:
- `concat`
- `cross_attn`
- `transformer`

Your experiment YAMLs use:
- `fusion_type: cross_attn`

Cross-attention fusion behavior:
- text feature queries vision feature token(s)
- residual + FFN blocks
- fused output dimension is `proj_dim` (default 512)

## 6) Action head variants (all implemented)

`action_head_type` supports:
- `linear`
- `mlp`
- `moe`
- `act`
- `act_moe`

### 6.1 `mlp` (no_moe baseline)
Definition:
- feed-forward MLP over `action_input` (`fused [+ geometry]`)
- outputs single action `[B, A]`

Used by:
- `experiments/no_moe.yaml`
- `experiments/no_moe_unfreeze.yaml`

### 6.2 `moe` (Mixture-of-Experts action head)
Definition:
- multiple MLP experts map `action_input -> action`
- router computes softmax weights over experts
- weighted sum of expert outputs is final action `[B, A]`

Router conditioning options (`router_condition`):
- `text`: router input is projected text feature
- `action_input`: router input is fused+geometry action input
- `text_geometry`: router input is text concatenated with geometry

Variants:
- `moe_text`: `router_condition: text`
- `moe_full`: `router_condition: action_input`

Auxiliary MoE objective:
- load-balancing term from mean routing probabilities and hard assignments
- weighted in training by `moe_load_balance_weight`

Router logging:
- `moe_router_weights.csv`
- `moe_router_entropy.csv`

### 6.3 `act` (chunked action decoder)
Definition:
- ACT-style head predicts action chunk `[B, K, A]`
- context projection + learnable query embeddings + decoder MLP

Training target:
- uses `action_chunks` `[B, K, A]`

Inference behavior in current code:
- rollout/infer executes first predicted action only (`pred[:, 0, :]`) at each step.

### 6.4 `act_moe` (ACT + MoE, newly integrated)
Implemented design in `model.py`:
- `z = TextMoEContext(text_features)` where `z` is MoE-produced latent context
- `actions = ACTActionHead(concat(obs_features, z))`

More explicitly:
1. Text-only MoE context generator:
   - experts map text feature -> context vector
   - router maps text feature -> expert weights
   - weighted sum gives `z` (size `act_moe_context_dim`)
2. ACT decoder input:
   - `[action_input, z]` where `action_input = fused [+ geometry]`
3. ACT outputs chunked actions `[B, K, A]`

This exactly matches requested pattern:
- `z = MoE(text_features)`
- `actions = ACT_decoder(obs, z)`

## 7) Training objectives and normalization

Primary action loss:
- configurable via `loss_type`:
  - `mse`
  - `huber` (used in experiment YAMLs)

Target normalization:
- if `normalize_action_targets: true`, targets normalized by dataset stats
- stats saved in checkpoint under `action_stats` and in:
  - `action_mean.npy`
  - `action_std.npy`

ACT-family supervision:
- `act` and `act_moe` use chunk targets (`action_chunks`)

Non-ACT supervision:
- `mlp` and `moe` use single-step targets (`actions`)

MoE auxiliary loss:
- enabled for `moe` and `act_moe` via `moe_load_balance_weight`

Early stopping:
- `early_stopping_patience`
- `early_stopping_min_delta`

## 8) Evaluation metrics collected

Per rollout summary columns:
- `success_rate`
- `best_camera`
- `safety_aborts`
- `avg_ep_max_success`
- `avg_ep_max_reward`
- `avg_ep_min_obj_to_target`
- `video_path`
- `log_path`

Training/checkpoint summary columns:
- `val_loss`
- `epoch`
- `router_mean_gap_last_epoch`
- `router_mean_entropy_last_epoch`

Unified merged metrics file:
- `Output/presentation/all_metrics_master.csv`

## 9) Safety Layer (Safe Fallback)

This repository includes a runtime safety gate in inference/evaluation.

Where it is implemented:
- `infer.py`
- `eval_rollout.py`

How it works:
1. Parse target object candidates from instruction text (or explicit override).
2. Build text prompts like "a robot scene containing a <object>".
3. Compute vision-text cosine score using model vision/text embeddings.
4. If best score is below threshold, abort before action execution.

Safety decision rule:
- If `presence_score < safety_threshold`:
  - return `SAFETY_ABORT ... message='No <object> detected; exiting.'`
- Otherwise safety check passes and normal action prediction proceeds.

Runtime controls:
- `--enable-safety-check`
- `--safety-threshold` (default around `0.20` in scripts/examples)
- `--safety-object` (manual object override, e.g., `drawer`, `button`, `peg`)

Behavioral note:
- The current safety gate is conservative and threshold-sensitive.
- High threshold can trigger false aborts even when object is present.
- In practice, threshold calibration per environment/task is recommended.

Recorded safety metric in rollout summaries:
- `safety_aborts`

## 10) Architecture-by-architecture quick map

### `no_moe`
- Head: `mlp`
- Output: `[B, A]`
- Router: none
- Chunking: no

### `moe_text`
- Head: `moe`
- Router input: text feature
- Output: `[B, A]`
- Chunking: no

### `moe_full`
- Head: `moe`
- Router input: action input (fused + geometry)
- Output: `[B, A]`
- Chunking: no

### `act`
- Head: `act`
- Router: none
- Output: `[B, K, A]`
- Chunking: yes

### `act_moe`
- Head: `act_moe`
- Router: text-only MoE context branch
- Output: `[B, K, A]`
- Chunking: yes
- Core formula:
  - `z = MoE(text_features)`
  - `actions = ACT(obs_features, z)`

## 11) Practical note for visualization slides

Recommended architecture diagram blocks:
1. Inputs:
   - RGB image
   - instruction text
   - geometry history
2. Shared encoders:
   - SigLIP2 vision/text streams
3. Projection + fusion:
   - cross-attn fusion
4. Head branch:
   - `mlp` / `moe` / `act` / `act_moe`
5. Outputs:
   - single-step or chunked actions

For `act_moe` visualization:
- draw a separate text-to-context MoE branch feeding latent `z` into ACT decoder.
