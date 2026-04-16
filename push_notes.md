# Vision-Language-Action (VLA) Project Report

## 1) Summary
This project builds a compact Vision-Language-Action (VLA) policy for MetaWorld manipulation tasks using SigLIP2-based visual-language representations and a learned action policy head.

The main goal is to map:
- RGB image observation
- Language instruction
- Geometry/state features

to continuous 4D robot actions.

Two model variants were explored:
- **Baseline (SigLIP2 + DistilBERT + standard head)**
- **Improved model (SigLIP2 shared backbone + cross-attention fusion + 4-expert MoE head + temporal geometry context)**

Observed outcome from experiments:
- The policy learns coarse manipulation on seen tasks.
- It still struggles on precision contact / pull-style behaviors (for example peg-insert and stick/handle pull settings), indicating limitations in interaction geometry reasoning and skill specialization.

---

## 2) Architecture

### 2.1 Input/Encoding
The policy uses a dual-modality VLA pipeline:
- **Vision encoder**: `google/siglip2-base-patch16-224`
- **Text encoder**:
  - Baseline: `distilbert-base-uncased`
  - MoE run: shared SigLIP2 text branch (`separate_backbones=false`)

Both vision/text outputs are projected to a shared embedding dimension (`proj_dim=512`) and optionally L2-normalized.

### 2.2 Fusion
Three fusion options are implemented (`concat`, `cross_attn`, `transformer`), with the MoE experiment using:
- **Fusion type**: `cross_attn`
- **Layers**: 3
- **Heads**: 8
- **Dropout**: 0.1

### 2.3 Geometry and Temporal Context
In addition to image+language, geometry features are derived from state:
- End-effector to object direction (3D)
- Object to goal direction (3D)

So each step contributes 6 geometry values. With `temporal_context=4`, the model receives a stacked 24D geometry history.

### 2.4 Action Head
Implemented action heads:
- Linear
- MLP
- **MoE (used in final run)**

MoE details:
- 4 experts, each MLP-based
- Router conditioned on language/text representation
- Auxiliary switch-style load-balancing loss
- Learnable action scaling after prediction

### 2.5 Training Setup (MoE run config)
From `checkpoints_3moe/train_config.json`:
- Loss: Huber (`delta=0.5`)
- Epochs: 10
- Batch size: 32, grad accumulation: 2
- LR: `5e-5`, AdamW, weight decay `1e-2`
- Mixed precision: fp16
- Frozen vision/text backbones
- Action target normalization enabled

---

## 3) Dataset Details

### 3.1 Dataset Source and Format
Primary training source is `short-metaworld-vla` loaded through `ShortMetaWorldDataset`.

Data are indexed from:
- Per-task image trajectories (`img_only/.../<task>/<traj>/<step>.jpg`)
- Matching pickle files containing action/state sequences (`r3m_MT10_20/*.pkl`)

Each training sample contains:
- Image
- Natural language task prompt
- Action vector (4D)
- State-derived geometry
- Metadata (task, episode, timestep)

### 3.2 Train/Validation Split
Split is done by **trajectory key** `(task_name, trajectory_id)` to avoid leakage across train/val.

For default dataset inspection in this project:
- Train samples: **62,987**
- Validation samples: **7,013**
- Total tasks: **18**

Task set includes:
`basketball-v3, button-press-topdown-v2, button-press-topdown-v3, door-open-v2, door-open-v3, drawer-close-v2, drawer-close-v3, drawer-open-v2, handle-pull-v3, peg-insert-side-v2, peg-insert-side-v3, pick-place-v2, pick-place-v3, push-v3, reach-v3, stick-pull-v3, sweep-v3, window-open-v3`

### 3.3 Action/Feature Handling
- Action dimension: 4
- Action targets are normalized using dataset mean/std before loss computation.
- Geometry vectors are fit/padded to configured size and temporally stacked.

---

## 4) Evaluation

### 4.1 Offline Validation (Checkpoint Metrics)
From saved checkpoints:

**MoE run (`checkpoints_3moe`)**
- Best checkpoint: epoch 8
- Best validation loss: **0.03394**
- Latest checkpoint (epoch 10) validation loss: **0.03489**
- Latest train loss: **0.00671**

**Baseline run (`Output/SigLip2_DistilBERT/checkpoints`)**
- Best checkpoint: epoch 49
- Validation loss: **0.01635**
- Train loss: **0.01334**

Interpretation:
- Baseline reports lower offline val loss on its run configuration.
- MoE run shows lower train loss but higher val loss in this specific setup, suggesting tuning/generalization is still in progress.

### 4.2 MoE Routing Behavior Analysis
Router logs in `checkpoints_3moe/moe_router_weights.csv` and `moe_router_entropy.csv` show:
- 10 logged epochs
- 18 tasks tracked
- Final-epoch average expert-weight gap (max minus min per task): **0.0074**
- Final entropy range: **1.385741 to 1.386291**
- Reference uniform entropy for 4 experts: `ln(4)=1.386294`

Interpretation:
- Routing remains close to uniform across experts (high entropy near `ln(4)`).
- The MoE router has only weak specialization in the current training setup.

### 4.3 Rollout/Behavioral Evidence
Project notes and generated rollout videos indicate:
- Success on multiple seen tasks (coarse manipulation).
- Persistent failure modes on pull and precision-contact behaviors.
- Typical issue: good object/goal localization but suboptimal interaction direction and force transfer.

This suggests current bottlenecks are not only perception, but also action grounding and contact-aware control.

---

## 5) Final Takeaway
The project demonstrates an end-to-end VLA training and evaluation pipeline with modular fusion and action-head design, including MoE routing diagnostics. The model can learn broad manipulation behaviors from short-MetaWorld data, but stronger skill specialization and contact dynamics reasoning are still needed for robust precision and pull-based tasks.
