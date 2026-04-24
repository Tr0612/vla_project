# VLA Project (SigLIP2 + MoE/ACT)

Compact Vision-Language-Action training/evaluation stack for MetaWorld-style manipulation tasks.

## Repo Structure

- Core source package: `src/vla_stack/` (`train.py`, `infer.py`, `eval_rollout.py`, `viz_actions.py`, `model.py`, `dataset.py`, `config.py`)
- Root `*.py` files are lightweight compatibility launchers (so commands like `python train.py` still work)
- Config-driven experiments in `experiments/`
- Pre-collected dataset roots under `data/`
- Rollout/analysis outputs under `Output/`

 [VLA Project (SigLIP2 + MoE/ACT)](#vla-project-siglip2--moeact)
- [VLA Project (SigLIP2 + MoE/ACT)](#vla-project-siglip2--moeact)
  - [Repo Structure](#repo-structure)
  - [Environment setup](#environment-setup)
    - [Option A: `uv` (recommended)](#option-a-uv-recommended)
    - [Option B: existing helper script](#option-b-existing-helper-script)
  - [Quick Demo (Run First)](#quick-demo-run-first)
  - [Dataset: how to obtain it](#dataset-how-to-obtain-it)
    - [source](#source)
    - [Fresh clone with Git LFS](#fresh-clone-with-git-lfs)
    - [Verify dataset structure](#verify-dataset-structure)
  - [Quick start: single training run](#quick-start-single-training-run)
  - [Run experiment suites](#run-experiment-suites)
    - [Full presentation pipeline (train + held-out train + eval + summaries)](#full-presentation-pipeline-train--held-out-train--eval--summaries)
    - [Re-run only holdout evaluation and summaries](#re-run-only-holdout-evaluation-and-summaries)
    - [Seen-task evaluation only](#seen-task-evaluation-only)
    - [Peg-only stress test](#peg-only-stress-test)
  - [Inference and rollout](#inference-and-rollout)
    - [Single image inference](#single-image-inference)
    - [Button-press mini demo (self-contained)](#button-press-mini-demo-self-contained)
    - [Rollout eval with video recording](#rollout-eval-with-video-recording)
    - [Visualize prediction vs ground truth](#visualize-prediction-vs-ground-truth)
  - [Checkpoint layout](#checkpoint-layout)
  - [Rollout videos and outputs](#rollout-videos-and-outputs)
  - [Notes](#notes)

## Environment setup

### Option A: `uv` (recommended)

```bash
uv venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow tqdm pyyaml matplotlib tensorboard pandas opencv-python
```

### Option B: existing helper script

```bash
source env_setup.bash
```

## Quick Demo (Run First)

Run the button-press mini demo:

```bash
./.venv/bin/python demo/run_button_press_demo.py
```

Run all 5 samples:

```bash
./.venv/bin/python demo/run_button_press_demo.py --max-samples 0
```

Requires:

- Dataset: `https://huggingface.co/datasets/Tr0612/ShortMetaWorld`
- Checkpoints: `https://huggingface.co/Tr0612/vla-checkpoints`

## Dataset: how to obtain it

This project is wired for `short_metaworld` format using:

- `data/short-metaworld-vla/short-MetaWorld/img_only/...`
- `data/short-metaworld-vla/short-MetaWorld/r3m-processed/r3m_MT10_20/*.pkl`

### source

The dataset remote used here is:

- `https://huggingface.co/datasets/Tr0612/ShortMetaWorld`

### Fresh clone with Git LFS

```bash
cd data
git clone https://huggingface.co/datasets/Tr0612/ShortMetaWorld
mv ShortMetaWorld short-metaworld-vla
```

If your clone did not fetch large files, run:

```bash
cd data/short-metaworld-vla
git lfs install
git lfs pull
```

### Verify dataset structure

```bash
cd /path/to/vla_project
find data/short-metaworld-vla -maxdepth 3 -type d | head -n 30
du -sh data/short-metaworld-vla
```

If you already have the dataset on another disk, copy/symlink it into:

- `data/short-metaworld-vla`

## Quick start: single training run

```bash
python train.py \
  --config experiments/siglip2_config.yaml \
  --dataset-type short_metaworld \
  --data-root data/short-metaworld-vla \
  --out-dir checkpoints/checkpoints_linear
```

Useful overrides:

```bash
python train.py --config experiments/no_moe.yaml --out-dir checkpoints/checkpoints_stage2_no_moe
python train.py --config experiments/moe_text_unfreeze.yaml --out-dir checkpoints/checkpoints_stage2_moe_text_unfreeze
python train.py --config experiments/act_moe_unfreeze.yaml --out-dir checkpoints/checkpoints_stage2_act_moe_unfreeze
```

## Run experiment suites

### Full presentation pipeline (train + held-out train + eval + summaries)

```bash
bash experiments/run_all.sh
```

### Re-run only holdout evaluation and summaries

```bash
RUN_CORE_TRAIN=0 RUN_HOLDOUT_TRAIN=0 RUN_HOLDOUT_EVAL=1 bash experiments/run_all.sh
```

### Seen-task evaluation only

```bash
bash experiments/run_seen_eval.sh
```

### Peg-only stress test

```bash
bash experiments/run_peg_only.sh
```

## Inference and rollout

### Single image inference

```bash
python infer.py \
  --ckpt checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt \
  --image data/short-metaworld-vla/short-MetaWorld/img_only/button-press-topdown-v3/0/0.jpg \
  --instruction "Press the button"
```

### Button-press mini demo (self-contained)

Use the tiny dataset in `demo/button_press_mini/` to run quick inference across the 4 selected checkpoints.

Demo files:

- `demo/button_press_mini/samples.jsonl` (5 samples with image/instruction/state/gt_action)
- `demo/button_press_mini/images/` (copied JPEG frames)
- `demo/button_press_mini/manifest.json` (metadata)
- `demo/run_button_press_demo.py` (runner script)

Quick run (default: 1 sample):

```bash
./.venv/bin/python demo/run_button_press_demo.py
```

Run all 5 samples:

```bash
./.venv/bin/python demo/run_button_press_demo.py --max-samples 0
```

Output file:

- `demo/button_press_mini/inference_results.jsonl`

Checkpoints used by the runner:

- `checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt`
- `checkpoints/checkpoints_stage2_moe_text_unfreeze/best.pt`
- `checkpoints/checkpoints_stage2_moe_full_unfreeze/best.pt`
- `checkpoints/checkpoints_stage2_moe_full_peg_only/best.pt`

If you do not have local checkpoints, download from:

- `https://huggingface.co/Tr0612/vla-checkpoints`

```bash
cd /path/to/vla_project
git clone https://huggingface.co/Tr0612/vla-checkpoints checkpoints_hf
cd checkpoints_hf && git lfs pull && cd ..
mkdir -p checkpoints
cp -r checkpoints_hf/checkpoints_stage2_* checkpoints/
```

### Rollout eval with video recording

```bash
python eval_rollout.py \
  --config experiments/no_moe_unfreeze.yaml \
  --ckpt checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt \
  --task button-press-topdown-v3 \
  --episodes 5 \
  --max-steps 250 \
  --camera corner2 \
  --record-video Output/presentation/no_moe_button.mp4
```

### Visualize prediction vs ground truth

```bash
python viz_actions.py \
  --ckpt checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt \
  --config experiments/no_moe_unfreeze.yaml \
  --out-dir plots
```

## Checkpoint layout

Checkpoint source (published):

- `https://huggingface.co/Tr0612/vla-checkpoints`

Expected local layout for this repo:

- `checkpoints/`
- `checkpoints/checkpoints_stage2_no_moe_unfreeze/`
- `checkpoints/checkpoints_stage2_moe_text_unfreeze/`
- `checkpoints/checkpoints_stage2_moe_full_unfreeze/`
- `checkpoints/checkpoints_stage2_moe_full_peg_only/`

Minimal files required per checkpoint folder for inference:

- `best.pt`
- `action_mean.npy`
- `action_std.npy`

Optional auxiliary files:

- `latest.pt`
- `train_config.json`
- `tensorboard/`
- router logs (`moe_router_weights.csv`, `moe_router_entropy.csv`)

## Rollout videos and outputs

Generated artifacts are under:

- `Output/presentation/...` (pipeline runs)
- `Output/.../videos/` (rollout videos)
- `plots*/` (diagnostic plots)

To inspect size quickly:

```bash
du -sh checkpoints Output plots*
```

## Notes

- Keep backbone freezing for lower VRAM runs; unfreeze last layers only after stable baseline.
- Prefer explicit `--config` in all commands.
- For RTX 50-series (`sm_120`), use CUDA 12.8+ PyTorch wheels.
