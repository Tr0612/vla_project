# VLA Project (SigLIP2 + MoE/ACT)

Compact Vision-Language-Action training/evaluation stack for MetaWorld-style manipulation tasks.

## Table of Contents
- [VLA Project (SigLIP2 + MoE/ACT)](#vla-project-siglip2--moeact)
  - [Table of Contents](#table-of-contents)
  - [Repo Structure](#repo-structure)
  - [Important Links](#important-links)
  - [Environment setup](#environment-setup)
    - [Option A: uv (recommended)](#option-a-uv-recommended)
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
  - [File-by-file guide](#file-by-file-guide)
    - [Core source (`src/vla_stack/`)](#core-source-srcvla_stack)
    - [Root compatibility launchers](#root-compatibility-launchers)
    - [Experiment and workflow files](#experiment-and-workflow-files)
    - [Data and outputs](#data-and-outputs)
  - [File Trees (Separate Reference)](#file-trees-separate-reference)
    - [Project tree (after cloning this repo)](#project-tree-after-cloning-this-repo)

## Repo Structure

- Core source package: `src/vla_stack/` (`train.py`, `infer.py`, `eval_rollout.py`, `viz_actions.py`, `model.py`, `dataset.py`, `config.py`)
- Root `*.py` files are lightweight compatibility launchers (so commands like `python train.py` still work)
- Config-driven experiments in `experiments/`
- Pre-collected dataset roots under `data/`
- Rollout/analysis outputs under `Output/`

## Important Links

Dataset : [ShortMetaWorld](https://huggingface.co/datasets/Tr0612/ShortMetaWorld/tree/main)

Checkpoints : [VLA Checkpoints](https://huggingface.co/Tr0612/vla-checkpoints/tree/main)

Experiment Outputs : https://huggingface.co/datasets/Tr0612/vla-outputs

Demo (small dataset with inference) : [Button Press Demo](https://huggingface.co/datasets/Tr0612/button-press-mini-demo/tree/main)

## Environment setup

Clone this repo : 

Make sure the folder structure is same as given in [Folder Structure](#file-trees-separate-reference)

### Option A: uv (recommended)

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

- Dataset: [ShortMetaWorld Dataset](https://huggingface.co/datasets/Tr0612/ShortMetaWorld)
- Checkpoints: [VLA Checkpoints](https://huggingface.co/Tr0612/vla-checkpoints)

## Dataset: how to obtain it

This project is wired for `short_metaworld` format using:

- `data/short-metaworld-vla/short-MetaWorld/img_only/...`
- `data/short-metaworld-vla/short-MetaWorld/r3m-processed/r3m_MT10_20/*.pkl`

### source

The dataset remote used here is:

- [ShortMetaWorld Dataset](https://huggingface.co/datasets/Tr0612/ShortMetaWorld)

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

- [VLA Checkpoints](https://huggingface.co/Tr0612/vla-checkpoints)

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

- [VLA Checkpoints](https://huggingface.co/Tr0612/vla-checkpoints)

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

## File-by-file guide

### Core source (`src/vla_stack/`)

- `src/vla_stack/config.py`: Defines `TrainConfig`, default hyperparameters, and config loading/override helpers.
- `src/vla_stack/dataset.py`: Dataset adapters and loaders; converts raw samples into the unified schema used by training and evaluation.
- `src/vla_stack/model.py`: VLA policy architecture (SigLIP/text encoders, fusion blocks, action heads including MLP/MoE/ACT variants).
- `src/vla_stack/train.py`: Main training entrypoint, dataloader setup, optimization loop, checkpoint saving, and validation metrics.
- `src/vla_stack/infer.py`: Single-image inference script; loads checkpoint + prompt/image/state and prints predicted action.
- `src/vla_stack/eval_rollout.py`: MetaWorld rollout evaluator; runs episodes, computes success, and can record videos.
- `src/vla_stack/viz_actions.py`: Offline diagnostics for prediction vs ground-truth actions; writes plots/arrays for analysis.
- `src/vla_stack/main.py`: Minimal package-level starter entrypoint.
- `src/vla_stack/__init__.py`: Package marker for `vla_stack`.

### Root compatibility launchers

- `train.py`, `infer.py`, `eval_rollout.py`, `viz_actions.py`, `main.py`: Thin wrappers that add `src/` to `PYTHONPATH` and call the corresponding `vla_stack.*` module, so existing commands still work.
- `config.py`, `dataset.py`, `model.py`: Compatibility re-export shims for legacy imports.

### Experiment and workflow files

- `experiments/`: YAML configs and runnable shell pipelines (`run_all.sh`, `run_seen_eval.sh`, `run_peg_only.sh`).
- `demo/run_button_press_demo.py`: Small demo runner that executes inference over a compact button-press mini dataset.
- `dataset_scripts/`: One-off dataset collection/conversion utilities.
- `env_setup.bash`: Helper script for environment setup.

### Data and outputs

- `data/`: Local dataset root (`short-metaworld-vla` expected by default config).
- `checkpoints/`: Model checkpoints and normalization stats (`best.pt`, `action_mean.npy`, `action_std.npy`).
- `Output/`: Rollout videos and experiment outputs.
- `plots/`, `plots_moe/`, `plots_door_open_v3/`: Action-quality visualization outputs.

## File Trees (Separate Reference)

### Project tree (after cloning this repo)

```text
vla_project/
├── README.md
├── pyproject.toml
├── env_setup.bash
├── src/
│   └── vla_stack/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       ├── infer.py
│       ├── eval_rollout.py
│       ├── viz_actions.py
│       └── main.py
├── train.py
├── infer.py
├── eval_rollout.py
├── viz_actions.py
├── main.py
├── config.py
├── dataset.py
├── model.py
├── demo/
│   ├── run_button_press_demo.py
│   └── button_press_mini/
├── checkpoints/
│   └── checkpoints_stage2_*/
├── experiments/
├── dataset_scripts/
├── data/
│   └── short-metaworld-vla/
│       ├── mt50_task_prompts.json
│       ├── short-MetaWorld/
│       │   ├── img_only/
│       │   │   ├── button-press-topdown-v3/
│       │   │   ├── door-open-v3/
│       │   │   └── <other-tasks>/
│       │   └── r3m-processed/
│       │       └── r3m_MT10_20/
│       │           └── *.pkl
│       └── README.md
└── Output/
