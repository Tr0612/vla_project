# VLA Project (SigLIP2 + MoE/ACT)

Compact Vision-Language-Action training/evaluation stack for MetaWorld-style manipulation tasks.

## What this repo includes

- Core training/inference scripts: `train.py`, `infer.py`, `eval_rollout.py`, `viz_actions.py`
- Config-driven experiments in `experiments/`
- Pre-collected dataset roots under `data/`
- Rollout/analysis outputs under `Output/`

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

## Dataset: how to obtain it

This project is wired for `short_metaworld` format using:

- `data/short-metaworld-vla/short-MetaWorld/img_only/...`
- `data/short-metaworld-vla/short-MetaWorld/r3m-processed/r3m_MT10_20/*.pkl`

### Preferred source

The dataset remote used here is:

- `[https://huggingface.co/datasets/hz1919810/short-metaworld-vla](https://huggingface.co/datasets/Tr0612/ShortMetaWorld)`

### Fresh clone with Git LFS

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/hz1919810/short-metaworld-vla
cd short-metaworld-vla
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

Top-level checkpoint directory:

- `checkpoints/`

Examples:

- `checkpoints/best.pt` (legacy/base run)
- `checkpoints/checkpoints_linear/`
- `checkpoints/checkpoints_stage2_no_moe_unfreeze/`
- `checkpoints/checkpoints_stage2_moe_text_unfreeze/`
- `checkpoints/checkpoints_stage2_moe_full_unfreeze/`
- `checkpoints/checkpoints_stage2_act_unfreeze/`
- `checkpoints/checkpoints_stage2_act_moe_unfreeze/`

Each experiment folder typically contains:

- `best.pt`
- `latest.pt`
- `train_config.json`
- optional `tensorboard/`
- optional router logs (`moe_router_weights.csv`, `moe_router_entropy.csv`)

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
