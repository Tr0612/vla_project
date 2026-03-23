# VLA Stack (SigLIP2 Starter)

This starter is configured for RTX 4070 8GB constraints.

## Defaults baked in

- Vision encoder: pretrained SigLIP2 (google/siglip2-base-patch16-224)
- Input resolution: 224x224
- Freeze vision encoder: True
- Trainable modules: fusion + action head (text frozen by default)
- Mixed precision: fp16 when CUDA is available
- Batch size: 8
- Gradient accumulation: 4 (effective batch size 32)
- Data workers: 2
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-2
- Epochs: 30 (change to 20-50 as needed)
- Gradient clipping: 1.0
- Checkpointing: saves latest.pt each epoch and best.pt by validation loss

Exact defaults are also listed in default_config.yaml.

## Dataset options

1. short_metaworld (default): loads local short-metaworld-vla directly and creates train/val split.
2. jsonl: uses explicit train/val JSONL files.

For short_metaworld, expected local path:
/media/thanush/ubuntu_project/vla/vla_stack/data/short-metaworld-vla

## Train

Default (short-metaworld-vla):

python train.py --dataset-type short_metaworld --data-root data/short-metaworld-vla

JSONL mode:

python train.py --dataset-type jsonl --train-jsonl /path/to/train.jsonl --val-jsonl /path/to/val.jsonl

Optional overrides:

python train.py --dataset-type short_metaworld --data-root data/short-metaworld-vla --val-ratio 0.1 --epochs 50 --batch-size 8 --num-workers 4 --out-dir /path/to/checkpoints

## Inference

python infer.py --ckpt /path/to/best.pt --image /path/to/image.png --instruction "push block to target"

## Visualize Predictions

python viz_actions.py --ckpt checkpoints/best.pt --config default_config.yaml --out-dir plots

Outputs:
- plots/pred_vs_gt.png
- plots/mae_per_dim.png
- plots/pred_actions.npy
- plots/gt_actions.npy

## Rollout Success (MetaWorld)

python eval_rollout.py --ckpt checkpoints/best.pt --task button-press-topdown-v2 --episodes 20 --max-steps 200

Optional video recording:

python eval_rollout.py --ckpt checkpoints/best.pt --task button-press-topdown-v2 --episodes 5 --record-video videos/rollout.mp4

## Notes

- action_dim default is 4 (MetaWorld-friendly).
- Keep SigLIP2 frozen for 8GB VRAM stability.

## RTX 50-series (sm_120) PyTorch fix

If you see:

`NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation`

your PyTorch build is too old (for example `2.6.0+cu124`).

Install a CUDA 12.8+ PyTorch build:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'cuda-not-available')"
```

## Phase-2 fine-tuning (optional)

If phase-1 is stable and you want more performance:
- Unfreeze only the last 1-2 SigLIP2 blocks.
- Reduce micro-batch to 2-4.
- Keep accumulation to maintain effective batch.
- Use lower LR for vision (1e-5) and 1e-4 for head.
