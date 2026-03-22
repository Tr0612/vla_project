# VLA Stack (SigLIP2 Starter)

This starter is configured for your RTX 4070 8GB constraints.

## Defaults baked in

- Vision encoder: pretrained `SigLIP2` (`google/siglip2-base-patch16-224`)
- Input resolution: `224x224`
- Freeze vision encoder: `True`
- Trainable modules: fusion + action head (text is frozen by default too)
- Mixed precision: `fp16` when CUDA is available
- Batch size: `8`
- Gradient accumulation: `4` (effective batch size `32`)
- Data workers: `2`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Weight decay: `1e-2`
- Epochs: `30` (change to 20-50 as needed)
- Gradient clipping: `1.0`
- Checkpointing: saves `latest.pt` each epoch and `best.pt` by validation loss

Exact defaults are also listed in `default_config.yaml`.

## Dataset format

Use JSONL for train/val files.

```json
{"image_path":"images/0001.png","instruction":"push block to goal","action":[0.1,0.0,-0.2,0.0,0.0,0.0,1.0]}
```

Notes:
- `action` length must match `action_dim` (default 7).
- Relative `image_path` is resolved from the JSONL file folder.

## Train

```bash
cd /media/thanush/ubuntu_project/vla/vla_stack
python train.py --train-jsonl /path/to/train.jsonl --val-jsonl /path/to/val.jsonl
```

Optional overrides:

```bash
python train.py \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 4 \
  --out-dir /path/to/checkpoints
```

## Inference

```bash
cd /media/thanush/ubuntu_project/vla/vla_stack
python infer.py --ckpt /path/to/best.pt --image /path/to/image.png --instruction "push block to target"
```

## Phase-2 fine-tuning (optional)

If phase-1 is stable and you want more performance:
- Unfreeze only the last 1-2 SigLIP2 blocks.
- Reduce micro-batch to `2-4`.
- Keep accumulation to maintain effective batch.
- Use lower LR for vision (`1e-5`) and `1e-4` for head.
