from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from dataset import ShortMetaWorldDataset, VLAJsonlDataset
from model import VLAFusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default_config.yaml")
    parser.add_argument("--dataset-type", type=str, choices=["short_metaworld", "jsonl"], default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_collate_fn(processor, tokenizer, image_size: int):
    def collate(batch: list[dict]):
        images = [x["image"].resize((image_size, image_size)) for x in batch]
        texts = [x["instruction"] for x in batch]
        actions = torch.stack([x["action"] for x in batch], dim=0)

        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "actions": actions,
        }

    return collate


def compute_action_stats(dataset, action_dim: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    sum_vec = torch.zeros(action_dim, dtype=torch.float64)
    sum_sq_vec = torch.zeros(action_dim, dtype=torch.float64)

    n = len(dataset)
    for i in range(n):
        if hasattr(dataset, "get_action"):
            a = dataset.get_action(i).to(dtype=torch.float64)
        else:
            a = dataset[i]["action"].to(dtype=torch.float64)

        if a.numel() != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {a.numel()} at idx={i}")

        sum_vec += a
        sum_sq_vec += a * a

    mean = sum_vec / max(n, 1)
    var = (sum_sq_vec / max(n, 1)) - (mean * mean)
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    std = torch.clamp(std, min=eps)

    return mean.to(torch.float32), std.to(torch.float32)


def normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (actions - mean) / std


def evaluate(
    model,
    loader,
    loss_fn,
    device,
    use_fp16: bool,
    normalize_targets: bool,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_actions = batch["actions"].to(device)

            target_for_loss = (
                normalize_actions(target_actions, action_mean, action_std) if normalize_targets else target_actions
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_fp16 and device.type == "cuda"),
            ):
                pred_actions = model(pixel_values, input_ids, attention_mask)
                if step == 1:
                    print(f"pred std: {pred_actions.detach().float().std().item():.6f}", flush=True)
                    print(f"gt std: {target_for_loss.detach().float().std().item():.6f}", flush=True)
                loss = loss_fn(pred_actions, target_for_loss)

            bs = target_actions.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    cfg = TrainConfig.from_yaml(cfg_path) if cfg_path.exists() else TrainConfig()

    if args.dataset_type is not None:
        cfg.dataset_type = args.dataset_type
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.train_jsonl is not None:
        cfg.train_jsonl = args.train_jsonl
    if args.val_jsonl is not None:
        cfg.val_jsonl = args.val_jsonl
    if args.val_ratio is not None:
        cfg.val_ratio = args.val_ratio
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    resume_path = Path(args.resume) if args.resume else None

    print("Training started", flush=True)
    print(f"Loaded config from: {cfg_path if cfg_path.exists() else 'TrainConfig defaults'}", flush=True)

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(cfg.seed)
    print(f"Training on device: {device}", flush=True)
    print(f"Dataset type: {cfg.dataset_type}", flush=True)

    if cfg.dataset_type == "short_metaworld":
        print(f"Loading short-metaworld from: {cfg.data_root}", flush=True)
        train_ds = ShortMetaWorldDataset(
            data_root=cfg.data_root,
            split="train",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
        )
        val_ds = ShortMetaWorldDataset(
            data_root=cfg.data_root,
            split="val",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
        )
    else:
        if not cfg.train_jsonl or not cfg.val_jsonl:
            raise ValueError("For dataset_type=jsonl, provide --train-jsonl and --val-jsonl")
        train_ds = VLAJsonlDataset(cfg.train_jsonl, action_dim=cfg.action_dim)
        val_ds = VLAJsonlDataset(cfg.val_jsonl, action_dim=cfg.action_dim)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)

    action_mean, action_std = compute_action_stats(train_ds, cfg.action_dim, cfg.action_norm_eps)
    print(f"Action mean: {action_mean.tolist()}", flush=True)
    print(f"Action std: {action_std.tolist()}", flush=True)

    print("Loading processor/tokenizer...", flush=True)
    processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    collate_fn = build_collate_fn(processor, tokenizer, cfg.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    print("Building model...", flush=True)
    model = VLAFusionPolicy(cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=(cfg.use_fp16 and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model"])

        if isinstance(resume_ckpt, dict) and "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])

        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_val = float(resume_ckpt.get("val_loss", best_val))

        resume_action_stats = resume_ckpt.get("action_stats", {}) if isinstance(resume_ckpt, dict) else {}
        resume_mean = resume_action_stats.get("mean")
        resume_std = resume_action_stats.get("std")
        if resume_mean is not None and resume_std is not None:
            if len(resume_mean) != cfg.action_dim or len(resume_std) != cfg.action_dim:
                raise ValueError(
                    "Checkpoint action_stats shape does not match config action_dim. "
                    f"Expected {cfg.action_dim}, got mean={len(resume_mean)}, std={len(resume_std)}"
                )
            action_mean = torch.tensor(resume_mean, dtype=torch.float32)
            action_std = torch.tensor(resume_std, dtype=torch.float32)
            print("Loaded action stats from checkpoint", flush=True)

        print(f"Resume start epoch: {start_epoch}", flush=True)

    action_mean = action_mean.to(device)
    action_std = action_std.to(device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "train_config.json")
    np.save(out_dir / "action_mean.npy", action_mean.detach().cpu().numpy())
    np.save(out_dir / "action_std.npy", action_std.detach().cpu().numpy())

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_actions = batch["actions"].to(device)

            target_for_loss = (
                normalize_actions(target_actions, action_mean, action_std)
                if cfg.normalize_action_targets
                else target_actions
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(cfg.use_fp16 and device.type == "cuda"),
            ):
                pred_actions = model(pixel_values, input_ids, attention_mask)
                loss = loss_fn(pred_actions, target_for_loss)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * cfg.grad_accum_steps
            steps += 1

        train_loss = running_loss / max(steps, 1)
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            cfg.use_fp16,
            cfg.normalize_action_targets,
            action_mean,
            action_std,
        )

        print(f"epoch={epoch + 1}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ckpt_payload = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "action_stats": {
                "mean": action_mean.detach().cpu().tolist(),
                "std": action_std.detach().cpu().tolist(),
                "normalized_targets": bool(cfg.normalize_action_targets),
            },
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        latest_path = out_dir / "latest.pt"
        torch.save(ckpt_payload, latest_path)

        if cfg.save_best_by_val and val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "best.pt"
            torch.save(ckpt_payload, best_path)
            print(f"saved best checkpoint: {best_path} (val_loss={best_val:.6f})")


if __name__ == "__main__":
    main()
