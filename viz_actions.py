from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from dataset import ShortMetaWorldDataset, VLAJsonlDataset
from model import VLAFusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize action prediction quality.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="default_config.yaml")
    parser.add_argument("--task", type=str, default=None, help="Optional task filter for short_metaworld (e.g. door-open-v3)")
    parser.add_argument("--dataset-type", type=str, choices=["short_metaworld", "jsonl"], default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--out-dir", type=str, default="plots")
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


def load_cfg(args: argparse.Namespace, ckpt: dict) -> TrainConfig:
    cfg_path = Path(args.config)
    cfg = TrainConfig.from_yaml(cfg_path) if cfg_path.exists() else TrainConfig()
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        cfg.apply_overrides(ckpt["cfg"])

    if args.dataset_type is not None:
        cfg.dataset_type = args.dataset_type
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.val_jsonl is not None:
        cfg.val_jsonl = args.val_jsonl
    if args.val_ratio is not None:
        cfg.val_ratio = args.val_ratio

    return cfg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = load_cfg(args, ckpt)

    action_stats = ckpt.get("action_stats", {}) if isinstance(ckpt, dict) else {}
    action_mean = torch.tensor(action_stats.get("mean", [0.0] * cfg.action_dim), dtype=torch.float32)
    action_std = torch.tensor(action_stats.get("std", [1.0] * cfg.action_dim), dtype=torch.float32)
    normalized_targets = bool(action_stats.get("normalized_targets", False))

    model = VLAFusionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    collate_fn = build_collate_fn(processor, tokenizer, cfg.image_size)

    if cfg.dataset_type == "short_metaworld":
        val_ds = ShortMetaWorldDataset(
            data_root=cfg.data_root,
            split="val",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
        )
        if args.task:
            val_ds.samples = [s for s in val_ds.samples if s.get("task_name") == args.task]
            if len(val_ds.samples) == 0:
                raise RuntimeError(
                    f"No validation samples found for task='{args.task}' under data_root='{cfg.data_root}'."
                )
            print(f"task filter: {args.task} | samples: {len(val_ds.samples)}")
    else:
        if not cfg.val_jsonl:
            raise ValueError("For jsonl mode, provide --val-jsonl")
        val_ds = VLAJsonlDataset(cfg.val_jsonl, action_dim=cfg.action_dim)

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    preds = []
    gts = []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_actions = batch["actions"].to(device)

            pred_actions = model(pixel_values, input_ids, attention_mask)
            pred_actions = pred_actions.detach().cpu()

            if normalized_targets:
                pred_actions = pred_actions * action_std + action_mean

            preds.append(pred_actions)
            gts.append(target_actions.detach().cpu())

            seen += target_actions.size(0)
            if seen >= args.max_samples:
                break

    pred = torch.cat(preds, dim=0)[: args.max_samples]
    gt = torch.cat(gts, dim=0)[: args.max_samples]

    mae = (pred - gt).abs()
    rmse = ((pred - gt) ** 2).mean(dim=0).sqrt()

    print(f"samples: {pred.shape[0]}")
    print(f"mean_mae: {mae.mean().item():.6f}")
    print(f"per_dim_mae: {mae.mean(dim=0).tolist()}")
    print(f"per_dim_rmse: {rmse.tolist()}")

    np.save(out_dir / "pred_actions.npy", pred.numpy())
    np.save(out_dir / "gt_actions.npy", gt.numpy())

    try:
        import matplotlib.pyplot as plt

        n_dims = pred.shape[1]

        n_plot = min(200, pred.shape[0])
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.4 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        x = np.arange(n_plot)
        pred_np = pred.numpy()
        gt_np = gt.numpy()

        for d in range(n_dims):
            axes[d].plot(x, gt_np[:n_plot, d], label="gt", linewidth=1.5)
            axes[d].plot(x, pred_np[:n_plot, d], label="pred", linewidth=1.0)
            axes[d].set_ylabel(f"a{d}")
            axes[d].grid(alpha=0.25)
            if d == 0:
                axes[d].legend(loc="upper right")

        axes[-1].set_xlabel("sample index")
        fig.tight_layout()
        fig.savefig(out_dir / "pred_vs_gt.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        vals = mae.mean(dim=0).numpy()
        ax.bar(np.arange(len(vals)), vals)
        ax.set_xlabel("action dimension")
        ax.set_ylabel("MAE")
        ax.set_title("Per-dimension MAE")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "mae_per_dim.png", dpi=150)
        plt.close(fig)

        print(f"saved plots to: {out_dir}")
    except ImportError:
        print("matplotlib not installed: saved only .npy outputs")


if __name__ == "__main__":
    main()
