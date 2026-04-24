from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from vla_stack.config import TrainConfig
from vla_stack.dataset import (
    ShortMetaWorldDataset,
    VLAJsonlDataset,
    sample_get_action,
    sample_get_action_chunk,
    sample_get_dataset_name,
    sample_get_image,
    sample_get_prompt,
    sample_get_state,
)
from vla_stack.model import VLAFusionPolicy


def resolve_text_tokenizer_name(cfg: TrainConfig) -> str:
    return cfg.text_model_name if cfg.separate_backbones else cfg.vision_model_name


def build_loss_fn(cfg: TrainConfig) -> nn.Module:
    loss_type = str(cfg.loss_type).lower().strip()
    if loss_type == "huber":
        return nn.HuberLoss(delta=float(cfg.huber_delta))
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss_type: {cfg.loss_type}. Expected one of: mse, huber")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default_config.yaml")
    parser.add_argument("--task", type=str, default=None, help="Optional task filter for short_metaworld (e.g. door-open-v3)")
    parser.add_argument(
        "--include-tasks",
        type=str,
        default=None,
        help="Comma-separated short_metaworld task names to include (applied after --task).",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        default=None,
        help="Comma-separated short_metaworld task names to exclude (applied after --task/--include-tasks).",
    )
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
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <out-dir>/tensorboard.",
    )
    return parser.parse_args()


def _parse_task_csv(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def build_collate_fn(processor, tokenizer, image_size: int):
    def image_like_to_pil(image_like) -> Image.Image:
        if isinstance(image_like, Image.Image):
            return image_like.convert("RGB")
        if torch.is_tensor(image_like):
            img = image_like.detach().cpu()
            if img.ndim == 4:
                img = img[-1]
            if img.ndim != 3:
                raise ValueError(f"Expected image tensor with shape [C,H,W] or [T,C,H,W], got {tuple(img.shape)}")
            if img.shape[0] in {1, 3, 4}:
                chw = img
            elif img.shape[-1] in {1, 3, 4}:
                chw = img.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported image tensor shape: {tuple(img.shape)}")
            if chw.shape[0] == 1:
                chw = chw.repeat(3, 1, 1)
            if chw.shape[0] == 4:
                chw = chw[:3]
            arr = (chw.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr, mode="RGB")
        raise TypeError(f"Unsupported image type in batch: {type(image_like)}")

    def collate(batch: list[dict]):
        images = [image_like_to_pil(sample_get_image(x)).resize((image_size, image_size)) for x in batch]
        texts = [sample_get_prompt(x) for x in batch]
        actions = torch.stack([sample_get_action(x) for x in batch], dim=0)
        action_chunks = torch.stack([sample_get_action_chunk(x) for x in batch], dim=0)
        geometry = torch.stack([sample_get_state(x) for x in batch], dim=0)
        task_names = [sample_get_dataset_name(x) for x in batch]

        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(text_inputs["input_ids"])

        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": attention_mask,
            "geometry_features": geometry,
            "task_names": task_names,
            "actions": actions,
            "action_chunks": action_chunks,
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
            a = sample_get_action(dataset[i]).to(dtype=torch.float64)

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
    action_head_type: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            geometry_features = batch["geometry_features"].to(device)
            target_actions = batch["actions"].to(device)
            target_action_chunks = batch["action_chunks"].to(device)

            head_type = str(action_head_type).lower().strip()
            is_act = head_type in {"act", "act_moe"}
            if is_act:
                target_for_loss = (
                    normalize_actions(target_action_chunks, action_mean, action_std)
                    if normalize_targets
                    else target_action_chunks
                )
            else:
                target_for_loss = (
                    normalize_actions(target_actions, action_mean, action_std) if normalize_targets else target_actions
                )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_fp16 and device.type == "cuda"),
            ):
                pred_actions = model(pixel_values, input_ids, attention_mask, geometry_features=geometry_features)
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
            geometry_dim=cfg.geometry_dim,
            temporal_context=cfg.temporal_context,
            action_chunk_size=cfg.act_chunk_size,
        )
        val_ds = ShortMetaWorldDataset(
            data_root=cfg.data_root,
            split="val",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            action_dim=cfg.action_dim,
            geometry_dim=cfg.geometry_dim,
            temporal_context=cfg.temporal_context,
            action_chunk_size=cfg.act_chunk_size,
        )
        if args.task:
            train_ds.samples = [s for s in train_ds.samples if s.get("task_name") == args.task]
            val_ds.samples = [s for s in val_ds.samples if s.get("task_name") == args.task]
            if len(train_ds.samples) == 0:
                raise RuntimeError(
                    f"No training samples found for task='{args.task}' under data_root='{cfg.data_root}'."
                )
            if len(val_ds.samples) == 0:
                raise RuntimeError(
                    f"No validation samples found for task='{args.task}' under data_root='{cfg.data_root}'. "
                    "Try increasing --val-ratio or using a dataset with more trajectories for this task."
                )
            print(
                f"task filter: {args.task} | train samples: {len(train_ds.samples)} | val samples: {len(val_ds.samples)}",
                flush=True,
            )

        include_tasks = _parse_task_csv(args.include_tasks)
        if include_tasks:
            train_ds.samples = [s for s in train_ds.samples if str(s.get("task_name", "")) in include_tasks]
            val_ds.samples = [s for s in val_ds.samples if str(s.get("task_name", "")) in include_tasks]
            print(
                f"include task filter ({len(include_tasks)} tasks) "
                f"| train samples: {len(train_ds.samples)} | val samples: {len(val_ds.samples)}",
                flush=True,
            )

        exclude_tasks = _parse_task_csv(args.exclude_tasks)
        if exclude_tasks:
            train_ds.samples = [s for s in train_ds.samples if str(s.get("task_name", "")) not in exclude_tasks]
            val_ds.samples = [s for s in val_ds.samples if str(s.get("task_name", "")) not in exclude_tasks]
            print(
                f"exclude task filter ({len(exclude_tasks)} tasks) "
                f"| train samples: {len(train_ds.samples)} | val samples: {len(val_ds.samples)}",
                flush=True,
            )

        if len(train_ds.samples) == 0:
            raise RuntimeError("Task filtering removed all training samples. Adjust --task/--include-tasks/--exclude-tasks.")
        if len(val_ds.samples) == 0:
            raise RuntimeError("Task filtering removed all validation samples. Adjust --val-ratio or task filters.")
    else:
        if not cfg.train_jsonl or not cfg.val_jsonl:
            raise ValueError("For dataset_type=jsonl, provide --train-jsonl and --val-jsonl")
        train_ds = VLAJsonlDataset(
            cfg.train_jsonl,
            action_dim=cfg.action_dim,
            geometry_dim=cfg.geometry_dim,
            temporal_context=cfg.temporal_context,
            action_chunk_size=cfg.act_chunk_size,
        )
        val_ds = VLAJsonlDataset(
            cfg.val_jsonl,
            action_dim=cfg.action_dim,
            geometry_dim=cfg.geometry_dim,
            temporal_context=cfg.temporal_context,
            action_chunk_size=cfg.act_chunk_size,
        )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)

    action_mean, action_std = compute_action_stats(train_ds, cfg.action_dim, cfg.action_norm_eps)
    print(f"Action mean: {action_mean.tolist()}", flush=True)
    print(f"Action std: {action_std.tolist()}", flush=True)

    print("Loading processor/tokenizer...", flush=True)
    processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolve_text_tokenizer_name(cfg))

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
    loss_fn = build_loss_fn(cfg)
    if str(cfg.loss_type).lower().strip() == "huber":
        print(f"Loss: Huber (delta={cfg.huber_delta})", flush=True)
    else:
        print("Loss: MSE", flush=True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(cfg.use_fp16 and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")
    early_stop_patience = int(max(0, getattr(cfg, "early_stopping_patience", 0)))
    early_stop_min_delta = float(max(0.0, getattr(cfg, "early_stopping_min_delta", 0.0)))
    epochs_without_improvement = 0

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

    writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            print(
                "TensorBoard logging disabled: torch.utils.tensorboard is unavailable. "
                "Install with: pip install tensorboard",
                flush=True,
            )
        else:
            tb_dir = Path(args.tb_logdir) if args.tb_logdir else (out_dir / "tensorboard")
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}", flush=True)

    global_step = start_epoch * max(len(train_loader), 1)
    head_type = str(cfg.action_head_type).lower().strip()
    use_moe_head = head_type in {"moe", "act_moe"}
    is_act_head = head_type in {"act", "act_moe"}
    router_csv_path = out_dir / "moe_router_weights.csv"
    router_entropy_csv_path = out_dir / "moe_router_entropy.csv"
    if use_moe_head and start_epoch == 0:
        with router_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "task_name", "expert_idx", "mean_router_weight", "sample_count"])
        with router_entropy_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "task_name", "mean_router_entropy", "sample_count"])

    if early_stop_patience > 0:
        print(
            f"Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta:.6f}",
            flush=True,
        )
    else:
        print("Early stopping disabled", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        sample_count = 0
        router_sum_by_task: dict[str, torch.Tensor] = {}
        router_count_by_task: dict[str, int] = {}
        router_entropy_sum_by_task: dict[str, float] = {}

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            geometry_features = batch["geometry_features"].to(device)
            task_names = batch.get("task_names", [])
            target_actions = batch["actions"].to(device)
            target_action_chunks = batch["action_chunks"].to(device)

            if is_act_head:
                target_for_loss = (
                    normalize_actions(target_action_chunks, action_mean, action_std)
                    if cfg.normalize_action_targets
                    else target_action_chunks
                )
            else:
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
                pred_actions = model(pixel_values, input_ids, attention_mask, geometry_features=geometry_features)
                main_loss = loss_fn(pred_actions, target_for_loss)
                aux_loss = None
                if use_moe_head and float(cfg.moe_load_balance_weight) > 0.0:
                    aux_raw = getattr(model, "last_moe_load_balance_loss", None)
                    if aux_raw is not None:
                        aux_loss = float(cfg.moe_load_balance_weight) * aux_raw
                loss = main_loss if aux_loss is None else (main_loss + aux_loss)
                loss = loss / cfg.grad_accum_steps

            if use_moe_head and hasattr(model.action_head, "last_router_weights"):
                router_weights = getattr(model.action_head, "last_router_weights", None)
                if router_weights is not None and len(task_names) == router_weights.size(0):
                    rw = router_weights.detach().cpu()
                    ent = -(rw * torch.log(rw + 1e-8)).sum(dim=-1)
                    for i, task_name in enumerate(task_names):
                        t = str(task_name)
                        if t not in router_sum_by_task:
                            router_sum_by_task[t] = torch.zeros(rw.size(1), dtype=torch.float32)
                            router_count_by_task[t] = 0
                            router_entropy_sum_by_task[t] = 0.0
                        router_sum_by_task[t] += rw[i]
                        router_count_by_task[t] += 1
                        router_entropy_sum_by_task[t] += float(ent[i].item())

            scaler.scale(loss).backward()

            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = target_actions.size(0)
            batch_loss = loss.item() * cfg.grad_accum_steps
            running_loss += batch_loss * bs
            sample_count += bs
            if writer is not None:
                writer.add_scalar("train/step_loss", batch_loss, global_step)
                writer.add_scalar("train/step_main_loss", main_loss.item(), global_step)
                if aux_loss is not None:
                    writer.add_scalar("train/step_moe_load_balance_loss", aux_loss.item(), global_step)
            global_step += 1

        # Flush remainder micro-batches when len(train_loader) is not divisible by grad_accum_steps.
        if len(train_loader) % cfg.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            cfg.use_fp16,
            cfg.normalize_action_targets,
            action_mean,
            action_std,
            cfg.action_head_type,
        )

        print(f"epoch={epoch + 1}/{cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if writer is not None:
            writer.add_scalar("train/epoch_loss", train_loss, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)
            if use_moe_head:
                for task_name, weight_sum in router_sum_by_task.items():
                    count = max(1, router_count_by_task.get(task_name, 0))
                    mean_weights = weight_sum / count
                    safe_task = task_name.replace("/", "_")
                    for expert_idx, val in enumerate(mean_weights.tolist()):
                        writer.add_scalar(f"moe/router/{safe_task}/expert_{expert_idx}", float(val), epoch + 1)
                    mean_entropy = float(router_entropy_sum_by_task.get(task_name, 0.0)) / count
                    writer.add_scalar(f"moe/router_entropy/{safe_task}", mean_entropy, epoch + 1)
        if use_moe_head and router_sum_by_task:
            with router_csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for task_name, weight_sum in router_sum_by_task.items():
                    count = max(1, router_count_by_task.get(task_name, 0))
                    mean_weights = weight_sum / count
                    for expert_idx, val in enumerate(mean_weights.tolist()):
                        w.writerow([epoch + 1, task_name, expert_idx, float(val), count])
            with router_entropy_csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for task_name in router_sum_by_task.keys():
                    count = max(1, router_count_by_task.get(task_name, 0))
                    mean_entropy = float(router_entropy_sum_by_task.get(task_name, 0.0)) / count
                    w.writerow([epoch + 1, task_name, mean_entropy, count])

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

        is_improved = val_loss < (best_val - early_stop_min_delta)
        if cfg.save_best_by_val and is_improved:
            best_val = val_loss
            best_path = out_dir / "best.pt"
            torch.save(ckpt_payload, best_path)
            print(f"saved best checkpoint: {best_path} (val_loss={best_val:.6f})")
            if writer is not None:
                writer.add_scalar("val/best_loss", best_val, epoch + 1)
        elif is_improved:
            best_val = val_loss

        if is_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"no val improvement for {epochs_without_improvement} epoch(s) "
                f"(best={best_val:.6f}, current={val_loss:.6f})",
                flush=True,
            )

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(
                f"early stopping triggered at epoch {epoch + 1}: "
                f"no improvement for {epochs_without_improvement} epochs",
                flush=True,
            )
            break

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
