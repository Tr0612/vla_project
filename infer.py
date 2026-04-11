from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from dataset import compute_geometry_features_from_state, fit_geometry_dim
from model import VLAFusionPolicy


def resolve_text_tokenizer_name(cfg: TrainConfig) -> str:
    return cfg.text_model_name if cfg.separate_backbones else cfg.vision_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument(
        "--state",
        type=str,
        default="",
        help="Optional comma-separated env state vector for geometry features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = TrainConfig()
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        cfg.apply_overrides(ckpt["cfg"])

    model = VLAFusionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    action_stats = ckpt.get("action_stats", {}) if isinstance(ckpt, dict) else {}
    action_mean = torch.tensor(action_stats.get("mean", [0.0] * cfg.action_dim), dtype=torch.float32)
    action_std = torch.tensor(action_stats.get("std", [1.0] * cfg.action_dim), dtype=torch.float32)
    normalized_targets = bool(action_stats.get("normalized_targets", False))

    processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolve_text_tokenizer_name(cfg))

    image = Image.open(args.image).convert("RGB").resize((cfg.image_size, cfg.image_size))
    image_inputs = processor(images=[image], return_tensors="pt")
    text_inputs = tokenizer(
        [args.instruction],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(text_inputs["input_ids"])

    with torch.no_grad():
        k = max(1, int(cfg.temporal_context))
        geom_dim = int(cfg.geometry_dim)
        if args.state.strip():
            state_vals = [float(x.strip()) for x in args.state.split(",") if x.strip()]
            cur = fit_geometry_dim(compute_geometry_features_from_state(state_vals), geom_dim)
            geom = torch.zeros((k * geom_dim,), dtype=torch.float32)
            geom[-geom_dim:] = cur
            geometry_features = geom.unsqueeze(0).to(device)
        else:
            geometry_features = torch.zeros((1, k * geom_dim), dtype=torch.float32, device=device)

        pred = model(
            pixel_values=image_inputs["pixel_values"].to(device),
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
            geometry_features=geometry_features,
        )

    pred_action = pred[0].detach().cpu()

    # If targets were normalized during training, map back to action space.
    if normalized_targets:
        pred_action = pred_action * action_std + action_mean
    print("pred_action:", pred_action.tolist())


if __name__ == "__main__":
    main()
