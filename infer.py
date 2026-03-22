from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from model import VLAFusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)

    parser.add_argument("--clip-action", dest="clip_action", action="store_true")
    parser.add_argument("--no-clip-action", dest="clip_action", action="store_false")
    parser.set_defaults(clip_action=True)

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    image = Image.open(args.image).convert("RGB").resize((cfg.image_size, cfg.image_size))
    image_inputs = processor(images=[image], return_tensors="pt")
    text_inputs = tokenizer(
        [args.instruction],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    with torch.no_grad():
        pred = model(
            pixel_values=image_inputs["pixel_values"].to(device),
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )

    pred_action = pred[0].detach().cpu()

    # If targets were normalized during training, map back to action space.
    if normalized_targets:
        pred_action = pred_action * action_std + action_mean

    if args.clip_action:
        pred_action = pred_action.clamp(-1.0, 1.0)

    print("pred_action:", pred_action.tolist())


if __name__ == "__main__":
    main()
