from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from vla_stack.config import TrainConfig
from vla_stack.dataset import compute_geometry_features_from_state, fit_geometry_dim
from vla_stack.model import VLAFusionPolicy


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
    parser.add_argument(
        "--enable-safety-check",
        action="store_true",
        help="Verify instructed object exists before returning action.",
    )
    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=0.20,
        help="Minimum vision-text cosine score to allow action output.",
    )
    parser.add_argument(
        "--safety-object",
        type=str,
        default="",
        help="Optional explicit object name for safety check (e.g., drawer, door).",
    )
    return parser.parse_args()


def _extract_object_candidates(instruction: str, override: str = "") -> list[str]:
    if override.strip():
        return [override.strip().lower()]

    import re

    tokens = re.findall(r"[a-z][a-z0-9_]*", instruction.lower())
    if not tokens:
        return []

    stop = {
        "the",
        "a",
        "an",
        "to",
        "for",
        "of",
        "on",
        "in",
        "at",
        "with",
        "and",
        "or",
        "from",
        "by",
        "robot",
        "scene",
        "task",
        "perform",
        "execute",
        "move",
        "moving",
        "reach",
        "open",
        "close",
        "pull",
        "push",
        "press",
        "insert",
        "pick",
        "place",
        "put",
        "drag",
        "gripper",
    }

    candidates: list[str] = []
    seen = set()
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 in stop or w2 in stop:
            continue
        phrase = f"{w1} {w2}"
        if phrase not in seen:
            seen.add(phrase)
            candidates.append(phrase)
    for w in tokens:
        if w in stop or len(w) < 3:
            continue
        if w not in seen:
            seen.add(w)
            candidates.append(w)
    return candidates


def _compute_presence_score(model, processor, tokenizer, image: Image.Image, object_name: str, device) -> float:
    prompt = f"a robot scene containing a {object_name}"
    image_inputs = processor(images=[image], return_tensors="pt")
    text_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(text_inputs["input_ids"])

    with torch.no_grad():
        vision_out = model.vision(pixel_values=image_inputs["pixel_values"].to(device))
        text_out = model.text(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
        )
        vision_pool = model._pool_vision_output(vision_out)
        text_pool = model._pool_text_output(text_out)
        vision_feat = model.vision_proj_ln(model.vision_proj(vision_pool))
        text_feat = model.text_proj_ln(model.text_proj(text_pool))
        vision_feat = torch.nn.functional.normalize(vision_feat, p=2, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=-1)
        score = torch.sum(vision_feat * text_feat, dim=-1)[0]
    return float(score.detach().cpu().item())


def _select_present_object(model, processor, tokenizer, image: Image.Image, candidates: list[str], device) -> tuple[str | None, float]:
    if not candidates:
        return None, float("-inf")
    best_obj = None
    best_score = float("-inf")
    for obj in candidates:
        s = _compute_presence_score(model, processor, tokenizer, image, obj, device)
        if s > best_score:
            best_score = s
            best_obj = obj
    return best_obj, best_score


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

    if args.enable_safety_check:
        object_candidates = _extract_object_candidates(args.instruction, args.safety_object)
        if not object_candidates:
            print("safety_check=enabled but target object is unknown. Use --safety-object for explicit verification.")
            return
        target_object, presence_score = _select_present_object(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image=image,
            candidates=object_candidates,
            device=device,
        )
        if presence_score < float(args.safety_threshold):
            print(
                f"SAFETY_ABORT object={target_object} score={presence_score:.4f} "
                f"threshold={float(args.safety_threshold):.4f} message='No {target_object} detected; exiting.'"
            )
            return
        print(
            f"safety_check=pass object={target_object} score={presence_score:.4f} "
            f"threshold={float(args.safety_threshold):.4f}"
        )
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

    if pred.ndim == 3:
        pred_action = pred[0, 0].detach().cpu()
    else:
        pred_action = pred[0].detach().cpu()

    # If targets were normalized during training, map back to action space.
    if normalized_targets:
        pred_action = pred_action * action_std + action_mean
    print("pred_action:", pred_action.tolist())


if __name__ == "__main__":
    main()
