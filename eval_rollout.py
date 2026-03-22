from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from model import VLAFusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task success in MetaWorld rollouts.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="default_config.yaml")
    parser.add_argument("--task", type=str, default="button-press-topdown-v3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="corner2")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--clip-action", dest="clip_action", action="store_true")
    parser.add_argument("--no-clip-action", dest="clip_action", action="store_false")
    parser.set_defaults(clip_action=True)
    parser.add_argument("--record-video", type=str, default="")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace, ckpt: dict) -> TrainConfig:
    cfg_path = Path(args.config)
    cfg = TrainConfig.from_yaml(cfg_path) if cfg_path.exists() else TrainConfig()
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        cfg.apply_overrides(ckpt["cfg"])
    return cfg


def make_instruction(task: str) -> str:
    return f"Perform the task: {task.replace('-', ' ')}"


def _task_candidates(task: str) -> list[str]:
    cands = [task]
    if task.endswith("-v2"):
        cands.append(task[:-3] + "-v3")
    elif task.endswith("-v3"):
        cands.append(task[:-3] + "-v2")
    return cands


def _pick_task_obj(benchmark, env_name: str):
    tasks = list(getattr(benchmark, "train_tasks", []))
    if not tasks:
        return None

    for t in tasks:
        if getattr(t, "env_name", None) == env_name:
            return t

    return tasks[0]


def make_metaworld_env(task: str, seed: int):
    try:
        import metaworld
    except ImportError as e:
        raise ImportError("metaworld is not installed. Install it to run rollout success evaluation.") from e

    last_err: Exception | None = None
    for cand in _task_candidates(task):
        try:
            benchmark = metaworld.MT1(cand, seed=seed)
            if cand not in benchmark.train_classes:
                continue
            env_cls = benchmark.train_classes[cand]
            try:
                env = env_cls(render_mode="rgb_array")
            except TypeError:
                env = env_cls()

            env._freeze_rand_vec = False
            env.seed(seed)

            task_obj = _pick_task_obj(benchmark, cand)
            if task_obj is not None:
                env.set_task(task_obj)

            return env, cand, task_obj
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not create MetaWorld MT1 env for task '{task}'. "
        f"Tried: {_task_candidates(task)}. Last error: {last_err}"
    )


def render_rgb(env, width: int, height: int, camera: str) -> np.ndarray:
    try:
        frame = env.render(offscreen=True, camera_name=camera, resolution=(width, height))
        if frame is not None:
            return frame
    except TypeError:
        pass
    except Exception:
        pass

    try:
        frame = env.render(width=width, height=height, camera_name=camera)
        if frame is not None:
            return frame
    except Exception:
        pass

    frame = env.render()
    if frame is None:
        raise RuntimeError("Could not render RGB frame from MetaWorld env with current render API.")
    return frame


def reset_env(env, seed: int):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _init_video_writer(path: str, fps: int):
    if not path:
        return None
    try:
        import imageio.v2 as imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Force a concrete codec to avoid pyav passing None codec.
        try:
            return imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264")
        except Exception:
            return imageio.get_writer(path, fps=fps, format="FFMPEG", codec="mpeg4")
    except Exception as e:
        print(f"video disabled: {e}")
        return None


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = load_cfg(args, ckpt)

    action_stats = ckpt.get("action_stats", {}) if isinstance(ckpt, dict) else {}
    action_mean = np.array(action_stats.get("mean", [0.0] * cfg.action_dim), dtype=np.float32)
    action_std = np.array(action_stats.get("std", [1.0] * cfg.action_dim), dtype=np.float32)
    normalized_targets = bool(action_stats.get("normalized_targets", False))

    model = VLAFusionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    env, resolved_task, task_obj = make_metaworld_env(args.task, args.seed)
    instruction = make_instruction(resolved_task)
    print(f"Requested task: {args.task} | Using task: {resolved_task}")

    video_writer = _init_video_writer(args.record_video, args.fps)

    successes = 0

    for ep in range(args.episodes):
        if task_obj is not None:
            env.set_task(task_obj)

        _, _ = reset_env(env, args.seed + ep)
        ep_success = False

        for _ in range(args.max_steps):
            frame = render_rgb(env, args.width, args.height, args.camera)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(frame).convert("RGB").resize((cfg.image_size, cfg.image_size))

            image_inputs = processor(images=[pil_image], return_tensors="pt")
            text_inputs = tokenizer(
                [instruction],
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

            action = pred[0].detach().cpu().numpy()
            if normalized_targets:
                action = action * action_std + action_mean
            if args.clip_action:
                action = np.clip(action, -1.0, 1.0)

            step_out = env.step(action)
            if len(step_out) == 5:
                _, _, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                _, _, done, info = step_out

            if info.get("success", 0.0) > 0.0:
                ep_success = True

            if video_writer is not None:
                video_writer.append_data(frame)

            if done:
                break

        successes += int(ep_success)
        print(f"episode {ep + 1}/{args.episodes} success={int(ep_success)}")

    if video_writer is not None:
        video_writer.close()

    success_rate = successes / max(args.episodes, 1)
    print(f"task={resolved_task} episodes={args.episodes} success_rate={success_rate:.4f}")


if __name__ == "__main__":
    main()
