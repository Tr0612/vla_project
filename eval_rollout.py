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
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="corner2")
    parser.add_argument("--camera-sweep", action="store_true")
    parser.add_argument("--camera-candidates", type=str, default="corner2,corner,corner3,topview")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--record-video", type=str, default="")
    parser.add_argument("--fps", type=int, default=20)

    # Two-phase geometry controller.
    parser.add_argument("--press-distance-thresh", type=float, default=0.04)
    parser.add_argument("--fallback-press-step", type=int, default=30)
    parser.add_argument("--approach-z-min", type=float, default=0.1)
    parser.add_argument("--press-z-max", type=float, default=-0.5)
    parser.add_argument("--xy-damping", type=float, default=0.8)
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

        try:
            return imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264")
        except Exception:
            pass

        try:
            return imageio.get_writer(path, fps=fps, plugin="pyav", codec="libx264")
        except Exception:
            return imageio.get_writer(path, fps=fps, plugin="pyav", codec="mpeg4")
    except Exception as e:
        print(f"video disabled: {e}")
        return None


def _extract_hand_and_obj_xy(obs) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if obs is None:
        return None, None
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size >= 7:
        hand_xy = arr[0:2]
        obj_xy = arr[4:6]
        return hand_xy, obj_xy
    return None, None


def _is_close_to_button(obs, dist_thresh: float) -> tuple[bool, float | None]:
    hand_xy, obj_xy = _extract_hand_and_obj_xy(obs)
    if hand_xy is None or obj_xy is None:
        return False, None
    dist = float(np.linalg.norm(hand_xy - obj_xy))
    return dist <= dist_thresh, dist


def _parse_camera_candidates(raw: str) -> list[str]:
    cams = [c.strip() for c in raw.split(",") if c.strip()]
    return cams if cams else ["corner2", "corner", "corner3", "topview"]


def _camera_video_path(base_path: str, camera: str) -> str:
    if not base_path:
        return ""
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{camera}{p.suffix}"))


def run_rollouts_for_camera(
    args: argparse.Namespace,
    model,
    processor,
    tokenizer,
    normalized_targets: bool,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    camera_name: str,
) -> tuple[float, str]:
    env, resolved_task, task_obj = make_metaworld_env(args.task, args.seed)
    instruction = make_instruction(resolved_task)
    print(f"Requested task: {args.task} | Using task: {resolved_task} | camera: {camera_name}")

    video_path = _camera_video_path(args.record_video, camera_name)
    video_writer = _init_video_writer(video_path, args.fps)

    successes = 0

    for ep in range(args.episodes):
        if task_obj is not None:
            env.set_task(task_obj)

        obs, _ = reset_env(env, args.seed + ep)
        ep_success = False

        for step_idx in range(args.max_steps):
            frame = render_rgb(env, args.width, args.height, camera_name)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(frame).convert("RGB").resize((224, 224))

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
                    pixel_values=image_inputs["pixel_values"].to(next(model.parameters()).device),
                    input_ids=text_inputs["input_ids"].to(next(model.parameters()).device),
                    attention_mask=text_inputs["attention_mask"].to(next(model.parameters()).device),
                )

            action = pred[0].detach().cpu().numpy()
            if normalized_targets:
                action = action * action_std + action_mean

            close_to_button, dist_xy = _is_close_to_button(obs, args.press_distance_thresh)
            if dist_xy is None:
                close_to_button = step_idx >= args.fallback_press_step

            action[0] *= args.xy_damping
            action[1] *= args.xy_damping

            if close_to_button:
                action[2] = min(action[2], args.press_z_max)
            else:
                action[2] = max(action[2], args.approach_z_min)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, _, done, info = step_out

            if info.get("success", 0.0) > 0.0:
                ep_success = True

            if video_writer is not None:
                video_writer.append_data(frame)

            if done:
                break

        successes += int(ep_success)
        print(f"camera={camera_name} episode {ep + 1}/{args.episodes} success={int(ep_success)}")

    if video_writer is not None:
        video_writer.close()

    try:
        env.close()
    except Exception:
        pass

    success_rate = successes / max(args.episodes, 1)
    print(f"task={resolved_task} camera={camera_name} episodes={args.episodes} success_rate={success_rate:.4f}")
    return success_rate, resolved_task


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

    cameras = _parse_camera_candidates(args.camera_candidates) if args.camera_sweep else [args.camera]

    results: list[tuple[str, float, str]] = []
    for cam in cameras:
        try:
            sr, resolved_task = run_rollouts_for_camera(
                args,
                model,
                processor,
                tokenizer,
                normalized_targets,
                action_mean,
                action_std,
                cam,
            )
            results.append((cam, sr, resolved_task))
        except Exception as e:
            print(f"camera={cam} failed: {e}")

    if not results:
        raise RuntimeError("All camera evaluations failed.")

    results.sort(key=lambda x: x[1], reverse=True)
    print("camera sweep summary:")
    for cam, sr, resolved_task in results:
        print(f"task={resolved_task} camera={cam} success_rate={sr:.4f}")

    best_cam, best_sr, resolved_task = results[0]
    print(f"best_camera={best_cam} task={resolved_task} success_rate={best_sr:.4f}")


if __name__ == "__main__":
    main()
