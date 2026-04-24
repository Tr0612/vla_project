from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config import TrainConfig
from dataset import compute_geometry_features_from_state, fit_geometry_dim
from model import VLAFusionPolicy


def resolve_text_tokenizer_name(cfg: TrainConfig) -> str:
    return cfg.text_model_name if cfg.separate_backbones else cfg.vision_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task success in MetaWorld rollouts.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="default_config.yaml")
    parser.add_argument("--task", type=str, default="button-press-topdown-v3")
    parser.add_argument("--instruction", type=str, default="", help="Optional language override for rollout.")
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
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-image-var", action="store_true")
    parser.add_argument("--debug-every", type=int, default=1)
    parser.add_argument(
        "--action-quant-step",
        type=float,
        default=0.0,
        help="Optional post-policy action quantization step size (0 disables). Useful for tokenization/coarsening ablations.",
    )
    parser.add_argument(
        "--action-quant-bins",
        type=int,
        default=0,
        help="Optional uniform quantization bins over [-1,1] (0 disables). Useful for tokenization/coarsening ablations.",
    )
    parser.add_argument(
        "--enable-safety-check",
        action="store_true",
        help="Verify target object exists before acting; abort episode if not detected.",
    )
    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=0.20,
        help="Minimum vision-text cosine score to allow execution.",
    )
    parser.add_argument(
        "--safety-object",
        type=str,
        default="",
        help="Optional manual object name for safety check (e.g., drawer, door, button).",
    )

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


def _load_task_prompts(data_root: str) -> dict:
    candidates = [
        Path(data_root) / "mt50_task_prompts.json",
        Path("data/short-metaworld-vla/mt50_task_prompts.json"),
        Path("data/short-metaworld-v3/mt50_task_prompts.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    prompts = json.load(f)
                print(f"Loaded task prompts: {p}")
                return prompts
            except Exception as e:
                print(f"Failed loading prompts from {p}: {e}")
    print("Task prompts not found; using fallback instruction template.")
    return {}


def make_instruction(task: str, task_prompts: dict | None = None) -> str:
    prompts = task_prompts or {}
    info = prompts.get(task)
    if isinstance(info, dict):
        text = info.get("simple")
        if isinstance(text, str) and text.strip():
            return text.strip()

    # Try counterpart naming if task prompt exists only in v2/v3 variant.
    if task.endswith("-v2"):
        info = prompts.get(task[:-3] + "-v3")
    elif task.endswith("-v3"):
        info = prompts.get(task[:-3] + "-v2")
    else:
        info = None

    if isinstance(info, dict):
        text = info.get("simple")
        if isinstance(text, str) and text.strip():
            return text.strip()

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
            return imageio.get_writer(path, fps=fps, codec="libx264")
        except Exception:
            return imageio.get_writer(path, fps=fps, codec="mpeg4")
    except Exception as e:
        print(f"video disabled: {e}")
        return None


def _append_video_frame(video_writer, frame):
    if video_writer is None:
        return None
    try:
        video_writer.append_data(frame)
        return video_writer
    except Exception as e:
        print(f"video disabled during write: {e}")
        try:
            video_writer.close()
        except Exception:
            pass
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


def _extract_object_candidates(task: str, instruction: str, override: str = "") -> list[str]:
    if override.strip():
        return [override.strip().lower()]

    import re

    raw = f"{task.replace('-', ' ')} {instruction}".lower()
    tokens = re.findall(r"[a-z][a-z0-9_]*", raw)
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

    # Favor adjacent two-word noun-ish phrases first.
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 in stop or w2 in stop:
            continue
        phrase = f"{w1} {w2}"
        if phrase not in seen:
            seen.add(phrase)
            candidates.append(phrase)

    # Then fallback to single-token candidates.
    for w in tokens:
        if w in stop or len(w) < 3:
            continue
        if w not in seen:
            seen.add(w)
            candidates.append(w)

    return candidates


def _compute_presence_score(
    model,
    processor,
    tokenizer,
    pil_image: Image.Image,
    object_name: str,
    cfg: TrainConfig,
    device: torch.device,
) -> float:
    prompt = f"a robot scene containing a {object_name}"
    image_inputs = processor(images=[pil_image], return_tensors="pt")
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
        pixel_values = image_inputs["pixel_values"].to(device)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = attention_mask.to(device)

        vision_out = model.vision(pixel_values=pixel_values)
        text_out = model.text(input_ids=input_ids, attention_mask=attention_mask)
        vision_pool = model._pool_vision_output(vision_out)
        text_pool = model._pool_text_output(text_out)

        vision_feat = model.vision_proj_ln(model.vision_proj(vision_pool))
        text_feat = model.text_proj_ln(model.text_proj(text_pool))

        vision_feat = torch.nn.functional.normalize(vision_feat, p=2, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=-1)
        score = torch.sum(vision_feat * text_feat, dim=-1)[0]
    return float(score.detach().cpu().item())


def _select_present_object(
    model,
    processor,
    tokenizer,
    pil_image: Image.Image,
    candidates: list[str],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[str | None, float]:
    if not candidates:
        return None, float("-inf")
    best_obj = None
    best_score = float("-inf")
    for obj in candidates:
        s = _compute_presence_score(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            pil_image=pil_image,
            object_name=obj,
            cfg=cfg,
            device=device,
        )
        if s > best_score:
            best_score = s
            best_obj = obj
    return best_obj, best_score


def _build_temporal_geometry(history: deque[torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
    k = max(1, int(cfg.temporal_context))
    geom_dim = int(cfg.geometry_dim)
    out = torch.zeros(k * geom_dim, dtype=torch.float32)
    recent = list(history)[-k:]
    start_slot = k - len(recent)
    for i, g in enumerate(recent):
        s = (start_slot + i) * geom_dim
        out[s : s + geom_dim] = fit_geometry_dim(g, geom_dim)
    return out


def _quantize_action(action: np.ndarray, quant_step: float, quant_bins: int) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if quant_step > 0:
        out = np.round(out / quant_step) * quant_step
    if quant_bins and quant_bins > 1:
        delta = 2.0 / float(quant_bins - 1)
        out = np.clip(out, -1.0, 1.0)
        out = np.round((out + 1.0) / delta) * delta - 1.0
    return out.astype(np.float32)


def run_rollouts_for_camera(
    args: argparse.Namespace,
    cfg: TrainConfig,
    model,
    processor,
    tokenizer,
    normalized_targets: bool,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    camera_name: str,
    task_prompts: dict,
) -> tuple[float, str, dict[str, float]]:
    env, resolved_task, task_obj = make_metaworld_env(args.task, args.seed)
    instruction = args.instruction.strip() if args.instruction.strip() else make_instruction(resolved_task, task_prompts)
    print(f"Requested task: {args.task} | Using task: {resolved_task} | camera: {camera_name}")
    print(f"instruction: {instruction}")

    video_path = _camera_video_path(args.record_video, camera_name)
    video_writer = _init_video_writer(video_path, args.fps)

    successes = 0
    safety_aborts = 0
    sum_ep_max_success = 0.0
    sum_ep_max_reward = 0.0
    sum_ep_min_obj_to_target = 0.0
    count_ep_min_obj_to_target = 0
    device = next(model.parameters()).device
    object_candidates = _extract_object_candidates(resolved_task, instruction, args.safety_object)
    if args.enable_safety_check:
        if not object_candidates:
            print(
                "safety_check=enabled but target object is unknown; "
                "provide --safety-object to force explicit object verification."
            )
        else:
            print(
                f"safety_check=enabled candidates={object_candidates[:8]} "
                f"threshold={float(args.safety_threshold):.3f}"
            )

    for ep in range(args.episodes):
        if task_obj is not None:
            env.set_task(task_obj)

        obs, _ = reset_env(env, args.seed + ep)
        geometry_history: deque[torch.Tensor] = deque(maxlen=max(1, int(cfg.temporal_context)))
        ep_success = False
        ep_max_success = 0.0
        ep_max_reward = float("-inf")
        ep_min_obj_to_target = float("inf")

        if args.enable_safety_check and object_candidates:
            pre_frame = render_rgb(env, args.width, args.height, camera_name)
            if pre_frame.dtype != np.uint8:
                pre_frame = np.clip(pre_frame, 0, 255).astype(np.uint8)
            pre_image = Image.fromarray(pre_frame).convert("RGB").resize((cfg.image_size, cfg.image_size))
            target_object, presence_score = _select_present_object(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                pil_image=pre_image,
                candidates=object_candidates,
                cfg=cfg,
                device=device,
            )
            if presence_score < float(args.safety_threshold):
                safety_aborts += 1
                print(
                    f"camera={camera_name} episode {ep + 1}/{args.episodes} "
                    f"SAFETY_ABORT object={target_object} score={presence_score:.4f} "
                    f"threshold={float(args.safety_threshold):.4f} message='No {target_object} detected; exiting.'"
                )
                if video_writer is not None:
                    video_writer = _append_video_frame(video_writer, pre_frame)
                continue

        for step_idx in range(args.max_steps):
            frame = render_rgb(env, args.width, args.height, camera_name)
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
            attention_mask = text_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(text_inputs["input_ids"])

            with torch.no_grad():
                pixel_values = image_inputs["pixel_values"].to(device)
                input_ids = text_inputs["input_ids"].to(device)
                attention_mask_device = attention_mask.to(device)
                cur_geom = compute_geometry_features_from_state(obs)
                geometry_history.append(fit_geometry_dim(cur_geom, int(cfg.geometry_dim)))
                geometry_features = _build_temporal_geometry(geometry_history, cfg).unsqueeze(0).to(device)

                pred = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask_device,
                    geometry_features=geometry_features,
                )

            if pred.ndim == 3:
                action = pred[0, 0].detach().cpu().numpy()
            else:
                action = pred[0].detach().cpu().numpy()
            if normalized_targets:
                action = action * action_std + action_mean
            action = _quantize_action(
                action=action,
                quant_step=float(max(0.0, args.action_quant_step)),
                quant_bins=int(max(0, args.action_quant_bins)),
            )

            if args.print_actions and (step_idx % max(1, args.debug_every) == 0):
                action_list = [round(float(x), 5) for x in action.tolist()]
                print(f"camera={camera_name} ep={ep + 1} step={step_idx} action={action_list}")

            if args.print_image_var and (step_idx % max(1, args.debug_every) == 0):
                with torch.no_grad():
                    vision_out = model.vision(pixel_values=pixel_values)
                    vision_pool = model._pool_vision_output(vision_out).detach().float()
                    vision_proj = model.vision_proj_ln(model.vision_proj(vision_pool)).detach().float()
                    pool_var = float(vision_pool.var(dim=-1, unbiased=False).mean().item())
                    proj_var = float(vision_proj.var(dim=-1, unbiased=False).mean().item())
                print(
                    f"camera={camera_name} ep={ep + 1} step={step_idx} "
                    f"vision_var_raw={pool_var:.8f} vision_var_proj={proj_var:.8f}"
                )

            close_to_button, dist_xy = _is_close_to_button(obs, args.press_distance_thresh)
            if dist_xy is None:
                close_to_button = step_idx >= args.fallback_press_step

            # action[0] *= args.xy_damping
            # action[1] *= args.xy_damping

            # if close_to_button:
            #     action[2] = min(action[2], args.press_z_max)
            # else:
            #     action[2] = max(action[2], args.approach_z_min)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            step_success = float(info.get("success", 0.0))
            ep_max_success = max(ep_max_success, step_success)
            ep_max_reward = max(ep_max_reward, float(reward))
            if "obj_to_target" in info:
                try:
                    ep_min_obj_to_target = min(ep_min_obj_to_target, float(info["obj_to_target"]))
                except Exception:
                    pass

            if step_success > 0.0:
                ep_success = True

            if video_writer is not None:
                video_writer = _append_video_frame(video_writer, frame)

            if done:
                break

        successes += int(ep_success)
        sum_ep_max_success += float(ep_max_success)
        sum_ep_max_reward += float(ep_max_reward)
        if np.isfinite(ep_min_obj_to_target):
            sum_ep_min_obj_to_target += float(ep_min_obj_to_target)
            count_ep_min_obj_to_target += 1
        extra = (
            f" max_success={ep_max_success:.3f} max_reward={ep_max_reward:.4f}"
            + (
                f" min_obj_to_target={ep_min_obj_to_target:.4f}"
                if np.isfinite(ep_min_obj_to_target)
                else ""
            )
        )
        print(f"camera={camera_name} episode {ep + 1}/{args.episodes} success={int(ep_success)}{extra}")

    if video_writer is not None:
        video_writer.close()

    try:
        env.close()
    except Exception:
        pass

    success_rate = successes / max(args.episodes, 1)
    avg_ep_max_success = sum_ep_max_success / max(args.episodes, 1)
    avg_ep_max_reward = sum_ep_max_reward / max(args.episodes, 1)
    avg_ep_min_obj_to_target = (
        sum_ep_min_obj_to_target / max(count_ep_min_obj_to_target, 1)
        if count_ep_min_obj_to_target > 0
        else float("nan")
    )
    print(
        f"task={resolved_task} camera={camera_name} episodes={args.episodes} "
        f"success_rate={success_rate:.4f} safety_aborts={safety_aborts} "
        f"avg_ep_max_success={avg_ep_max_success:.4f} avg_ep_max_reward={avg_ep_max_reward:.4f} "
        f"avg_ep_min_obj_to_target={avg_ep_min_obj_to_target:.4f}"
    )
    return success_rate, resolved_task, {
        "safety_aborts": float(safety_aborts),
        "avg_ep_max_success": float(avg_ep_max_success),
        "avg_ep_max_reward": float(avg_ep_max_reward),
        "avg_ep_min_obj_to_target": float(avg_ep_min_obj_to_target),
    }


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
    tokenizer = AutoTokenizer.from_pretrained(resolve_text_tokenizer_name(cfg))
    task_prompts = _load_task_prompts(cfg.data_root)

    cameras = _parse_camera_candidates(args.camera_candidates) if args.camera_sweep else [args.camera]

    results: list[tuple[str, float, str, dict[str, float]]] = []
    for cam in cameras:
        try:
            sr, resolved_task, stats = run_rollouts_for_camera(
                args,
                cfg,
                model,
                processor,
                tokenizer,
                normalized_targets,
                action_mean,
                action_std,
                cam,
                task_prompts,
            )
            results.append((cam, sr, resolved_task, stats))
        except Exception as e:
            print(f"camera={cam} failed: {e}")

    if not results:
        raise RuntimeError("All camera evaluations failed.")

    results.sort(key=lambda x: x[1], reverse=True)
    print("camera sweep summary:")
    for cam, sr, resolved_task, stats in results:
        print(
            f"task={resolved_task} camera={cam} success_rate={sr:.4f} "
            f"safety_aborts={int(stats.get('safety_aborts', 0.0))} "
            f"avg_ep_max_success={float(stats.get('avg_ep_max_success', float('nan'))):.4f} "
            f"avg_ep_max_reward={float(stats.get('avg_ep_max_reward', float('nan'))):.4f} "
            f"avg_ep_min_obj_to_target={float(stats.get('avg_ep_min_obj_to_target', float('nan'))):.4f}"
        )

    best_cam, best_sr, resolved_task, best_stats = results[0]
    print(
        f"best_camera={best_cam} task={resolved_task} success_rate={best_sr:.4f} "
        f"safety_aborts={int(best_stats.get('safety_aborts', 0.0))} "
        f"avg_ep_max_success={float(best_stats.get('avg_ep_max_success', float('nan'))):.4f} "
        f"avg_ep_max_reward={float(best_stats.get('avg_ep_max_reward', float('nan'))):.4f} "
        f"avg_ep_min_obj_to_target={float(best_stats.get('avg_ep_min_obj_to_target', float('nan'))):.4f}"
    )


if __name__ == "__main__":
    main()
