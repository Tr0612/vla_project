from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EpisodeBuffer:
    actions: list[list[float]]
    states: list[list[float]]
    success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect button-press-topdown-v3 demonstrations and save in short-MetaWorld-like format."
    )
    parser.add_argument("--out-root", type=str, default="data/button-press-topdown-v3")
    parser.add_argument("--trajectories", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="corner2")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--save-prompts", action="store_true")
    return parser.parse_args()


def make_env(seed: int):
    import metaworld

    task_name = "button-press-topdown-v3"
    benchmark = metaworld.MT1(task_name, seed=seed)
    env_cls = benchmark.train_classes[task_name]

    try:
        env = env_cls(render_mode="rgb_array")
    except TypeError:
        env = env_cls()

    env._freeze_rand_vec = False
    env.seed(seed)

    task_obj = None
    for t in benchmark.train_tasks:
        if getattr(t, "env_name", None) == task_name:
            task_obj = t
            break
    if task_obj is None and benchmark.train_tasks:
        task_obj = benchmark.train_tasks[0]

    if task_obj is not None:
        env.set_task(task_obj)

    return env, task_obj, task_name


def reset_env(env, seed: int):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


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
        raise RuntimeError("Could not render frame from environment.")
    return frame


def collect_one_episode(
    env,
    task_obj,
    policy,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    frame_stride: int,
    seed: int,
    traj_dir: Path,
) -> EpisodeBuffer:
    if task_obj is not None:
        env.set_task(task_obj)

    obs, _ = reset_env(env, seed)

    actions: list[list[float]] = []
    states: list[list[float]] = []
    ep_success = False

    traj_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, frame_stride)
    saved_step_count = 0

    for step_idx in range(max_steps):
        frame = render_rgb(env, width, height, camera)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        action = policy.get_action(obs).astype(np.float32)

        # Save fewer, non-redundant frames and keep action/state aligned.
        if step_idx % frame_stride == 0:
            Image.fromarray(frame).convert("RGB").save(traj_dir / f"{saved_step_count}.jpg", quality=95)
            states.append(np.asarray(obs, dtype=np.float32).tolist())
            actions.append(action.tolist())
            saved_step_count += 1

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step_out

        if float(info.get("success", 0.0)) > 0.0:
            ep_success = True

        if done:
            break

    return EpisodeBuffer(actions=actions, states=states, success=ep_success)


def main() -> None:
    args = parse_args()

    try:
        from metaworld.policies import SawyerButtonPressTopdownV3Policy
    except Exception as e:
        raise ImportError(
            "Could not import SawyerButtonPressTopdownV3Policy. Check metaworld installation."
        ) from e

    out_root = Path(args.out_root)
    img_root = out_root / "short-MetaWorld" / "img_only" / "button-press-topdown-v3"
    pkl_root = out_root / "short-MetaWorld" / "r3m-processed" / "r3m_MT10_20"

    img_root.mkdir(parents=True, exist_ok=True)
    pkl_root.mkdir(parents=True, exist_ok=True)

    env, task_obj, task_name = make_env(args.seed)
    policy = SawyerButtonPressTopdownV3Policy()

    all_actions: list[list[list[float]]] = []
    all_states: list[list[list[float]]] = []

    successes = 0
    attempts = 0

    while successes < args.trajectories and attempts < args.max_attempts:
        attempt_seed = args.seed + attempts
        tmp_dir = img_root / f"_tmp_attempt_{attempts}"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        ep = collect_one_episode(
            env=env,
            task_obj=task_obj,
            policy=policy,
            camera=args.camera,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            frame_stride=max(1, args.frame_stride),
            seed=attempt_seed,
            traj_dir=tmp_dir,
        )

        attempts += 1

        if ep.success and len(ep.actions) > 0 and len(ep.states) > 0:
            traj_id = successes
            final_dir = img_root / str(traj_id)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)

            all_actions.append(ep.actions)
            all_states.append(ep.states)
            successes += 1
            print(
                f"saved traj {traj_id + 1}/{args.trajectories} "
                f"(steps={len(ep.actions)}, seed={attempt_seed})"
            )
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"discarded attempt {attempts}/{args.max_attempts} (no success)")

    try:
        env.close()
    except Exception:
        pass

    if successes == 0:
        raise RuntimeError("No successful trajectories collected. Try increasing --max-attempts or --max-steps.")

    pkl_path = pkl_root / f"{task_name}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"actions": all_actions, "state": all_states}, f)

    if args.save_prompts:
        prompt_path = out_root / "mt50_task_prompts.json"
        prompt_payload = {
            task_name: {
                "simple": "Press the top-down button",
                "detailed": "Move above the button and press it down from the top.",
                "task_specific": "Approach the red button from above and press down until activated.",
            }
        }
        with prompt_path.open("w", encoding="utf-8") as f:
            json.dump(prompt_payload, f, indent=2)

    print("collection complete")
    print(f"successful trajectories: {successes}")
    print(f"attempts: {attempts}")
    print(f"images root: {img_root}")
    print(f"pkl: {pkl_path}")


if __name__ == "__main__":
    main()
