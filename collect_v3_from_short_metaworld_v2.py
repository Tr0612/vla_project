from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Explicit mapping for tasks present in your short-metaworld-vla v2 set.
TASK_TO_POLICY = {
    "basketball-v3": "SawyerBasketballV3Policy",
    "button-press-topdown-v3": "SawyerButtonPressTopdownV3Policy",
    "door-open-v3": "SawyerDoorOpenV3Policy",
    "drawer-close-v3": "SawyerDrawerCloseV3Policy",
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
    "pick-place-v3": "SawyerPickPlaceV3Policy",
    "push-v3": "SawyerPushV3Policy",
    "reach-v3": "SawyerReachV3Policy",
    "sweep-v3": "SawyerSweepV3Policy",
    "window-open-v3": "SawyerWindowOpenV3Policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect v3 trajectories for v2 tasks found in short-metaworld-vla."
    )
    parser.add_argument("--source-root", type=str, default="data/short-metaworld-vla")
    parser.add_argument("--out-root", type=str, default="data/short-metaworld-v3")
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="corner2")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--tasks", type=str, default="")
    return parser.parse_args()


def resolve_source_pkl_root(source_root: Path) -> Path:
    cands = [
        source_root / "short-MetaWorld" / "r3m-processed" / "r3m_MT10_20",
        source_root / "r3m-processed" / "r3m_MT10_20",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find source r3m_MT10_20 under {source_root}")


def list_source_v2_tasks(source_root: Path) -> list[str]:
    pkl_root = resolve_source_pkl_root(source_root)
    v2_tasks = sorted([p.stem for p in pkl_root.glob("*.pkl") if p.stem.endswith("-v2")])
    return v2_tasks


def to_v3_task(v2_task: str) -> str:
    if not v2_task.endswith("-v2"):
        return v2_task
    return v2_task[:-3] + "-v3"


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
        raise RuntimeError("Could not render frame from environment")
    return frame


def reset_env(env, seed: int):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def make_env(task_v3: str, seed: int):
    import metaworld

    benchmark = metaworld.MT1(task_v3, seed=seed)
    env_cls = benchmark.train_classes[task_v3]

    try:
        env = env_cls(render_mode="rgb_array")
    except TypeError:
        env = env_cls()

    env._freeze_rand_vec = False
    env.seed(seed)

    task_obj = None
    for t in benchmark.train_tasks:
        if getattr(t, "env_name", None) == task_v3:
            task_obj = t
            break
    if task_obj is None and benchmark.train_tasks:
        task_obj = benchmark.train_tasks[0]

    if task_obj is not None:
        env.set_task(task_obj)

    return env, task_obj


def make_policy(task_v3: str):
    from metaworld import policies

    policy_name = TASK_TO_POLICY.get(task_v3)
    if policy_name is None:
        raise KeyError(
            f"No policy mapping for {task_v3}. Add it to TASK_TO_POLICY in this script."
        )

    policy_cls = getattr(policies, policy_name, None)
    if policy_cls is None:
        raise AttributeError(f"Policy class not found in metaworld.policies: {policy_name}")

    return policy_cls()


def collect_task(
    task_v3: str,
    out_img_task_dir: Path,
    trajectories: int,
    max_steps: int,
    max_attempts: int,
    frame_stride: int,
    seed: int,
    camera: str,
    width: int,
    height: int,
) -> tuple[list[list[list[float]]], list[list[list[float]]], int, int]:
    frame_stride = max(1, frame_stride)

    env, task_obj = make_env(task_v3, seed=seed)
    policy = make_policy(task_v3)

    all_actions: list[list[list[float]]] = []
    all_states: list[list[list[float]]] = []

    successes = 0
    attempts = 0

    while successes < trajectories and attempts < max_attempts:
        attempt_seed = seed + attempts
        tmp_dir = out_img_task_dir / f"_tmp_attempt_{attempts}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if task_obj is not None:
            env.set_task(task_obj)

        obs, _ = reset_env(env, attempt_seed)

        actions: list[list[float]] = []
        states: list[list[float]] = []
        ep_success = False
        saved_step_count = 0

        for step_idx in range(max_steps):
            frame = render_rgb(env, width, height, camera)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            action = policy.get_action(obs).astype(np.float32)

            if step_idx % frame_stride == 0:
                Image.fromarray(frame).convert("RGB").save(tmp_dir / f"{saved_step_count}.jpg", quality=95)
                actions.append(action.tolist())
                states.append(np.asarray(obs, dtype=np.float32).tolist())
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

        attempts += 1

        if ep_success and len(actions) > 0 and len(states) > 0:
            final_dir = out_img_task_dir / str(successes)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)

            all_actions.append(actions)
            all_states.append(states)
            successes += 1
            print(
                f"[{task_v3}] saved traj {successes}/{trajectories} "
                f"(steps={len(actions)}, seed={attempt_seed})"
            )
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[{task_v3}] discarded attempt {attempts}/{max_attempts} (no success)")

    try:
        env.close()
    except Exception:
        pass

    return all_actions, all_states, successes, attempts


def convert_prompts(source_root: Path, out_root: Path, selected_v2: list[str]) -> None:
    src_prompt = source_root / "mt50_task_prompts.json"
    if not src_prompt.exists():
        return

    with src_prompt.open("r", encoding="utf-8") as f:
        src = json.load(f)

    dst: dict[str, dict] = {}
    for v2 in selected_v2:
        v3 = to_v3_task(v2)
        if v2 in src and isinstance(src[v2], dict):
            dst[v3] = src[v2]

    if not dst:
        return

    out_prompt = out_root / "mt50_task_prompts.json"
    with out_prompt.open("w", encoding="utf-8") as f:
        json.dump(dst, f, indent=2)


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)

    source_v2 = list_source_v2_tasks(source_root)
    if not source_v2:
        raise RuntimeError(f"No *-v2 tasks found under {source_root}")

    if args.tasks.strip():
        requested_v2 = [x.strip() for x in args.tasks.split(",") if x.strip()]
        selected_v2 = [t for t in requested_v2 if t in source_v2]
        missing = [t for t in requested_v2 if t not in source_v2]
        if missing:
            print(f"warning: these tasks were not found in source and will be skipped: {missing}")
    else:
        selected_v2 = source_v2

    if not selected_v2:
        raise RuntimeError("No valid tasks selected.")

    out_img_root = out_root / "short-MetaWorld" / "img_only"
    out_pkl_root = out_root / "short-MetaWorld" / "r3m-processed" / "r3m_MT10_20"
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_pkl_root.mkdir(parents=True, exist_ok=True)

    convert_prompts(source_root, out_root, selected_v2)

    summary = []

    for v2_task in selected_v2:
        v3_task = to_v3_task(v2_task)
        if v3_task not in TASK_TO_POLICY:
            print(f"skip {v2_task} -> {v3_task}: no policy mapping")
            continue

        print(f"\n=== collecting {v2_task} -> {v3_task} ===")
        out_img_task_dir = out_img_root / v3_task
        out_img_task_dir.mkdir(parents=True, exist_ok=True)

        all_actions, all_states, successes, attempts = collect_task(
            task_v3=v3_task,
            out_img_task_dir=out_img_task_dir,
            trajectories=args.trajectories,
            max_steps=args.max_steps,
            max_attempts=args.max_attempts,
            frame_stride=args.frame_stride,
            seed=args.seed,
            camera=args.camera,
            width=args.width,
            height=args.height,
        )

        pkl_path = out_pkl_root / f"{v3_task}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump({"actions": all_actions, "state": all_states}, f)

        summary.append((v2_task, v3_task, successes, attempts, str(pkl_path)))
        print(
            f"finished {v3_task}: successes={successes}/{args.trajectories}, "
            f"attempts={attempts}/{args.max_attempts}"
        )

    print("\n=== summary ===")
    for v2_task, v3_task, successes, attempts, pkl_path in summary:
        print(
            f"{v2_task} -> {v3_task} | "
            f"successes={successes}/{args.trajectories} | "
            f"attempts={attempts}/{args.max_attempts} | pkl={pkl_path}"
        )


if __name__ == "__main__":
    main()
