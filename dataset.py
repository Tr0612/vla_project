from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import pickle
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def compute_geometry_features_from_state(state, eps: float = 1e-6) -> torch.Tensor:
    s = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if s.numel() >= 7:
        ee_pos = s[0:3]
        obj_pos = s[4:7]
    else:
        ee_pos = torch.zeros(3, dtype=torch.float32)
        obj_pos = torch.zeros(3, dtype=torch.float32)

    goal_pos = s[-3:] if s.numel() >= 3 else torch.zeros(3, dtype=torch.float32)

    ee_to_obj = obj_pos - ee_pos
    obj_to_goal = goal_pos - obj_pos

    ee_to_obj = ee_to_obj / torch.clamp(torch.linalg.norm(ee_to_obj), min=eps)
    obj_to_goal = obj_to_goal / torch.clamp(torch.linalg.norm(obj_to_goal), min=eps)
    return torch.cat([ee_to_obj, obj_to_goal], dim=0)


def fit_geometry_dim(geom: torch.Tensor, geometry_dim: int) -> torch.Tensor:
    g = geom.to(dtype=torch.float32).reshape(-1)
    if g.numel() == geometry_dim:
        return g
    if g.numel() > geometry_dim:
        return g[:geometry_dim]
    out = torch.zeros(geometry_dim, dtype=torch.float32)
    out[: g.numel()] = g
    return out


def pil_to_chw_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return tensor


def sample_get_image(sample: dict[str, Any]) -> Any:
    obs = sample.get("obs")
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"]
    return sample["image"]


def sample_get_state(sample: dict[str, Any]) -> torch.Tensor:
    obs = sample.get("obs")
    if isinstance(obs, dict) and "state" in obs:
        return torch.as_tensor(obs["state"], dtype=torch.float32).reshape(-1)
    if "geometry" in sample:
        return torch.as_tensor(sample["geometry"], dtype=torch.float32).reshape(-1)
    return torch.zeros(0, dtype=torch.float32)


def sample_get_prompt(sample: dict[str, Any]) -> str:
    task = sample.get("task")
    if isinstance(task, dict) and "prompt" in task:
        return str(task["prompt"])
    return str(sample["instruction"])


def sample_get_task_id(sample: dict[str, Any]) -> int:
    task = sample.get("task")
    if isinstance(task, dict):
        return int(task.get("task_id", -1))
    return -1


def sample_get_dataset_name(sample: dict[str, Any]) -> str:
    meta = sample.get("meta")
    if isinstance(meta, dict) and "dataset" in meta:
        return str(meta["dataset"])
    return str(sample.get("task_name", "unknown"))


def sample_get_action(sample: dict[str, Any]) -> torch.Tensor:
    return torch.as_tensor(sample["action"], dtype=torch.float32).reshape(-1)


class UnifiedSampleAdapter(Dataset):
    """Plugin interface for converting any dataset to the unified single-step schema."""

    dataset_name: str = "unknown"

    def to_unified_schema(self, raw: Any, idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def get_action(self, idx: int) -> torch.Tensor:
        return sample_get_action(self[idx])


class RLBenchAdapter(UnifiedSampleAdapter):
    """
    Minimal RLBench adapter scaffold.

    Pass records where each item contains enough data for `load_fn(record)` to return a raw sample.
    """

    dataset_name = "rlbench"

    def __init__(
        self,
        records: list[Any],
        load_fn: Callable[[Any], dict[str, Any]] | None = None,
        action_dim: int | None = None,
    ):
        self.records = records
        self.load_fn = load_fn if load_fn is not None else (lambda x: x)
        self.action_dim = action_dim

    def __len__(self) -> int:
        return len(self.records)

    def load_rlbench(self, idx: int) -> dict[str, Any]:
        return self.load_fn(self.records[idx])

    def to_unified_schema(self, raw: dict[str, Any], idx: int) -> dict[str, Any]:
        image_raw = raw.get("image")
        if isinstance(image_raw, Image.Image):
            image_pil = image_raw.convert("RGB")
            image_tensor = pil_to_chw_tensor(image_pil)
            legacy_image = image_pil
        elif torch.is_tensor(image_raw):
            image_tensor = image_raw.to(dtype=torch.float32)
            legacy_image = image_raw
        else:
            image_path = raw.get("image_path")
            if image_path is None:
                raise ValueError("RLBench sample must provide `image`, `image_path`, or custom `load_fn` output.")
            image_pil = Image.open(str(image_path)).convert("RGB")
            image_tensor = pil_to_chw_tensor(image_pil)
            legacy_image = image_pil

        state = torch.as_tensor(raw.get("state", []), dtype=torch.float32).reshape(-1)
        action = torch.as_tensor(raw.get("action", []), dtype=torch.float32).reshape(-1)
        if self.action_dim is not None and action.numel() != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.numel()} at idx={idx}")

        prompt = str(raw.get("prompt", raw.get("instruction", "execute task")))
        task_id = int(raw.get("task_id", -1))
        episode_id = str(raw.get("episode_id", f"episode_{idx}"))
        timestep = int(raw.get("timestep", 0))

        return {
            "obs": {
                "image": image_tensor,
                "state": state,
            },
            "task": {
                "prompt": prompt,
                "task_id": task_id,
            },
            "action": action,
            "meta": {
                "dataset": self.dataset_name,
                "episode_id": episode_id,
                "timestep": timestep,
            },
            # Legacy compatibility keys used by existing training/eval code paths.
            "image": legacy_image,
            "instruction": prompt,
            "geometry": state,
            "task_name": self.dataset_name,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self.load_rlbench(idx)
        return self.to_unified_schema(raw, idx)


@dataclass
class VLASample:
    image_path: str
    instruction: str
    action: list[float]
    task_id: int = -1
    episode_id: str = ""
    timestep: int = 0
    state: list[float] | None = None


class VLAJsonlDataset(UnifiedSampleAdapter):
    """JSONL format:
    {"image_path": "...", "instruction": "push block", "action": [..float..]}
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        action_dim: int | None = None,
        geometry_dim: int = 6,
        temporal_context: int = 1,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.samples: list[VLASample] = []
        self.action_dim = action_dim
        self.geometry_dim = int(max(1, geometry_dim))
        self.temporal_context = int(max(1, temporal_context))
        self.dataset_name = "jsonl"

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                self.samples.append(
                    VLASample(
                        image_path=raw["image_path"],
                        instruction=raw["instruction"],
                        action=raw["action"],
                        task_id=int(raw.get("task_id", -1)),
                        episode_id=str(raw.get("episode_id", "")),
                        timestep=int(raw.get("timestep", 0)),
                        state=raw.get("state"),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def get_action(self, idx: int) -> torch.Tensor:
        a = torch.tensor(self.samples[idx].action, dtype=torch.float32)
        if self.action_dim is not None and a.numel() != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {a.numel()} at idx={idx}")
        return a

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        p = Path(s.image_path)
        if not p.is_absolute():
            p = self.root / p

        image = Image.open(p).convert("RGB")
        image_tensor = pil_to_chw_tensor(image)
        action = self.get_action(idx)
        state = s.state
        if state is None:
            state_tensor = torch.zeros(self.geometry_dim * self.temporal_context, dtype=torch.float32)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        episode_id = s.episode_id if s.episode_id else f"jsonl_{idx}"

        return {
            "obs": {
                "image": image_tensor,
                "state": state_tensor,
            },
            "task": {
                "prompt": s.instruction,
                "task_id": int(s.task_id),
            },
            "action": action,
            "meta": {
                "dataset": self.dataset_name,
                "episode_id": episode_id,
                "timestep": int(s.timestep),
            },
            # Legacy compatibility keys used by existing training/eval code paths.
            "image": image,
            "instruction": s.instruction,
            "action": action,
            "geometry": state_tensor,
            "task_name": self.dataset_name,
        }


class ShortMetaWorldDataset(UnifiedSampleAdapter):
    """Loader for folder-based short-MetaWorld data."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        val_ratio: float = 0.1,
        seed: int = 42,
        action_dim: int = 4,
        geometry_dim: int = 6,
        temporal_context: int = 1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.action_dim = action_dim
        self.geometry_dim = int(max(1, geometry_dim))
        self.temporal_context = int(max(1, temporal_context))

        if split not in {"train", "val"}:
            raise ValueError(f"split must be train/val, got: {split}")

        self.prompts = self._load_prompts()
        self.img_root, self.pkl_root = self._resolve_data_paths()

        all_samples = self._index_all_samples()
        self.samples = self._split_samples_by_trajectory(all_samples, split, val_ratio, seed)
        self._geometry_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
        for row in self.samples:
            key = (row["task_name"], int(row["trajectory_id"]), int(row["step_id"]))
            geom = compute_geometry_features_from_state(row.get("state", []))
            self._geometry_by_key[key] = fit_geometry_dim(geom, self.geometry_dim)
        task_names = sorted({str(row["task_name"]) for row in self.samples})
        self._task_to_id = {task_name: i for i, task_name in enumerate(task_names)}
        self.dataset_name = "short_metaworld"

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split={split} at {self.data_root}. "
                "Check dataset paths and file availability."
            )

    def _load_prompts(self) -> dict[str, Any]:
        prompt_file = self.data_root / "mt50_task_prompts.json"
        if not prompt_file.exists():
            return {}
        with prompt_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_data_paths(self) -> tuple[Path, Path]:
        img_candidates = [
            self.data_root / "short-MetaWorld" / "short-MetaWorld" / "img_only",
            self.data_root / "short-MetaWorld" / "img_only",
        ]
        pkl_candidates = [
            self.data_root / "short-MetaWorld" / "r3m-processed" / "r3m_MT10_20",
            self.data_root / "r3m-processed" / "r3m_MT10_20",
        ]

        img_root = next((p for p in img_candidates if p.exists()), None)
        pkl_root = next((p for p in pkl_candidates if p.exists()), None)

        if img_root is None:
            raise FileNotFoundError(f"Could not find img_only directory under {self.data_root}")
        if pkl_root is None:
            raise FileNotFoundError(f"Could not find r3m_MT10_20 directory under {self.data_root}")

        return img_root, pkl_root

    def _get_prompt(self, task_name: str) -> str:
        info = self.prompts.get(task_name)
        if isinstance(info, dict):
            return info.get("simple", f"Perform the task: {task_name.replace('-', ' ')}")
        return f"Perform the task: {task_name.replace('-', ' ')}"

    def _index_all_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        pkl_files = sorted(self.pkl_root.glob("*.pkl"))
        for pkl_file in pkl_files:
            task_name = pkl_file.stem
            task_img_dir = self.img_root / task_name
            if not task_img_dir.exists():
                continue

            with pkl_file.open("rb") as f:
                data = pickle.load(f)

            actions_all = data.get("actions", [])
            states_all = data.get("state", [])

            traj_dirs = sorted([p for p in task_img_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name))

            for traj_idx, traj_dir in enumerate(traj_dirs):
                if traj_idx >= len(actions_all) or traj_idx >= len(states_all):
                    continue

                img_paths = sorted(traj_dir.glob("*.jpg"), key=lambda p: int(p.stem))
                action_seq = actions_all[traj_idx]
                state_seq = states_all[traj_idx]

                min_steps = min(len(img_paths), len(action_seq), len(state_seq))
                if min_steps < 1:
                    continue

                for step_idx in range(min_steps):
                    samples.append(
                        {
                            "task_name": task_name,
                            "trajectory_id": traj_idx,
                            "step_id": step_idx,
                            "image_path": str(img_paths[step_idx]),
                            "action": action_seq[step_idx],
                            "state": state_seq[step_idx],
                            "instruction": self._get_prompt(task_name),
                        }
                    )

        return samples

    @staticmethod
    def _split_samples_by_trajectory(
        samples: list[dict[str, Any]],
        split: str,
        val_ratio: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        traj_keys = sorted({(s["task_name"], s["trajectory_id"]) for s in samples})
        rng = random.Random(seed)
        rng.shuffle(traj_keys)

        n_val = max(1, int(len(traj_keys) * val_ratio)) if len(traj_keys) > 1 else 0
        val_keys = set(traj_keys[:n_val])

        if split == "train":
            return [s for s in samples if (s["task_name"], s["trajectory_id"]) not in val_keys]
        return [s for s in samples if (s["task_name"], s["trajectory_id"]) in val_keys]

    def __len__(self) -> int:
        return len(self.samples)

    def get_action(self, idx: int) -> torch.Tensor:
        row = self.samples[idx]
        action = torch.tensor(row["action"], dtype=torch.float32)
        if action.numel() != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action.numel()} at idx={idx}. "
                "Set action_dim to match the dataset."
            )
        return action

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = pil_to_chw_tensor(image)
        action = self.get_action(idx)
        task_name = str(row.get("task_name", ""))
        trajectory_id = int(row.get("trajectory_id", -1))
        step_id = int(row.get("step_id", 0))

        temporal_geoms: list[torch.Tensor] = []
        for offset in range(self.temporal_context - 1, -1, -1):
            hist_step = step_id - offset
            if hist_step < 0:
                temporal_geoms.append(torch.zeros(self.geometry_dim, dtype=torch.float32))
                continue
            geom = self._geometry_by_key.get((task_name, trajectory_id, hist_step))
            if geom is None:
                geom = torch.zeros(self.geometry_dim, dtype=torch.float32)
            temporal_geoms.append(geom)
        geometry = torch.cat(temporal_geoms, dim=0)
        prompt = str(row["instruction"])
        task_id = int(self._task_to_id.get(task_name, -1))
        episode_id = f"{task_name}:{trajectory_id}"

        return {
            "obs": {
                "image": image_tensor,
                "state": geometry,
            },
            "task": {
                "prompt": prompt,
                "task_id": task_id,
            },
            "action": action,
            "meta": {
                "dataset": self.dataset_name,
                "episode_id": episode_id,
                "timestep": step_id,
            },
            # Legacy compatibility keys used by existing training/eval code paths.
            "image": image,
            "instruction": prompt,
            "action": action,
            "geometry": geometry,
            "task_name": task_name,
        }
