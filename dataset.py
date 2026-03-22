from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle
import random

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class VLASample:
    image_path: str
    instruction: str
    action: list[float]


class VLAJsonlDataset(Dataset):
    """JSONL format:
    {"image_path": "...", "instruction": "push block", "action": [..float..]}
    """

    def __init__(self, jsonl_path: str | Path, action_dim: int | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.samples: list[VLASample] = []
        self.action_dim = action_dim

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
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        p = Path(s.image_path)
        if not p.is_absolute():
            p = self.root / p

        image = Image.open(p).convert("RGB")
        action = torch.tensor(s.action, dtype=torch.float32)
        if self.action_dim is not None and action.numel() != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.numel()} at idx={idx}")

        return {
            "image": image,
            "instruction": s.instruction,
            "action": action,
        }


class ShortMetaWorldDataset(Dataset):
    """Loader for folder-based short-MetaWorld data.

    Expected files under data_root:
    - mt50_task_prompts.json
    - short-MetaWorld/short-MetaWorld/img_only/<task>/<traj>/<step>.jpg
    - short-MetaWorld/r3m-processed/r3m_MT10_20/<task>.pkl

    Each sample is a single step with keys:
    image (PIL), instruction (str), action (torch.float32 tensor)
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        val_ratio: float = 0.1,
        seed: int = 42,
        action_dim: int = 4,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.action_dim = action_dim

        if split not in {"train", "val"}:
            raise ValueError(f"split must be train/val, got: {split}")

        self.prompts = self._load_prompts()
        self.img_root, self.pkl_root = self._resolve_data_paths()

        all_samples = self._index_all_samples()
        self.samples = self._split_samples_by_trajectory(all_samples, split, val_ratio, seed)

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
                if traj_idx >= len(actions_all):
                    continue
                if traj_idx >= len(states_all):
                    continue

                img_paths = sorted(traj_dir.glob("*.jpg"), key=lambda p: int(p.stem))
                action_seq = actions_all[traj_idx]
                state_seq = states_all[traj_idx]

                min_steps = min(len(img_paths), len(action_seq), len(state_seq))
                if min_steps < 1:
                    continue

                for step_idx in range(min_steps):
                    action = action_seq[step_idx]
                    samples.append(
                        {
                            "task_name": task_name,
                            "trajectory_id": traj_idx,
                            "step_id": step_idx,
                            "image_path": str(img_paths[step_idx]),
                            "action": action,
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        action = torch.tensor(row["action"], dtype=torch.float32)

        if action.numel() != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action.numel()} at idx={idx}. "
                "Set action_dim to match the dataset."
            )

        return {
            "image": image,
            "instruction": row["instruction"],
            "action": action,
        }
