from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

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

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.samples: list[VLASample] = []

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

        return {
            "image": image,
            "instruction": s.instruction,
            "action": action,
        }
