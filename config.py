from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


def _parse_yaml_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""

    # Try JSON-compatible scalars first (numbers, booleans, null, quoted strings).
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback to plain string.
    return text


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Minimal YAML loader for flat key-value config files.

    Supports lines like:
      key: value
    Ignores empty lines and # comments.
    """
    p = Path(path)
    if not p.exists():
        return {}

    out: dict[str, Any] = {}
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Strip inline comments for unquoted values.
        if "#" in value and not (value.startswith('"') or value.startswith("'")):
            value = value.split("#", 1)[0].strip()

        out[key] = _parse_yaml_scalar(value)

    return out


@dataclass
class TrainConfig:
    # Data source
    dataset_type: str = "short_metaworld"  # short_metaworld | jsonl
    data_root: str = "data/short-metaworld-vla"
    train_jsonl: str = ""
    val_jsonl: str = ""
    val_ratio: float = 0.1

    # Model
    vision_model_name: str = "google/siglip2-base-patch16-224"
    text_model_name: str = "distilbert-base-uncased"
    image_size: int = 224
    freeze_vision: bool = True
    freeze_text: bool = True

    # Data shape
    action_dim: int = 4
    num_workers: int = 2

    # Optimization
    epochs: int = 30
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    use_fp16: bool = True

    # Runtime
    seed: int = 42
    device: str = "cuda"
    out_dir: str = "checkpoints"
    save_best_by_val: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        cfg = cls()
        values = _load_simple_yaml(path)
        cfg.apply_overrides(values)
        return cfg

    def apply_overrides(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
