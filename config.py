from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainConfig:
    # Model
    vision_model_name: str = "google/siglip2-base-patch16-224"
    text_model_name: str = "distilbert-base-uncased"
    image_size: int = 224
    freeze_vision: bool = True
    freeze_text: bool = True

    # Data
    action_dim: int = 7
    num_workers: int = 2

    # Optimization (requested recipe)
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
    out_dir: str = "vla_stack/checkpoints"
    save_best_by_val: bool = True

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
