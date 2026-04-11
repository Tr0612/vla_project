from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


def _parse_yaml_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""

    try:
        return json.loads(text)
    except Exception:
        pass

    return text


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
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
    separate_backbones: bool = True
    image_size: int = 224
    freeze_vision: bool = True
    freeze_text: bool = True

    # Fusion/projection
    fusion_type: str = "concat"  # concat | cross_attn | transformer
    proj_dim: int = 512
    fusion_hidden_dim: int = 1024
    fusion_out_dim: int = 512
    fusion_num_layers: int = 2
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    normalize_embeddings: bool = True

    # Action head
    action_head_type: str = "linear"  # linear | mlp | moe
    action_mlp_hidden_dim: int = 256
    action_mlp_layers: int = 2
    action_mlp_dropout: float = 0.1
    moe_num_experts: int = 4
    moe_hidden_dim: int = 512
    moe_load_balance_weight: float = 0.01
    router_condition: str = "text"  # action_input | text | text_geometry
    use_geometry_features: bool = True
    geometry_dim: int = 6
    temporal_context: int = 3

    # Data shape
    action_dim: int = 4
    num_workers: int = 2

    # Action scaling/normalization
    normalize_action_targets: bool = True
    action_norm_eps: float = 1e-6
    learnable_action_scale: bool = True
    action_scale_init: float = 1.0

    # Optimization
    loss_type: str = "mse"  # mse | huber
    huber_delta: float = 1.0
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
