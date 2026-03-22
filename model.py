from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from config import TrainConfig


class VLAFusionPolicy(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        # For SigLIP/SigLIP2 checkpoints, AutoModel may return a joint model
        # that expects both text and image. We extract the vision tower when present.
        vision_backbone = AutoModel.from_pretrained(cfg.vision_model_name)
        self.vision = vision_backbone.vision_model if hasattr(vision_backbone, "vision_model") else vision_backbone

        self.text = AutoModel.from_pretrained(cfg.text_model_name)

        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        if cfg.freeze_text:
            for p in self.text.parameters():
                p.requires_grad = False

        vision_dim = self._get_hidden_size(self.vision)
        text_dim = self._get_hidden_size(self.text)

        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )
        self.action_head = nn.Linear(512, cfg.action_dim)

    @staticmethod
    def _get_hidden_size(model: nn.Module) -> int:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return 768
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
            return int(cfg.vision_config.hidden_size)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return int(cfg.text_config.hidden_size)
        return 768

    @staticmethod
    def _pool_vision_output(vision_out) -> torch.Tensor:
        # Prefer model-provided pooled features if available.
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            return vision_out.pooler_output

        if hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            # Mean-pool tokens to avoid assumptions about CLS token conventions.
            return vision_out.last_hidden_state.mean(dim=1)

        if isinstance(vision_out, (tuple, list)) and len(vision_out) > 0:
            x = vision_out[0]
            if x.ndim == 3:
                return x.mean(dim=1)
            return x

        raise ValueError("Unsupported vision output format")

    @staticmethod
    def _pool_text_output(text_out) -> torch.Tensor:
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            return text_out.pooler_output

        if hasattr(text_out, "last_hidden_state") and text_out.last_hidden_state is not None:
            # BERT-style models generally use token 0 as sentence representation.
            return text_out.last_hidden_state[:, 0]

        if isinstance(text_out, (tuple, list)) and len(text_out) > 0:
            x = text_out[0]
            if x.ndim == 3:
                return x[:, 0]
            return x

        raise ValueError("Unsupported text output format")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.vision(pixel_values=pixel_values)
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)

        vision_pool = self._pool_vision_output(vision_out)
        text_pool = self._pool_text_output(text_out)

        fused = self.fusion(torch.cat([vision_pool, text_pool], dim=-1))
        actions = self.action_head(fused)
        return actions
