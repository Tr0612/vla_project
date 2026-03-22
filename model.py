from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from config import TrainConfig


class VLAFusionPolicy(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        self.vision = AutoModel.from_pretrained(cfg.vision_model_name)
        self.text = AutoModel.from_pretrained(cfg.text_model_name)

        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        if cfg.freeze_text:
            for p in self.text.parameters():
                p.requires_grad = False

        vision_dim = self.vision.config.hidden_size
        text_dim = self.text.config.hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )
        self.action_head = nn.Linear(512, cfg.action_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.vision(pixel_values=pixel_values)
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)

        vision_pool = vision_out.last_hidden_state[:, 0]
        text_pool = text_out.last_hidden_state[:, 0]

        fused = self.fusion(torch.cat([vision_pool, text_pool], dim=-1))
        actions = self.action_head(fused)
        return actions
