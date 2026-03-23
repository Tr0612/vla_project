from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from config import TrainConfig


class VLAFusionPolicy(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

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

        # Projection layers to align modalities before fusion.
        self.vision_proj = nn.Linear(vision_dim, cfg.proj_dim)
        self.text_proj = nn.Linear(text_dim, cfg.proj_dim)
        self.vision_proj_ln = nn.LayerNorm(cfg.proj_dim)
        self.text_proj_ln = nn.LayerNorm(cfg.proj_dim)

        self.fusion = nn.Sequential(
            nn.Linear(cfg.proj_dim * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )
        self.action_head = nn.Linear(512, cfg.action_dim)

        init = max(float(cfg.action_scale_init), 1e-6)
        log_init = math.log(init)
        if cfg.learnable_action_scale:
            self.action_log_scale = nn.Parameter(torch.full((cfg.action_dim,), log_init))
        else:
            self.register_buffer("action_log_scale", torch.full((cfg.action_dim,), log_init))

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
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            return vision_out.pooler_output
        if hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
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

        vision_feat = self.vision_proj_ln(self.vision_proj(vision_pool))
        text_feat = self.text_proj_ln(self.text_proj(text_pool))

        if self.cfg.normalize_embeddings:
            vision_feat = F.normalize(vision_feat, p=2, dim=-1)
            text_feat = F.normalize(text_feat, p=2, dim=-1)

        fused = self.fusion(torch.cat([vision_feat, text_feat], dim=-1))

        # Bounded base action plus learnable per-dimension scaling.
        base_action = self.action_head(fused)
        action_scale = torch.exp(self.action_log_scale)
        actions = base_action * action_scale
        return actions


# class MoEActionHead(nn.Module):
#     def __init__(self, input_dim, num_experts=4, action_dim=7):
#         self.experts = nn.ModuleList([
#             MLPExpert(input_dim, action_dim) for _ in range(num_experts)
#         ])
#         self.router = nn.Linear(input_dim, num_experts)
        
#     def forward(self, x):
#         # Compute expert weights
#         router_logits = self.router(x)
#         router_weights = F.softmax(router_logits, dim=-1)
        
#         # Weighted combination of expert outputs
#         expert_outputs = torch.stack([e(x) for e in self.experts])
#         action = torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=0)
        
#         return action, router_weights  # Return routing for analysis