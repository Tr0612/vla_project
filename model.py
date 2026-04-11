from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from config import TrainConfig


class MLPExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainMLPHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        hidden_dim = max(32, int(hidden_dim))
        num_layers = max(1, int(num_layers))
        dropout = float(dropout)

        layers: list[nn.Module] = []
        cur_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(cur_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float,
        router_input_dim: int | None = None,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [MLPExpert(input_dim, hidden_dim, action_dim, dropout=dropout) for _ in range(num_experts)]
        )
        if router_input_dim is None:
            router_input_dim = input_dim
        self.router = nn.Linear(router_input_dim, num_experts)
        self.num_experts = num_experts
        self.last_router_weights: torch.Tensor | None = None
        self.last_load_balance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, router_input: torch.Tensor | None = None) -> torch.Tensor:
        if router_input is None:
            router_input = x
        router_logits = self.router(router_input)
        router_weights = F.softmax(router_logits, dim=-1)
        self.last_router_weights = router_weights.detach()

        # Switch-style load balancing objective: minimize E * sum(mean_prob * mean_assign).
        hard_assign = F.one_hot(torch.argmax(router_weights, dim=-1), num_classes=self.num_experts).to(
            router_weights.dtype
        )
        mean_prob = router_weights.mean(dim=0)
        mean_assign = hard_assign.mean(dim=0)
        self.last_load_balance_loss = self.num_experts * torch.sum(mean_prob * mean_assign)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=max(1, num_heads),
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(max(1, num_layers))])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(max(1, num_layers))])

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        x = text_feat.unsqueeze(1)
        kv = vision_feat.unsqueeze(1)

        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers, self.ln1, self.ln2):
            attn_out, _ = attn(query=x, key=kv, value=kv, need_weights=False)
            x = ln1(x + attn_out)
            x = ln2(x + ffn(x))

        return x.squeeze(1)


class TransformerFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, num_heads),
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([vision_feat, text_feat], dim=1)
        encoded = self.encoder(tokens)
        return encoded.mean(dim=1)


class VLAFusionPolicy(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.last_moe_load_balance_loss: torch.Tensor | None = None

        self.vision, self.text = self._build_encoders(cfg)

        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        if cfg.freeze_text:
            for p in self.text.parameters():
                p.requires_grad = False

        vision_dim = self._get_hidden_size(self.vision)
        text_dim = self._get_hidden_size(self.text)

        self.vision_proj = nn.Linear(vision_dim, cfg.proj_dim)
        self.text_proj = nn.Linear(text_dim, cfg.proj_dim)
        self.vision_proj_ln = nn.LayerNorm(cfg.proj_dim)
        self.text_proj_ln = nn.LayerNorm(cfg.proj_dim)

        fusion_type = cfg.fusion_type.lower().strip()
        if fusion_type == "concat":
            self.fusion, self.fused_dim = self._build_concat_fusion()
        elif fusion_type == "cross_attn":
            self.fusion = CrossAttentionFusion(
                d_model=cfg.proj_dim,
                num_heads=cfg.fusion_num_heads,
                hidden_dim=cfg.fusion_hidden_dim,
                num_layers=cfg.fusion_num_layers,
                dropout=cfg.fusion_dropout,
            )
            self.fused_dim = cfg.proj_dim
        elif fusion_type == "transformer":
            self.fusion = TransformerFusion(
                d_model=cfg.proj_dim,
                num_heads=cfg.fusion_num_heads,
                hidden_dim=cfg.fusion_hidden_dim,
                num_layers=cfg.fusion_num_layers,
                dropout=cfg.fusion_dropout,
            )
            self.fused_dim = cfg.proj_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {cfg.fusion_type}")

        geom_input_dim = cfg.geometry_dim * max(1, int(cfg.temporal_context))
        self.action_input_dim = self.fused_dim + (geom_input_dim if cfg.use_geometry_features else 0)

        head_type = cfg.action_head_type.lower().strip()
        if head_type == "moe":
            self.action_head = MoEActionHead(
                input_dim=self.action_input_dim,
                action_dim=cfg.action_dim,
                num_experts=max(2, int(cfg.moe_num_experts)),
                hidden_dim=max(64, int(cfg.moe_hidden_dim)),
                dropout=cfg.fusion_dropout,
                router_input_dim=cfg.proj_dim,
            )
        elif head_type == "mlp":
            self.action_head = PlainMLPHead(
                input_dim=self.action_input_dim,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.action_mlp_hidden_dim,
                num_layers=cfg.action_mlp_layers,
                dropout=cfg.action_mlp_dropout,
            )
        else:
            self.action_head = nn.Linear(self.action_input_dim, cfg.action_dim)

        init = max(float(cfg.action_scale_init), 1e-6)
        log_init = math.log(init)
        if cfg.learnable_action_scale:
            self.action_log_scale = nn.Parameter(torch.full((cfg.action_dim,), log_init))
        else:
            self.register_buffer("action_log_scale", torch.full((cfg.action_dim,), log_init))

    def _build_encoders(self, cfg: TrainConfig) -> tuple[nn.Module, nn.Module]:
        if cfg.separate_backbones:
            vision_model = AutoModel.from_pretrained(cfg.vision_model_name)
            text_model = AutoModel.from_pretrained(cfg.text_model_name)
            return self._extract_vision_model(vision_model), self._extract_text_model(text_model)

        shared = AutoModel.from_pretrained(cfg.vision_model_name)
        vision = self._extract_vision_model(shared)
        if hasattr(shared, "text_model"):
            text = shared.text_model
        else:
            fallback_text = AutoModel.from_pretrained(cfg.text_model_name)
            text = self._extract_text_model(fallback_text)
        return vision, text

    def _build_concat_fusion(self) -> tuple[nn.Module, int]:
        in_dim = self.cfg.proj_dim * 2
        hidden_dim = max(64, int(self.cfg.fusion_hidden_dim))
        out_dim = max(32, int(self.cfg.fusion_out_dim))
        num_layers = max(1, int(self.cfg.fusion_num_layers))
        dropout = float(self.cfg.fusion_dropout)

        layers: list[nn.Module] = []
        cur_dim = in_dim
        for layer_idx in range(num_layers):
            next_dim = out_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.extend(
                [
                    nn.Linear(cur_dim, next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            cur_dim = next_dim
        return nn.Sequential(*layers), cur_dim

    @staticmethod
    def _extract_vision_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "vision_model"):
            return model.vision_model
        return model

    @staticmethod
    def _extract_text_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "text_model"):
            return model.text_model
        return model

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

    def _fuse(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        fusion_type = self.cfg.fusion_type.lower().strip()
        if fusion_type == "concat":
            return self.fusion(torch.cat([vision_feat, text_feat], dim=-1))
        return self.fusion(vision_feat, text_feat)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        geometry_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vision_out = self.vision(pixel_values=pixel_values)
        if attention_mask is None:
            text_out = self.text(input_ids=input_ids)
        else:
            text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)

        vision_pool = self._pool_vision_output(vision_out)
        text_pool = self._pool_text_output(text_out)

        vision_feat = self.vision_proj_ln(self.vision_proj(vision_pool))
        text_feat = self.text_proj_ln(self.text_proj(text_pool))

        if self.cfg.normalize_embeddings:
            vision_feat = F.normalize(vision_feat, p=2, dim=-1)
            text_feat = F.normalize(text_feat, p=2, dim=-1)

        fused = self._fuse(vision_feat, text_feat)
        if self.cfg.use_geometry_features:
            if geometry_features is None:
                geometry_features = torch.zeros(
                    fused.size(0),
                    self.cfg.geometry_dim * max(1, int(self.cfg.temporal_context)),
                    device=fused.device,
                    dtype=fused.dtype,
                )
            else:
                geometry_features = geometry_features.to(device=fused.device, dtype=fused.dtype)
            action_input = torch.cat([fused, geometry_features], dim=-1)
        else:
            action_input = fused

        if self.cfg.action_head_type.lower().strip() == "moe":
            # Force language-only routing for task specialization.
            router_input = text_feat
            base_action = self.action_head(action_input, router_input=router_input)
            self.last_moe_load_balance_loss = self.action_head.last_load_balance_loss
        else:
            base_action = self.action_head(action_input)
            self.last_moe_load_balance_loss = None

        action_scale = torch.exp(self.action_log_scale)
        return base_action * action_scale
