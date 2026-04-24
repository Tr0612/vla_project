from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from vla_stack.config import TrainConfig


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


class TextMoEContext(nn.Module):
    def __init__(
        self,
        text_dim: int,
        context_dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.context_dim = int(max(16, context_dim))
        self.num_experts = int(max(2, num_experts))
        hidden_dim = int(max(64, hidden_dim))
        dropout = float(dropout)

        self.experts = nn.ModuleList(
            [MLPExpert(text_dim, hidden_dim, self.context_dim, dropout=dropout) for _ in range(self.num_experts)]
        )
        self.router = nn.Linear(text_dim, self.num_experts)
        self.last_router_weights: torch.Tensor | None = None
        self.last_load_balance_loss: torch.Tensor | None = None

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(text_feat)
        router_weights = F.softmax(router_logits, dim=-1)
        self.last_router_weights = router_weights.detach()

        hard_assign = F.one_hot(torch.argmax(router_weights, dim=-1), num_classes=self.num_experts).to(
            router_weights.dtype
        )
        mean_prob = router_weights.mean(dim=0)
        mean_assign = hard_assign.mean(dim=0)
        self.last_load_balance_loss = self.num_experts * torch.sum(mean_prob * mean_assign)

        expert_outputs = torch.stack([expert(text_feat) for expert in self.experts], dim=1)
        return torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=1)


class ACTActionHead(nn.Module):
    """
    ACT-style chunked action decoder.

    Given a single fused context vector per timestep, predicts a horizon of K actions.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.chunk_size = int(max(1, chunk_size))
        hidden_dim = int(max(64, hidden_dim))
        num_layers = int(max(1, num_layers))
        dropout = float(dropout)

        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_embed = nn.Parameter(torch.randn(self.chunk_size, hidden_dim) * 0.02)

        layers: list[nn.Module] = []
        cur_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(cur_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, action_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        ctx = self.context_proj(x)  # [B, H]
        q = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, K, H]
        h = q + ctx.unsqueeze(1)  # [B, K, H]
        return self.decoder(h)  # [B, K, A]


class ACTMoEHead(nn.Module):
    """
    z = MoE(text_features)
    actions = ACT_decoder(obs_features, z)
    """

    def __init__(
        self,
        obs_dim: int,
        text_dim: int,
        action_dim: int,
        chunk_size: int,
        act_hidden_dim: int,
        act_num_layers: int,
        act_dropout: float,
        moe_num_experts: int,
        moe_hidden_dim: int,
        moe_context_dim: int,
    ):
        super().__init__()
        self.text_moe = TextMoEContext(
            text_dim=text_dim,
            context_dim=moe_context_dim,
            num_experts=moe_num_experts,
            hidden_dim=moe_hidden_dim,
            dropout=act_dropout,
        )
        self.act = ACTActionHead(
            input_dim=obs_dim + int(max(16, moe_context_dim)),
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=act_hidden_dim,
            num_layers=act_num_layers,
            dropout=act_dropout,
        )
        self.last_router_weights: torch.Tensor | None = None
        self.last_load_balance_loss: torch.Tensor | None = None

    def forward(self, obs_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        z = self.text_moe(text_feat)
        self.last_router_weights = self.text_moe.last_router_weights
        self.last_load_balance_loss = self.text_moe.last_load_balance_loss
        x = torch.cat([obs_feat, z], dim=-1)
        return self.act(x)


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
        self._unfreeze_last_n_layers(self.vision, int(max(0, cfg.unfreeze_vision_last_n_layers)))
        self._unfreeze_last_n_layers(self.text, int(max(0, cfg.unfreeze_text_last_n_layers)))

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
        self.geom_input_dim = geom_input_dim
        self.action_input_dim = self.fused_dim + (geom_input_dim if cfg.use_geometry_features else 0)
        self.router_condition = str(cfg.router_condition).lower().strip()
        if self.router_condition not in {"action_input", "text", "text_geometry"}:
            raise ValueError(
                f"Unsupported router_condition: {cfg.router_condition}. "
                "Expected one of: action_input, text, text_geometry"
            )

        head_type = cfg.action_head_type.lower().strip()
        if head_type == "moe":
            self.action_head = MoEActionHead(
                input_dim=self.action_input_dim,
                action_dim=cfg.action_dim,
                num_experts=max(2, int(cfg.moe_num_experts)),
                hidden_dim=max(64, int(cfg.moe_hidden_dim)),
                dropout=cfg.fusion_dropout,
                router_input_dim=self._get_router_input_dim(),
            )
        elif head_type == "mlp":
            self.action_head = PlainMLPHead(
                input_dim=self.action_input_dim,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.action_mlp_hidden_dim,
                num_layers=cfg.action_mlp_layers,
                dropout=cfg.action_mlp_dropout,
            )
        elif head_type == "act":
            self.action_head = ACTActionHead(
                input_dim=self.action_input_dim,
                action_dim=cfg.action_dim,
                chunk_size=max(1, int(cfg.act_chunk_size)),
                hidden_dim=max(64, int(cfg.act_hidden_dim)),
                num_layers=max(1, int(cfg.act_num_layers)),
                dropout=float(cfg.act_dropout),
            )
        elif head_type == "act_moe":
            self.action_head = ACTMoEHead(
                obs_dim=self.action_input_dim,
                text_dim=cfg.proj_dim,
                action_dim=cfg.action_dim,
                chunk_size=max(1, int(cfg.act_chunk_size)),
                act_hidden_dim=max(64, int(cfg.act_hidden_dim)),
                act_num_layers=max(1, int(cfg.act_num_layers)),
                act_dropout=float(cfg.act_dropout),
                moe_num_experts=max(2, int(cfg.moe_num_experts)),
                moe_hidden_dim=max(64, int(cfg.moe_hidden_dim)),
                moe_context_dim=max(16, int(cfg.act_moe_context_dim)),
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
    def _find_transformer_blocks(module: nn.Module) -> list[nn.Module]:
        candidates: list[list[nn.Module]] = []

        def _as_blocks(obj) -> list[nn.Module] | None:
            if obj is None:
                return None
            if isinstance(obj, nn.ModuleList):
                return [m for m in obj if isinstance(m, nn.Module)]
            if isinstance(obj, (list, tuple)) and obj and all(isinstance(m, nn.Module) for m in obj):
                return list(obj)
            return None

        attr_paths = [
            ("encoder", "layers"),
            ("encoder", "layer"),
            ("transformer", "layers"),
            ("transformer", "layer"),
            ("layers",),
            ("layer",),
            ("blocks",),
            ("h",),
        ]
        for path in attr_paths:
            cur = module
            ok = True
            for name in path:
                if not hasattr(cur, name):
                    ok = False
                    break
                cur = getattr(cur, name)
            if not ok:
                continue
            blocks = _as_blocks(cur)
            if blocks:
                candidates.append(blocks)

        if candidates:
            # Prefer the longest stack to target actual transformer depth.
            candidates.sort(key=len, reverse=True)
            return candidates[0]
        return []

    @classmethod
    def _unfreeze_last_n_layers(cls, module: nn.Module, n_last: int) -> None:
        if n_last <= 0:
            return
        blocks = cls._find_transformer_blocks(module)
        if not blocks:
            return
        for blk in blocks[-n_last:]:
            for p in blk.parameters():
                p.requires_grad = True

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

    def _get_router_input_dim(self) -> int:
        if self.router_condition == "text":
            return self.cfg.proj_dim
        if self.router_condition == "action_input":
            return self.action_input_dim
        if self.router_condition == "text_geometry":
            return self.cfg.proj_dim + (self.geom_input_dim if self.cfg.use_geometry_features else 0)
        raise ValueError(f"Unsupported router_condition: {self.router_condition}")

    def _build_router_input(
        self,
        text_feat: torch.Tensor,
        action_input: torch.Tensor,
        geometry_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.router_condition == "text":
            return text_feat
        if self.router_condition == "action_input":
            return action_input
        if self.router_condition == "text_geometry":
            if self.cfg.use_geometry_features:
                if geometry_features is None:
                    geometry_features = torch.zeros(
                        text_feat.size(0),
                        self.geom_input_dim,
                        device=text_feat.device,
                        dtype=text_feat.dtype,
                    )
                return torch.cat([text_feat, geometry_features], dim=-1)
            return text_feat
        raise ValueError(f"Unsupported router_condition: {self.router_condition}")

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

        head_type = self.cfg.action_head_type.lower().strip()
        if head_type == "moe":
            router_input = self._build_router_input(text_feat, action_input, geometry_features)
            base_action = self.action_head(action_input, router_input=router_input)
            self.last_moe_load_balance_loss = self.action_head.last_load_balance_loss
        elif head_type == "act_moe":
            base_action = self.action_head(action_input, text_feat)
            self.last_moe_load_balance_loss = self.action_head.last_load_balance_loss
        else:
            base_action = self.action_head(action_input)
            self.last_moe_load_balance_loss = None

        action_scale = torch.exp(self.action_log_scale)
        return base_action * action_scale
