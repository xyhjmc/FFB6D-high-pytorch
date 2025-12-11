"""
Enhanced MiniPointNet with Global Context Aggregation & FCAttention.

- Shared MLP 提取局部特征
- MaxPool 提取全局特征，并可选择 concat 回每个点
- 可选 SE / FCAttention 通道注意力 (FCAttention 更轻量且效果通常更好)
- 支持 norm / dropout / return_global
"""
from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试从你的 blocks.py 导入 FCAttention
try:
    from lgff.models.blocks import FCAttention
except ImportError:
    FCAttention = None  # 如果没找到文件，回退到 None


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = (norm or "bn").lower()
    if norm == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm == "gn":
        num_groups = min(8, num_channels)
        return nn.GroupNorm(num_groups, num_channels)
    elif norm == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm}")


class MiniPointNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        feat_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (64, 128),
        norm: str = "bn",
        use_se: bool = True,        # 是否使用注意力
        use_fc_attn: bool = True,   # [新增] 优先使用 FCAttention (如果 use_se=True)
        dropout: float = 0.0,
        concat_global: bool = True,
        return_global: bool = False,
    ) -> None:
        super().__init__()
        self.concat_global = concat_global
        self.return_global = return_global

        # 1. Local Feature Encoder (Shared MLP)
        layers = []
        in_ch = input_dim
        for h in hidden_dims:
            layers.append(nn.Conv1d(in_ch, h, 1, bias=False))
            layers.append(_make_norm(norm, h))
            layers.append(nn.ReLU(inplace=True))
            in_ch = h
        self.local_encoder = nn.Sequential(*layers)

        # local 最终维度
        self.local_dim = in_ch

        # 2. Projection Layer (local + optional global)
        proj_in_dim = self.local_dim + (self.local_dim if concat_global else 0)

        self.project = nn.Sequential(
            nn.Conv1d(proj_in_dim, feat_dim, 1, bias=False),
            _make_norm(norm, feat_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Dropout（可选）
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # 4. Attention (SE or FCAttention)
        self.attn = None
        if use_se:
            if use_fc_attn and FCAttention is not None:
                # [升级] 使用 FCAttention
                # FCAttention 是针对 2D [B, C, H, W] 设计的
                # 但它的核心是 GlobalPool + 1D Conv，所以我们可以通过 Reshape 来复用它
                self.attn = FCAttention(feat_dim)
                self.attn_type = "fc"
            else:
                # [回退] 标准 SE Block
                hidden_se = max(8, feat_dim // 4)
                self.attn = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),              # [B,C,N] -> [B,C,1]
                    nn.Conv1d(feat_dim, hidden_se, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_se, feat_dim, 1),
                    nn.Sigmoid(),
                )
                self.attn_type = "se"

        # 5. 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x:    [B, C_in, N]
        """
        B, C, N = x.shape

        # 1. Local features
        local_feat = self.local_encoder(x)  # [B, local_dim, N]

        # 2. Global feature + concat
        if self.concat_global:
            if mask is None:
                global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]  # [B, local_dim, 1]
            else:
                # 简单处理：假设无 mask
                global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]

            global_expand = global_feat.expand(-1, -1, N)  # [B, local_dim, N]
            feat = torch.cat([local_feat, global_expand], dim=1)  # [B, 2*local_dim, N]
        else:
            feat = local_feat

        # 3. Project
        feat = self.project(feat)          # [B, feat_dim, N]
        feat = self.dropout(feat)

        # 4. Attention (FCAttention / SE)
        if self.attn is not None:
            if self.attn_type == "fc":
                # FCAttention 需要 4D 输入 [B, C, H, W]
                # 我们把 N 视作 H，W=1
                feat_4d = feat.unsqueeze(-1)  # [B, C, N, 1]
                feat_4d = self.attn(feat_4d)  # [B, C, N, 1]
                feat = feat_4d.squeeze(-1)    # [B, C, N]
            else:
                # SE Block 直接处理 3D
                w = self.attn(feat)           # [B, C, 1]
                feat = feat * w

        if self.return_global:
            global_feat_out = torch.max(feat, dim=2)[0]  # [B, feat_dim]
            return feat, global_feat_out

        return feat