"""
Lightweight & Extensible PointNet-style encoder for LGFF.

Features:
- Custom hidden_dims & feat_dim.
- Supports BN / GroupNorm / Identity normalization.
- Optional SE-style channel attention (AvgPool based).
- Optional Global Feature extraction (MaxPool or AvgPool).
- Mask support for ROI/padding handling.

"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    """构建 1D 归一化层."""
    norm = (norm or "bn").lower()
    if norm == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm == "gn":
        # GroupNorm: 对小 batch 较友好
        num_groups = min(8, num_channels)
        return nn.GroupNorm(num_groups, num_channels)
    elif norm == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm}")


class ChannelSE1d(nn.Module):
    """
    轻量级 1D 通道注意力 (SE-Block).

    使用全局平均池化提取通道全局信息，然后通过两层全连接生成通道权重。
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)  # 避免 hidden 太小
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, C, N]
            mask: [B, N] (optional) True/1 为有效点
        """
        B, C, N = x.shape

        # Global Pooling (Squeeze)
        if mask is None:
            g = x.mean(dim=2)  # [B, C]
        else:
            m = mask.float().unsqueeze(1)     # [B,1,N]
            denom = m.sum(dim=2).clamp(min=1.0)  # [B,1]
            g = (x * m).sum(dim=2) / denom       # [B,C]

        # Excitation
        w = F.relu(self.fc1(g), inplace=True)
        w = self.fc2(w)
        w = torch.sigmoid(w).unsqueeze(2)        # [B,C,1]

        return x * w


class MiniPointNet(nn.Module):
    """
    强化版轻量 PointNet。

    Structure:
        Input -> MLP(Conv1d) -> [SE] -> [Dropout] -> feat
                                         |
                                         +-> [GlobalPool] (optional)

    Args:
        input_dim: 输入点云通道数 (3: xyz, 或 6: xyz+normals 等)
        feat_dim:  输出逐点特征通道数
        hidden_dims: 中间层通道数列表，如 (64, 128)
        norm: 'bn' / 'gn' / 'none'
        use_se: 是否启用 SE 通道注意力
        dropout: dropout 概率
        return_global: 是否同时返回全局特征 (PointNet 风格)
        global_pool_method: 'max' 或 'avg'
    """

    def __init__(
        self,
        input_dim: int = 3,
        feat_dim: int = 128,
        hidden_dims: Iterable[int] = (64, 128),
        norm: str = "bn",
        use_se: bool = True,
        dropout: float = 0.0,
        return_global: bool = False,
        global_pool_method: str = "max",
    ) -> None:
        super().__init__()
        self.return_global = return_global
        self.use_se = use_se

        # ---- 1. 校验 global_pool_method ----
        valid_pools = {"max", "avg"}
        method = (global_pool_method or "max").lower()
        if method not in valid_pools:
            raise ValueError(f"Unknown global_pool_method={global_pool_method}, "
                             f"expected one of {valid_pools}")
        self.global_pool_method = method

        # ---- 2. Shared MLP Encoder ----
        layers = []
        in_ch = input_dim
        for h in hidden_dims:
            layers.append(nn.Conv1d(in_ch, h, kernel_size=1, bias=False))
            layers.append(_make_norm(norm, h))
            layers.append(nn.ReLU(inplace=True))
            in_ch = h

        # Final projection to feat_dim
        layers.append(nn.Conv1d(in_ch, feat_dim, kernel_size=1, bias=False))
        layers.append(_make_norm(norm, feat_dim))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

        # ---- 3. Optional Modules ----
        self.se = ChannelSE1d(feat_dim) if use_se else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # ---- 4. Weight Init ----
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
            x:    [B, C_in, N] 点云 (已经是 transpose 后的格式)
            mask: [B, N] (optional)，True/1 表示有效点，False/0 表示无效/填充点

        Returns:
            若 return_global=False:
                feat: [B, C_out, N]
            若 return_global=True:
                (feat, global_feat)
                - feat:        [B, C_out, N]
                - global_feat: [B, C_out]
        """
        # 1. Per-point features
        feat = self.encoder(x)  # [B, C_out, N]

        # 2. SE Attention
        if self.se is not None:
            feat = self.se(feat, mask=mask)

        # 3. Dropout
        feat = self.dropout(feat)

        if not self.return_global:
            return feat

        # 4. Global Feature (PointNet-style)
        if self.global_pool_method == "max":
            if mask is None:
                # 标准 Max Pool
                global_feat = torch.max(feat, dim=2)[0]  # [B, C_out]
            else:
                # Masked Max Pool：无效点赋极小值，再做 max
                m_expanded = mask.unsqueeze(1).to(dtype=torch.bool)  # [B,1,N]
                # 使用 dtype-safe 的最小值，兼容 fp16/fp32
                min_val = torch.finfo(feat.dtype).min
                feat_masked = feat.masked_fill(~m_expanded, min_val)
                global_feat = torch.max(feat_masked, dim=2)[0]
        else:  # "avg"
            if mask is None:
                global_feat = feat.mean(dim=2)
            else:
                m_expanded = mask.float().unsqueeze(1)         # [B,1,N]
                denom = m_expanded.sum(dim=2).clamp(min=1.0)   # [B,1]
                global_feat = (feat * m_expanded).sum(dim=2) / denom

        return feat, global_feat
