from __future__ import annotations

from typing import Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utility: kNN & graph feature (EdgeConv)
# ============================================================================

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: [B, C, N]  (point features)
    return: idx [B, N, k]  (indices of k nearest neighbors for each point)
    """
    # Use pairwise distance via (x - y)^2 = x^2 + y^2 - 2 x^T y
    # x: [B, C, N] -> [B, N, C]
    x_t = x.transpose(1, 2)  # [B, N, C]
    # pairwise distance: [B, N, N]
    # d(i,j) = ||x_i - x_j||^2
    xx = torch.sum(x_t ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    yy = xx.transpose(1, 2)  # [B, 1, N]
    inner = torch.bmm(x_t, x_t.transpose(1, 2))  # [B, N, N]
    dist = xx + yy - 2 * inner  # [B, N, N]

    # 在自己的实现里可以考虑加 eps / clamp 防止数值问题
    _, idx = torch.topk(-dist, k=k, dim=-1)  # 取负号 -> k 个最小距离
    return idx  # [B, N, k]


def get_graph_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    构造 EdgeConv 使用的图特征:
    输入:
        x:   [B, C, N]      点特征
        idx: [B, N, k]      kNN 索引
    输出:
        edge_feat: [B, 2C, N, k]
        形式: cat( x_j - x_i, x_i ), 其中 j 为邻居, i 为中心点
    """
    B, C, N = x.size()
    k = idx.size(-1)

    # 把 batch 维合并到点索引上，方便 gather
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # [B, 1, 1]
    idx = idx + idx_base  # [B, N, k]
    idx = idx.view(-1)  # [B*N*k]

    # [B, C, N] -> [B*N, C]
    x = x.transpose(1, 2).contiguous()  # [B, N, C]
    feature = x.view(B * N, C)  # [B*N, C]

    # 取邻居点特征
    neighbors = feature[idx, :]  # [B*N*k, C]
    neighbors = neighbors.view(B, N, k, C)  # [B, N, k, C]

    # 中心点特征
    x_c = x.view(B, N, 1, C).expand(-1, -1, k, -1)  # [B, N, k, C]

    # edge feature: concat( neighbor - center, center )
    edge = torch.cat((neighbors - x_c, x_c), dim=-1)  # [B, N, k, 2C]
    edge = edge.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]

    return edge


# ============================================================================
# Optional: SE block for 1D feature (channel attention)
# ============================================================================

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        w = x.mean(dim=-1, keepdim=True)  # [B, C, 1]
        w = F.relu(self.fc1(w), inplace=True)  # [B, hidden, 1]
        w = torch.sigmoid(self.fc2(w))  # [B, C, 1]
        return x * w


# ============================================================================
# EdgeConv Block
# ============================================================================

class EdgeConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            k: int = 16,
            use_se: bool = False,
    ) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock1D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, N]
        return: [B, C_out, N]
        """
        # 1. kNN
        idx = knn(x, self.k)  # [B, N, k]

        # 2. 构造 Edge 特征
        edge_feat = get_graph_feature(x, idx)  # [B, 2*C_in, N, k]

        # 3. 卷积 + max pooling over k
        x = self.conv(edge_feat)  # [B, C_out, N, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [B, C_out, N]

        if self.use_se:
            x = self.se(x)  # [B, C_out, N]

        return x


# ============================================================================
# MiniDGCNN
# ============================================================================

class MiniDGCNN(nn.Module):
    """
    轻量版 DGCNN，用作 LGFF 的点云分支。

    输入:
        - x: [B, input_dim, N]  (通常 input_dim = 3)

    输出:
        - feat: [B, feat_dim, N]
        - (可选) global_feat: [B, feat_dim]  当 return_global=True 时返回

    主要超参数:
        - hidden_dims: 每个 EdgeConv block 的输出 channel 列表, 如 (64, 64, 128)
        - k: kNN 中的邻居个数
        - concat_global: 是否把各层的输出 concat 后再做线性投影
        - return_global: 是否返回 max-pool 全局特征
    """

    def __init__(
            self,
            input_dim: int = 3,
            feat_dim: int = 128,
            hidden_dims: Sequence[int] = (64, 64, 128),
            k: int = 16,
            use_se: bool = True,
            dropout: float = 0.0,
            concat_global: bool = True,
            return_global: bool = False,
            # 为了兼容你现在 MiniPointNet 的参数名：
            norm: str = "bn",
            point_norm: Optional[str] = None,
            point_use_se: Optional[bool] = None,
            point_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        # 兼容旧参数名（如果传了就覆盖）
        if point_use_se is not None:
            use_se = point_use_se
        if point_dropout is not None:
            dropout = point_dropout
        if point_norm is not None:
            norm = point_norm  # 当前实现只用 BN，不做额外区分

        self.k = k
        self.concat_global = concat_global
        self.return_global = return_global

        # EdgeConv blocks
        self.blocks = nn.ModuleList()
        in_ch = input_dim
        for h in hidden_dims:
            self.blocks.append(EdgeConvBlock(in_ch, h, k=k, use_se=use_se))
            in_ch = h

        # 输出维度
        if concat_global:
            total_ch = sum(hidden_dims)
        else:
            total_ch = hidden_dims[-1]

        self.out_conv = nn.Sequential(
            nn.Conv1d(total_ch, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, C_in, N]
        return:
            feat: [B, feat_dim, N]
            (optional) global_feat: [B, feat_dim]
        """
        B, C, N = x.size()

        feats_per_layer = []
        out = x
        for block in self.blocks:
            out = block(out)  # [B, C_h, N]
            feats_per_layer.append(out)

        if self.concat_global:
            feat_cat = torch.cat(feats_per_layer, dim=1)  # [B, sum(C_h), N]
        else:
            feat_cat = feats_per_layer[-1]  # [B, C_last, N]

        feat = self.out_conv(feat_cat)  # [B, feat_dim, N]

        if self.return_global:
            # 全局 max-pool 得到一个向量，可选给后续模块用
            global_feat = feat.max(dim=-1)[0]  # [B, feat_dim]
            return feat, global_feat

        return feat


# 在 mini_dgcnn.py 末尾加上

class MiniDGCNN_APE(MiniDGCNN):
    """
    APE 专用默认配置的小 DGCNN：
    - input_dim = 3 (XYZ 相机坐标)
    - feat_dim = 128 (和 RGB 分支对齐)
    - hidden_dims = (64, 64, 128) 三层 EdgeConv
    - k = 16 邻居数，兼顾细节和速度
    - use_se = True 适度通道注意力
    - dropout = 0.1 防一点过拟合
    - concat_global = True 把三层特征拼接再投影
    - return_global = False 保持和 MiniPointNet 接口一致（返回 [B,C,N]）
    """

    def __init__(self, **kwargs):
        defaults = dict(
            input_dim=3,
            feat_dim=128,
            hidden_dims=(64, 64, 128),
            k=16,
            use_se=True,
            dropout=0.1,
            concat_global=True,
            return_global=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
