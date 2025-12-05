"""
Single-class LGFF model implementation.
Integrates MobileNetV3 (RGB) + Simplified PointNet (Geometry) + Dense Fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_base import LGFFBase


class LGFF_SC(LGFFBase):
    """
    LGFF_SC: 单类轻量 6D 位姿网络
    - RGB 分支: MobileNetV3-Small
    - 点云分支: 简化 PointNet (per-point MLP)
    - 融合: 门控加权 + Dense Head 做逐点姿态预测
    """

    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg

        # --------------------------------------------------------------
        # 1. RGB Branch: MobileNetV3-Small (Backbone)
        # --------------------------------------------------------------
        # 使用 ImageNet 预训练权重
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # 提取特征层 (去掉最后的分类器)
        self.rgb_backbone = mobilenet.features

        # 动态获取最后一层通道数 (通常为 576)
        last_channels = mobilenet.features[-1].out_channels
        rgb_feat_dim = getattr(cfg, "rgb_feat_dim", 128)

        # 1x1 Conv 做降维：C_last -> rgb_feat_dim
        self.rgb_reduce = nn.Conv2d(last_channels, rgb_feat_dim, kernel_size=1)

        # --------------------------------------------------------------
        # 2. Geometry Branch: Simplified PointNet
        # --------------------------------------------------------------
        geo_feat_dim = getattr(cfg, "geo_feat_dim", 128)

        # [稳健性检查] 加权融合要求 RGB 和 Geo 维度一致
        assert rgb_feat_dim == geo_feat_dim, \
            f"Fusion requires rgb_dim ({rgb_feat_dim}) == geo_dim ({geo_feat_dim})"

        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, geo_feat_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(geo_feat_dim, geo_feat_dim, 1),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------
        # 3. Fusion Gate (Cross-Modality Gating)
        # --------------------------------------------------------------
        # 输入: cat(RGB_emb, Geo_emb)
        fusion_in_dim = rgb_feat_dim + geo_feat_dim

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(fusion_in_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid(),
        )

        # --------------------------------------------------------------
        # 4. Dense Heads (Per-Point Prediction)
        # --------------------------------------------------------------
        # 计算 Head 输入维度:
        # feat_fused 是加权和，维度 = rgb_feat_dim (128)
        # feat_ready = cat([feat_fused, geo_emb]) -> 128 + 128 = 256
        head_in_dim = rgb_feat_dim + geo_feat_dim

        self.head_shared = nn.Sequential(
            nn.Conv1d(head_in_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
        )

        # 独立输出头（逐点预测）
        self.pose_r = nn.Conv1d(64, 4, 1)  # Quaternion: 4
        self.pose_t = nn.Conv1d(64, 3, 1)  # Translation: 3
        self.pose_c = nn.Conv1d(64, 1, 1)  # Confidence: 1

    def forward(self, batch: dict) -> dict:
        """
        Forward pass for LGFF_SC.
        """
        rgb = batch["rgb"]            # [B,3,H,W]
        points = batch["point_cloud"] # [B,N,3]
        intrinsic = batch["intrinsic"]

        B, _, H_in, W_in = rgb.shape
        _, N, _ = points.shape

        # ----------------------------------------------------------
        # A. RGB Feature Extraction
        # ----------------------------------------------------------
        feat_map = self.rgb_backbone(rgb)
        feat_map = self.rgb_reduce(feat_map)

        # ----------------------------------------------------------
        # B. Geometry Feature Extraction
        # ----------------------------------------------------------
        points_t = points.transpose(1, 2) # [B,3,N]
        geo_emb = self.point_encoder(points_t) # [B,128,N]

        # ----------------------------------------------------------
        # C. Projection & Sampling
        # ----------------------------------------------------------
        # fused_raw: [B, 256, N]
        # rgb_emb:   [B, 128, N]
        fused_raw, rgb_emb = self.extract_and_fuse(
            feat_map, geo_emb, points, intrinsic, (H_in, W_in)
        )

        # ----------------------------------------------------------
        # D. Gated Fusion
        # ----------------------------------------------------------
        gate = self.fusion_gate(fused_raw) # [B,1,N]

        # 加权融合: 维度保持为 128
        feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)

        # 拼接: 128 + 128 = 256
        feat_ready = torch.cat([feat_fused, geo_emb], dim=1)

        # ----------------------------------------------------------
        # E. Dense Head
        # ----------------------------------------------------------
        feat_shared = self.head_shared(feat_ready) # [B,64,N]

        pred_q = self.pose_r(feat_shared)  # [B,4,N]
        pred_t = self.pose_t(feat_shared)  # [B,3,N]
        pred_c = self.pose_c(feat_shared)  # [B,1,N]

        # 格式调整
        pred_q = pred_q.permute(0, 2, 1)  # [B,N,4]
        pred_t = pred_t.permute(0, 2, 1)  # [B,N,3]
        pred_c = pred_c.permute(0, 2, 1)  # [B,N,1]

        # 归一化
        pred_q = F.normalize(pred_q, dim=-1)
        pred_c = torch.sigmoid(pred_c)

        return {
            "pred_quat": pred_q,    # [B, N, 4]
            "pred_trans": pred_t,   # [B, N, 3]
            "pred_conf": pred_c,    # [B, N, 1]
            "points": points,       # [B, N, 3]
        }