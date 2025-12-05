"""
Single-class LGFF model implementation.
Integrates the robust MobileNetV3 (RGB) and MiniPointNet (Geometry)
into a dense fusion architecture for 6D pose estimation.
"""
from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_base import LGFFBase

# 自定义模块
from lgff.models.backbone.mobilenet import MobileNetV3Extractor
from lgff.models.pointnet.mini_pointnet import MiniPointNet


class LGFF_SC(LGFFBase):
    """
    LGFF_SC: 单类轻量 6D 位姿网络

    架构:
        1. RGB 分支: MobileNetV3 (可配置 arch / OS / 冻结 BN 等)
        2. 点云分支: MiniPointNet (Shared MLP + 可选 SE 注意力)
        3. 融合模块: 像素-点云对齐采样 + 门控注意力融合 (Gated Fusion)
        4. 预测头: 逐点密集预测 [R, t, conf]
    """

    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg

        # --------------------------------------------------------------
        # 1. RGB Branch: MobileNetV3 Backbone
        # --------------------------------------------------------------
        backbone_arch = getattr(cfg, "backbone_arch", "small")           # 'small' / 'large'
        backbone_os = getattr(cfg, "backbone_output_stride", 8)          # 8 / 16 / 32
        backbone_pretrained = getattr(cfg, "backbone_pretrained", True)
        backbone_freeze_bn = getattr(cfg, "backbone_freeze_bn", True)
        backbone_return_inter = getattr(cfg, "backbone_return_intermediate", False)
        backbone_low_level_idx = getattr(cfg, "backbone_low_level_index", 2)

        self.rgb_backbone = MobileNetV3Extractor(
            arch=backbone_arch,
            output_stride=backbone_os,
            pretrained=backbone_pretrained,
            freeze_bn=backbone_freeze_bn,
            return_intermediate=backbone_return_inter,
            low_level_index=backbone_low_level_idx,
        )

        # 动态获取输出通道数 (例如 small: 576)
        last_channels = self.rgb_backbone.out_channels
        rgb_feat_dim = getattr(cfg, "rgb_feat_dim", 128)

        # 1x1 Conv 降维: C_last -> rgb_feat_dim
        # 这是我们新增的模块之一（需要自定义初始化）
        self.rgb_reduce = nn.Conv2d(last_channels, rgb_feat_dim, kernel_size=1, bias=False)

        # --------------------------------------------------------------
        # 2. Geometry Branch: MiniPointNet
        # --------------------------------------------------------------
        geo_feat_dim = getattr(cfg, "geo_feat_dim", 128)

        # 融合要求 RGB / Geo 特征维度一致
        assert rgb_feat_dim == geo_feat_dim, (
            f"Fusion requires rgb_dim ({rgb_feat_dim}) == geo_dim ({geo_feat_dim})"
        )

        point_input_dim = getattr(cfg, "point_input_dim", 3)
        point_hidden_dims = getattr(cfg, "point_hidden_dims", (64, 128))
        point_norm = getattr(cfg, "point_norm", "bn")
        point_use_se = getattr(cfg, "point_use_se", True)
        point_dropout = getattr(cfg, "point_dropout", 0.0)

        # MiniPointNet 本身内部也有自己的初始化；我们在 _init_weights 里会再次
        # 用 Kaiming 刷一遍，不影响稳定性
        self.point_encoder = MiniPointNet(
            input_dim=point_input_dim,
            feat_dim=geo_feat_dim,
            hidden_dims=point_hidden_dims,
            norm=point_norm,
            use_se=point_use_se,
            dropout=point_dropout,
            return_global=False,        # 目前只用逐点特征
            global_pool_method="max",
        )

        # --------------------------------------------------------------
        # 3. Fusion Gate (Cross-Modality Gating)
        # --------------------------------------------------------------
        fusion_in_dim = rgb_feat_dim + geo_feat_dim  # 128 + 128 = 256

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(fusion_in_dim, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # --------------------------------------------------------------
        # 4. Dense Heads (Per-Point Prediction)
        # --------------------------------------------------------------
        head_in_dim = rgb_feat_dim + geo_feat_dim     # 256
        head_hidden_dim = getattr(cfg, "head_hidden_dim", 128)
        head_feat_dim = getattr(cfg, "head_feat_dim", 64)
        head_dropout = getattr(cfg, "head_dropout", 0.0)

        # 这里你改成 bias=True 是合理的：没有 Norm，保留 bias 有帮助
        self.head_shared = nn.Sequential(
            nn.Conv1d(head_in_dim, head_hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden_dim, head_feat_dim, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.head_dropout = (
            nn.Dropout(p=head_dropout) if head_dropout > 0.0 else nn.Identity()
        )

        # 输出头（保持偏置默认 True）
        self.pose_r = nn.Conv1d(head_feat_dim, 4, 1)  # Quaternion: 4
        self.pose_t = nn.Conv1d(head_feat_dim, 3, 1)  # Translation: 3
        self.pose_c = nn.Conv1d(head_feat_dim, 1, 1)  # Confidence: 1

        # --------------------------------------------------------------
        # 5. Weight Init (只初始化“新模块”，不动预训练 backbone)
        # --------------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        对新增模块做一个干净的初始化：
        - 不触碰 self.rgb_backbone 里的预训练权重
        """
        # 1) RGB 降维层
        for m in [self.rgb_reduce]:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(mod.weight, 1.0)
                    nn.init.constant_(mod.bias, 0.0)

        # 2) 点云编码器（额外刷一遍没问题）
        for mod in self.point_encoder.modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1.0)
                nn.init.constant_(mod.bias, 0.0)

        # 3) Fusion Gate
        for mod in self.fusion_gate.modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1.0)
                nn.init.constant_(mod.bias, 0.0)

        # 4) Head Shared
        for mod in self.head_shared.modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(mod.weight, 1.0)
                nn.init.constant_(mod.bias, 0.0)

        # 5) 输出头
        for head in [self.pose_r, self.pose_t, self.pose_c]:
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            if head.bias is not None:
                nn.init.constant_(head.bias, 0.0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
                - rgb:         [B, 3, H, W]
                - point_cloud: [B, N, 3]
                - intrinsic:   [B, 3, 3]

        Returns:
            dict:
                - pred_quat:  [B, N, 4]
                - pred_trans: [B, N, 3]
                - pred_conf:  [B, N, 1]
                - points:     [B, N, 3]
        """
        rgb: torch.Tensor = batch["rgb"]
        points: torch.Tensor = batch["point_cloud"]
        intrinsic: torch.Tensor = batch["intrinsic"]

        B, _, H_in, W_in = rgb.shape
        _, N, _ = points.shape

        # ----------------------------------------------------------
        # A. RGB Feature Extraction
        # ----------------------------------------------------------
        feat_map = self.rgb_backbone(rgb)
        # 兼容 return_intermediate=True 的情况
        if isinstance(feat_map, (tuple, list)):
            feat_map, _ = feat_map  # 暂时忽略浅层特征，后续可拓展

        # [B, C_back, H/OS, W/OS] -> [B, rgb_feat_dim, ...]
        feat_map = self.rgb_reduce(feat_map)

        # ----------------------------------------------------------
        # B. Geometry Feature Extraction
        # ----------------------------------------------------------
        # [B, N, 3] -> [B, 3, N]
        points_t = points.transpose(1, 2)
        # [B, C_geo, N]
        geo_emb = self.point_encoder(points_t)

        # ----------------------------------------------------------
        # C. Projection & Sampling (Pixel-Point Alignment)
        # ----------------------------------------------------------
        # 调用 LGFFBase 中的通用对齐采样：将 RGB 特征和点云特征对齐到 N 个点上
        # fused_raw: [B, C_rgb+C_geo, N]
        # rgb_emb:   [B, C_rgb,       N]
        fused_raw, rgb_emb = self.extract_and_fuse(
            feat_map, geo_emb, points, intrinsic, (H_in, W_in)
        )

        # ----------------------------------------------------------
        # D. Gated Fusion
        # ----------------------------------------------------------
        # 计算门控权重 [B,1,N]
        gate = self.fusion_gate(fused_raw)

        # 残差式加权融合：RGB vs Geo
        feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)

        # 保留原始 Geo，再次 concat
        feat_ready = torch.cat([feat_fused, geo_emb], dim=1)  # [B, C_fused+C_geo, N]

        # ----------------------------------------------------------
        # E. Dense Head Prediction
        # ----------------------------------------------------------
        feat_shared = self.head_shared(feat_ready)   # [B, head_feat_dim, N]
        feat_shared = self.head_dropout(feat_shared)

        pred_q = self.pose_r(feat_shared)           # [B,4,N]
        pred_t = self.pose_t(feat_shared)           # [B,3,N]
        pred_c = self.pose_c(feat_shared)           # [B,1,N]

        # 调整维度到 [B,N,C]
        pred_q = pred_q.permute(0, 2, 1).contiguous()
        pred_t = pred_t.permute(0, 2, 1).contiguous()
        pred_c = pred_c.permute(0, 2, 1).contiguous()

        # 归一化四元数 & 置信度激活
        pred_q = F.normalize(pred_q, dim=-1)
        pred_c = torch.sigmoid(pred_c)

        return {
            "pred_quat": pred_q,    # [B, N, 4]
            "pred_trans": pred_t,   # [B, N, 3]
            "pred_conf": pred_c,    # [B, N, 1]
            "points": points,       # [B, N, 3]
        }
