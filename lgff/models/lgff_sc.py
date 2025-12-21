"""
Single-class LGFF model implementation
(Decoupled Heads + PC Normalization + Z-bias Init).

- RGB Branch: MobileNetV3 / ResNet
- Geometry Branch: MiniPointNet over normalized point cloud
- Fusion: feature-level gating between RGB & Geometry
- Heads:
    * Rotation Head: uses fused features (RGB+Geo)
    * Translation Head: uses fused features + raw points
    * Confidence Head: uses fused features + raw points
    * [New] Keypoint Offset Head: uses fused features + raw points
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

# 尝试导入 ResNet
try:
    from lgff.models.backbone.resnet import ResNetExtractor
except ImportError:
    ResNetExtractor = None


class LGFF_SC(LGFFBase):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg

        # ==================================================================
        # 0. 一些基础配置
        # ==================================================================
        # 关键点个数：优先使用 cfg.num_keypoints，其次兼容 cfg.n_keypoints
        self.num_keypoints: int = int(
            getattr(cfg, "num_keypoints", getattr(cfg, "n_keypoints", 9))
        )

        # ==================================================================
        # 1. RGB Branch
        # ==================================================================
        backbone_name = getattr(cfg, "backbone_name", "mobilenet_v3_large")
        backbone_os = getattr(cfg, "backbone_output_stride", 8)
        pretrained = getattr(cfg, "backbone_pretrained", True)
        freeze_bn = getattr(cfg, "backbone_freeze_bn", True)
        mobilenet_arch = getattr(cfg, "backbone_arch", "large")

        if "resnet" in backbone_name and ResNetExtractor is not None:
            self.rgb_backbone = ResNetExtractor(
                arch=backbone_name,
                output_stride=backbone_os,
                pretrained=pretrained,
                freeze_bn=freeze_bn,
            )
        else:
            self.rgb_backbone = MobileNetV3Extractor(
                arch=mobilenet_arch,
                output_stride=backbone_os,
                pretrained=pretrained,
                freeze_bn=freeze_bn,
                return_intermediate=False,
            )

        last_channels = self.rgb_backbone.out_channels
        rgb_feat_dim = getattr(cfg, "rgb_feat_dim", 128)
        self.rgb_reduce = nn.Conv2d(last_channels, rgb_feat_dim, 1, bias=False)

        # ==================================================================
        # 2. Geometry Branch (PointNet over normalized PC)
        # ==================================================================
        geo_feat_dim = getattr(cfg, "geo_feat_dim", 128)
        assert (
            rgb_feat_dim == geo_feat_dim
        ), "RGB feat dim and Geo feat dim must match for fusion."

        point_input_dim = getattr(cfg, "point_input_dim", 3)
        point_hidden_dims = getattr(cfg, "point_hidden_dims", (64, 128))

        self.point_encoder = MiniPointNet(
            input_dim=point_input_dim,
            feat_dim=geo_feat_dim,
            hidden_dims=point_hidden_dims,
            norm=getattr(cfg, "point_norm", "bn"),
            use_se=getattr(cfg, "point_use_se", True),
            dropout=getattr(cfg, "point_dropout", 0.0),
            concat_global=True,
            return_global=False,
        )

        # ==================================================================
        # 3. Fusion Module
        # ==================================================================
        fusion_in_dim = rgb_feat_dim + geo_feat_dim  # [rgb_emb, geo_emb]
        #yuanban
        # self.fusion_gate = nn.Sequential(
        #     nn.Conv1d(fusion_in_dim, 128, 1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 1, 1, bias=True),
        #     nn.Sigmoid(),
        # )
        gate_mode = getattr(cfg, "gate_mode", "channel")
        gate_hidden = int(getattr(cfg, "gate_hidden", 128))
        gate_out_ch = 1 if gate_mode == "point" else rgb_feat_dim  # C_fused
        #实验D
        self.split_fusion_heads = bool(getattr(cfg, "split_fusion_heads", False))

        if not self.split_fusion_heads:
            # 旧：一套 gate
            # 实验C
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(fusion_in_dim, gate_hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(gate_hidden, gate_out_ch, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            # 新：两套 gate
            gate_hidden_rot = int(getattr(cfg, "gate_hidden_rot", gate_hidden))
            gate_hidden_tc = int(getattr(cfg, "gate_hidden_tc", gate_hidden))

            self.fusion_gate_rot = nn.Sequential(
                nn.Conv1d(fusion_in_dim, gate_hidden_rot, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(gate_hidden_rot, gate_out_ch, 1, bias=True),
                nn.Sigmoid(),
            )
            self.fusion_gate_tc = nn.Sequential(
                nn.Conv1d(fusion_in_dim, gate_hidden_tc, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(gate_hidden_tc, gate_out_ch, 1, bias=True),
                nn.Sigmoid(),
            )

        # ==================================================================
        # 4. Decoupled Heads
        # ==================================================================
        head_hidden = getattr(cfg, "head_hidden_dim", 128)
        head_dropout = getattr(cfg, "head_dropout", 0.0)

        # [关键] 解耦输入：
        #   Rot Head:  feat_fused (C_fused = rgb_feat_dim)
        #   Trans Head: [feat_fused, points_t]   (C_fused + 3)
        #   Conf Head:  [feat_fused, points_t]   (C_fused + 3)
        #   KpOf Head:  [feat_fused, points_t]   (C_fused + 3)
        rot_in_dim = rgb_feat_dim
        trans_in_dim = rgb_feat_dim + 3
        conf_in_dim = rgb_feat_dim + 3
        kpof_in_dim = rgb_feat_dim + 3  # 和 trans/conf 共享输入

        # --- Rotation Head ---
        self.rot_head = nn.Sequential(
            nn.Conv1d(rot_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 4, 1),  # Quaternion
        )

        # --- Translation Head (per-point translation, later 聚合) ---
        self.trans_head = nn.Sequential(
            nn.Conv1d(trans_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 3, 1),  # Translation (Direct Regression)
        )

        # --- Confidence Head ---
        self.conf_head = nn.Sequential(
            nn.Conv1d(conf_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, 1, 1),  # Logits
        )

        # --- [New] Keypoint Offset Head ---
        # 目标：输出 pred_kp_ofs: [B, n_kpts, N, 3]
        # 实现：conv 输出通道数 = 3 * n_kpts，然后 reshape。
        self.kp_of_head = nn.Sequential(
            nn.Conv1d(kpof_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 3 * self.num_keypoints, 1),  # 3D offset for each keypoint
        )

        # Z 轴初始化偏置（物体平均距离，可在 cfg 里改）
        self.init_z_bias = float(getattr(cfg, "init_z_bias", 0.7))

        self._init_weights()

    # ======================================================================
    # 初始化
    # ======================================================================
    def _init_weights(self) -> None:
        modules = [
            self.rgb_reduce,
            self.point_encoder,
            # self.fusion_gate,
            self.rot_head,
            self.trans_head,
            self.conf_head,
            self.kp_of_head,  # [New]
        ]

        # -------- Fusion gate(s): route D compatible --------
        if getattr(self, "split_fusion_heads", False):
            # route D: two gates
            if hasattr(self, "fusion_gate_rot"):
                modules.append(self.fusion_gate_rot)
            if hasattr(self, "fusion_gate_tc"):
                modules.append(self.fusion_gate_tc)
        else:
            # baseline: single gate
            if hasattr(self, "fusion_gate"):
                modules.append(self.fusion_gate)
        for m in modules:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(
                        mod.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(mod.weight, 1.0)
                    nn.init.constant_(mod.bias, 0.0)

        # Rot Head: 最后一层正常小随机 + 0 bias
        nn.init.normal_(self.rot_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.rot_head[-1].bias, 0.0)

        # Trans Head: Direct Regression + Z-bias 先验
        nn.init.normal_(self.trans_head[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.trans_head[-1].bias, 0.0)
        # [重要] 为 Z 轴注入初始深度先验（单位与数据一致）
        with torch.no_grad():
            # bias shape: [3] -> [x, y, z]
            self.trans_head[-1].bias.data[2] = self.init_z_bias

        # Conf Head
        nn.init.normal_(self.conf_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.conf_head[-1].bias, 0.0)

        # [New] KpOf Head
        nn.init.normal_(self.kp_of_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.kp_of_head[-1].bias, 0.0)

    # ======================================================================
    # 前向
    # ======================================================================
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rgb = batch["rgb"]               # [B, 3, H, W]
        points = batch["point_cloud"]    # [B, N, 3]  原始相机坐标
        intrinsic = batch["intrinsic"]   # [B, 3, 3]
        B, _, H, W = rgb.shape
        N = points.shape[1]

        # ------------------------------------------------------------------
        # 0. 点云预处理：居中 + 半径归一化 (只对几何编码用)
        # ------------------------------------------------------------------
        pc_centering = getattr(self.cfg, "pc_centering", True)
        pc_scale_norm = getattr(self.cfg, "pc_scale_norm", True)

        if pc_centering:
            pc_center = points.mean(dim=1, keepdim=True)      # [B, 1, 3]
        else:
            pc_center = torch.zeros_like(points[:, :1, :])

        pc_centered = points - pc_center                      # [B, N, 3]

        if pc_scale_norm:
            dist = torch.norm(pc_centered, dim=-1, keepdim=True)  # [B, N, 1]
            pc_radius = dist.max(dim=1, keepdim=True)[0].clamp(min=1e-6)  # [B, 1, 1]
            pc_normed = pc_centered / pc_radius               # [B, N, 3]
        else:
            pc_normed = pc_centered

        # ------------------------------------------------------------------
        # 1. RGB Feature
        # ------------------------------------------------------------------
        feat_map = self.rgb_backbone(rgb)
        if isinstance(feat_map, (tuple, list)):
            feat_map = feat_map[0]
        feat_map = self.rgb_reduce(feat_map)                  # [B, C_rgb, H', W']

        # ------------------------------------------------------------------
        # 2. Geometry Feature (用归一化点云)
        # ------------------------------------------------------------------
        points_norm_t = pc_normed.transpose(1, 2)             # [B, 3, N]
        geo_emb = self.point_encoder(points_norm_t)           # [B, C_geo, N]

        # ------------------------------------------------------------------
        # 3. Fusion (投影 & 对齐使用原始 points)
        # ------------------------------------------------------------------
        # extract_and_fuse: 通常会根据 points 投影到特征图上取 RGB features
        fused_raw, rgb_emb = self.extract_and_fuse(
            feat_map, geo_emb, points, intrinsic, (H, W)
        )   # fused_raw: [B, C_rgb+C_geo, N], rgb_emb: [B, C_rgb, N]
        if not self.split_fusion_heads:
            gate = self.fusion_gate(fused_raw)
            feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)

            feat_rot = feat_fused
            feat_tc = feat_fused
        else:
            gate_rot = self.fusion_gate_rot(fused_raw)
            gate_tc = self.fusion_gate_tc(fused_raw)

            feat_rot = rgb_emb * gate_rot + geo_emb * (1.0 - gate_rot)
            feat_tc = rgb_emb * gate_tc + geo_emb * (1.0 - gate_tc)

        # if gate.shape[1] == 1:
        #     # point-wise gate, broadcast to channels
        #     feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)
        # else:
        #     # channel-wise gate
        #     feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)

        # ------------------------------------------------------------------
        # 4. 构造 Head 输入
        # ------------------------------------------------------------------
        points_t = points.transpose(1, 2)                     # [B, 3, N]
        # Rot Head 输入用 feat_rot
        rot_input = feat_rot

        # Trans/Conf/KpOf 输入用 feat_tc
        trans_conf_kp_input = torch.cat([feat_tc, points_t], dim=1)


        # ------------------------------------------------------------------
        # 5. 解耦预测
        # ------------------------------------------------------------------

        # A. Rotation
        pred_q = self.rot_head(rot_input)                     # [B, 4, N]
        pred_q = pred_q.permute(0, 2, 1).contiguous()         # [B, N, 4]
        pred_q = F.normalize(pred_q, dim=-1)

        # B. Translation (per-point, direct regression)
        pred_t = self.trans_head(trans_conf_kp_input)         # [B, 3, N]
        pred_t = pred_t.permute(0, 2, 1).contiguous()         # [B, N, 3]

        # #原来的代码
        # # C. Confidence
        # pred_c = self.conf_head(trans_conf_kp_input)          # [B, 1, N]
        # pred_c = pred_c.permute(0, 2, 1).contiguous()         # [B, N, 1]
        # pred_c = torch.sigmoid(pred_c)

        #E1
        conf_detach = bool(getattr(self.cfg, "conf_detach_trunk", False))  # E1 开关
        conf_in = trans_conf_kp_input.detach() if conf_detach else trans_conf_kp_input

        pred_c = self.conf_head(conf_in)  # [B, 1, N]
        pred_c = pred_c.permute(0, 2, 1).contiguous()  # [B, N, 1]
        pred_c = torch.sigmoid(pred_c)

        # D. [New] Keypoint Offsets
        # kp_of_raw: [B, 3*K, N]
        #路线A
        # kp_of_raw = self.kp_of_head(trans_conf_kp_input)
        #路线B
        lambda_kp_of = float(getattr(self.cfg, "lambda_kp_of", 0.3))
        kp_detach = bool(getattr(self.cfg, "kp_of_detach_trunk", False))  # 路线B默认 True

        pred_kp_ofs = None
        if lambda_kp_of > 0.0:
            kp_in = trans_conf_kp_input.detach() if kp_detach else trans_conf_kp_input
            kp_of_raw = self.kp_of_head(kp_in)  # [B, 3K, N]
            kp_of_raw = kp_of_raw.view(B, self.num_keypoints, 3, N)
            pred_kp_ofs = kp_of_raw.permute(0, 1, 3, 2).contiguous()  # [B, K, N, 3]

        # # 先 reshape 成 [B, K, 3, N] 再转成 [B, K, N, 3]，方便直接给 OFLoss 用
        # kp_of_raw = kp_of_raw.view(
        #     B, self.num_keypoints, 3, N
        # )  # [B, K, 3, N]
        # pred_kp_ofs = kp_of_raw.permute(0, 1, 3, 2).contiguous()  # [B, K, N, 3]

        return {
            "pred_quat": pred_q,
            "pred_trans": pred_t,
            "pred_conf": pred_c,
            "pred_kp_ofs": pred_kp_ofs,   # [B, K, N, 3] for OFLoss
            "points": points,             # 方便 Loss / Eval 使用
        }


__all__ = ["LGFF_SC"]
