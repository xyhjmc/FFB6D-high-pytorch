# -*- coding: utf-8 -*-
"""
Single-class LGFF model implementation
(Decoupled Heads + PC Normalization + Z-bias Init + 2D Seg Head + seg->point sampling).
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_base import LGFFBase

from lgff.models.backbone.mobilenet import MobileNetV3Extractor
from lgff.models.pointnet.mini_pointnet import MiniPointNet

try:
    from lgff.models.backbone.resnet import ResNetExtractor
except ImportError:
    ResNetExtractor = None


class SegHead2D(nn.Module):
    """
    Very light ROI-level segmentation head.
    Input:  feat_map [B, C, Hf, Wf]
    Output: mask_logits [B, 1, H_out, W_out]  (H_out/W_out 由 forward 指定)
    """
    def __init__(self, in_ch: int, mid_ch: int = 64, use_gn: bool = True) -> None:
        super().__init__()

        if use_gn:
            def Norm(c: int) -> nn.Module:
                g = 8 if c >= 8 else 1
                return nn.GroupNorm(g, c)
        else:
            def Norm(c: int) -> nn.Module:
                return nn.BatchNorm2d(c)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            Norm(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            Norm(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class LGFF_SC_SEG(LGFFBase):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg

        self.num_keypoints: int = int(getattr(cfg, "num_keypoints", getattr(cfg, "n_keypoints", 8)))

        # --------------------------
        # Seg switches
        # --------------------------
        self.use_seg_head: bool = bool(getattr(cfg, "use_seg_head", True))
        self.lambda_seg: float = float(getattr(cfg, "lambda_seg", 0.1))
        self.seg_detach_trunk: bool = bool(getattr(cfg, "seg_detach_trunk", False))
        self.seg_point_thresh: float = float(getattr(cfg, "seg_point_thresh", 0.5))  # 用于 pred_valid_mask_bool

        # pose-fusion mask policy (只负责产出，不强行修改 conf)
        self.pose_fusion_use_valid_mask: bool = bool(getattr(cfg, "pose_fusion_use_valid_mask", False))
        self.pose_fusion_valid_mask_source: str = str(getattr(cfg, "pose_fusion_valid_mask_source", "labels")).lower()
        # 允许 cfg 里写 pose_fusion_conf_floor（你已在 yaml 里写）
        self.pose_fusion_conf_floor: float = float(getattr(cfg, "pose_fusion_conf_floor", 1e-4))
        # 是否在模型内部用 seg mask 直接抑制 conf
        self.pose_fusion_mask_conf_in_model: bool = bool(getattr(cfg, "pose_fusion_mask_conf_in_model", False))

        # ==================================================================
        # 1. RGB Branch
        # ==================================================================
        backbone_name = getattr(cfg, "backbone_name", "mobilenet_v3_large")
        backbone_os = getattr(cfg, "backbone_output_stride", 8)
        pretrained = getattr(cfg, "backbone_pretrained", True)
        freeze_bn = getattr(cfg, "backbone_freeze_bn", True)
        mobilenet_arch = getattr(cfg, "backbone_arch", "large")

        self.return_intermediate = getattr(cfg, "backbone_return_intermediate", False)
        low_level_index = getattr(cfg, "backbone_low_level_index", 2)

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
                return_intermediate=self.return_intermediate,
                low_level_index=low_level_index,
            )

        rgb_feat_dim = getattr(cfg, "rgb_feat_dim", 128)

        with torch.no_grad():
            dummy_in = torch.zeros(1, 3, 128, 128)
            dummy_out = self.rgb_backbone(dummy_in)

        if self.return_intermediate and isinstance(dummy_out, (tuple, list)):
            high_ch = dummy_out[0].shape[1]
            low_ch = dummy_out[1].shape[1]
            self.high_dim = rgb_feat_dim // 2
            self.low_dim = rgb_feat_dim - self.high_dim
            self.rgb_reduce = nn.Conv2d(high_ch, self.high_dim, 1, bias=False)
            self.rgb_low_reduce = nn.Conv2d(low_ch, self.low_dim, 1, bias=False)
            print(f"[LGFF_SC_SEG] Multi-scale Enabled: High({high_ch}->{self.high_dim}) + Low({low_ch}->{self.low_dim})")
        else:
            last_channels = self.rgb_backbone.out_channels
            self.rgb_reduce = nn.Conv2d(last_channels, rgb_feat_dim, 1, bias=False)
            self.rgb_low_reduce = None

        # ==================================================================
        # 2. Geometry Branch
        # ==================================================================
        geo_feat_dim = getattr(cfg, "geo_feat_dim", 128)
        assert rgb_feat_dim == geo_feat_dim, "RGB feat dim and Geo feat dim must match for fusion."

        self.point_encoder = MiniPointNet(
            input_dim=getattr(cfg, "point_input_dim", 3),
            feat_dim=geo_feat_dim,
            hidden_dims=getattr(cfg, "point_hidden_dims", (64, 128)),
            norm=getattr(cfg, "point_norm", "bn"),
            use_se=getattr(cfg, "point_use_se", True),
            dropout=getattr(cfg, "point_dropout", 0.0),
            concat_global=True,
            return_global=False,
        )

        # ==================================================================
        # 3. Fusion Gate
        # ==================================================================
        fusion_in_dim = rgb_feat_dim + geo_feat_dim
        gate_mode = getattr(cfg, "gate_mode", "channel")
        gate_hidden = int(getattr(cfg, "gate_hidden", 128))
        gate_out_ch = 1 if gate_mode == "point" else rgb_feat_dim

        self.split_fusion_heads = bool(getattr(cfg, "split_fusion_heads", False))
        if not self.split_fusion_heads:
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(fusion_in_dim, gate_hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(gate_hidden, gate_out_ch, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
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
        # 4. Heads
        # ==================================================================
        head_hidden = getattr(cfg, "head_hidden_dim", 128)
        head_dropout = getattr(cfg, "head_dropout", 0.0)

        rot_in_dim = rgb_feat_dim
        trans_in_dim = rgb_feat_dim + 3
        conf_in_dim = rgb_feat_dim + 3
        kpof_in_dim = rgb_feat_dim + 3

        self.rot_head = nn.Sequential(
            nn.Conv1d(rot_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 4, 1),
        )

        self.trans_head = nn.Sequential(
            nn.Conv1d(trans_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 3, 1),
        )

        self.conf_head = nn.Sequential(
            nn.Conv1d(conf_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, 1, 1),
        )

        self.kp_of_head = nn.Sequential(
            nn.Conv1d(kpof_in_dim, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Conv1d(head_hidden, head_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_hidden, 3 * self.num_keypoints, 1),
        )

        # ==================================================================
        # 4.5 Seg Head
        # ==================================================================
        seg_mid = int(getattr(cfg, "seg_head_channels", getattr(cfg, "seg_head_dim", 64)))
        seg_use_gn = bool(getattr(cfg, "seg_use_gn", True))
        self.seg_head = SegHead2D(in_ch=rgb_feat_dim, mid_ch=seg_mid, use_gn=seg_use_gn) if self.use_seg_head else None

        # Init Z-bias
        self.init_z_bias = float(getattr(cfg, "init_z_bias", 0.7))

        self._init_weights()

    def _init_weights(self) -> None:
        modules = [
            self.rgb_reduce,
            self.point_encoder,
            self.rot_head,
            self.trans_head,
            self.conf_head,
            self.kp_of_head,
        ]
        if getattr(self, "rgb_low_reduce", None) is not None:
            modules.append(self.rgb_low_reduce)

        if self.split_fusion_heads:
            modules.extend([self.fusion_gate_rot, self.fusion_gate_tc])
        else:
            modules.append(self.fusion_gate)

        if self.seg_head is not None:
            modules.append(self.seg_head)

        for m in modules:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    if hasattr(mod, "weight") and mod.weight is not None:
                        nn.init.constant_(mod.weight, 1.0)
                    if hasattr(mod, "bias") and mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)

        nn.init.normal_(self.rot_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.rot_head[-1].bias, 0.0)

        nn.init.normal_(self.trans_head[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.trans_head[-1].bias, 0.0)
        with torch.no_grad():
            self.trans_head[-1].bias.data[2] = self.init_z_bias

        nn.init.normal_(self.conf_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.conf_head[-1].bias, 0.0)

        nn.init.normal_(self.kp_of_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.kp_of_head[-1].bias, 0.0)

        if self.seg_head is not None:
            nn.init.normal_(self.seg_head.out_conv.weight, mean=0, std=0.01)
            nn.init.constant_(self.seg_head.out_conv.bias, 0.0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rgb = batch["rgb"]               # [B,3,H,W]
        points = batch["point_cloud"]    # [B,N,3]
        intrinsic = batch["intrinsic"]   # [B,3,3] or [1,3,3]

        B, _, H, W = rgb.shape
        N = points.shape[1]

        # --------------------------
        # 0) Point preprocess
        # --------------------------
        pc_centering = getattr(self.cfg, "pc_centering", True)
        pc_scale_norm = getattr(self.cfg, "pc_scale_norm", True)

        if pc_centering:
            pc_center = points.mean(dim=1, keepdim=True)
        else:
            pc_center = torch.zeros_like(points[:, :1, :])

        pc_centered = points - pc_center

        if pc_scale_norm:
            dist = torch.norm(pc_centered, dim=-1, keepdim=True)
            pc_radius = dist.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            pc_normed = pc_centered / pc_radius
        else:
            pc_normed = pc_centered

        # --------------------------
        # 1) RGB feat map
        # --------------------------
        feat_out = self.rgb_backbone(rgb)

        if self.return_intermediate and isinstance(feat_out, (tuple, list)):
            feat_high, feat_low = feat_out
            feat_high = self.rgb_reduce(feat_high)
            feat_low = self.rgb_low_reduce(feat_low)
            feat_high = F.interpolate(feat_high, size=feat_low.shape[-2:], mode="bilinear", align_corners=False)
            feat_map = torch.cat([feat_high, feat_low], dim=1)
        else:
            if isinstance(feat_out, (tuple, list)):
                feat_out = feat_out[0]
            feat_map = self.rgb_reduce(feat_out)

        # --------------------------
        # 1.5) 2D seg logits
        # --------------------------
        pred_mask_logits: Optional[torch.Tensor] = None
        if self.use_seg_head and (self.seg_head is not None) and (self.lambda_seg > 0.0):
            seg_in = feat_map.detach() if self.seg_detach_trunk else feat_map
            pred_mask_logits = self.seg_head(seg_in, out_hw=(H, W))  # [B,1,H,W]

        # --------------------------
        # 2) Geometry feat (point)
        # --------------------------
        points_norm_t = pc_normed.transpose(1, 2)          # [B,3,N]
        geo_emb = self.point_encoder(points_norm_t)        # [B,C,N]

        # --------------------------
        # 3) Fuse (and return uv for seg->point)
        # --------------------------
        fused_raw, rgb_emb, uv = self.extract_and_fuse(
            rgb_feat=feat_map,
            geo_feat=geo_emb,
            points=points,
            intrinsic=intrinsic,
            img_shape=(H, W),
            valid_mask=None,
            apply_valid_mask=False,
            return_uv=True,
        )  # fused_raw: [B,Crgb+Cgeo,N], rgb_emb: [B,C,N], uv: [B,N,2]

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

        # --------------------------
        # 3.5) seg -> per-point valid mask (for pose_fusion / debug)
        # --------------------------
        pred_valid_mask: Optional[torch.Tensor] = None      # [B,N] float
        pred_valid_mask_bool: Optional[torch.Tensor] = None # [B,N] bool
        if pred_mask_logits is not None:
            mask_prob_2d = torch.sigmoid(pred_mask_logits)  # [B,1,H,W]
            # 采样到点：返回 [B,N]
            pred_valid_mask = self.sample_map_to_points(mask_prob_2d, uv, img_size=(H, W))
            pred_valid_mask = pred_valid_mask.clamp(0.0, 1.0)

            # bool mask：阈值化 + z>0
            pred_valid_mask_bool = (pred_valid_mask > self.seg_point_thresh) & (points[..., 2] > 1e-6)

        # --------------------------
        # 4) Head inputs
        # --------------------------
        points_t = points.transpose(1, 2)  # [B,3,N]
        trans_conf_kp_input = torch.cat([feat_tc, points_t], dim=1)

        # --------------------------
        # 5) Predictions
        # --------------------------
        pred_q = self.rot_head(feat_rot).permute(0, 2, 1).contiguous()  # [B,N,4]
        pred_q = F.normalize(pred_q, dim=-1)

        pred_t = self.trans_head(trans_conf_kp_input).permute(0, 2, 1).contiguous()  # [B,N,3]

        conf_detach = bool(getattr(self.cfg, "conf_detach_trunk", False))
        conf_in = trans_conf_kp_input.detach() if conf_detach else trans_conf_kp_input
        pred_c = self.conf_head(conf_in).permute(0, 2, 1).contiguous()  # [B,N,1]
        pred_c = torch.sigmoid(pred_c)

        # 可选：如果你希望在模型内就用 seg 抑制无效点 conf（不是必须）
        if (
            self.pose_fusion_mask_conf_in_model
            and self.pose_fusion_use_valid_mask
            and self.pose_fusion_valid_mask_source == "seg"
            and pred_valid_mask is not None
        ):
            pred_c = pred_c * pred_valid_mask.unsqueeze(-1).clamp_min(self.pose_fusion_conf_floor)

        pred_kp_ofs = None
        lambda_kp_of = float(getattr(self.cfg, "lambda_kp_of", 0.3))
        kp_detach = bool(getattr(self.cfg, "kp_of_detach_trunk", False))
        if lambda_kp_of > 0.0:
            kp_in = trans_conf_kp_input.detach() if kp_detach else trans_conf_kp_input
            kp_of_raw = self.kp_of_head(kp_in)  # [B,3K,N]
            kp_of_raw = kp_of_raw.view(B, self.num_keypoints, 3, N)
            pred_kp_ofs = kp_of_raw.permute(0, 1, 3, 2).contiguous()  # [B,K,N,3]

        return {
            "pred_quat": pred_q,
            "pred_trans": pred_t,
            "pred_conf": pred_c,
            "pred_kp_ofs": pred_kp_ofs,

            "pred_mask_logits": pred_mask_logits,          # [B,1,H,W] or None
            "pred_valid_mask": pred_valid_mask,            # [B,N] or None (float prob)
            "pred_valid_mask_bool": pred_valid_mask_bool,  # [B,N] or None (bool)

            "points": points,
        }


# 兼容：如果你外部仍然写死 import LGFF_SC，则让它指向 seg 版本（可按需删除）
LGFF_SC = LGFF_SC_SEG

__all__ = ["LGFF_SC_SEG", "LGFF_SC", "SegHead2D"]
