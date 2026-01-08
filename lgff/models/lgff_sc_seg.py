# lgff/models/lgff_sc_seg.py
# -*- coding: utf-8 -*-
"""
Single-class LGFF model implementation (Seg Enhanced).
Uses LGFFConfigSeg and supports Robust Index-based Sampling.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# [CHANGED] Import correct config and renamed Base class
from lgff.utils.config_seg import LGFFConfigSeg
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_base_seg import LGFFBaseSeg  # [CHANGED]

from lgff.models.backbone.mobilenet import MobileNetV3Extractor
from lgff.models.pointnet.mini_pointnet import MiniPointNet

try:
    from lgff.models.backbone.resnet import ResNetExtractor
except ImportError:
    ResNetExtractor = None


class SegHead2D(nn.Module):
    """
    Lightweight Segmentation Head.
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, use_gn: bool = True) -> None:
        super().__init__()
        if use_gn:
            def Norm(c):
                return nn.GroupNorm(8 if c >= 8 else 1, c)
        else:
            def Norm(c):
                return nn.BatchNorm2d(c)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False), Norm(mid_ch), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False), Norm(mid_ch), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        # Always upsample to full ROI size for sharp mask supervision
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class LGFF_SC_SEG(LGFFBaseSeg):  # [CHANGED] Inherit from LGFFBaseSeg
    def __init__(self, cfg: LGFFConfigSeg, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg

        self.num_keypoints: int = int(getattr(cfg, "num_keypoints", 8))

        # Seg Configs
        self.use_seg_head: bool = bool(getattr(cfg, "use_seg_head", True))
        self.lambda_seg: float = float(getattr(cfg, "lambda_seg", 0.1))
        self.seg_detach_trunk: bool = bool(getattr(cfg, "seg_detach_trunk", False))
        self.seg_point_thresh: float = float(getattr(cfg, "seg_point_thresh", 0.5))

        # Fusion Policy
        self.pose_fusion_use_valid_mask: bool = bool(getattr(cfg, "pose_fusion_use_valid_mask", False))
        self.pose_fusion_valid_mask_source: str = str(getattr(cfg, "pose_fusion_valid_mask_source", "labels")).lower()
        self.pose_fusion_conf_floor: float = float(getattr(cfg, "pose_fusion_conf_floor", 1e-4))
        self.pose_fusion_mask_conf_in_model: bool = bool(getattr(cfg, "pose_fusion_mask_conf_in_model", False))

        # 1. RGB Branch
        backbone_name = getattr(cfg, "backbone_name", "mobilenet_v3_large")
        backbone_os = getattr(cfg, "backbone_output_stride", 8)
        self.return_intermediate = getattr(cfg, "backbone_return_intermediate", False)

        if "resnet" in backbone_name and ResNetExtractor:
            self.rgb_backbone = ResNetExtractor(
                arch=backbone_name, output_stride=backbone_os,
                pretrained=getattr(cfg, "backbone_pretrained", True),
                freeze_bn=getattr(cfg, "backbone_freeze_bn", True),
            )
        else:
            self.rgb_backbone = MobileNetV3Extractor(
                arch=getattr(cfg, "backbone_arch", "large"),
                output_stride=backbone_os,
                pretrained=getattr(cfg, "backbone_pretrained", True),
                freeze_bn=getattr(cfg, "backbone_freeze_bn", True),
                return_intermediate=self.return_intermediate,
                low_level_index=getattr(cfg, "backbone_low_level_index", 2),
            )

        rgb_feat_dim = getattr(cfg, "rgb_feat_dim", 128)

        # Dimension Check & Reducers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self.rgb_backbone(dummy)

        if self.return_intermediate and isinstance(out, (tuple, list)):
            self.high_dim = rgb_feat_dim // 2
            self.low_dim = rgb_feat_dim - self.high_dim
            self.rgb_reduce = nn.Conv2d(out[0].shape[1], self.high_dim, 1, bias=False)
            self.rgb_low_reduce = nn.Conv2d(out[1].shape[1], self.low_dim, 1, bias=False)
        else:
            ch = out[0].shape[1] if isinstance(out, (tuple, list)) else out.shape[1]
            self.rgb_reduce = nn.Conv2d(ch, rgb_feat_dim, 1, bias=False)
            self.rgb_low_reduce = None

        # 2. Geometry Branch
        geo_feat_dim = getattr(cfg, "geo_feat_dim", 128)
        self.point_encoder = MiniPointNet(
            input_dim=getattr(cfg, "point_input_dim", 3),
            feat_dim=geo_feat_dim,
            hidden_dims=getattr(cfg, "point_hidden_dims", (64, 128)),
            norm=getattr(cfg, "point_norm", "bn"),
            use_se=getattr(cfg, "point_use_se", True),
            dropout=getattr(cfg, "point_dropout", 0.0),
        )

        # 3. Fusion Gate
        fusion_dim = rgb_feat_dim + geo_feat_dim
        self.split_fusion_heads = bool(getattr(cfg, "split_fusion_heads", False))

        if not self.split_fusion_heads:
            self.fusion_gate = self._make_gate(fusion_dim, rgb_feat_dim)
        else:
            self.fusion_gate_rot = self._make_gate(fusion_dim, rgb_feat_dim)
            self.fusion_gate_tc = self._make_gate(fusion_dim, rgb_feat_dim)

        # 4. Heads
        head_dim = getattr(cfg, "head_hidden_dim", 128)
        self.rot_head = self._make_head(rgb_feat_dim, head_dim, 4)
        self.trans_head = self._make_head(rgb_feat_dim + 3, head_dim, 3)
        self.conf_head = self._make_head(rgb_feat_dim + 3, head_dim, 1)  # Sigmoid later
        self.kp_of_head = self._make_head(rgb_feat_dim + 3, head_dim, 3 * self.num_keypoints)

        # 4.5 Seg Head
        seg_mid = int(getattr(cfg, "seg_head_channels", 64))
        self.seg_head = SegHead2D(rgb_feat_dim, seg_mid) if self.use_seg_head else None

        # Init Z-bias
        self.init_z_bias = float(getattr(cfg, "init_z_bias", 0.7))
        self._init_weights()

    def _make_gate(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, 128, 1, bias=False), nn.ReLU(True),
            nn.Conv1d(128, out_dim, 1, bias=True), nn.Sigmoid()
        )

    def _make_head(self, in_dim, hid_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, hid_dim, 1, bias=True), nn.ReLU(True),
            nn.Conv1d(hid_dim, hid_dim, 1, bias=True), nn.ReLU(True),
            nn.Conv1d(hid_dim, out_dim, 1)
        )

    def _init_weights(self) -> None:
        # Standard initialization
        modules = [
            self.rgb_reduce, self.point_encoder, self.rot_head,
            self.trans_head, self.conf_head, self.kp_of_head
        ]
        if self.rgb_low_reduce: modules.append(self.rgb_low_reduce)
        if self.split_fusion_heads:
            modules.extend([self.fusion_gate_rot, self.fusion_gate_tc])
        else:
            modules.append(self.fusion_gate)
        if self.seg_head: modules.append(self.seg_head)

        for m in modules:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None: nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    if hasattr(mod, "weight") and mod.weight is not None: nn.init.constant_(mod.weight, 1.0)
                    if hasattr(mod, "bias") and mod.bias is not None: nn.init.constant_(mod.bias, 0.0)

        # Head init
        for head in [self.rot_head, self.conf_head, self.kp_of_head]:
            nn.init.normal_(head[-1].weight, mean=0, std=0.01)
            nn.init.constant_(head[-1].bias, 0.0)

        nn.init.normal_(self.trans_head[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.trans_head[-1].bias, 0.0)
        # Z-bias
        with torch.no_grad():
            self.trans_head[-1].bias.data[2] = self.init_z_bias

        if self.seg_head:
            nn.init.normal_(self.seg_head.out_conv.weight, mean=0, std=0.01)
            nn.init.constant_(self.seg_head.out_conv.bias, 0.0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rgb = batch["rgb"]  # [B,3,H,W]
        points = batch["point_cloud"]  # [B,N,3]
        intrinsic = batch["intrinsic"]  # [B,3,3]

        # [NEW] Retrieve choose indices if available (for robust mapping)
        choose = batch.get("choose", None)  # [B,N] or None

        B, _, H, W = rgb.shape
        N = points.shape[1]

        # 0. Point Normalize
        if getattr(self.cfg, "pc_centering", True):
            pc_c = points.mean(dim=1, keepdim=True)
            pc_normed = points - pc_c
        else:
            pc_normed = points

        if getattr(self.cfg, "pc_scale_norm", True):
            r = torch.norm(pc_normed, dim=-1, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            pc_normed = pc_normed / r

        # 1. RGB Backbone
        feat_out = self.rgb_backbone(rgb)

        if self.return_intermediate and isinstance(feat_out, (tuple, list)):
            h, l = feat_out
            h = self.rgb_reduce(h)
            l = self.rgb_low_reduce(l)
            h = F.interpolate(h, size=l.shape[-2:], mode="bilinear", align_corners=False)
            feat_map = torch.cat([h, l], dim=1)
        else:
            if isinstance(feat_out, (tuple, list)): feat_out = feat_out[0]
            feat_map = self.rgb_reduce(feat_out)

        # 2. Seg Head (2D)
        pred_mask_logits = None
        if self.use_seg_head and self.seg_head is not None:
            seg_in = feat_map.detach() if self.seg_detach_trunk else feat_map
            # Force output size to original H,W for strict supervision
            pred_mask_logits = self.seg_head(seg_in, out_hw=(H, W))  # [B,1,H,W]

        # 3. Geo Branch
        geo_emb = self.point_encoder(pc_normed.transpose(1, 2))

        # 4. Fusion
        # Note: We likely upsampled feat_map in backbone, but check if it matches H,W
        # If feat_map is smaller, we rely on UV interpolation.
        # If SegHead is present, pred_mask_logits is strictly H,W.
        fused_raw, rgb_emb, uv = self.extract_and_fuse(
            rgb_feat=feat_map,
            geo_feat=geo_emb,
            points=points,
            intrinsic=intrinsic,
            img_shape=(H, W),
            choose=choose,  # [NEW] Pass choose for exact sampling if dimensions match
            return_uv=True
        )

        if not self.split_fusion_heads:
            gate = self.fusion_gate(fused_raw)
            feat_fused = rgb_emb * gate + geo_emb * (1.0 - gate)
            feat_rot, feat_tc = feat_fused, feat_fused
        else:
            gr = self.fusion_gate_rot(fused_raw)
            gtc = self.fusion_gate_tc(fused_raw)
            feat_rot = rgb_emb * gr + geo_emb * (1.0 - gr)
            feat_tc = rgb_emb * gtc + geo_emb * (1.0 - gtc)

        # 4.5 Seg -> Points (Validation Mask)
        pred_valid_mask, pred_valid_mask_bool = None, None
        if pred_mask_logits is not None:
            mask_prob_2d = torch.sigmoid(pred_mask_logits)
            # [CRITICAL] Use choose for mask sampling! Mask is strictly H,W.
            pred_valid_mask = self.sample_map_to_points(
                mask_prob_2d, uv, img_size=(H, W), choose=choose
            )
            pred_valid_mask = pred_valid_mask.clamp(0.0, 1.0)
            pred_valid_mask_bool = (pred_valid_mask > self.seg_point_thresh) & (points[..., 2] > 1e-6)

        # 5. Heads
        points_t = points.transpose(1, 2)
        base_in = torch.cat([feat_tc, points_t], dim=1)

        pred_q = F.normalize(self.rot_head(feat_rot).permute(0, 2, 1), dim=-1)
        pred_t = self.trans_head(base_in).permute(0, 2, 1)
        pred_c = torch.sigmoid(self.conf_head(base_in).permute(0, 2, 1))

        # In-model suppression (optional)
        if (self.pose_fusion_mask_conf_in_model and
                self.pose_fusion_use_valid_mask and
                self.pose_fusion_valid_mask_source == "seg" and
                pred_valid_mask is not None):
            pred_c = pred_c * pred_valid_mask.unsqueeze(-1).clamp_min(self.pose_fusion_conf_floor)

        # KP Offset
        pred_kp_ofs = None
        if float(getattr(self.cfg, "lambda_kp_of", 0.0)) > 0:
            k_raw = self.kp_of_head(base_in)
            k_raw = k_raw.view(B, self.num_keypoints, 3, N)
            pred_kp_ofs = k_raw.permute(0, 1, 3, 2).contiguous()

        return {
            "pred_quat": pred_q,
            "pred_trans": pred_t,
            "pred_conf": pred_c,
            "pred_kp_ofs": pred_kp_ofs,
            "pred_mask_logits": pred_mask_logits,
            "pred_valid_mask": pred_valid_mask,
            "pred_valid_mask_bool": pred_valid_mask_bool,
            "points": points,
        }


__all__ = ["LGFF_SC_SEG", "SegHead2D"]