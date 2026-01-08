# lgff/models/lgff_base_seg.py
"""
Base LGFF network components (Segmentation Variant).
Encapsulates the core RGB-D fusion logic (Projection & Sampling).
Renamed to LGFFBaseSeg to avoid conflict with original LGFFBase.
"""
from __future__ import annotations

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lgff.utils.geometry import GeometryToolkit


class LGFFBaseSeg(nn.Module):
    """
    LGFF Seg 专用几何辅助基类：
    - [NEW] 基于索引的精确采样 (Exact Index Sampling)
    - 3D → 2D 投影 (Legacy/Fallback)
    - 特征图 UV 采样 (Legacy/Fallback)
    - 简单的 RGB / 几何特征拼接
    - 四元数 + 平移 → 位姿矩阵
    """

    def __init__(self, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.geometry = geometry

    # ------------------------------------------------------------------
    # [NEW & CRITICAL] 精确索引采样 (比 UV 投影更准)
    # ------------------------------------------------------------------
    def sample_by_indices(
        self,
        feature_map: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample features using flattened indices (e.g. 'choose' from dataloader).
        Directly gathers pixels without reprojection error.

        Args:
            feature_map: [B, C, H, W]
            indices:     [B, N]  (values in 0..H*W-1)

        Returns:
            sampled:     [B, C, N]
        """
        B, C, H, W = feature_map.shape
        N = indices.shape[1]

        # Flatten features: [B, C, H, W] -> [B, C, H*W]
        flat_feats = feature_map.view(B, C, -1)
        
        # Expand indices: [B, N] -> [B, C, N]
        idx_expanded = indices.unsqueeze(1).expand(-1, C, -1)
        
        # Gather: [B, C, N]
        sampled = torch.gather(flat_feats, 2, idx_expanded)
        return sampled

    # ------------------------------------------------------------------
    # 3D → 2D 投影 (Fallback)
    # ------------------------------------------------------------------
    def project_points(
        self,
        points: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D points to 2D pixel coordinates.
        """
        eps = 1e-6
        z = points[..., 2].clamp(min=eps)
        x = points[..., 0]
        y = points[..., 1]

        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)
        if intrinsic.size(0) == 1 and points.size(0) > 1:
            intrinsic = intrinsic.expand(points.size(0), -1, -1)

        fx = intrinsic[:, 0, 0].unsqueeze(1)
        fy = intrinsic[:, 1, 1].unsqueeze(1)
        cx = intrinsic[:, 0, 2].unsqueeze(1)
        cy = intrinsic[:, 1, 2].unsqueeze(1)

        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        uv = torch.stack([u, v], dim=-1)
        return uv

    # ------------------------------------------------------------------
    # 在特征图上采样点特征 (Interpolated)
    # ------------------------------------------------------------------
    def sample_features(
        self,
        feature_map: torch.Tensor,
        uv: torch.Tensor,
        img_size: Tuple[int, int],
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Sample features using UV coordinates (Grid Sample).
        """
        B, C, H_f, W_f = feature_map.shape
        H_in, W_in = img_size

        u = uv[..., 0]
        v = uv[..., 1]

        if W_in > 1: scale_x = (W_f - 1.0) / (W_in - 1.0)
        else: scale_x = 1.0
        
        if H_in > 1: scale_y = (H_f - 1.0) / (H_in - 1.0)
        else: scale_y = 1.0

        u_feat = u * scale_x
        v_feat = v * scale_y

        # Normalize to [-1, 1]
        if W_f > 1: u_norm = 2.0 * (u_feat / (W_f - 1.0)) - 1.0
        else: u_norm = torch.zeros_like(u_feat)
        
        if H_f > 1: v_norm = 2.0 * (v_feat / (H_f - 1.0)) - 1.0
        else: v_norm = torch.zeros_like(v_feat)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2) # [B,N,1,2]

        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
        ) # [B, C, N, 1]

        return sampled.squeeze(3)

    # ------------------------------------------------------------------
    # Map (Seg) -> Points
    # ------------------------------------------------------------------
    def sample_map_to_points(
        self,
        map_2d: torch.Tensor,
        uv: torch.Tensor,
        img_size: Tuple[int, int],
        choose: Optional[torch.Tensor] = None, # [NEW]
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Smart sampling: uses 'choose' indices if available (Exact), else UV (Interpolated).
        """
        if choose is not None:
            # Exact sampling
            sampled = self.sample_by_indices(map_2d, choose)
        else:
            # Fallback projection sampling
            sampled = self.sample_features(map_2d, uv, img_size, align_corners=align_corners)
        
        if sampled.size(1) == 1:
            return sampled[:, 0, :] # [B, N]
        return sampled # [B, C, N]

    # ------------------------------------------------------------------
    # 投影 + 采样 + 融合
    # ------------------------------------------------------------------
    def extract_and_fuse(
        self,
        rgb_feat: torch.Tensor,
        geo_feat: torch.Tensor,
        points: torch.Tensor,
        intrinsic: torch.Tensor,
        img_shape: Tuple[int, int],
        choose: Optional[torch.Tensor] = None, # [NEW]
        valid_mask: Optional[torch.Tensor] = None,
        apply_valid_mask: bool = False,
        return_uv: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        uv = self.project_points(points, intrinsic) # Always compute UV for fallback or debug
        
        # Use Exact Index Sampling if available and resolution matches
        use_choose = (choose is not None) and (rgb_feat.shape[-2:] == img_shape)
        
        if use_choose:
            rgb_emb = self.sample_by_indices(rgb_feat, choose)
        else:
            rgb_emb = self.sample_features(rgb_feat, uv, img_shape)

        if apply_valid_mask and valid_mask is not None:
            m = valid_mask.to(dtype=rgb_emb.dtype, device=rgb_emb.device).unsqueeze(1)
            rgb_emb = rgb_emb * m
            geo_feat = geo_feat * m

        fused = torch.cat([rgb_emb, geo_feat], dim=1)

        if return_uv:
            return fused, rgb_emb, uv
        return fused, rgb_emb

    # ------------------------------------------------------------------
    # Pose Construction
    # ------------------------------------------------------------------
    def build_pose(self, quat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        if quat.dim() == 2 and quat.size(1) == 4 and trans.dim() == 2 and trans.size(1) == 3:
            rot = self.geometry.quat_to_rot(quat)
            trans_ = trans.view(-1, 3, 1)
            return torch.cat([rot, trans_], dim=-1)

        if (
            quat.dim() == 3 and quat.size(-1) == 4 and
            trans.dim() == 3 and trans.size(-1) == 3
        ):
            B, N, _ = quat.shape
            quat_flat = quat.reshape(-1, 4)
            rot_flat = self.geometry.quat_to_rot(quat_flat)
            rot = rot_flat.view(B, N, 3, 3)

            trans_flat = trans.reshape(-1, 3, 1)
            trans_ = trans_flat.view(B, N, 3, 1)
            return torch.cat([rot, trans_], dim=-1)

        raise ValueError("Unsupported shape for build_pose")

    def forward(self, *args, **kwargs):
        raise NotImplementedError