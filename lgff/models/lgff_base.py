"""
Base LGFF network components.
Encapsulates the core RGB-D fusion logic (Projection & Sampling).
"""
from __future__ import annotations

from typing import Tuple, Optional, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from lgff.utils.geometry import GeometryToolkit


class LGFFBase(nn.Module):
    """
    LGFF 的几何辅助基类：
    - 3D → 2D 投影
    - 在特征图上按点采样
    - 简单的 RGB / 几何特征拼接
    - 四元数 + 平移 → 位姿矩阵（支持全局 / 逐点两种形式）
    """

    def __init__(self, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.geometry = geometry

    # ------------------------------------------------------------------
    # 3D → 2D 投影
    # ------------------------------------------------------------------
    def project_points(
        self,
        points: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            points:    [B, N, 3]  点云 (x, y, z) in camera coordinates.
            intrinsic: [B, 3, 3]  或 [1, 3, 3] 相机内参 (fx, fy, cx, cy).

        Returns:
            uv: [B, N, 2]  投影后的像素坐标 (u, v) in image space.
        """
        eps = 1e-6
        z = points[..., 2].clamp(min=eps)  # [B, N]
        x = points[..., 0]
        y = points[..., 1]

        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)  # [1,3,3]
        if intrinsic.size(0) == 1 and points.size(0) > 1:
            intrinsic = intrinsic.expand(points.size(0), -1, -1)

        fx = intrinsic[:, 0, 0].unsqueeze(1)
        fy = intrinsic[:, 1, 1].unsqueeze(1)
        cx = intrinsic[:, 0, 2].unsqueeze(1)
        cy = intrinsic[:, 1, 2].unsqueeze(1)

        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        uv = torch.stack([u, v], dim=-1)  # [B, N, 2]
        return uv

    # ------------------------------------------------------------------
    # 在特征图上采样点特征
    # ------------------------------------------------------------------
    def sample_features(
        self,
        feature_map: torch.Tensor,
        uv: torch.Tensor,
        img_size: Tuple[int, int],
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Sample features from a 2D feature map using 2D coordinates.

        Args:
            feature_map: [B, C, H_f, W_f]
            uv:          [B, N, 2]        原图像素坐标 (u, v).
            img_size:    (H_in, W_in)     uv 所在图像分辨率
            align_corners: grid_sample 参数

        Returns:
            sampled_features: [B, C, N]
        """
        B, C, H_f, W_f = feature_map.shape
        H_in, W_in = img_size

        u = uv[..., 0]  # [B,N]
        v = uv[..., 1]  # [B,N]

        # 原图坐标 → 特征图坐标
        if W_in > 1:
            scale_x = (W_f - 1.0) / (W_in - 1.0)
        else:
            scale_x = 1.0
        if H_in > 1:
            scale_y = (H_f - 1.0) / (H_in - 1.0)
        else:
            scale_y = 1.0

        u_feat = u * scale_x
        v_feat = v * scale_y

        # 特征图坐标 → [-1,1]
        if W_f > 1:
            u_norm = 2.0 * (u_feat / (W_f - 1.0)) - 1.0
        else:
            u_norm = torch.zeros_like(u_feat)
        if H_f > 1:
            v_norm = 2.0 * (v_feat / (H_f - 1.0)) - 1.0
        else:
            v_norm = torch.zeros_like(v_feat)

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)  # [B,N,1,2]

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )  # [B, C, N, 1]

        return sampled.squeeze(3)  # [B, C, N]

    # ------------------------------------------------------------------
    # 2D map(例如 seg logits/prob) → 点级采样
    # ------------------------------------------------------------------
    def sample_map_to_points(
        self,
        map_2d: torch.Tensor,
        uv: torch.Tensor,
        img_size: Tuple[int, int],
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        Sample a 2D map (e.g., seg logits/prob) to per-point values.

        Args:
            map_2d:   [B, 1, H, W] or [B, C, H, W]
            uv:       [B, N, 2] pixel coords in the same image space as img_size
            img_size: (H, W) of the image space where uv is defined

        Returns:
            sampled:  [B, N] if C==1 else [B, C, N]
        """
        sampled = self.sample_features(map_2d, uv, img_size, align_corners=align_corners)  # [B,C,N]
        if sampled.size(1) == 1:
            return sampled[:, 0, :]  # [B,N]
        return sampled  # [B,C,N]

    # ------------------------------------------------------------------
    # 投影 + 采样 + 简单融合
    # ------------------------------------------------------------------
    def extract_and_fuse(
        self,
        rgb_feat: torch.Tensor,
        geo_feat: torch.Tensor,
        points: torch.Tensor,
        intrinsic: torch.Tensor,
        img_shape: Tuple[int, int],
        valid_mask: Optional[torch.Tensor] = None,      # [B,N] bool/0-1
        apply_valid_mask: bool = False,
        return_uv: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Perform projection, sampling, and basic fusion.

        Returns:
            fused_feat: [B, C_rgb + C_geo, N]
            rgb_emb:    [B, C_rgb, N]
            uv:         [B, N, 2] (only if return_uv=True)
        """
        uv = self.project_points(points, intrinsic)  # [B,N,2]
        rgb_emb = self.sample_features(rgb_feat, uv, img_shape)  # [B,C_rgb,N]

        if apply_valid_mask and valid_mask is not None:
            m = valid_mask.to(dtype=rgb_emb.dtype, device=rgb_emb.device).unsqueeze(1)  # [B,1,N]
            rgb_emb = rgb_emb * m
            geo_feat = geo_feat * m

        fused = torch.cat([rgb_emb, geo_feat], dim=1)  # [B,C_rgb+C_geo,N]

        if return_uv:
            return fused, rgb_emb, uv
        return fused, rgb_emb

    # ------------------------------------------------------------------
    # 四元数 + 平移 → 位姿
    # ------------------------------------------------------------------
    def build_pose(self, quat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        if quat.dim() == 2 and quat.size(1) == 4 and trans.dim() == 2 and trans.size(1) == 3:
            rot = self.geometry.quat_to_rot(quat)
            trans_ = trans.view(-1, 3, 1)
            return torch.cat([rot, trans_], dim=-1)

        if (
            quat.dim() == 3 and quat.size(-1) == 4 and
            trans.dim() == 3 and trans.size(-1) == 3 and
            quat.shape[:2] == trans.shape[:2]
        ):
            B, N, _ = quat.shape
            quat_flat = quat.reshape(-1, 4)
            rot_flat = self.geometry.quat_to_rot(quat_flat)  # [B*N,3,3]
            rot = rot_flat.view(B, N, 3, 3)

            trans_flat = trans.reshape(-1, 3, 1)
            trans_ = trans_flat.view(B, N, 3, 1)
            return torch.cat([rot, trans_], dim=-1)

        raise ValueError(
            f"Unsupported shape for build_pose: quat {quat.shape}, trans {trans.shape}. "
            "Expected [B,4]+[B,3] or [B,N,4]+[B,N,3]."
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
