"""
Base LGFF network components.
Encapsulates the core RGB-D fusion logic (Projection & Sampling).
"""
from __future__ import annotations

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

    子类（LGFF_SC / LGFF_CC）负责：
    - backbone / PointNet / fusion / head 的具体实现
    - forward 的完整数据流
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

        # points: [B, N, 3]
        z = points[..., 2].clamp(min=eps)  # [B, N]
        x = points[..., 0]
        y = points[..., 1]

        # intrinsic: [B, 3, 3] or [1, 3, 3]
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)  # [1,3,3]

        if intrinsic.size(0) == 1 and points.size(0) > 1:
            # 扩展成 batch 大小
            intrinsic = intrinsic.expand(points.size(0), -1, -1)

        fx = intrinsic[:, 0, 0].unsqueeze(1)  # [B,1]
        fy = intrinsic[:, 1, 1].unsqueeze(1)
        cx = intrinsic[:, 0, 2].unsqueeze(1)
        cy = intrinsic[:, 1, 2].unsqueeze(1)

        # u = (x * fx) / z + cx, v = (y * fy) / z + cy
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
        img_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Sample features from a 2D feature map using 2D coordinates.
        负责：
        1) 将原图坐标 (u, v) 映射到特征图坐标系；
        2) 再归一化到 [-1,1]，传给 grid_sample。

        Args:
            feature_map: [B, C, H_f, W_f]  backbone 输出的特征图.
            uv:          [B, N, 2]        原图像素坐标 (u, v).
            img_size:    (H_in, W_in)     对应 uv 的原始图像分辨率。

        Returns:
            sampled_features: [B, C, N]   每个点插值得到的特征向量.
        """
        B, C, H_f, W_f = feature_map.shape
        H_in, W_in = img_size

        # 拆分 u, v (像素坐标)
        u = uv[..., 0]  # [B,N]
        v = uv[..., 1]  # [B,N]

        # 1) 原图坐标 → 特征图坐标
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

        # 2) 特征图坐标 → [-1,1]
        if W_f > 1:
            u_norm = 2.0 * (u_feat / (W_f - 1.0)) - 1.0
        else:
            u_norm = torch.zeros_like(u_feat)

        if H_f > 1:
            v_norm = 2.0 * (v_feat / (H_f - 1.0)) - 1.0
        else:
            v_norm = torch.zeros_like(v_feat)

        # grid: [B, N, 1, 2]  (treat N as "width", height=1)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)  # [B,N,1,2]

        # 3) bilinear sample
        # output: [B, C, N, 1]
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # -> [B, C, N]
        sampled = sampled.squeeze(3)
        return sampled

    # ------------------------------------------------------------------
    # 投影 + 采样 + 简单融合
    # ------------------------------------------------------------------
    def extract_and_fuse(
        self,
        rgb_feat: torch.Tensor,
        geo_feat: torch.Tensor,
        points: torch.Tensor,
        intrinsic: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        High-level helper to perform projection, sampling, and basic fusion.

        Args:
            rgb_feat:  [B, C_rgb, H_f, W_f]  RGB feature map from backbone.
            geo_feat:  [B, C_geo, N]        点级几何特征 (例如 PointNet 输出).
            points:    [B, N, 3]            点云 (camera coords).
            intrinsic: [B, 3, 3] or [1,3,3] 相机内参.
            img_shape: (H_in, W_in)         原图分辨率 (与 uv 对应).

        Returns:
            fused_feat: [B, C_rgb + C_geo, N]  拼接后的特征 (用于后续 MLP/Attention).
            rgb_emb:   [B, C_rgb, N]          纯 RGB 点特征 (可用于可视化/辅助 loss).
        """
        # 1. 3D → 2D 投影 (像素坐标)
        uv = self.project_points(points, intrinsic)  # [B,N,2]

        # 2. 在特征图上采样 RGB 特征
        rgb_emb = self.sample_features(rgb_feat, uv, img_shape)  # [B,C_rgb,N]

        # 3. 简单拼接 (更多花活可以在子类覆写)
        fused = torch.cat([rgb_emb, geo_feat], dim=1)  # [B,C_rgb+C_geo,N]

        return fused, rgb_emb

    # ------------------------------------------------------------------
    # 四元数 + 平移 → 位姿（支持全局 / 逐点）
    # ------------------------------------------------------------------
    def build_pose(
        self,
        quat: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct pose matrix [R|t] from quaternion and translation.

        支持两种形式：
        1) 全局预测（per-object）:
           quat:  [B, 4]
           trans: [B, 3]
           -> pose: [B, 3, 4]

        2) 逐点预测（per-point / per-hypothesis）:
           quat:  [B, N, 4]
           trans: [B, N, 3]
           -> pose: [B, N, 3, 4]

        Args:
            quat:  [B,4] 或 [B,N,4]
            trans: [B,3] 或 [B,N,3]

        Returns:
            pose:  [B,3,4] 或 [B,N,3,4]，与输入形式对应。
        """
        # -------- case 1: 全局预测 [B,4] + [B,3] --------
        if quat.dim() == 2 and quat.size(1) == 4 and trans.dim() == 2 and trans.size(1) == 3:
            # [B,4] -> [B,3,3]
            rot = self.geometry.quat_to_rot(quat)  # 约定 geometry 支持 [B,4] 输入
            trans_ = trans.view(-1, 3, 1)          # [B,3] -> [B,3,1]
            pose = torch.cat([rot, trans_], dim=-1)  # [B,3,4]
            return pose

        # -------- case 2: 逐点预测 [B,N,4] + [B,N,3] --------
        if (
            quat.dim() == 3 and quat.size(-1) == 4 and
            trans.dim() == 3 and trans.size(-1) == 3 and
            quat.shape[:2] == trans.shape[:2]
        ):
            B, N, _ = quat.shape

            # [B,N,4] -> [B*N,4]
            quat_flat = quat.reshape(-1, 4)
            # [B*N,4] -> [B*N,3,3]
            rot_flat = self.geometry.quat_to_rot(quat_flat)
            # -> [B,N,3,3]
            rot = rot_flat.view(B, N, 3, 3)

            # [B,N,3] -> [B*N,3,1] -> [B,N,3,1]
            trans_flat = trans.reshape(-1, 3, 1)
            trans_ = trans_flat.view(B, N, 3, 1)

            # 拼 [R|t]: [B,N,3,4]
            pose = torch.cat([rot, trans_], dim=-1)
            return pose

        # 其它形状暂不支持，直接抛错方便调试
        raise ValueError(
            f"Unsupported shape for build_pose: quat {quat.shape}, trans {trans.shape}. "
            "Expected [B,4]+[B,3] or [B,N,4]+[B,N,3]."
        )

    # ------------------------------------------------------------------
    # 抽象接口：由 LGFF_SC / LGFF_CC 实现
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # pragma: no cover - interface only
        """
        Should be implemented by subclasses (LGFF_SC, LGFF_CC).
        """
        raise NotImplementedError
