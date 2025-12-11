"""
LGFF 的几何工具脚本，在 FFB6D 工具基础上提供轻量封装。
主要暴露 ``set_camera_intrinsics``、``pcld_processor`` 等接口，包装
相机内参设置、点云与深度图转换、位姿估计与点云采样流程；通过复用
FFB6D 的 ``Basic_Utils`` 与 ``ImgPcldUtils``，为上层模型提供统一的几
何处理入口。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from common.ffb6d_utils.basic_utils import (
    Basic_Utils,
    best_fit_transform,
    intrinsic_matrix as ffb6d_intrinsics,
)
from common.ffb6d_utils.dataset_tools.utils import ImgPcldUtils


class GeometryToolkit:
    """Convenience wrapper around FFB6D geometry utilities."""

    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        # ``Basic_Utils`` requires a config-like object with ``dataset_name``.
        cfg = SimpleNamespace(dataset_name="custom")
        self.bs_utils = Basic_Utils(cfg)
        self.img_utils = ImgPcldUtils()
        self.camera_matrix = (
            np.asarray(camera_matrix) if camera_matrix is not None else ffb6d_intrinsics["ycb_K1"]
        )

    def depth_to_point_cloud(
        self, depth: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a depth map to a point cloud using FFB6D's implementation."""

        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        return self.bs_utils.dpt_2_cld(depth, depth_scale, K)

    def depth_image_backproject(
        self, depth: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Dense back-projection using the dataset tools helper."""

        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        return self.img_utils.K_dpt_2_cld(depth, depth_scale, K)

    def project_points(
        self, points: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Project camera-frame points in **meters** with cam_scale=1.0 (no unit rescale)."""
        K = camera_matrix if camera_matrix is not None else self.camera_matrix
        return self.bs_utils.project_p3d(points, depth_scale, K)

    def compute_add(self, pred_rt: torch.Tensor, gt_rt: torch.Tensor, model_points: torch.Tensor) -> torch.Tensor:
        return self.bs_utils.cal_add_cuda(pred_rt, gt_rt, model_points)

    def compute_adds(
        self, pred_rt: torch.Tensor, gt_rt: torch.Tensor, model_points: torch.Tensor
    ) -> torch.Tensor:
        return self.bs_utils.cal_adds_cuda(pred_rt, gt_rt, model_points)

    def quat_to_rot(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert a quaternion to a rotation matrix."""

        quat = torch.nn.functional.normalize(quat, dim=-1)
        w, x, y, z = quat.unbind(-1)
        B = quat.shape[0]
        rot = torch.zeros((B, 3, 3), device=quat.device, dtype=quat.dtype)
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - z * w)
        rot[:, 0, 2] = 2 * (x * z + y * w)
        rot[:, 1, 0] = 2 * (x * y + z * w)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - x * w)
        rot[:, 2, 0] = 2 * (x * z - y * w)
        rot[:, 2, 1] = 2 * (y * z + x * w)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot

    @staticmethod
    def best_fit_transform(src: np.ndarray, dst: np.ndarray):
        return best_fit_transform(src, dst)

 # ---------- 新增：旋转误差（输入旋转矩阵） ----------
    @staticmethod
    def rotation_error_from_mats(
        R_pred: torch.Tensor,
        R_gt: torch.Tensor,
        return_deg: bool = True,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        计算旋转矩阵之间的角度误差（批量）。
        Args:
            R_pred: [..., 3, 3]
            R_gt:   [..., 3, 3]
        Returns:
            rot_err: [...], 单位为度(默认)或弧度
        """
        # 相对旋转 R_rel = R_gt^T * R_pred
        R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)  # [..., 3, 3]
        # trace = 1 + 2 cos(theta)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        # 数值稳定：裁剪到 [-1, 3] 的对应 cos 范围
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)  # 弧度
        if return_deg:
            theta = theta * (180.0 / np.pi)
        return theta

    # ---------- 新增：旋转误差（输入四元数） ----------
    @staticmethod
    def rotation_error_from_quat(
        q_pred: torch.Tensor,
        q_gt: torch.Tensor,
        return_deg: bool = True,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        计算四元数之间的角度误差（批量）。
        Args:
            q_pred: [..., 4] (w, x, y, z 或 x, y, z, w 均可，只要一致)
            q_gt:   [..., 4]
        Returns:
            rot_err: [...], 单位为度(默认)或弧度
        """
        # 规范化
        q_pred = F.normalize(q_pred, dim=-1)
        q_gt = F.normalize(q_gt, dim=-1)

        # 点积对应 cos(theta/2)，注意要取绝对值处理 q 和 -q 等价
        dot = torch.sum(q_pred * q_gt, dim=-1).abs()  # [...]
        dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

        # 相对旋转角: theta = 2 * arccos(dot)
        theta = 2.0 * torch.acos(dot)  # 弧度
        if return_deg:
            theta = theta * (180.0 / np.pi)
        return theta

def build_geometry(camera_intrinsic: Optional[np.ndarray] = None) -> GeometryToolkit:
    return GeometryToolkit(camera_intrinsic)


def depth_to_point_cloud(
    toolkit: GeometryToolkit, depth: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    return toolkit.depth_to_point_cloud(depth, depth_scale, camera_matrix)


def dense_backproject(
    toolkit: GeometryToolkit, depth: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    return toolkit.depth_image_backproject(depth, depth_scale, camera_matrix)


def project_points(
    toolkit: GeometryToolkit, points: np.ndarray, depth_scale: float, camera_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    return toolkit.project_points(points, depth_scale, camera_matrix)


def add_metric(toolkit: GeometryToolkit, pred_rt: torch.Tensor, gt_rt: torch.Tensor, model_points: torch.Tensor):
    return toolkit.compute_add(pred_rt, gt_rt, model_points)


def adds_metric(toolkit: GeometryToolkit, pred_rt: torch.Tensor, gt_rt: torch.Tensor, model_points: torch.Tensor):
    return toolkit.compute_adds(pred_rt, gt_rt, model_points)


def quaternion_to_matrix(toolkit: GeometryToolkit, quat: torch.Tensor) -> torch.Tensor:
    return toolkit.quat_to_rot(quat)


def best_fit_transform(src: np.ndarray, dst: np.ndarray):
    return GeometryToolkit.best_fit_transform(src, dst)

def rotation_error_from_mats(
    toolkit: GeometryToolkit,
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    return_deg: bool = True,
) -> torch.Tensor:
    return toolkit.rotation_error_from_mats(R_pred, R_gt, return_deg=return_deg)


def rotation_error_from_quat(
    toolkit: GeometryToolkit,
    q_pred: torch.Tensor,
    q_gt: torch.Tensor,
    return_deg: bool = True,
) -> torch.Tensor:
    return toolkit.rotation_error_from_quat(q_pred, q_gt, return_deg=return_deg)
__all__ = [
    "GeometryToolkit",
    "add_metric",
    "adds_metric",
    "best_fit_transform",
    "build_geometry",
    "dense_backproject",
    "depth_to_point_cloud",
    "project_points",
    "quaternion_to_matrix",
]

