"""Geometry helpers shared across LGFF and legacy FFB6D code.

A number of geometric routines already exist inside ``ffb6d``.  This
module provides a thin wrapper that reuses those implementations instead
of re-creating them from scratch.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch

from ffb6d.utils.basic_utils import (
    Basic_Utils,
    best_fit_transform,
    intrinsic_matrix as ffb6d_intrinsics,
)
from ffb6d.utils.dataset_tools.utils import ImgPcldUtils


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


__all__ = ["GeometryToolkit"]
