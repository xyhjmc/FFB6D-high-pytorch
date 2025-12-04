"""LGFF-facing geometry helpers built on top of FFB6D utilities."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from common.geometry import GeometryToolkit


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


__all__ = [
    "add_metric",
    "adds_metric",
    "best_fit_transform",
    "build_geometry",
    "dense_backproject",
    "depth_to_point_cloud",
    "project_points",
    "quaternion_to_matrix",
]
