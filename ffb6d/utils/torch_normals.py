"""
PyTorch implementation of depth to normal map conversion.

The interface mirrors the original ``normalSpeed.depth_normal`` helper but
uses differentiable Torch kernels so it can run on GPU when available.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def _sobel_kernels(device: torch.device):
    sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=device)
    sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=device)
    sobel_x = (sobel_x / 8.0).view(1, 1, 3, 3)
    sobel_y = (sobel_y / 8.0).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(3, 1, 1, 1)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)
    return sobel_x, sobel_y


def depth_normal(
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    patch: int = 5,
    max_depth: float = 2000.0,
    smooth_size: int = 21,
    use_gpu: bool | None = None,
):
    """
    Estimate surface normals from a depth map using PyTorch operations.

    Args:
        depth_mm: Depth image in millimeters, shape (H, W).
        fx: Focal length in x direction.
        fy: Focal length in y direction.
        patch: Unused but kept for API compatibility with normalSpeed.
        max_depth: Depth values larger than this are treated as invalid.
        smooth_size: Optional mean filter size applied before normal estimation.
        use_gpu: If True and CUDA is available, run on GPU; otherwise CPU.

    Returns:
        Normal map with shape (H, W, 3) in the range [-1, 1]. Invalid regions
        are filled with zeros.
    """

    if depth_mm.ndim != 2:
        raise ValueError("depth_normal expects a 2D depth array")

    device = torch.device("cuda" if (use_gpu is not False and torch.cuda.is_available()) else "cpu")

    depth = torch.as_tensor(depth_mm, dtype=torch.float32, device=device)
    depth = torch.clamp(depth, max=max_depth)

    valid = depth > 0

    if smooth_size and smooth_size > 1:
        smooth_size = smooth_size | 1

        pad = smooth_size // 2
        depth = depth.unsqueeze(0).unsqueeze(0)
        depth = F.avg_pool2d(depth, kernel_size=smooth_size, stride=1, padding=pad)
        depth = depth.squeeze(0).squeeze(0)

    depth = depth / 1000.0  # convert to meters for numerical stability

    h, w = depth.shape
    ys = torch.arange(h, device=device).view(-1, 1).expand(-1, w)
    xs = torch.arange(w, device=device).view(1, -1).expand(h, -1)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = torch.stack((x, y, z), dim=0).unsqueeze(0)  # (1, 3, H, W)

    sobel_x, sobel_y = _sobel_kernels(device)
    grad_x = F.conv2d(pts, sobel_x, padding=1, groups=3).squeeze(0)
    grad_y = F.conv2d(pts, sobel_y, padding=1, groups=3).squeeze(0)

    grad_x = grad_x.permute(1, 2, 0)
    grad_y = grad_y.permute(1, 2, 0)
    normals = torch.cross(grad_y, grad_x, dim=2)
    norms = torch.linalg.norm(normals, dim=2, keepdim=True) + 1e-6
    normals = normals / norms

    normals = normals * valid.unsqueeze(-1)

    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    return normals.detach().cpu().numpy().astype(np.float32)
