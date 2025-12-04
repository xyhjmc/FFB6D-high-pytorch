"""Base LGFF network components."""
from __future__ import annotations

import torch
from torch import nn

from lgff.utils.geometry import GeometryToolkit


class LGFFBase(nn.Module):
    def __init__(self, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.geometry = geometry

    def build_pose(self, quat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        rot = self.geometry.quat_to_rot(quat)
        return torch.cat([rot, trans.unsqueeze(-1)], dim=-1)

    def forward(self, batch):  # pragma: no cover - interface only
        raise NotImplementedError
