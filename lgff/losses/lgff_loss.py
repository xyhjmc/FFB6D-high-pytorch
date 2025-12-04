"""Loss functions for LGFF assembled from reusable loss blocks."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit


class LGFFLoss(nn.Module):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gt_pose = batch["pose"].to(outputs["pose"].device)
        rot_loss = self.mse(outputs["pose"][:, :, :3], gt_pose[:, :, :3])
        trans_loss = self.mse(outputs["pose"][:, :, 3], gt_pose[:, :, 3])

        kp_loss = self.l1(
            outputs["kps_pred"], batch["kps_3d"].to(outputs["pose"].device)
        )

        loss = rot_loss + trans_loss + kp_loss

        metrics = {
            "rot_l2": rot_loss.item(),
            "trans_l2": trans_loss.item(),
            "kp_l1": kp_loss.item(),
        }

        # Optional ADD metric when GT model points are available.
        if "model_points" in batch:
            add_vals = []
            for pred_rt, gt_rt, pts in zip(
                outputs["pose"], gt_pose, batch["model_points"].to(outputs["pose"].device)
            ):
                add_vals.append(self.geometry.compute_add(pred_rt, gt_rt, pts))
            add = torch.stack(add_vals).mean()
            metrics["add"] = add.item()
            loss = loss + add

        return loss, metrics


__all__ = ["LGFFLoss"]
