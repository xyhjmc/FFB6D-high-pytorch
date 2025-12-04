"""Single-class LGFF model skeleton."""
from __future__ import annotations

import torch
from torch import nn

from common.config import LGFFConfig
from common.geometry import GeometryToolkit
from lgff.models.lgff_base import LGFFBase


class LGFF_SC(LGFFBase):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__(geometry)
        self.cfg = cfg
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(inplace=True),
        )
        self.global_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.pose_head = nn.Linear(64, 7)
        self.kp_head = nn.Conv1d(256, cfg.num_keypoints * 3, 1)

    def forward(self, batch):
        points = batch["point_cloud"]  # B x N x 3
        points = points.transpose(1, 2)  # B x 3 x N
        feat = self.point_encoder(points)
        global_feat = torch.max(feat, dim=2)[0]

        latent = self.global_head(global_feat)
        pose_params = self.pose_head(latent)
        quat = pose_params[:, :4]
        trans = pose_params[:, 4:]
        pose = self.build_pose(quat, trans)

        kp_pred = self.kp_head(feat).view(
            points.shape[0], self.cfg.num_keypoints, 3, points.shape[2]
        )
        kp_pred = kp_pred.mean(dim=-1)

        return {
            "pose": pose,
            "quat": quat,
            "trans": trans,
            "kps_pred": kp_pred,
        }


__all__ = ["LGFF_SC"]
