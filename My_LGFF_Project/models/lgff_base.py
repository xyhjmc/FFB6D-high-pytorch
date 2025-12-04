# lgff/models/lgff_base.py

import torch
import torch.nn as nn
from .backbones import build_backbone
from .blocks import C2fBlock, SPPFLite  # 如果需要
# TODO: 引入 PointNetEncoder, PixelFeatureSampler, Fusion, KeypointHead (可拆文件或写在这里)

class LGFFBase(nn.Module):
    """
    Lightweight Geometric-Feature Fusion Network (Base)
    - 不含 class embedding / 条件头
    """
    def __init__(self, backbone_name="resnet18", num_keypoints=8, **kwargs):
        super().__init__()
        # 1. RGB backbone
        self.rgb_backbone = build_backbone(backbone_name, **kwargs)

        # 2. PointNetEncoder
        # TODO: 从你之前的简化 PointNet 设计中搬过来
        self.pc_encoder = ...

        # 3. PixelFeatureSampler
        self.pixel_sampler = ...

        # 4. Fusion (LBFM)
        self.fusion = ...

        # 5. Keypoint Head (不含类嵌入)
        self.head = ...

    def forward(self, rgb, pts_xyz, pts_uv_norm):
        """
        Args:
            rgb:        [B,3,H,W]
            pts_xyz:    [B,N,3]
            pts_uv_norm:[B,N,2] in [-1,1]
        Returns:
            kp_offsets: [B,N,K,3]
            mask_logits(optional): [B,N,1]
        """
        # TODO: 仿照之前给你的 FFB6D-light 流程填写：
        #  1. F_rgb = rgb_backbone(rgb)
        #  2. feat_pts = pixel_sampler(F_rgb, pts_uv_norm)
        #  3. g3d = pc_encoder(pts_xyz)
        #  4. fused = fusion(feat_pts, pts_xyz, g3d)
        #  5. kp_offsets, mask_logits = head(fused)
        raise NotImplementedError
