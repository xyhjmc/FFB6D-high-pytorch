# lgff/models/lgff_cc.py

import torch
import torch.nn as nn
from .lgff_base import LGFFBase

class LGFF_CC(LGFFBase):
    """
    Class-Conditional LGFF network (带类嵌入 + FiLM 调制)
    """
    def __init__(self, num_classes, cls_embed_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.cls_embed = nn.Embedding(num_classes, cls_embed_dim)
        # TODO: 定义 FiLM / 条件 MLP, 例如:
        # self.film_mlp = nn.Sequential(...)
        # 并在 fusion/head 之前插入调制逻辑

    def forward(self, rgb, pts_xyz, pts_uv_norm, cls_ids):
        """
        在 LGFFBase 流程基础上，加入 cls_ids 的条件调制。
        """
        # TODO: 你可以调用父类一些中间函数，或者复制 forward 逻辑，
        # 在 fusion 之前/之后注入 class embedding.
        raise NotImplementedError
