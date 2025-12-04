"""
单类别 LGFF 模型脚本，轻量封装基础融合网络供下游使用。
``LGFF_SC`` 继承 ``LGFFBase``，通过配置绑定类别数、点数与关键点数量，
并在初始化时传入 RandLA-Net 的配置，实现直接可训练的单类别模型。
"""
from __future__ import annotations

from typing import Any

from lgff.utils.config import LGFFConfig
from lgff.models.lgff_base import LGFFBase


class LGFF_SC(LGFFBase):
    def __init__(self, cfg: LGFFConfig, rndla_cfg: Any) -> None:
        super().__init__(
            n_classes=cfg.num_classes,
            n_pts=cfg.num_points,
            rndla_cfg=rndla_cfg,
            n_kps=cfg.num_keypoints,
        )
        self.cfg = cfg


__all__ = ["LGFF_SC"]
