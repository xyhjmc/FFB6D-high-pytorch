"""Single-class LGFF model that lightly wraps the base fusion network."""
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
