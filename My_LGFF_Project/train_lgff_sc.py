"""单类别LGFF模型的训练入口，负责构建数据、模型并驱动训练循环。"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common import load_config, get_logger
from lgff.datasets import SingleObjectDataset
from lgff.engines import TrainerSC
from lgff.losses import LGFFLoss
from lgff.models import LGFF_SC
from lgff.utils import build_geometry


def main():
    cfg = load_config()
    logger = get_logger("lgff.sc", log_file=f"{cfg.log_dir}/train.log")
    geometry = build_geometry(cfg.camera_intrinsic)

    train_ds = SingleObjectDataset(cfg, split="train")
    val_ds = SingleObjectDataset(cfg, split="val") if cfg.annotation_file else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        if val_ds
        else None
    )

    model = LGFF_SC(cfg, geometry)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = LGFFLoss(cfg, geometry)

    trainer = TrainerSC(model, optimizer, loss_fn, train_loader, val_loader, cfg, logger)
    trainer.train()


if __name__ == "__main__":
    main()
