# tools/train_lgff_sc_seg.py
"""
Training entry point for Single-Class LGFF (Segmentation Variant).
Updates:
- Uses LGFFConfigSeg from config_seg.py
- Uses TrainerSCSeg and SingleObjectDatasetSeg
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# [CHANGED] Correct imports for Seg components
from lgff.utils.config_seg import LGFFConfigSeg, load_config
from lgff.utils.logger import setup_logger

try:
    from common.geometry import GeometryToolkit
except ImportError:
    from lgff.utils.geometry import GeometryToolkit

# [CHANGED] Direct import of renamed classes
from lgff.datasets.single_loader_seg import SingleObjectDatasetSeg
from lgff.engines.trainer_sc_seg import TrainerSCSeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LGFF SC Seg", add_help=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Note: --config and --opt are parsed by load_config internally
    args, _ = parser.parse_known_args()
    return args


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _worker_init_fn_builder(base_seed: int):
    def _worker_init_fn(worker_id: int) -> None:
        ws = base_seed + worker_id
        np.random.seed(ws)
        random.seed(ws)

    return _worker_init_fn


def _build_model_and_loss(cfg: LGFFConfigSeg, geometry: GeometryToolkit):
    # Always use the Seg model/loss for this script
    from lgff.models.lgff_sc_seg import LGFF_SC_SEG
    from lgff.losses.lgff_loss_seg import LGFFLoss_SEG

    # Ensure flags are set (redundant safety)
    if not getattr(cfg, "use_seg_head", False):
        cfg.use_seg_head = True

    model = LGFF_SC_SEG(cfg, geometry)
    loss_fn = LGFFLoss_SEG(cfg, geometry)

    return model, loss_fn


def _check_seg_requirements(cfg: LGFFConfigSeg, train_ds: SingleObjectDatasetSeg, logger: logging.Logger) -> None:
    try:
        sample = train_ds[0]
    except Exception as e:
        logger.warning(f"[SegCheck] Failed to read train_ds[0]: {e}")
        return

    if "mask" not in sample:
        raise RuntimeError(
            "[SegCheck] Dataset sample has no 'mask' key. Check cfg.return_mask and seg_supervision_source."
        )
    logger.info("[SegCheck] Dataset provides 'mask' for training.")


def main() -> None:
    args = parse_args()

    # 1. Load Config (Seg version)
    cfg: LGFFConfigSeg = load_config()

    # 2. Work Dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if getattr(cfg, "work_dir", None) is None:
        cfg.work_dir = getattr(cfg, "log_dir", "output/debug_seg")
    os.makedirs(cfg.work_dir, exist_ok=True)

    # 3. Force Seg Flags
    cfg.use_seg_head = True
    cfg.return_mask = True
    if bool(getattr(cfg, "seg_ignore_invalid", True)):
        cfg.return_valid_mask = True

    # 4. Logger
    log_file = os.path.join(cfg.work_dir, "train.log")
    logger = setup_logger(log_file, name="lgff.train")
    logger.info(f"LGFF Seg Training | Work Dir: {cfg.work_dir}")

    # 5. Seed
    seed_value = int(args.seed) if args.seed is not None else int(getattr(cfg, "seed", 42))
    set_random_seed(seed_value, deterministic=bool(getattr(cfg, "deterministic", False)))

    worker_init_fn = _worker_init_fn_builder(seed_value)
    generator = torch.Generator()
    generator.manual_seed(seed_value)

    # 6. Geometry & Data
    geometry = GeometryToolkit()

    logger.info("Initializing Datasets...")
    train_ds = SingleObjectDatasetSeg(cfg, split="train")
    val_ds = SingleObjectDatasetSeg(cfg, split=getattr(cfg, "val_split", "test"))

    _check_seg_requirements(cfg, train_ds, logger)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    # 7. Model & Loss
    model, loss_fn = _build_model_and_loss(cfg, geometry)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model created on {device}")

    # 8. Trainer
    trainer = TrainerSCSeg(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=cfg.work_dir,
        resume_path=args.resume,
    )

    trainer.fit()


if __name__ == "__main__":
    main()