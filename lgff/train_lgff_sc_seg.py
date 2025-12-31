# tools/train_lgff_sc.py
"""
Training entry point for Single-Class LGFF (supports variants: sc / sc_seg, base loss / seg loss).

Key design:
- Dataloader/Trainer are shared.
- Model/Loss are selected by cfg.model_variant / cfg.loss_variant.
- If seg is enabled, this script EXPECTS the dataset to provide a segmentation GT tensor
  (e.g., batch["seg_mask"] or batch["seg_gt"]) — otherwise it will warn.

Usage examples:
  python tools/train_lgff_sc.py --config configs/ape.yaml
  python tools/train_lgff_sc.py --config configs/ape.yaml --opt model_variant="sc_seg" loss_variant="seg" lambda_seg=1.0
  python tools/train_lgff_sc.py --config configs/ape.yaml --resume output/exp/checkpoint_last.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import logging
from typing import Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Path patch: allow running from project root or tools/
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from lgff.utils.config import LGFFConfig, load_config
from lgff.utils.logger import setup_logger

try:
    from common.geometry import GeometryToolkit
except ImportError:
    from lgff.utils.geometry import GeometryToolkit

from lgff.datasets.single_loader import SingleObjectDataset
from lgff.engines.trainer_sc import TrainerSC


def parse_args() -> argparse.Namespace:
    """Only parse args specific to this train launcher. --config/--opt are handled by load_config()."""
    parser = argparse.ArgumentParser(description="Train LGFF Single-Class (variant-aware)", add_help=True)

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., output/exp/checkpoint_last.pth)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override output directory (default: config.work_dir or config.log_dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (default: config.seed)",
    )

    args, _ = parser.parse_known_args()
    return args


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed; deterministic=True is more reproducible but may be slower."""
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


def _build_model_and_loss(cfg: LGFFConfig, geometry: GeometryToolkit):
    """
    Select model/loss by config. Keeps Trainer/Dataloader shared.
    """
    # ----------------- Model -----------------
    mv = getattr(cfg, "model_variant", "sc")
    if mv == "sc_seg":
        # expects you created: lgff/models/lgff_sc_seg.py
        from lgff.models.lgff_sc_seg import LGFF_SC_SEG as ModelCls  # type: ignore
        # keep config consistent
        if not getattr(cfg, "use_seg_head", False):
            cfg.use_seg_head = True
    else:
        from lgff.models.lgff_sc import LGFF_SC as ModelCls  # type: ignore
        # allow use_seg_head True (for ablation) but do not force it

    # ----------------- Loss -----------------
    lv = getattr(cfg, "loss_variant", "base")
    if lv == "seg":
        # expects you created: lgff/losses/lgff_loss_seg.py
        from lgff.losses.lgff_loss_seg import LGFFLoss_SEG as LossCls  # type: ignore
        # if using seg loss, lambda_seg should be > 0
        if float(getattr(cfg, "lambda_seg", 0.0)) <= 0:
            # do not hard fail; allow you to sweep lambda_seg=0 for debug
            pass
    else:
        from lgff.losses.lgff_loss import LGFFLoss as LossCls  # type: ignore

    model = ModelCls(cfg, geometry)
    loss_fn = LossCls(cfg, geometry)
    return model, loss_fn


def _is_seg_enabled(cfg: LGFFConfig) -> bool:
    return bool(getattr(cfg, "use_seg_head", False)) or str(getattr(cfg, "model_variant", "sc")).endswith("seg")


def _check_seg_requirements(cfg: LGFFConfig, train_ds: SingleObjectDataset, logger: logging.Logger) -> None:
    """
    If seg is enabled, verify dataset returns seg GT keys.
    This does not modify dataset; it just warns early.
    """
    need_seg = _is_seg_enabled(cfg) or (getattr(cfg, "loss_variant", "base") == "seg")
    if not need_seg:
        return

    try:
        sample = train_ds[0]
    except Exception as e:
        logger.warning(f"[SegCheck] Failed to read train_ds[0] for seg check: {e}")
        return

    has_seg = ("mask" in sample)
    if not has_seg:
        raise RuntimeError(
            "[SegCheck] Seg is ENABLED but dataset sample has no 'mask' key. "
            "Check cfg.return_mask and seg_supervision_source."
        )
    if bool(getattr(cfg, "seg_ignore_invalid", True)) and bool(getattr(cfg, "return_valid_mask", False)):
        if "mask_valid" not in sample:
            raise RuntimeError("[SegCheck] seg_ignore_invalid=True but dataset sample has no 'mask_valid'.")
    logger.info("[SegCheck] Dataset provides required mask keys for seg training.")


def main() -> None:
    # 1) parse launcher args
    args = parse_args()

    # 2) load config (handles --config/--opt; writes config_used.yaml)
    cfg: LGFFConfig = load_config()

    # 3) resolve work_dir priority: CLI > cfg.work_dir/log_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if getattr(cfg, "work_dir", None) is None:
        cfg.work_dir = getattr(cfg, "log_dir", "output/debug")
    os.makedirs(cfg.work_dir, exist_ok=True)

    # 3.5) force seg dataset flags when seg is enabled
    if _is_seg_enabled(cfg):
        cfg.use_seg_head = True
        cfg.seg_supervision = "mask"
        if str(getattr(cfg, "seg_supervision_source", "mask_visib")).lower().strip() not in {"mask_visib", "mask_full"}:
            cfg.seg_supervision_source = "mask_visib"
        cfg.return_mask = True
        if bool(getattr(cfg, "seg_ignore_invalid", True)):
            cfg.return_valid_mask = True
        try:
            cfg.validate()
        except Exception as e:
            raise RuntimeError(f"[Config] Seg validation failed after enforcing seg flags: {e}")

    # 4) logger
    log_file = os.path.join(cfg.work_dir, "train.log")
    logger = setup_logger(log_file, name="lgff.train")

    logger.info("==========================================")
    logger.info("LGFF Training Launcher (Variant-aware)")
    logger.info(f"Work Dir      : {cfg.work_dir}")
    logger.info(f"Config Used   : {os.path.join(cfg.work_dir, 'config_used.yaml')}")
    logger.info(f"Model Variant : {getattr(cfg, 'model_variant', 'sc')}")
    logger.info(f"Loss Variant  : {getattr(cfg, 'loss_variant', 'base')}")
    logger.info("==========================================")

    # 5) seed
    seed_value = int(args.seed) if args.seed is not None else int(getattr(cfg, "seed", 42))
    set_random_seed(seed_value, deterministic=bool(getattr(cfg, "deterministic", False)))
    logger.info(f"Random Seed   : {seed_value} | deterministic={bool(getattr(cfg, 'deterministic', False))}")

    worker_init_fn = _worker_init_fn_builder(seed_value)
    generator = torch.Generator()
    generator.manual_seed(seed_value)

    # 6) geometry
    geometry = GeometryToolkit()

    # 7) datasets & loaders
    logger.info("Initializing Datasets...")
    train_ds = SingleObjectDataset(cfg, split="train")
    val_split_name = getattr(cfg, "val_split", "test")
    val_ds = SingleObjectDataset(cfg, split=val_split_name)

    logger.info(f"  - Train Set : {len(train_ds)} samples")
    logger.info(f"  - Val Set   : {len(val_ds)} samples (split='{val_split_name}')")

    # seg requirement check (only warning)
    _check_seg_requirements(cfg, train_ds, logger)

    batch_size = int(getattr(cfg, "batch_size", 8))
    num_workers = int(getattr(cfg, "num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    else:
        logger.warning("Validation dataset is empty! Training will proceed without validation.")

    # 8) build model/loss (variant-aware)
    logger.info("Building Model/Loss...")
    model, loss_fn = _build_model_and_loss(cfg, geometry)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - Trainable Params: {n_params / 1e6:.2f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    logger.info(f"Device        : {device}")

    # 9) trainer
    logger.info("Initializing Trainer...")
    trainer = TrainerSC(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=cfg.work_dir,
        resume_path=args.resume,
    )

    # 10) train
    logger.info("Start Training Loop.")
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise

    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
