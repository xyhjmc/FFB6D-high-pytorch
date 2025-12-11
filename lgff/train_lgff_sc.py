"""
Training entry point for Single-Class LGFF.
Uses LGFFConfig/load_config for configuration, sets up environment,
and launches the TrainerSC.
"""

import argparse
import os
import sys
import random
import logging

import numpy as np

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# 路径补丁：确保可以从项目根目录或 lgff 子目录运行
# ---------------------------------------------------------------------

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.insert(0, parent_dir)
# ---------------------------------------------------------------------
# 模块导入
# ---------------------------------------------------------------------
from lgff.utils.config import LGFFConfig, load_config
from lgff.utils.logger import setup_logger

try:
    from common.geometry import GeometryToolkit
except ImportError:
    from lgff.utils.geometry import GeometryToolkit

from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.losses.lgff_loss import LGFFLoss
from lgff.engines.trainer_sc import TrainerSC


def parse_args():
    """只解析与训练脚本本身相关的参数。"""
    parser = argparse.ArgumentParser(
        description="Train LGFF Single-Class Model",
        add_help=True,
    )

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
        help="Override output directory (default: defined in config.log_dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # 注意：不要在这里解析 --config / --opt，
    # 让 lgff.utils.config.load_config 去处理它们。
    args, _ = parser.parse_known_args()
    return args


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """设置随机种子；deterministic=True 更可复现，False 更快。"""
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


def main():
    # 1. 解析脚本参数
    args = parse_args()

    # 2. 加载配置 (会处理 --config / --opt，并保存 config_used.yaml)
    cfg: LGFFConfig = load_config()

    # 3. work_dir 优先级：CLI > config
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        os.makedirs(cfg.work_dir, exist_ok=True)

    # 4. 初始化日志
    log_file = os.path.join(cfg.work_dir, "train.log")
    logger = setup_logger(log_file, name="lgff.train")

    config_used_path = os.path.join(cfg.work_dir, "config_used.yaml")
    logger.info("==========================================")
    logger.info("LGFF Training Launcher")
    logger.info(f"Work Dir      : {cfg.work_dir}")
    logger.info(f"Config Used   : {config_used_path}")
    logger.info("==========================================")

    # 5. 随机数种子
    seed_value = args.seed if args.seed is not None else getattr(cfg, "seed", 42)
    set_random_seed(seed_value, deterministic=getattr(cfg, "deterministic", False))
    logger.info(
        f"Random Seed   : {seed_value} | deterministic={getattr(cfg, 'deterministic', False)}"
    )

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed_value + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed_value)

    # 6. 几何工具
    geometry = GeometryToolkit()

    # 7. 数据集 & DataLoader
    logger.info("Initializing Datasets...")

    train_ds = SingleObjectDataset(cfg, split="train")
    val_split_name = getattr(cfg, "val_split", "test")
    val_ds = SingleObjectDataset(cfg, split=val_split_name)

    logger.info(f"  - Train Set : {len(train_ds)} samples")
    logger.info(f"  - Val Set   : {len(val_ds)} samples (split='{val_split_name}')")

    batch_size = getattr(cfg, "batch_size", 8)
    num_workers = getattr(cfg, "num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
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
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )
    else:
        logger.warning("Validation dataset is empty! Training will proceed without validation.")

    # 8. 模型
    logger.info("Building Model (LGFF_SC)...")
    model = LGFF_SC(cfg, geometry)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - Trainable Params: {n_params / 1e6:.2f}M")

    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Using CUDA.")
    else:
        logger.info("CUDA not available, using CPU.")

    # 9. 损失
    logger.info("Building Loss (LGFFLoss)...")
    loss_fn = LGFFLoss(cfg, geometry)

    # 10. 训练器
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

    # 11. 开始训练
    logger.info("�� Start Training Loop!")
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
