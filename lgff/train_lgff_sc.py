"""
Training entry point for Single-Class LGFF.
Uses LGFFConfig/load_config for configuration, sets up environment,
and launches the TrainerSC.

Usage examples:
    python train_lgff_sc.py --config configs/helmet_config.yaml
    python train_lgff_sc.py --config configs/helmet_config.yaml --opt epochs=50 lr=5e-5
    python train_lgff_sc.py --config configs/drill.yaml --resume output/drill/checkpoint_last.pth
"""
import argparse
import os
import sys
import random
import logging

import torch
from torch.utils.data import DataLoader

# 确保项目根目录在 python path 中
sys.path.append(os.getcwd())

from common.ffb6d_utils.model_complexity import ModelComplexityLogger
from lgff.utils.config import LGFFConfig, load_config
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.losses.lgff_loss import LGFFLoss
from lgff.engines.trainer_sc import TrainerSC


def parse_args():
    """
    这里只解析和“训练入口”相关的额外参数；
    配置本身（--config, --opt）交给 common.config.load_config 处理。
    """
    parser = argparse.ArgumentParser(description="Train LGFF Single-Class Model")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override output directory (default: cfg.log_dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    # 用 parse_known_args 保留给 load_config 的 --config / --opt 等参数
    args, _ = parser.parse_known_args()
    return args


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 下面两句看需求：benchmark=True 更快但略减弱可复现性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    # 1. 解析入口参数（不含 config）
    args = parse_args()

    # 2. 使用 common.config.load_config() 读取 LGFFConfig
    #    load_config 内部会处理:
    #      --config xxx.yaml
    #      --opt key=value ...
    cfg: LGFFConfig = load_config()

    # 3. work_dir / log_dir 统一
    #    注意：logger 的正确使用顺序是：
    #      (1) 先确定 work_dir
    #      (2) 再 setup_logger()
    #      (3) 然后其他地方才能 get_logger("lgff.train")
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        # 默认使用 cfg.work_dir（load_config 内部保证至少和 log_dir 一致）
        work_dir = cfg.work_dir or cfg.log_dir

    os.makedirs(work_dir, exist_ok=True)

    # 4. 初始化 logger（务必在任何 get_logger 调用之前）
    log_file = os.path.join(work_dir, "train.log")
    setup_logger(log_file, name="lgff.train")
    logger = get_logger("lgff.train")
    logger.setLevel(logging.INFO)

    logger.info(f"Loaded LGFFConfig via common.config.load_config()")
    logger.info(f"Work directory: {work_dir}")

    # 5. 随机种子 & 设备信息
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(
            f"CUDA devices: {torch.cuda.device_count()}, "
            f"current: {torch.cuda.get_device_name(0)}"
        )

    # 6. 初始化 Geometry Toolkit
    # 如果你的 GeometryToolkit 以后支持传入内参，可以改成：
    # geometry = GeometryToolkit(intrinsic=cfg.camera_intrinsic)
    geometry = GeometryToolkit()

    # 7. 数据集与 DataLoader
    logger.info("Initializing Datasets...")

    train_ds = SingleObjectDataset(cfg, split="train")

    # 验证集 split 可以以后写进 LGFFConfig，这里给一个默认 / 可选方案
    val_split = getattr(cfg, "val_split", "val")  # cfg 里如果没有就用 "val"
    if val_split not in ("train", "val", "test"):
        logger.warning(f"Unknown val_split={val_split}, fallback to 'val'.")
        val_split = "val"
    val_ds = SingleObjectDataset(cfg, split=val_split)

    logger.info(f"Train Dataset Size: {len(train_ds)}")
    logger.info(f"Val Dataset Size: {len(val_ds)}")

    batch_size = getattr(cfg, "batch_size", 2)
    num_workers = getattr(cfg, "num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if len(val_ds) == 0:
        logger.warning("Validation dataset is empty, val_loader will be set to None.")
        val_loader = None

    # 8. 构建模型
    logger.info("Building LGFF_SC model...")
    model = LGFF_SC(cfg, geometry)

    # 8.1 模型复杂度统计（参数量 / FLOPs）
    complexity_logger = ModelComplexityLogger()
    try:
        sample_batch = next(iter(train_loader))
        complexity_info = complexity_logger.maybe_log(model, sample_batch, stage="train_init")
        if complexity_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=train_init",
                        f"Params: {complexity_info['params']:,} ({complexity_info['param_mb']:.2f} MB)",
                        f"GFLOPs: {complexity_info['gflops']:.3f}" if complexity_info.get("gflops") is not None else "GFLOPs: N/A",
                    ]
                )
            )
    except StopIteration:
        logger.warning("Train loader is empty; skip complexity logging.")
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")

    # 8.2 重新实例化一次 train_loader，避免消耗首个 batch
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 9. 构建损失函数
    logger.info("Building LGFFLoss...")
    loss_fn = LGFFLoss(cfg, geometry)

    # 10. 构造 Trainer
    logger.info("Initializing TrainerSC...")
    trainer = TrainerSC(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=work_dir,
        resume_path=args.resume,
    )

    # 11. 开始训练
    logger.info("�� Start Training Loop!")
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise

    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
