"""
Evaluation entry point for Single-Class LGFF.

This script mirrors the training entry (`train_lgff_sc.py`) but focuses on
loading a trained checkpoint, running inference on a chosen split, and
reporting simple metrics (ADD / ADD-S / <2cm accuracy).

Usage examples:
    python lgff/eval_sc.py --config configs/helmet_sc.yaml --checkpoint output/helmet_sc/checkpoint_best.pth
    python lgff/eval_sc.py --config configs/helmet_sc.yaml --checkpoint output/helmet_sc/checkpoint_last.pth --split val
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.getcwd())

from common.ffb6d_utils.model_complexity import ModelComplexityLogger
from lgff.utils.config import LGFFConfig, load_config
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.engines.evaluator_sc import EvaluatorSC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LGFF Single-Class Model")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to checkpoint (.pth) containing model weights",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate (default: cfg.val_split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (default: cfg.batch_size)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers (default: cfg.num_workers)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override output directory for logs/results",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to dump metric summary as JSON",
    )
    args, _ = parser.parse_known_args()
    return args


def load_model_weights(model: LGFF_SC, checkpoint_path: str, device: torch.device) -> None:
    """Load model weights from a TrainerSC checkpoint or a bare state_dict."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger = logging.getLogger("lgff.eval")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        # Fallback to assume checkpoint itself is a state_dict
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")


def build_dataloader(cfg: LGFFConfig, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = SingleObjectDataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def main() -> None:
    args = parse_args()

    # Load config via CLI (--config/--opt)
    cfg: LGFFConfig = load_config()

    work_dir = args.work_dir or cfg.work_dir or cfg.log_dir
    os.makedirs(work_dir, exist_ok=True)

    log_file = os.path.join(work_dir, "eval.log")
    setup_logger(log_file, name="lgff.eval")
    logger = get_logger("lgff.eval")
    logger.setLevel(logging.INFO)

    # Resolve split / dataloader params
    split = args.split or getattr(cfg, "val_split", "test")
    batch_size = args.batch_size or getattr(cfg, "batch_size", 2)
    num_workers = args.num_workers or getattr(cfg, "num_workers", 4)

    logger.info(f"Evaluating split={split} | batch_size={batch_size} | num_workers={num_workers}")

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    test_loader = build_dataloader(cfg, split, batch_size, num_workers)
    logger.info(f"Test dataset size: {len(test_loader.dataset)} samples")

    # Model
    model = LGFF_SC(cfg, geometry)
    load_model_weights(model, args.checkpoint, device)

    # Complexity stats (params/FLOPs)
    complexity_logger = ModelComplexityLogger()
    try:
        example_batch = next(iter(test_loader))
        complexity_info = complexity_logger.maybe_log(model, example_batch, stage="eval")
        if complexity_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=eval",
                        f"Params: {complexity_info['params']:,} ({complexity_info['param_mb']:.2f} MB)",
                        f"GFLOPs: {complexity_info['gflops']:.3f}" if complexity_info.get("gflops") is not None else "GFLOPs: N/A",
                    ]
                )
            )
    except StopIteration:
        logger.warning("Evaluation loader is empty; skip complexity logging.")
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")

    # Recreate loader so the consumed batch is not skipped during evaluation
    test_loader = build_dataloader(cfg, split, batch_size, num_workers)

    # Evaluator
    evaluator = EvaluatorSC(model=model, test_loader=test_loader, cfg=cfg, geometry=geometry)
    metrics = evaluator.run()

    # Pretty print
    metrics_str = json.dumps(metrics, indent=2)
    logger.info(f"Evaluation metrics:\n{metrics_str}")
    print(metrics_str)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
