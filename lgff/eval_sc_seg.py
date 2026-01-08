# tools/eval_sc_seg.py
"""
Evaluation entry point for Single-Class LGFF (Segmentation Variant).
Updates:
- Uses LGFFConfigSeg, EvaluatorSCSeg, SingleObjectDatasetSeg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# [CHANGED] Correct imports
from lgff.utils.config_seg import LGFFConfigSeg, load_config, merge_cfg_from_checkpoint
from lgff.utils.logger import setup_logger, get_logger
from lgff.utils.geometry import GeometryToolkit
from lgff.datasets.single_loader_seg import SingleObjectDatasetSeg
from lgff.engines.evaluator_sc_seg import EvaluatorSCSeg
from common.ffb6d_utils.model_complexity import ModelComplexityLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LGFF Seg")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--opt", type=str, nargs="*", default=[])
    args, _ = parser.parse_known_args()
    return args


def build_dataloader(cfg: LGFFConfigSeg, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = SingleObjectDatasetSeg(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def _build_model(cfg: LGFFConfigSeg, geometry: GeometryToolkit) -> torch.nn.Module:
    from lgff.models.lgff_sc_seg import LGFF_SC_SEG
    if not getattr(cfg, "use_seg_head", False):
        cfg.use_seg_head = True
    return LGFF_SC_SEG(cfg, geometry)


def main() -> None:
    args = parse_args()

    # 1. Load Config (Seg version)
    cfg_cli: LGFFConfigSeg = load_config()

    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "checkpoint_best.pth")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt_state = torch.load(ckpt_path, map_location="cpu")

    # Merge config
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt_state.get("config"))
    cfg.use_seg_head = True
    cfg.return_mask = True
    if bool(getattr(cfg, "seg_ignore_invalid", True)):
        cfg.return_valid_mask = True

    # Work dir
    work_dir = args.work_dir or getattr(cfg, "work_dir", None) or getattr(cfg, "log_dir", "output/eval_seg")
    os.makedirs(work_dir, exist_ok=True)

    setup_logger(os.path.join(work_dir, "eval.log"))
    logger = get_logger("lgff.eval")

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loader
    split = args.split or getattr(cfg, "val_split", "test")
    batch_size = args.batch_size or int(getattr(cfg, "batch_size", 1))
    test_loader = build_dataloader(cfg, split, batch_size, args.num_workers or 4)

    # Model
    model = _build_model(cfg, geometry)
    # Relax strict loading in case newer code has extra buffers (like Z-bias buffers)
    model.load_state_dict(ckpt_state["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

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
                        (
                            f"GFLOPs: {complexity_info['gflops']:.3f}"
                            if complexity_info.get("gflops") is not None
                            else "GFLOPs: N/A"
                        ),
                    ]
                )
            )
    except StopIteration:
        logger.warning("Evaluation loader is empty; skip complexity logging.")
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")

    # Evaluator
    evaluator = EvaluatorSCSeg(model, test_loader, cfg, geometry, save_dir=work_dir)
    metrics = evaluator.run()

    print(json.dumps(metrics, indent=2))
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
