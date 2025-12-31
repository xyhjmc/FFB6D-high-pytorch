# tools/eval_lgff_sc.py
"""
Evaluation entry point for Single-Class LGFF (supports variants: sc / sc_seg, base loss / seg loss).

Key design:
- Dataloader is shared.
- Model variant is selected by cfg.model_variant (same as training).
- EvaluatorSC is shared (your modified one with PnP/ICP/ADD+ADD-S CSV).

Usage examples:
  python tools/eval_lgff_sc.py --config configs/ape.yaml --checkpoint output/exp/checkpoint_best.pth --split test
  python tools/eval_lgff_sc.py --config configs/ape.yaml --checkpoint output/exp/ --split test_lmo
  python tools/eval_lgff_sc.py --config configs/ape.yaml --checkpoint output/exp/checkpoint_best.pth --opt model_variant="sc_seg"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lgff.utils.config import LGFFConfig, load_config, merge_cfg_from_checkpoint
from lgff.utils.logger import setup_logger, get_logger

# Same import strategy as training
try:
    from common.geometry import GeometryToolkit
except ImportError:
    from lgff.utils.geometry import GeometryToolkit

from lgff.datasets.single_loader_seg import SingleObjectDataset as SingleObjectDatasetSeg
from lgff.engines.evaluator_sc_seg import EvaluatorSC as EvaluatorSCSeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LGFF Single-Class Model (variant-aware)")

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to checkpoint (.pth) OR a directory containing checkpoint_best.pth / checkpoint_last.pth",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate (default: cfg.val_split or 'test')",
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
        help="Override output directory for logs/results (default: cfg.work_dir)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to dump metric summary as JSON",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_checkpoint_path(path: str) -> str:
    """Allow passing a directory; prefer checkpoint_best.pth then checkpoint_last.pth."""
    if os.path.isdir(path):
        best = os.path.join(path, "checkpoint_best.pth")
        last = os.path.join(path, "checkpoint_last.pth")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"No checkpoint_best.pth or checkpoint_last.pth under {path}")
    return path


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load model weights from a TrainerSC checkpoint or a bare state_dict."""
    if checkpoint is None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

    logger = logging.getLogger("lgff.eval")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    state_dict = checkpoint.get("state_dict", None)
    if state_dict is None:
        # Fallback: assume checkpoint itself is a state_dict
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

    return checkpoint


def build_dataloader(cfg: LGFFConfig, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = SingleObjectDatasetSeg(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def _build_model(cfg: LGFFConfig, geometry: GeometryToolkit) -> torch.nn.Module:
    """
    Select model by cfg.model_variant so that sc / sc_seg share the same eval launcher.
    """
    mv = getattr(cfg, "model_variant", "sc")
    if mv == "sc_seg":
        from lgff.models.lgff_sc_seg import LGFF_SC_SEG as ModelCls  # type: ignore
        if not getattr(cfg, "use_seg_head", False):
            cfg.use_seg_head = True
        return ModelCls(cfg, geometry)

    from lgff.models.lgff_sc import LGFF_SC as ModelCls  # type: ignore
    return ModelCls(cfg, geometry)


def main() -> None:
    args = parse_args()

    # Load config via CLI (--config/--opt). This will also write config_used.yaml under cfg.work_dir.
    cfg_cli: LGFFConfig = load_config()

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt_state = torch.load(ckpt_path, map_location="cpu")

    # Merge: checkpoint config as base (backbone dims etc.), then apply CLI overrides
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt_state.get("config"))

    # Work dir: CLI > cfg.work_dir > cfg.log_dir
    work_dir = args.work_dir or getattr(cfg, "work_dir", None) or getattr(cfg, "log_dir", "output/debug")
    os.makedirs(work_dir, exist_ok=True)

    log_file = os.path.join(work_dir, "eval.log")
    setup_logger(log_file, name="lgff.eval")
    logger = get_logger("lgff.eval")
    logger.setLevel(logging.INFO)

    logger.info("==========================================")
    logger.info("LGFF Evaluation Launcher (Variant-aware)")
    logger.info(f"Work Dir      : {work_dir}")
    logger.info(f"Checkpoint    : {ckpt_path}")
    logger.info(f"Model Variant : {getattr(cfg, 'model_variant', 'sc')}")
    logger.info("==========================================")

    # Resolve split / dataloader params
    split = args.split or getattr(cfg, "val_split", "test")
    batch_size = args.batch_size or int(getattr(cfg, "batch_size", 2))
    num_workers = args.num_workers or int(getattr(cfg, "num_workers", 4))

    logger.info(f"Evaluating split={split} | batch_size={batch_size} | num_workers={num_workers}")

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    test_loader = build_dataloader(cfg, split, batch_size, num_workers)
    logger.info(f"Test dataset size: {len(test_loader.dataset)} samples")

    # Model
    model = _build_model(cfg, geometry)
    _ = load_model_weights(model, ckpt_path, device, checkpoint=ckpt_state)
    model = model.to(device)
    model.eval()

    # Evaluator (your updated one: ADD+ADD-S + PnP + ICP + CSV)
    evaluator = EvaluatorSCSeg(
        model=model, test_loader=test_loader, cfg=cfg, geometry=geometry, save_dir=work_dir
    )
    metrics = evaluator.run()

    metrics_str = json.dumps(metrics, indent=2)
    logger.info(f"Evaluation metrics:\n{metrics_str}")
    print(metrics_str)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
