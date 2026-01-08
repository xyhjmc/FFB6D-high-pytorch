"""
Quick consistency check for LGFF Seg pipeline:
- Config + dataset sample (mask/mask_valid/choose)
- Model forward outputs
- Loss forward pass
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from lgff.utils.config_seg import load_config, merge_cfg_from_checkpoint, LGFFConfigSeg
from lgff.utils.geometry import GeometryToolkit
from lgff.datasets.single_loader_seg import SingleObjectDatasetSeg
from lgff.models.lgff_sc_seg import LGFF_SC_SEG
from lgff.losses.lgff_loss_seg import LGFFLoss_SEG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGFF Seg pipeline consistency check")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _ensure_seg_flags(cfg: LGFFConfigSeg) -> None:
    cfg.use_seg_head = True
    cfg.return_mask = True
    if bool(getattr(cfg, "seg_ignore_invalid", True)):
        cfg.return_valid_mask = True


def _check_batch(batch: Dict[str, torch.Tensor]) -> None:
    required = ["rgb", "points", "point_cloud", "pose", "intrinsic", "choose"]
    missing = [k for k in required if k not in batch]
    if missing:
        raise RuntimeError(f"Missing required batch keys: {missing}")

    if "mask" not in batch:
        raise RuntimeError("Missing 'mask' in batch (seg supervision).")

    if "mask_valid" not in batch:
        print("[Warn] 'mask_valid' missing; seg_ignore_invalid may be disabled.")

    if batch["points"].shape != batch["point_cloud"].shape:
        raise RuntimeError("points vs point_cloud shape mismatch.")


def _check_outputs(outputs: Dict[str, torch.Tensor]) -> None:
    required = ["pred_quat", "pred_trans", "pred_conf", "pred_mask_logits"]
    missing = [k for k in required if k not in outputs]
    if missing:
        raise RuntimeError(f"Missing required output keys: {missing}")


def main() -> None:
    args = parse_args()

    cfg_cli: LGFFConfigSeg = load_config()
    cfg = cfg_cli
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))
    _ensure_seg_flags(cfg)

    dataset = SingleObjectDatasetSeg(cfg, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    _check_batch(batch)

    device = torch.device(args.device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    geometry = GeometryToolkit()
    model = LGFF_SC_SEG(cfg, geometry).to(device)
    loss_fn = LGFFLoss_SEG(cfg, geometry).to(device)

    with torch.no_grad():
        outputs = model(batch)
        _check_outputs(outputs)
        loss, metrics = loss_fn(outputs, batch)

    H, W = batch["rgb"].shape[-2:]
    choose = batch["choose"]
    if int(choose.max().item()) >= H * W:
        print(f"[Warn] choose max={int(choose.max().item())} exceeds H*W-1={H*W-1}.")

    print("Seg pipeline check passed.")
    print(f"Loss: {float(loss.item()):.6f}")
    if isinstance(metrics, dict):
        print(f"Metrics keys: {list(metrics.keys())}")


if __name__ == "__main__":
    main()
