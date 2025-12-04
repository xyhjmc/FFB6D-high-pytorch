"""Evaluation utilities for single-class LGFF."""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            total += batch["point_cloud"].shape[0]
            if "pose" in outputs:
                metrics["pose_batches"] = metrics.get("pose_batches", 0) + batch["pose"].shape[0]
    if total > 0 and "pose_batches" in metrics:
        metrics["pose_batches"] /= float(total)
    return metrics


__all__ = ["evaluate"]
