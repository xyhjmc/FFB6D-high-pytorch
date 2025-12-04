"""
单类别 LGFF 的评估脚本，作为评估工具库提供批量测试逻辑。
主要包含 ``evaluate`` 函数：负责将 DataLoader 中的批次移到设备、
调用模型前向推理、累计与归一化指标（如姿态批次数等），最终输出
用于日志或监控的度量字典。
"""
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
