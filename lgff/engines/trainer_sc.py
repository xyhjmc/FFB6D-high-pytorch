"""Training loop for single-class LGFF."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from common.config import LGFFConfig
from common.logger import get_logger


class TrainerSC:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: LGFFConfig,
        logger=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or get_logger("lgff.trainer")
        self.model.to(self.device)

    def _run_loader(self, loader: DataLoader, train: bool = True) -> Dict[str, float]:
        running_loss = 0.0
        total = 0
        agg_metrics: Dict[str, float] = {}
        self.model.train(mode=train)

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss, metrics = self.loss_fn(outputs, batch)
            if train:
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * batch["point_cloud"].shape[0]
            total += batch["point_cloud"].shape[0]
            for key, value in metrics.items():
                agg_metrics[key] = agg_metrics.get(key, 0.0) + value * batch["point_cloud"].shape[0]

        if total > 0:
            agg_metrics = {k: v / total for k, v in agg_metrics.items()}
        agg_metrics["loss"] = running_loss / max(total, 1)
        return agg_metrics

    def train(self) -> None:
        for epoch in range(self.cfg.epochs):
            train_stats = self._run_loader(self.train_loader, train=True)
            self.logger.info("[Epoch %d] train: %s", epoch, train_stats)
            if self.val_loader:
                val_stats = self._run_loader(self.val_loader, train=False)
                self.logger.info("[Epoch %d]   val: %s", epoch, val_stats)


__all__ = ["TrainerSC"]
