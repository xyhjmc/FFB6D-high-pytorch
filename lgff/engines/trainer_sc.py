"""
Robust Single-Class Trainer for LGFF.
Features: AMP, Gradient Clipping, TensorBoard Logging, Checkpointing.
"""
from __future__ import annotations

import os
import logging
import math
from typing import Optional, Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lgff.utils.config import LGFFConfig
from lgff.models.lgff_sc import LGFF_SC
from lgff.losses.lgff_loss import LGFFLoss


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class TrainerSC:
    def __init__(
        self,
        model: LGFF_SC,
        loss_fn: LGFFLoss,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: LGFFConfig,
        output_dir: str = "output",
        resume_path: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("lgff.trainer")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Device & Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # 2. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(cfg, "lr", 1e-4),
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

        # 3. Scheduler
        # 默认使用 ReduceLROnPlateau（你也可以在外部自行覆盖 self.scheduler）
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # 4. Data Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 5. AMP Scaler
        self.scaler = GradScaler(enabled=getattr(cfg, "use_amp", True))
        self.use_amp = self.scaler.is_enabled()

        # 6. Logging / State
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        self.global_step: int = 0
        self.start_epoch: int = 0
        self.best_val_loss: float = float("inf")

        # 7. Resume (optional)
        if resume_path is not None and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------
    def fit(self) -> None:
        self.logger.info(f"Start training on device: {self.device}")
        epochs = getattr(self.cfg, "epochs", 50)

        for epoch in range(self.start_epoch, epochs):
            epoch_idx = epoch + 1  # for display

            # --- Training ---
            train_metrics = self._train_one_epoch(epoch)

            # --- Validation & Scheduler ---
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)
                val_loss = val_metrics.get("loss_total", float("nan"))

                # NaN/Inf 防御
                if not math.isfinite(val_loss):
                    self.logger.warning(
                        f"Epoch {epoch_idx}: val_loss is NaN/Inf "
                        f"({val_loss}), skip scheduler.step this epoch."
                    )
                    metric_for_sched: Optional[float] = None
                else:
                    metric_for_sched = val_loss

                self._step_scheduler(metric_for_sched)

                # best model 更新
                if math.isfinite(val_loss) and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            else:
                train_loss = train_metrics.get("loss_total", float("nan"))
                if not math.isfinite(train_loss):
                    self.logger.warning(
                        f"Epoch {epoch_idx}: train_loss is NaN/Inf "
                        f"({train_loss}), skip scheduler.step this epoch."
                    )
                    metric_for_sched = None
                else:
                    metric_for_sched = train_loss

                self._step_scheduler(metric_for_sched)

            # Regular checkpoint
            self._save_checkpoint(epoch, is_best=False)

            self.logger.info(
                f"Epoch [{epoch_idx}/{epochs}] finished. "
                f"Best Val Loss: {self.best_val_loss:.6f}"
            )

        self.writer.close()

    # ------------------------------------------------------------------
    # 单个 epoch 的训练
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}

        if len(self.train_loader) == 0:
            # 极端防御：train_loader 为空，直接报错提示配置问题
            raise RuntimeError("Train loader is empty. Please check your dataset / sampler.")

        for i, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none=True)

            # --- Forward (with AMP) ---
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch)
                loss, metrics = self.loss_fn(outputs, batch)
                # 防御: metrics 为空时，用 loss_total 填充
                if not isinstance(metrics, dict) or len(metrics) == 0:
                    metrics = {"loss_total": float(loss.detach().item())}

            # --- Backward (with AMP) ---
            self.scaler.scale(loss).backward()

            # 先 unscale 再做梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

            # Optimizer step + scaler update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- Logging & Meters ---
            bs = batch["rgb"].size(0)

            # 初始化 meters
            if not meters:
                for k in metrics.keys():
                    meters[k] = AverageMeter()

            for k, v in metrics.items():
                meters[k].update(v, bs)

            if i % getattr(self.cfg, "log_interval", 10) == 0:
                epoch_idx = epoch + 1
                log_str = f"Epoch [{epoch_idx}][{i}/{len(self.train_loader)}] "
                log_str += " | ".join([f"{k}: {m.val:.4f}" for k, m in meters.items()])
                self.logger.info(log_str)

                # TensorBoard: 当前 batch 的值
                for k, m in meters.items():
                    self.writer.add_scalar(f"Train/{k}", m.val, self.global_step)
                self.writer.add_scalar(
                    "Train/LR", self.optimizer.param_groups[0]["lr"], self.global_step
                )

            self.global_step += 1

        # End-of-epoch summary
        if not meters:
            # 理论上不会触发，除非 loss_fn 完全没返回 metrics
            self.logger.warning("No training metrics were recorded in this epoch.")
            avg_metrics = {"loss_total": float("nan")}
        else:
            avg_metrics = {k: m.avg for k, m in meters.items()}

        self.logger.info(f"Epoch {epoch + 1} Train Summary: {avg_metrics}")
        return avg_metrics

    # ------------------------------------------------------------------
    # 验证阶段
    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            self.logger.warning("Validation loader is None or empty. Skipping validation.")
            # 兜底返回一个只有 loss_total 的字典，防止外部访问报错
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 可以选择性使用 AMP 做验证
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    _, metrics = self.loss_fn(outputs, batch)
                    if not isinstance(metrics, dict) or len(metrics) == 0:
                        # 验证阶段至少要有一个 loss_total，方便 scheduler
                        metrics = {"loss_total": float("nan")}

                bs = batch["rgb"].size(0)

                if not meters:
                    for k in metrics.keys():
                        meters[k] = AverageMeter()

                for k, v in metrics.items():
                    meters[k].update(v, bs)

        if not meters:
            self.logger.warning("No validation metrics were recorded in this epoch.")
            avg_metrics = {"loss_total": float("nan")}
        else:
            avg_metrics = {k: m.avg for k, m in meters.items()}

        # TensorBoard: epoch 级别验证指标
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch + 1)

        self.logger.info(f"Epoch {epoch + 1} Val Summary: {avg_metrics}")
        return avg_metrics

    # ------------------------------------------------------------------
    # Scheduler 统一封装：兼容 ReduceLROnPlateau 和其他 LR scheduler
    # ------------------------------------------------------------------
    def _step_scheduler(self, metric: Optional[float]) -> None:
        if self.scheduler is None:
            return

        # ReduceLROnPlateau 需要 metric；如果 metric 无效，就直接跳过
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                self.logger.warning(
                    "Scheduler is ReduceLROnPlateau but metric is None; "
                    "skip scheduler.step() this epoch."
                )
                return
            self.scheduler.step(metric)
        else:
            # 其他 scheduler 一般不需要参数
            try:
                self.scheduler.step()
            except TypeError:
                # 极端情况：自定义 scheduler 需要参数，但我们没有
                self.logger.warning(
                    "scheduler.step() expected a metric but none was provided; "
                    "skip scheduler.step() this epoch."
                )

    # ------------------------------------------------------------------
    # Checkpoint 保存 / 恢复
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }
        last_path = os.path.join(self.output_dir, "checkpoint_last.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saved best model to {best_path}")

    def _load_checkpoint(self, path: str) -> None:
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")
        if "scaler" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except Exception as e:
                self.logger.warning(f"Failed to load scaler state: {e}")

        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.global_step = int(checkpoint.get("global_step", 0))

        self.logger.info(
            f"Resumed from epoch {self.start_epoch}, "
            f"best_val_loss={self.best_val_loss:.6f}, "
            f"global_step={self.global_step}"
        )


__all__ = ["TrainerSC"]
