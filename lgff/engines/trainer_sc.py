# lgff/engines/trainer_sc.py
from __future__ import annotations

import os
import logging
import math
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from torch.amp import autocast, GradScaler  # ✅ 新 AMP API
from tqdm.auto import tqdm  # ✅ 进度条

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

        # 3. Scheduler Configuration
        scheduler_type = getattr(cfg, "scheduler", "plateau").lower()
        self.logger.info(f"Initializing Scheduler: {scheduler_type}")

        if scheduler_type == "plateau":
            # ⚠️ 不加 verbose，兼容旧版 PyTorch
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=getattr(cfg, "lr_factor", 0.5),
                patience=getattr(cfg, "lr_patience", 5),
                min_lr=getattr(cfg, "lr_min", 1e-6),
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(cfg, "epochs", 50),
                eta_min=getattr(cfg, "lr_min", 1e-6),
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=getattr(cfg, "lr_step_size", 20),
                gamma=getattr(cfg, "lr_factor", 0.5),
            )
        elif scheduler_type == "none":
            self.scheduler = None
        else:
            self.logger.warning(
                f"Unknown scheduler type '{scheduler_type}', defaulting to None."
            )
            self.scheduler = None

        # 4. Data Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 5. AMP Scaler（新 API）
        self.scaler = GradScaler("cuda", enabled=getattr(cfg, "use_amp", True))
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
            metric_for_sched: Optional[float] = None
            val_metrics: Dict[str, float] = {"loss_total": float("nan")}

            if self.val_loader is not None:
                val_metrics = self._validate(epoch)
                val_loss = float(val_metrics.get("loss_total", float("nan")))

                # NaN/Inf 防御
                if not math.isfinite(val_loss):
                    self.logger.warning(
                        f"Epoch {epoch_idx}: val_loss is NaN/Inf ({val_loss}), "
                        f"skipping scheduler/save."
                    )
                else:
                    metric_for_sched = val_loss
                    # best model 更新
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, is_best=True)
            else:
                # 无验证集，使用 train loss
                train_loss = float(train_metrics.get("loss_total", float("nan")))
                if not math.isfinite(train_loss):
                    self.logger.warning(
                        f"Epoch {epoch_idx}: train_loss is NaN/Inf, skipping scheduler."
                    )
                else:
                    metric_for_sched = train_loss

            # 更新学习率
            self._step_scheduler(metric_for_sched)

            # Regular checkpoint
            self._save_checkpoint(epoch, is_best=False)

            # -------- 清晰的 Epoch 总结日志 --------
            train_loss_log = float(train_metrics.get("loss_total", float("nan")))
            val_loss_log = float(val_metrics.get("loss_total", float("nan")))

            train_str = f"{train_loss_log:.6f}" if math.isfinite(train_loss_log) else "NaN"
            if self.val_loader is not None:
                val_str = f"{val_loss_log:.6f}" if math.isfinite(val_loss_log) else "NaN"
            else:
                val_str = "N/A"

            best_info = (
                f"{self.best_val_loss:.6f}"
                if self.best_val_loss != float("inf")
                else "N/A"
            )
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch [{epoch_idx}/{epochs}] Summary | "
                f"Train loss_total={train_str} | "
                f"Val loss_total={val_str} | "
                f"LR={lr_now:.2e} | "
                f"Best Val={best_info}"
            )

        self.writer.close()

    # ------------------------------------------------------------------
    # 单个 epoch 的训练（带 tqdm 进度条，并保留每轮的最后一行）
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}

        if len(self.train_loader) == 0:
            raise RuntimeError(
                "Train loader is empty. Please check your dataset / sampler."
            )

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"[Train] Epoch {epoch + 1}",
            leave=True,   # ✅ 训练条保留在终端
        )

        for i, batch in pbar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(batch)
                loss, metrics = self.loss_fn(outputs, batch)

                # 确保 metrics 为 dict
                if not isinstance(metrics, dict) or len(metrics) == 0:
                    metrics = {"loss_total": loss.detach()}

            # 统一把 metric 转成 float（支持 0-dim tensor / float / int）
            processed_metrics: Dict[str, float] = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        v_val = float(v.detach().item())
                    else:
                        v_val = float(v.detach().mean().item())
                else:
                    v_val = float(v)
                processed_metrics[k] = v_val

            # backward + grad clip（max_grad_norm 可配置）
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            max_grad_norm = getattr(self.cfg, "max_grad_norm", 2.0)
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Logging
            bs = batch["rgb"].size(0)
            if not meters:
                for k in processed_metrics.keys():
                    meters[k] = AverageMeter()
            for k, v in processed_metrics.items():
                meters[k].update(v, bs)

            # 更新 tqdm 状态栏
            show_keys = list(processed_metrics.keys())
            main_key = "loss_total" if "loss_total" in show_keys else show_keys[0]
            pbar.set_postfix({
                main_key: f"{meters[main_key].val:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # 按 log_interval 写 TensorBoard 日志 + logger
            if i % getattr(self.cfg, "log_interval", 10) == 0:
                epoch_idx = epoch + 1
                log_str = f"Epoch [{epoch_idx}][{i}/{len(self.train_loader)}] "
                log_str += " | ".join(
                    [f"{k}: {m.val:.4f}" for k, m in meters.items()]
                )
                self.logger.info(log_str)

                for k, m in meters.items():
                    self.writer.add_scalar(f"Train/{k}", m.val, self.global_step)
                self.writer.add_scalar(
                    "Train/LR",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )

            self.global_step += 1

        if not meters:
            avg_metrics: Dict[str, float] = {"loss_total": float("nan")}
        else:
            avg_metrics = {k: m.avg for k, m in meters.items()}

        self.logger.info(f"Epoch {epoch + 1} Train Summary: {avg_metrics}")
        return avg_metrics

    # ------------------------------------------------------------------
    # 验证阶段（带 tqdm 进度条，并保留每轮的最后一行）
    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"[Val]   Epoch {epoch + 1}",
                leave=True,   # ✅ 验证条保留在终端
            )

            for i, batch in pbar:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(batch)
                    _, metrics = self.loss_fn(outputs, batch)

                    if not isinstance(metrics, dict) or len(metrics) == 0:
                        metrics = {"loss_total": float("nan")}

                processed_metrics: Dict[str, float] = {}
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        if v.ndim == 0:
                            v_val = float(v.detach().item())
                        else:
                            v_val = float(v.detach().mean().item())
                    else:
                        v_val = float(v)
                    processed_metrics[k] = v_val

                bs = batch["rgb"].size(0)
                if not meters:
                    for k in processed_metrics.keys():
                        meters[k] = AverageMeter()
                for k, v in processed_metrics.items():
                    meters[k].update(v, bs)

                # 更新 tqdm 状态
                main_key = (
                    "loss_total"
                    if "loss_total" in meters
                    else list(meters.keys())[0]
                )
                pbar.set_postfix({main_key: f"{meters[main_key].val:.4f}"})

        if not meters:
            avg_metrics: Dict[str, float] = {"loss_total": float("nan")}
        else:
            avg_metrics = {k: m.avg for k, m in meters.items()}

        for k, v in avg_metrics.items():
            # Val 用 epoch 作为 X 轴
            self.writer.add_scalar(f"Val/{k}", v, epoch + 1)

        self.logger.info(f"Epoch {epoch + 1} Val Summary: {avg_metrics}")
        return avg_metrics

    # ------------------------------------------------------------------
    # Scheduler 统一封装
    # ------------------------------------------------------------------
    def _step_scheduler(self, metric: Optional[float]) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                self.logger.warning(
                    "ReduceLROnPlateau is enabled but metric is None, "
                    "scheduler.step skipped for this epoch."
                )
                return
            self.scheduler.step(metric)
        else:
            # Cosine / Step 等不需要 metric
            try:
                self.scheduler.step()
            except TypeError:
                pass

    # ------------------------------------------------------------------
    # Checkpoint 保存 / 恢复
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state: Dict[str, Any] = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": (
                asdict(self.cfg)
                if hasattr(self.cfg, "__dataclass_fields__")
                else self.cfg.__dict__
            ),
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

        if (
            "scheduler" in checkpoint
            and checkpoint["scheduler"] is not None
            and self.scheduler is not None
        ):
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
