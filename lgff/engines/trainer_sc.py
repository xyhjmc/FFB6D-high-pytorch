# lgff/engines/trainer_sc.py
from __future__ import annotations

import os
import logging
import math
import json
import csv
from typing import Optional, Dict, Any
from dataclasses import asdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from lgff.utils.config import LGFFConfig
from lgff.models.lgff_sc import LGFF_SC
from lgff.losses.lgff_loss import LGFFLoss
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.pose_metrics import (
    fuse_pose_from_outputs,
    compute_batch_pose_metrics,
    summarize_pose_metrics,
)


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

        self.loss_components_path = os.path.join(
            self.output_dir, "loss_components_history.csv"
        )

        # 1. Device & Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # 2. Geometry
        self.geometry = GeometryToolkit()
        self.obj_diameter: Optional[float] = getattr(cfg, "obj_diameter_m", None)
        if self.obj_diameter is None:
            self.obj_diameter = getattr(cfg, "obj_diameter", None)

        self.cmd_acc_threshold_m: float = getattr(
            cfg, "cmd_threshold_m", getattr(cfg, "eval_cmd_threshold_m", 0.02)
        )

        # 3. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(cfg, "lr", 1e-4),
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

        # 4. Scheduler
        scheduler_type = getattr(cfg, "scheduler", "plateau").lower()
        self.logger.info(f"Initializing Scheduler: {scheduler_type}")

        if scheduler_type == "plateau":
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
        else:
            self.scheduler = None

        # 5. Data Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 6. AMP
        self.scaler = GradScaler("cuda", enabled=getattr(cfg, "use_amp", True))
        self.use_amp = self.scaler.is_enabled()

        # 7. Logging
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        self.global_step: int = 0
        self.start_epoch: int = 0
        self.best_val_loss: float = float("inf")

        self.history: Dict[str, list[Dict[str, float]]] = {
            "train": [],
            "val": [],
        }

        # ====== Curriculum Loss 调度（你原先只调 lambda_t） ======
        self.total_epochs: int = getattr(self.cfg, "epochs", 50)
        self.use_curriculum_loss: bool = getattr(self.cfg, "use_curriculum_loss", False)

        self.curriculum_warmup_frac: float = getattr(self.cfg, "curriculum_warmup_frac", 0.4)
        self.curriculum_final_factor_t: float = getattr(self.cfg, "curriculum_final_factor_t", 0.3)

        # ====== [新增] Seg warmup（与 curriculum 分离，默认开启但不破坏旧版本） ======
        # 只有当 loss_fn 有 lambda_seg 时才生效
        self.seg_warmup_frac: float = float(getattr(self.cfg, "seg_warmup_frac", 0.02))
        self.seg_warmup_mode: str = str(getattr(self.cfg, "seg_warmup_mode", "linear")).lower()

        # 记录基准权重（从 loss_fn 优先读取）
        self.base_lambda_dense: float = float(
            getattr(self.loss_fn, "lambda_dense", getattr(self.cfg, "lambda_add", 1.0))
        )
        self.base_lambda_t: float = float(
            getattr(self.loss_fn, "lambda_t", getattr(self.cfg, "lambda_t", 2.0))
        )
        self.base_lambda_conf: float = float(
            getattr(self.loss_fn, "lambda_conf", getattr(self.cfg, "lambda_conf", 0.1))
        )
        self.base_lambda_kp_of: float = float(
            getattr(self.loss_fn, "lambda_kp_of", getattr(self.cfg, "lambda_kp_of", 2.0))
        )
        # seg（若无则为 0，不影响）
        self.base_lambda_seg: float = float(getattr(self.loss_fn, "lambda_seg", 0.0))

        # 8. Resume
        if resume_path is not None and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Curriculum：根据 epoch 调整 lambda_t +（可选）lambda_seg warmup
    # ------------------------------------------------------------------
    def _update_loss_schedule(self, epoch: int) -> None:
        """
        - Translation curriculum: warmup 前保持 1.0，之后线性衰减到 final_factor_t
        - Seg warmup: 前 seg_warmup_frac 线性从 0 升到 1（乘 base_lambda_seg）
        """
        epoch_idx = epoch + 1
        frac = epoch_idx / max(1, self.total_epochs)

        # ---- (A) Translation curriculum ----
        if self.use_curriculum_loss:
            warm = float(self.curriculum_warmup_frac)
            warm = min(max(warm, 0.0), 1.0)

            if frac <= warm:
                scale_t = 1.0
            else:
                progress = (frac - warm) / max(1e-6, 1.0 - warm)
                progress = min(max(progress, 0.0), 1.0)

                ft = float(self.curriculum_final_factor_t)
                ft = min(max(ft, 0.0), 1.0)
                scale_t = 1.0 - progress * (1.0 - ft)

            if hasattr(self.loss_fn, "lambda_t"):
                self.loss_fn.lambda_t = self.base_lambda_t * scale_t

            # 恢复其他权重为常数（避免被别处意外改）
            if hasattr(self.loss_fn, "lambda_dense"):
                self.loss_fn.lambda_dense = self.base_lambda_dense
            if hasattr(self.loss_fn, "lambda_conf"):
                self.loss_fn.lambda_conf = self.base_lambda_conf
            if hasattr(self.loss_fn, "lambda_kp_of"):
                self.loss_fn.lambda_kp_of = self.base_lambda_kp_of

        # ---- (B) Seg warmup（仅当 loss_fn 支持 seg）----
        if hasattr(self.loss_fn, "lambda_seg") and self.base_lambda_seg > 0.0:
            w = min(max(self.seg_warmup_frac, 0.0), 1.0)

            if w <= 1e-9:
                scale_seg = 1.0
            else:
                if self.seg_warmup_mode == "linear":
                    scale_seg = min(max(frac / w, 0.0), 1.0)
                else:
                    # fallback
                    scale_seg = min(max(frac / w, 0.0), 1.0)

            self.loss_fn.lambda_seg = self.base_lambda_seg * scale_seg

        # 日志打印（可选，避免太吵：你也可以改成每 N epoch 打一次）
        seg_now = float(getattr(self.loss_fn, "lambda_seg", 0.0))
        self.logger.info(
            f"[Schedule] Epoch {epoch_idx}: "
            f"lambda_t={float(getattr(self.loss_fn, 'lambda_t', self.base_lambda_t)):.4f}, "
            f"lambda_kp={float(getattr(self.loss_fn, 'lambda_kp_of', self.base_lambda_kp_of)):.4f}, "
            f"lambda_dense={float(getattr(self.loss_fn, 'lambda_dense', self.base_lambda_dense)):.4f}, "
            f"lambda_seg={seg_now:.4f}"
        )

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------
    def fit(self) -> None:
        self.logger.info(f"Start training on device: {self.device}")

        epochs = int(getattr(self.cfg, "epochs", 50))

        for epoch in range(self.start_epoch, epochs):
            epoch_idx = epoch + 1

            self._update_loss_schedule(epoch)

            # --- Training ---
            train_metrics = self._train_one_epoch(epoch)

            # --- Validation ---
            metric_for_sched: Optional[float] = None
            val_metrics: Dict[str, float] = {"loss_total": float("nan")}

            if self.val_loader is not None:
                val_metrics = self._validate(epoch)
                val_loss = float(val_metrics.get("loss_total", float("nan")))

                if not math.isfinite(val_loss):
                    self.logger.warning(f"Epoch {epoch_idx}: val_loss is NaN/Inf.")
                else:
                    metric_for_sched = val_loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, is_best=True)
            else:
                # 无验证集用 train loss
                train_loss = float(train_metrics.get("loss_total", float("nan")))
                if math.isfinite(train_loss):
                    metric_for_sched = train_loss

            self._step_scheduler(metric_for_sched)
            self._save_checkpoint(epoch, is_best=False)
            self._record_epoch_metrics(epoch_idx, train_metrics, val_metrics)

            # 记录 Loss 分量（自动包含 seg，若存在）
            self._append_loss_components(epoch_idx, train_metrics)

            # 日志
            self._log_epoch_summary(epoch_idx, epochs, train_metrics, val_metrics)

        self._save_metrics_history()
        self.writer.close()

    def _log_epoch_summary(self, epoch_idx, epochs, train_metrics, val_metrics):
        train_loss = train_metrics.get("loss_total", float("nan"))
        val_loss = val_metrics.get("loss_total", float("nan"))
        lr_now = self.optimizer.param_groups[0]["lr"]

        t_err_mean_val = val_metrics.get("t_err_mean", float("nan"))
        acc_5mm = val_metrics.get("acc_add<5mm", float("nan"))
        acc_adds_5mm = val_metrics.get("acc_adds<5mm", float("nan"))

        extra_val = ""
        if math.isfinite(t_err_mean_val):
            extra_val = (
                f" | Val t_err={t_err_mean_val:.4f}, "
                f"acc<5mm={acc_5mm:.2%}, "
                f"acc_s<5mm={acc_adds_5mm:.2%}"
            )

        self.logger.info(
            f"Epoch [{epoch_idx}/{epochs}] Summary | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"LR={lr_now:.2e} | Best={self.best_val_loss:.6f}"
            f"{extra_val}"
        )

    # ------------------------------------------------------------------
    # Train One Epoch
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"[Train] Epoch {epoch + 1}",
            leave=True,
        )

        log_interval = int(getattr(self.cfg, "log_interval", 10))

        for i, batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(batch)
                loss, metrics = self.loss_fn(outputs, batch)
                if not isinstance(metrics, dict):
                    metrics = {"loss_total": loss.detach()}

            if loss is None or (not torch.isfinite(loss)):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            max_grad_norm = float(getattr(self.cfg, "max_grad_norm", 2.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Meters update
            bs = int(batch["rgb"].size(0))
            if not meters:
                for k in metrics.keys():
                    meters[k] = AverageMeter()
            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(val, bs)

            # Tqdm update（自动包含 seg）
            postfix = {
                "loss": f"{meters.get('loss_total', AverageMeter()).val:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            }
            for k in ["loss_t", "loss_kp", "loss_dense", "loss_conf", "loss_seg"]:
                if k in meters:
                    postfix[k] = f"{meters[k].val:.4f}"
            pbar.set_postfix(postfix)

            # Tensorboard Logging
            if i % log_interval == 0:
                self.global_step += 1
                for k, m in meters.items():
                    self.writer.add_scalar(f"Train/{k}", m.val, self.global_step)

                # 同步记录当前 lambda（便于观察 warmup）
                if hasattr(self.loss_fn, "lambda_seg"):
                    self.writer.add_scalar("Train/lambda_seg", float(getattr(self.loss_fn, "lambda_seg", 0.0)), self.global_step)
                self.writer.add_scalar("Train/lambda_t", float(getattr(self.loss_fn, "lambda_t", 0.0)), self.global_step)

        return {k: m.avg for k, m in meters.items()} if meters else {}

    # ------------------------------------------------------------------
    # Validate: loss + pose metrics
    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}
        pose_meter: Dict[str, list[float]] = {}

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"[Val]   Epoch {epoch + 1}",
                leave=True
            )

            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(batch)
                    loss, metrics = self.loss_fn(outputs, batch)
                    if not isinstance(metrics, dict):
                        metrics = {"loss_total": loss.detach()} if loss is not None else {}

                # 1) Update loss meters
                bs = int(batch["rgb"].size(0))
                if not meters:
                    for k in metrics.keys():
                        meters[k] = AverageMeter()
                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    if k not in meters:
                        meters[k] = AverageMeter()
                    meters[k].update(val, bs)

                # 2) Pose metrics
                pred_rt = fuse_pose_from_outputs(
                    outputs, self.geometry, self.cfg, stage="eval",
                    valid_mask=batch.get("labels", None)  # [B,N]
                )

                if self.obj_diameter is None and batch["model_points"].size(1) > 1:
                    mp0 = batch["model_points"][0]
                    self.obj_diameter = float(torch.cdist(mp0.unsqueeze(0), mp0.unsqueeze(0)).max().item())

                batch_pose_metrics = compute_batch_pose_metrics(
                    pred_rt=pred_rt,
                    gt_rt=batch["pose"],
                    model_points=batch["model_points"],
                    cls_ids=batch.get("cls_id"),
                    geometry=self.geometry,
                    cfg=self.cfg,
                )

                for name, val_list in batch_pose_metrics.items():
                    if name not in pose_meter:
                        pose_meter[name] = []
                    # 处理不同类型的返回值
                    if torch.is_tensor(val_list):
                        if val_list.numel() == 1:  # 单个值张量
                            pose_meter[name].append(val_list.item())
                        else:  # 多元素张量
                            pose_meter[name].extend(val_list.detach().cpu().numpy().tolist())
                    else:  # 标量值
                        pose_meter[name].append(float(val_list))

                # tqdm postfix（验证看 avg）
                if "loss_total" in meters:
                    postfix = {"loss": f"{meters['loss_total'].avg:.4f}"}
                    for k in ["loss_t", "loss_kp", "loss_dense", "loss_conf", "loss_seg"]:
                        if k in meters:
                            postfix[k] = f"{meters[k].avg:.4f}"
                    pbar.set_postfix(postfix)

        avg_metrics = {k: m.avg for k, m in meters.items()} if meters else {}

        obj_diam = float(self.obj_diameter) if self.obj_diameter else 0.0
        pose_summary = summarize_pose_metrics(pose_meter, obj_diam, self.cmd_acc_threshold_m)
        avg_metrics.update(pose_summary)

        # Tensorboard (epoch level)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"Val/{k}", float(v), epoch + 1)

        return avg_metrics

    # ------------------------------------------------------------------
    # Helper: Record Metrics
    # ------------------------------------------------------------------
    def _record_epoch_metrics(self, epoch_idx, train_metrics, val_metrics):
        self.history["train"].append({"epoch": int(epoch_idx), **train_metrics})
        if self.val_loader:
            self.history["val"].append({"epoch": int(epoch_idx), **val_metrics})

    # ------------------------------------------------------------------
    # Loss Components CSV (auto supports seg if present)
    # ------------------------------------------------------------------
    def _append_loss_components(self, epoch_idx: int, train_metrics: Dict[str, float]) -> None:
        # 统一 key（允许不存在）
        component_keys = ["loss_dense", "loss_t", "loss_kp", "loss_conf", "loss_seg"]

        lambdas = {
            "loss_dense": float(getattr(self.loss_fn, "lambda_dense", self.base_lambda_dense)),
            "loss_t": float(getattr(self.loss_fn, "lambda_t", self.base_lambda_t)),
            "loss_kp": float(getattr(self.loss_fn, "lambda_kp_of", self.base_lambda_kp_of)),
            "loss_conf": float(getattr(self.loss_fn, "lambda_conf", self.base_lambda_conf)),
            "loss_seg": float(getattr(self.loss_fn, "lambda_seg", self.base_lambda_seg)),
        }

        total_loss = float(train_metrics.get("loss_total", 0.0))
        row: Dict[str, float] = {"epoch": float(epoch_idx)}

        for k in component_keys:
            if k not in train_metrics:
                continue
            val = float(train_metrics.get(k, 0.0))
            lam = float(lambdas.get(k, 1.0))
            w_val = val * lam
            ratio = w_val / total_loss if total_loss > 1e-8 else 0.0

            row[k] = val
            row[f"w_{k}"] = w_val
            row[f"ratio_{k}"] = ratio

        # Also log current lambdas (useful)
        row["lambda_t"] = float(getattr(self.loss_fn, "lambda_t", self.base_lambda_t))
        row["lambda_dense"] = float(getattr(self.loss_fn, "lambda_dense", self.base_lambda_dense))
        row["lambda_conf"] = float(getattr(self.loss_fn, "lambda_conf", self.base_lambda_conf))
        row["lambda_kp"] = float(getattr(self.loss_fn, "lambda_kp_of", self.base_lambda_kp_of))
        if hasattr(self.loss_fn, "lambda_seg"):
            row["lambda_seg"] = float(getattr(self.loss_fn, "lambda_seg", self.base_lambda_seg))

        file_exists = os.path.exists(self.loss_components_path)
        keys = sorted(row.keys())
        with open(self.loss_components_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Save / Scheduler / Checkpoint
    # ------------------------------------------------------------------
    def _save_metrics_history(self) -> None:
        if not self.history["train"]:
            return
        with open(os.path.join(self.output_dir, "metrics_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def _step_scheduler(self, metric):
        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else self.cfg.__dict__,
        }
        torch.save(state, os.path.join(self.output_dir, "checkpoint_last.pth"))
        if is_best:
            torch.save(state, os.path.join(self.output_dir, "checkpoint_best.pth"))

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))


__all__ = ["TrainerSC"]
