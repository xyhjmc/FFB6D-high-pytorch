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

from torch.amp import autocast, GradScaler  # ✅ 新 AMP API
from tqdm.auto import tqdm  # ✅ 进度条

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

        # loss 分量历史记录（仅统计，不影响训练）
        self.loss_components_path = os.path.join(
            self.output_dir, "loss_components_history.csv"
        )

        # 1. Device & Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # 2. Geometry / Pose metrics 相关（与 EvaluatorSC 对齐）
        self.geometry = GeometryToolkit()

        # 物体直径（单位 m），优先读取 obj_diameter_m
        self.obj_diameter: Optional[float] = getattr(cfg, "obj_diameter_m", None)
        if self.obj_diameter is None:
            # 向后兼容旧字段 obj_diameter
            self.obj_diameter = getattr(cfg, "obj_diameter", None)

        # CMD accuracy 阈值（单位 m），与 EvaluatorSC / pose_metrics 统一
        self.cmd_acc_threshold_m: float = getattr(
            cfg,
            "cmd_threshold_m",
            getattr(cfg, "eval_cmd_threshold_m", 0.02),
        )

        # 3. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(cfg, "lr", 1e-4),
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

        # 4. Scheduler Configuration
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
        elif scheduler_type == "none":
            self.scheduler = None
        else:
            self.logger.warning(
                f"Unknown scheduler type '{scheduler_type}', defaulting to None."
            )
            self.scheduler = None

        # 5. Data Loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 6. AMP Scaler（新 API）
        self.scaler = GradScaler("cuda", enabled=getattr(cfg, "use_amp", True))
        self.use_amp = self.scaler.is_enabled()

        # 7. Logging / State
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
        self.global_step: int = 0
        self.start_epoch: int = 0
        self.best_val_loss: float = float("inf")

        # 保存 epoch 级别的指标历史（train / val）
        self.history: Dict[str, list[Dict[str, float]]] = {
            "train": [],
            "val": [],
        }

        # ====== Curriculum Loss 调度：对 lambda_t / lambda_rot 做 epoch 级调节 ======
        self.total_epochs: int = getattr(self.cfg, "epochs", 50)
        self.use_curriculum_loss: bool = getattr(self.cfg, "use_curriculum_loss", False)
        # 在前 warmup_frac 的 epoch，中等权重保持不变；之后线性衰减到 final_factor
        self.curriculum_warmup_frac: float = getattr(
            self.cfg, "curriculum_warmup_frac", 0.4
        )
        self.curriculum_final_factor_t: float = getattr(
            self.cfg, "curriculum_final_factor_t", 0.3
        )
        self.curriculum_final_factor_rot: float = getattr(
            self.cfg, "curriculum_final_factor_rot", 0.3
        )

        # 记录初始的 loss 权重，作为 curriculum 的基准
        # 注意：这里直接从 loss_fn 上读，避免 cfg 与 loss 内部默认不一致
        self.base_lambda_dense: float = float(
            getattr(self.loss_fn, "lambda_dense", getattr(self.cfg, "lambda_add", 1.0))
        )
        self.base_lambda_t: float = float(
            getattr(self.loss_fn, "lambda_t", getattr(self.cfg, "lambda_t", 0.5))
        )
        self.base_lambda_rot: float = float(
            getattr(self.loss_fn, "lambda_rot", getattr(self.cfg, "lambda_rot", 0.5))
        )
        self.base_lambda_conf: float = float(
            getattr(self.loss_fn, "lambda_conf", getattr(self.cfg, "lambda_conf", 0.1))
        )
        self.base_lambda_add_cad: float = float(
            getattr(self.loss_fn, "lambda_add_cad", getattr(self.cfg, "lambda_add_cad", 0.0))
        )
        self.base_lambda_kp_of: float = float(
            getattr(self.loss_fn, "lambda_kp_of", getattr(self.cfg, "lambda_kp_of", 0.6))
        )

        # 8. Resume (optional)
        if resume_path is not None and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Curriculum：根据 epoch 调整 loss_fn 内部的 lambda_t / lambda_rot
    # ------------------------------------------------------------------
    def _update_loss_schedule(self, epoch: int) -> None:
        """基于 epoch 对 loss 分支权重做一个简单的 coarse-to-fine 调度。"""
        if not self.use_curriculum_loss:
            return

        epoch_idx = epoch + 1
        frac = epoch_idx / max(1, self.total_epochs)

        warm = self.curriculum_warmup_frac
        warm = min(max(warm, 0.0), 1.0)

        if frac <= warm:
            scale_t = 1.0
            scale_rot = 1.0
        else:
            # 在 [warm, 1.0] 区间线性从 1.0 衰减到 final_factor
            progress = (frac - warm) / max(1e-6, 1.0 - warm)
            progress = min(max(progress, 0.0), 1.0)

            ft = self.curriculum_final_factor_t
            fr = self.curriculum_final_factor_rot
            ft = min(max(ft, 0.0), 1.0)
            fr = min(max(fr, 0.0), 1.0)

            scale_t = 1.0 - progress * (1.0 - ft)
            scale_rot = 1.0 - progress * (1.0 - fr)

        # 应用到 loss_fn 内部权重
        if hasattr(self.loss_fn, "lambda_t"):
            self.loss_fn.lambda_t = self.base_lambda_t * scale_t
        if hasattr(self.loss_fn, "lambda_rot"):
            self.loss_fn.lambda_rot = self.base_lambda_rot * scale_rot

        # 其他分支保持常数（如有需要，也可以在此对 lambda_dense / lambda_kp_of 做反向增强）
        if hasattr(self.loss_fn, "lambda_dense"):
            self.loss_fn.lambda_dense = self.base_lambda_dense
        if hasattr(self.loss_fn, "lambda_conf"):
            self.loss_fn.lambda_conf = self.base_lambda_conf
        if hasattr(self.loss_fn, "lambda_add_cad"):
            self.loss_fn.lambda_add_cad = self.base_lambda_add_cad
        if hasattr(self.loss_fn, "lambda_kp_of"):
            self.loss_fn.lambda_kp_of = self.base_lambda_kp_of

        self.logger.info(
            f"[Curriculum] Epoch {epoch_idx}: "
            f"lambda_t={getattr(self.loss_fn, 'lambda_t', self.base_lambda_t):.4f}, "
            f"lambda_rot={getattr(self.loss_fn, 'lambda_rot', self.base_lambda_rot):.4f}, "
            f"lambda_dense={getattr(self.loss_fn, 'lambda_dense', self.base_lambda_dense):.4f}, "
            f"lambda_kp_of={getattr(self.loss_fn, 'lambda_kp_of', self.base_lambda_kp_of):.4f}"
        )

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------
    def fit(self) -> None:
        self.logger.info(f"Start training on device: {self.device}")

        epochs = getattr(self.cfg, "epochs", 50)

        for epoch in range(self.start_epoch, epochs):
            epoch_idx = epoch + 1  # for display

            # 更新当前 epoch 的 loss 权重（Curriculum）
            self._update_loss_schedule(epoch)

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

            # 记录 epoch 级别的指标到 self.history
            self._record_epoch_metrics(epoch_idx, train_metrics, val_metrics)

            # 仅统计：记录 loss 分量、权重与占比（与新 loss 命名保持一致）
            self._append_loss_components(epoch_idx, train_metrics)

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

            # 额外关心的几个指标（如果存在的话就顺带打印）
            t_err_mean_tr = train_metrics.get("t_err_mean", float("nan"))
            t_err_z_tr = train_metrics.get("t_err_z", float("nan"))
            t_bias_z_tr = train_metrics.get("t_bias_z", float("nan"))

            t_err_mean_val = val_metrics.get("t_err_mean", float("nan"))
            t_err_z_val = val_metrics.get("t_err_z", float("nan"))
            t_bias_z_val = val_metrics.get("t_bias_z", float("nan"))

            extra_train = (
                f" | Train t_err_mean={t_err_mean_tr:.4f}, "
                f"t_err_z={t_err_z_tr:.4f}, t_bias_z={t_bias_z_tr:.4f}"
                if math.isfinite(float(t_err_mean_tr))
                else ""
            )
            extra_val = (
                f" | Val t_err_mean={t_err_mean_val:.4f}, "
                f"t_err_z={t_err_z_val:.4f}, t_bias_z={t_bias_z_val:.4f}"
                if self.val_loader is not None and math.isfinite(float(t_err_mean_val))
                else ""
            )

            # 顺带把统一指标中的 mean_add_s / mean_rot_err 打印一下（如果存在）
            mean_add_s_val = val_metrics.get("mean_add_s", float("nan"))
            mean_rot_err_val = val_metrics.get("mean_rot_err", float("nan"))
            extra_pose_val = ""
            if math.isfinite(float(mean_add_s_val)) and math.isfinite(float(mean_rot_err_val)):
                extra_pose_val = (
                    f" | Val mean_add_s={mean_add_s_val:.4f}, "
                    f"mean_rot_err={mean_rot_err_val:.2f}"
                )

            self.logger.info(
                f"Epoch [{epoch_idx}/{epochs}] Summary | "
                f"Train loss_total={train_str} | "
                f"Val loss_total={val_str} | "
                f"LR={lr_now:.2e} | "
                f"Best Val={best_info}"
                f"{extra_train}{extra_val}{extra_pose_val}"
            )

        # 训练完一次性把历史指标存成 JSON / CSV
        self._save_metrics_history()

        self.writer.close()

    # ------------------------------------------------------------------
    # 单个 epoch 的训练（仅用 loss 相关指标，保持开销可控）
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
            leave=True,
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

            # 若 loss 出现 NaN/Inf，直接跳过该 step（防止梯度爆炸污染模型）
            if not torch.isfinite(loss):
                self.logger.warning(
                    f"[Train] Epoch {epoch + 1}, step {i}: loss is NaN/Inf，skip this batch."
                )
                continue

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

            # 更新 tqdm 状态栏：主指标 + 若干关键辅助指标
            show_keys = list(processed_metrics.keys())
            main_key = "loss_total" if "loss_total" in show_keys else show_keys[0]

            postfix = {
                main_key: f"{meters[main_key].val:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            }

            # 如果这些指标存在，就顺带展示
            for k in ["loss_t", "loss_rot", "loss_dense", "loss_conf", "loss_add_cad", "loss_kp_of"]:
                if k in meters:
                    postfix[k] = f"{meters[k].val:.4f}"

            pbar.set_postfix(postfix)

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
    # 验证阶段：loss + 统一 pose 指标（与 EvaluatorSC 一致）
    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        # pose 级别指标的累积容器（逐样本）
        pose_meter: Dict[str, list[float]] = {
            "add_s": [],
            "add": [],
            "t_err": [],
            "t_err_x": [],
            "t_err_y": [],
            "t_err_z": [],
            "rot_err": [],
            "cmd_acc": [],
        }

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"[Val]   Epoch {epoch + 1}",
                leave=True,
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

                # 1) loss 相关 metrics（与原来一致）
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

                # 2) 统一的姿态指标（和 EvaluatorSC / pose_metrics 一致）
                pred_rt = fuse_pose_from_outputs(
                    outputs, self.geometry, self.cfg, stage="eval"
                )
                gt_rt = batch["pose"]
                model_points = batch["model_points"]  # [B, M, 3]

                # 若 Trainer 侧还没有直径，且 M>1，则用当前 batch 的 CAD 点云估算一次
                if self.obj_diameter is None and model_points.size(1) > 1:
                    with torch.no_grad():
                        mp0 = model_points[0]  # [M, 3]
                        dist_mat = torch.cdist(
                            mp0.unsqueeze(0), mp0.unsqueeze(0)
                        ).squeeze(0)
                        self.obj_diameter = float(dist_mat.max().item())
                        self.logger.info(
                            f"[TrainerSC] Estimated obj_diameter from CAD (val): "
                            f"{self.obj_diameter:.6f} m"
                        )

                cls_ids = batch.get("cls_id", None)

                batch_pose_metrics = compute_batch_pose_metrics(
                    pred_rt=pred_rt,
                    gt_rt=gt_rt,
                    model_points=model_points,
                    cls_ids=cls_ids,
                    geometry=self.geometry,
                    cfg=self.cfg,
                )

                # 累计到 pose_meter（numpy -> list）
                for name, tensor_1d in batch_pose_metrics.items():
                    if name not in pose_meter:
                        pose_meter[name] = []
                    pose_meter[name].extend(tensor_1d.numpy().tolist())

                # 更新 tqdm 状态：同样展示多个关键指标
                main_key = "loss_total" if "loss_total" in meters else list(meters.keys())[0]

                postfix = {
                    main_key: f"{meters[main_key].val:.4f}",
                }
                for k in ["loss_t", "loss_rot", "loss_dense", "loss_conf", "loss_add_cad", "loss_kp_of"]:
                    if k in meters:
                        postfix[k] = f"{meters[k].val:.4f}"
                pbar.set_postfix(postfix)

        # 3) 汇总 loss 级别的平均指标
        if not meters:
            avg_metrics: Dict[str, float] = {"loss_total": float("nan")}
        else:
            avg_metrics = {k: m.avg for k, m in meters.items()}

        # 4) 使用 summarize_pose_metrics 汇总统一的姿态指标
        obj_diam = self.obj_diameter if self.obj_diameter is not None else 0.0
        pose_summary = summarize_pose_metrics(
            meter=pose_meter,
            obj_diameter=obj_diam,
            cmd_threshold_m=self.cmd_acc_threshold_m,
        )

        # 5) 合并：loss + pose 指标，作为最终验证指标字典返回
        avg_metrics.update(pose_summary)

        # 写入 TensorBoard（Val 用 epoch 作为 X 轴）
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch + 1)

        self.logger.info(f"Epoch {epoch + 1} Val Summary: {avg_metrics}")
        return avg_metrics

    # ------------------------------------------------------------------
    # 记录每个 epoch 的指标到 self.history
    # ------------------------------------------------------------------
    def _record_epoch_metrics(
        self,
        epoch_idx: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        train_entry = {"epoch": epoch_idx}
        train_entry.update({k: float(v) for k, v in train_metrics.items()})
        self.history["train"].append(train_entry)

        if self.val_loader is not None:
            val_entry = {"epoch": epoch_idx}
            val_entry.update({k: float(v) for k, v in val_metrics.items()})
            self.history["val"].append(val_entry)

    # ------------------------------------------------------------------
    # 记录 loss 分量、权重与占比（仅用于监控，已适配新 loss 命名）
    # ------------------------------------------------------------------
    def _append_loss_components(self, epoch_idx: int, train_metrics: Dict[str, float]) -> None:
        """
        注意：
        - 新版 LGFFLoss 中主要分量命名为：
          loss_dense, loss_t, loss_rot, loss_conf, loss_add_cad, loss_kp_of
        - 这里使用 loss_fn 当前的 lambda_*（包含 curriculum 或 uncertainty 的影响），
          以便统计“实际有效权重”下各分量占比。
        """
        component_keys = [
            "loss_dense",
            "loss_t",
            "loss_rot",
            "loss_conf",
            "loss_add_cad",
            "loss_kp_of",
        ]
        short_names = {
            "loss_dense": "dense",
            "loss_t": "t",
            "loss_rot": "rot",
            "loss_conf": "conf",
            "loss_add_cad": "add_cad",
            "loss_kp_of": "kp_of",
        }

        # 从 loss_fn 上读取当前有效权重（包括 curriculum 调整后的值）
        lambdas = {
            "loss_dense": float(
                getattr(self.loss_fn, "lambda_dense", self.base_lambda_dense)
            ),
            "loss_t": float(
                getattr(self.loss_fn, "lambda_t", self.base_lambda_t)
            ),
            "loss_rot": float(
                getattr(self.loss_fn, "lambda_rot", self.base_lambda_rot)
            ),
            "loss_conf": float(
                getattr(self.loss_fn, "lambda_conf", self.base_lambda_conf)
            ),
            "loss_add_cad": float(
                getattr(self.loss_fn, "lambda_add_cad", self.base_lambda_add_cad)
            ),
            "loss_kp_of": float(
                getattr(self.loss_fn, "lambda_kp_of", self.base_lambda_kp_of)
            ),
        }

        total_loss = float(train_metrics.get("loss_total", 0.0))
        components = {k: float(train_metrics.get(k, 0.0)) for k in component_keys}

        # 若后续启用了 uncertainty weighting，可以考虑将 lambdas 再乘以 exp(-log_var)
        # 这里先保持简单，实现一个“近似占比”统计。
        weighted = {
            f"w_{short_names[k]}": lambdas[k] * components[k]
            for k in component_keys
        }

        ratios = {}
        for k in component_keys:
            num = weighted[f"w_{short_names[k]}"]
            if math.isfinite(total_loss) and abs(total_loss) > 1e-8:
                ratios[f"ratio_{short_names[k]}"] = num / total_loss
            else:
                ratios[f"ratio_{short_names[k]}"] = 0.0

        row = {"epoch": epoch_idx}
        row.update(components)
        row.update(weighted)
        row.update(ratios)

        write_header = not os.path.exists(self.loss_components_path)
        with open(self.loss_components_path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [
                "epoch",
                "loss_dense",
                "loss_t",
                "loss_rot",
                "loss_conf",
                "loss_add_cad",
                "loss_kp_of",
                "w_dense",
                "w_t",
                "w_rot",
                "w_conf",
                "w_add_cad",
                "w_kp_of",
                "ratio_dense",
                "ratio_t",
                "ratio_rot",
                "ratio_conf",
                "ratio_add_cad",
                "ratio_kp_of",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, 0.0) for k in fieldnames})

        # 简短日志，帮助人工检查量纲/权重
        def _fmt_ratio(full_key: str) -> str:
            short = short_names.get(full_key, full_key)
            return f"{ratios.get(f'ratio_{short}', 0.0) * 100:.0f}%"

        summary_lines = [
            f"[LossBreakdown] epoch={epoch_idx} | total={total_loss:.6f}",
            f"  - dense: {components['loss_dense']:.6f} ({_fmt_ratio('loss_dense')})",
            f"  - t:     {components['loss_t']:.6f} ({_fmt_ratio('loss_t')})",
            f"  - rot:   {components['loss_rot']:.6f} ({_fmt_ratio('loss_rot')})",
            f"  - conf:  {components['loss_conf']:.6f} ({_fmt_ratio('loss_conf')})",
            f"  - cad:   {components['loss_add_cad']:.6f} ({_fmt_ratio('loss_add_cad')})",
            f"  - kp_of: {components['loss_kp_of']:.6f} ({_fmt_ratio('loss_kp_of')})",
        ]
        self.logger.info("\n".join(summary_lines))

    # ------------------------------------------------------------------
    # 把 history 存成 JSON / CSV（epoch 维度）
    # ------------------------------------------------------------------
    def _save_metrics_history(self) -> None:
        if not self.history["train"] and not self.history["val"]:
            self.logger.warning("No metrics history to save.")
            return

        json_path = os.path.join(self.output_dir, "metrics_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metrics history (JSON) to {json_path}")

        # 统一所有键，写一份 CSV，方便画图 / 对比
        all_rows = []
        all_keys = set(["split", "epoch"])

        for split in ("train", "val"):
            for row in self.history[split]:
                r = {"split": split}
                r.update(row)
                all_rows.append(r)
                all_keys.update(r.keys())

        all_keys = sorted(all_keys)

        csv_path = os.path.join(self.output_dir, "metrics_history.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

        self.logger.info(f"Saved metrics history (CSV) to {csv_path}")

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
            "epoch_idx": epoch + 1,
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
