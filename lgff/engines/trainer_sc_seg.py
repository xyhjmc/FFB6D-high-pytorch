# lgff/engines/trainer_sc_seg.py
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional, Any

# 引入配置与基类
from lgff.utils.config_seg import LGFFConfigSeg
from lgff.engines.trainer_sc import TrainerSC, AverageMeter
from lgff.utils.pose_metrics_seg import (
    fuse_pose_from_outputs,
    compute_batch_pose_metrics,
    summarize_pose_metrics,
)

# [FIX] 使用新的 amp 接口
from torch.amp import autocast


class TrainerSCSeg(TrainerSC):
    """
    Trainer for Segmentation-Enhanced LGFF.

    Updates:
    - Overrides _train_one_epoch: dynamic logging for train.
    - Overrides _validate: dynamic logging for val (shows detailed losses like loss_seg).
    - Fixes FutureWarning for torch.amp.autocast.
    """

    def __init__(
            self,
            model,
            loss_fn,
            train_loader,
            val_loader,
            cfg: LGFFConfigSeg,
            output_dir: str = "output",
            resume_path: Optional[str] = None,
    ) -> None:
        super().__init__(model, loss_fn, train_loader, val_loader, cfg, output_dir, resume_path)
        self.logger = logging.getLogger("lgff.trainer_seg")
        self.cfg = cfg

    # -------------------------------------------------------------------------
    # [NEW] 重写训练循环
    # -------------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        meters: Dict[str, AverageMeter] = {}

        # TQDM 进度条
        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch + 1}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            # 1. Move data
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 2. Forward & Backward
            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(batch)
                loss, metrics = self.loss_fn(outputs, batch)

            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
                if hasattr(self.cfg, "max_grad_norm") and self.cfg.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if hasattr(self.cfg, "max_grad_norm") and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            # 3. Metrics Tracking
            if not isinstance(metrics, dict):
                metrics = {"loss": loss.item()}

            if "loss" not in metrics:
                metrics["loss"] = loss.item()

            bs = batch["rgb"].size(0)
            for k, v in metrics.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(float(v), bs)

            # 4. Update Progress Bar (Train: show .val)
            postfix = {
                "loss": f"{meters['loss'].val:.4f}",
            }

            ignore_keys = ["loss", "sym_rate", "sym_r"]
            for k in meters:
                if k not in ignore_keys:
                    postfix[k] = f"{meters[k].val:.4f}"

            pbar.set_postfix(postfix)

        return {k: m.avg for k, m in meters.items()}

    # -------------------------------------------------------------------------
    # Seg Metrics Helper
    # -------------------------------------------------------------------------
    def _compute_seg_metrics(self, pred_seg: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
        """Compute IoU and Accuracy for binary segmentation."""
        if pred_seg.shape[1] == 1:
            pred_mask = (torch.sigmoid(pred_seg) > 0.5).float()
        else:
            pred_mask = torch.argmax(pred_seg, dim=1, keepdim=True).float()

        gt_mask = gt_mask.float()

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)

        correct = (pred_mask == gt_mask).sum()
        total = gt_mask.numel()
        acc = correct.float() / total

        return {"seg_iou": iou.item(), "seg_acc": acc.item()}

    # -------------------------------------------------------------------------
    # Validation Loop (With Detailed Losses)
    # -------------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}
        pose_meter: Dict[str, list[float]] = {}

        seg_meters = {
            "seg_iou": AverageMeter(),
            "seg_acc": AverageMeter()
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[Val] Epoch {epoch + 1}", leave=True)

            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 1. Forward
                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(batch)
                    loss, metrics = self.loss_fn(outputs, batch)
                    if not isinstance(metrics, dict):
                        metrics = {"loss_total": loss.detach()} if loss is not None else {}

                # 2. Update Loss Meters
                bs = int(batch["rgb"].size(0))
                if not meters:
                    for k in metrics.keys():
                        meters[k] = AverageMeter()
                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    if k not in meters: meters[k] = AverageMeter()
                    meters[k].update(val, bs)

                # 3. Update Seg Metrics
                if "mask" in batch and "pred_mask_logits" in outputs and outputs["pred_mask_logits"] is not None:
                    seg_res = self._compute_seg_metrics(outputs["pred_mask_logits"], batch["mask"])
                    for k, v in seg_res.items():
                        seg_meters[k].update(v, bs)

                # 4. Pose Metrics
                pred_rt = fuse_pose_from_outputs(
                    outputs, self.geometry, self.cfg, stage="eval",
                    valid_mask=batch.get("labels", None)
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
                    if name not in pose_meter: pose_meter[name] = []
                    if torch.is_tensor(val_list):
                        if val_list.numel() == 1:
                            pose_meter[name].append(val_list.item())
                        else:
                            pose_meter[name].extend(val_list.detach().cpu().numpy().tolist())
                    else:
                        pose_meter[name].append(float(val_list))

                # Update PBar (Dynamic Detailed Losses)
                postfix = {}

                # 1. Total Loss (Renamed)
                if "loss_total" in meters:
                    postfix["val_loss"] = f"{meters['loss_total'].avg:.4f}"

                # 2. Detailed Losses (loss_seg, loss_t, etc.)
                # 过滤掉 sym_rate 和已经显示过的 loss_total
                ignore_keys = ["loss_total", "sym_rate", "sym_r"]
                for k in meters:
                    if k not in ignore_keys:
                        # Val 显示 accumulative average (.avg) 而非 instantaneous value (.val)
                        postfix[k] = f"{meters[k].avg:.4f}"

                # 3. Seg IoU
                if seg_meters["seg_iou"].count > 0:
                    postfix["mIoU"] = f"{seg_meters['seg_iou'].avg:.2%}"

                pbar.set_postfix(postfix)

        # Summarize
        avg_metrics = {k: m.avg for k, m in meters.items()} if meters else {}

        if seg_meters["seg_iou"].count > 0:
            avg_metrics["val_seg_iou"] = seg_meters["seg_iou"].avg
            avg_metrics["val_seg_acc"] = seg_meters["seg_acc"].avg

        obj_diam = float(self.obj_diameter) if self.obj_diameter else 0.0
        pose_summary = summarize_pose_metrics(pose_meter, obj_diam, self.cmd_acc_threshold_m)
        avg_metrics.update(pose_summary)

        # Tensorboard
        if hasattr(self, "writer") and self.writer:
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f"Val/{k}", float(v), epoch + 1)

        return avg_metrics

    def _log_epoch_summary(self, epoch_idx, epochs, train_metrics, val_metrics):
        """Enhanced logging with Seg info."""
        train_loss = train_metrics.get("loss", float("nan"))
        train_seg = train_metrics.get("loss_seg", float("nan"))

        val_loss = val_metrics.get("loss_total", float("nan"))
        val_seg = val_metrics.get("loss_seg", float("nan"))  # [NEW] Log val seg loss
        val_iou = val_metrics.get("val_seg_iou", float("nan"))

        add_s_5mm = val_metrics.get("acc_adds<5mm", float("nan"))

        lr_now = self.optimizer.param_groups[0]["lr"]

        # [CHANGED] 增加 Val Seg Loss 的日志
        msg = (
            f"Epoch [{epoch_idx}/{epochs}] | LR: {lr_now:.2e} | "
            f"T-Loss: {train_loss:.4f} (Seg: {train_seg:.4f}) | "
            f"V-Loss: {val_loss:.4f} (Seg: {val_seg:.4f}) | V-mIoU: {val_iou:.2%} | V-ADD-S(5mm): {add_s_5mm:.2%}"
        )
        self.logger.info(msg)


__all__ = ["TrainerSCSeg"]