# lgff/engines/trainer_sc_seg.py
from __future__ import annotations

import logging
import os
from dataclasses import asdict
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
from lgff.engines.evaluator_sc_seg import EvaluatorSCSeg

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
        self.best_metric_name = self._resolve_best_metric_name()
        self.best_metric_value = float("-inf")

    def _resolve_best_metric_name(self) -> str:
        sym_ids = getattr(self.cfg, "sym_class_ids", [])
        obj_id = getattr(self.cfg, "obj_id", None)
        if obj_id is not None and obj_id in sym_ids:
            return "acc_adds<0.100d"
        return "acc_add<0.100d"

    def _get_best_metric_value(self, val_metrics: Dict[str, float]) -> Optional[float]:
        metric_name = self._resolve_best_metric_name()
        metric_value = val_metrics.get(metric_name, None)
        if metric_value is None:
            return None
        if not torch.isfinite(torch.tensor(metric_value)):
            return None
        return float(metric_value)

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "config": asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else self.cfg.__dict__,
        }
        torch.save(state, os.path.join(self.output_dir, "checkpoint_last.pth"))
        if is_best:
            torch.save(state, os.path.join(self.output_dir, "checkpoint_best.pth"))

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_metric_name = ckpt.get("best_metric_name", self._resolve_best_metric_name())
        self.best_metric_value = ckpt.get("best_metric_value", float("-inf"))

    # -------------------------------------------------------------------------
    # [NEW] 重写训练循环
    # -------------------------------------------------------------------------
    def fit(self) -> None:
        self.logger.info(f"Start training on device: {self.device}")

        epochs = getattr(self.cfg, "epochs", 50)

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
                best_metric_value = self._get_best_metric_value(val_metrics)

                if not torch.isfinite(torch.tensor(val_loss)):
                    self.logger.warning(f"Epoch {epoch_idx}: val_loss is NaN/Inf.")
                else:
                    metric_for_sched = val_loss
                    if best_metric_value is None:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_metric_name = "loss_total"
                            self.best_metric_value = val_loss
                            self._save_checkpoint(epoch, is_best=True)
                    else:
                        if best_metric_value > self.best_metric_value:
                            self.best_metric_name = self._resolve_best_metric_name()
                            self.best_metric_value = best_metric_value
                            self._save_checkpoint(epoch, is_best=True)
            else:
                train_loss = float(train_metrics.get("loss_total", float("nan")))
                if torch.isfinite(torch.tensor(train_loss)):
                    metric_for_sched = train_loss

            self._step_scheduler(metric_for_sched)
            self._save_checkpoint(epoch, is_best=False)
            self._record_epoch_metrics(epoch_idx, train_metrics, val_metrics)

            self._append_loss_components(epoch_idx, train_metrics)

            self._log_epoch_summary(epoch_idx, epochs, train_metrics, val_metrics)

        self._save_metrics_history()
        if hasattr(self, "writer") and self.writer:
            self.writer.close()

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
    def _seg_binary_metrics_from_logits(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        valid_mask_2d: Optional[torch.Tensor] = None,
        thr: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        if pred_logits.shape[1] == 1:
            prob = torch.sigmoid(pred_logits)[:, 0]  # [B,H,W]
        else:
            prob = torch.softmax(pred_logits, dim=1)[:, 1]  # [B,H,W]

        pred = (prob > float(thr)).to(dtype=torch.bool)  # [B,H,W]

        gt = gt_mask
        if gt.dim() == 4:
            gt = gt[:, 0]
        gt = (gt > 0.5).to(dtype=torch.bool)  # [B,H,W]

        if valid_mask_2d is not None:
            vm = valid_mask_2d
            if vm.dim() == 4:
                vm = vm[:, 0]
            vm = (vm > 0.5)
        else:
            vm = None

        def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            return num / den.clamp_min(eps)

        def compute_on(mask_eval: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if mask_eval is None:
                pe = pred
                ge = gt
            else:
                pe = pred & mask_eval
                ge = gt & mask_eval

            tp = (pe & ge).flatten(1).sum(dim=1).float()
            fp = (pe & (~ge)).flatten(1).sum(dim=1).float()
            fn = ((~pe) & ge).flatten(1).sum(dim=1).float()

            iou = _safe_div(tp, tp + fp + fn)
            dice = _safe_div(2 * tp, 2 * tp + fp + fn)
            prec = _safe_div(tp, tp + fp)
            rec = _safe_div(tp, tp + fn)
            return iou, dice, prec, rec

        iou_full, dice_full, prec_full, rec_full = compute_on(None)
        iou_valid, dice_valid, prec_valid, rec_valid = compute_on(vm)

        return {
            "seg_iou_full": iou_full,
            "seg_dice_full": dice_full,
            "seg_prec_full": prec_full,
            "seg_rec_full": rec_full,
            "seg_iou_valid": iou_valid,
            "seg_dice_valid": dice_valid,
            "seg_prec_valid": prec_valid,
            "seg_rec_valid": rec_valid,
        }

    def _compute_valid_mask_from_seg(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        pred_logits = outputs.get("pred_mask_logits", None)
        if not isinstance(pred_logits, torch.Tensor):
            return None
        choose = batch.get("choose", None)
        if not isinstance(choose, torch.Tensor):
            return None

        if pred_logits.shape[1] == 1:
            probs = torch.sigmoid(pred_logits)  # [B,1,H,W]
        else:
            probs = torch.softmax(pred_logits, dim=1)[:, 1:2]  # [B,1,H,W]

        B, _, H, W = probs.shape
        HW = H * W
        if int(choose.max().item()) >= HW:
            return None

        probs_flat = probs.view(B, -1)
        point_probs = torch.gather(probs_flat, 1, choose)
        thr = float(getattr(self.cfg, "seg_point_thresh", 0.5))
        return point_probs > thr

    def _resolve_pose_valid_mask(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self.cfg, "pose_fusion_use_valid_mask", False)):
            return None
        mask_src = str(getattr(self.cfg, "pose_fusion_valid_mask_source", "")).lower().strip()
        if mask_src == "seg":
            vm = outputs.get("pred_valid_mask_bool", None)
            if isinstance(vm, torch.Tensor):
                if vm.dim() == 3 and vm.shape[-1] == 1:
                    return vm.squeeze(-1).bool()
                return vm.bool()
            return self._compute_valid_mask_from_seg(outputs, batch)
        if mask_src == "labels":
            lbl = batch.get("labels", None)
            if isinstance(lbl, torch.Tensor):
                valid_mask = lbl > 0
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0).expand(outputs["pred_quat"].shape[0], -1)
                return valid_mask
        return None

    # -------------------------------------------------------------------------
    # Validation Loop (With Detailed Losses)
    # -------------------------------------------------------------------------
    def _validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss_total": float("nan")}

        self.model.eval()
        meters: Dict[str, AverageMeter] = {}
        pose_meter: Dict[str, list[float]] = {}
        use_eval_metrics = bool(getattr(self.cfg, "best_metric_use_eval", False))

        seg_meters = {
            "seg_iou_full": AverageMeter(),
            "seg_dice_full": AverageMeter(),
            "seg_prec_full": AverageMeter(),
            "seg_rec_full": AverageMeter(),
            "seg_iou_valid": AverageMeter(),
            "seg_dice_valid": AverageMeter(),
            "seg_prec_valid": AverageMeter(),
            "seg_rec_valid": AverageMeter(),
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
                    valid_2d = batch.get("mask_valid", None)
                    seg_res = self._seg_binary_metrics_from_logits(
                        outputs["pred_mask_logits"], batch["mask"], valid_mask_2d=valid_2d, thr=0.5
                    )
                    for k, v in seg_res.items():
                        if k in seg_meters:
                            seg_meters[k].update(v.mean().item(), bs)

                # 4. Pose Metrics
                if not use_eval_metrics:
                    valid_mask = self._resolve_pose_valid_mask(outputs, batch)
                    pred_rt = fuse_pose_from_outputs(
                        outputs, self.geometry, self.cfg, stage="eval", valid_mask=valid_mask
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
                if seg_meters["seg_iou_full"].count > 0:
                    postfix["mIoU"] = f"{seg_meters['seg_iou_full'].avg:.2%}"

                pbar.set_postfix(postfix)

        # Summarize
        avg_metrics = {k: m.avg for k, m in meters.items()} if meters else {}

        if seg_meters["seg_iou_full"].count > 0:
            for k, meter in seg_meters.items():
                avg_metrics[f"val_{k}"] = meter.avg

        if use_eval_metrics:
            evaluator = EvaluatorSCSeg(
                model=self.model,
                test_loader=self.val_loader,
                cfg=self.cfg,
                geometry=self.geometry,
                save_dir=self.output_dir,
                save_per_image=bool(getattr(self.cfg, "best_metric_eval_save_csv", False)),
            )
            avg_metrics.update(evaluator.run())
        else:
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
        val_iou = val_metrics.get("val_seg_iou_full", float("nan"))

        add_s_5mm = val_metrics.get("acc_adds<5mm", float("nan"))

        lr_now = self.optimizer.param_groups[0]["lr"]

        # [CHANGED] 增加 Val Seg Loss 的日志
        msg = (
            f"Epoch [{epoch_idx}/{epochs}] | LR: {lr_now:.2e} | "
            f"T-Loss: {train_loss:.4f} (Seg: {train_seg:.4f}) | "
            f"V-Loss: {val_loss:.4f} (Seg: {val_seg:.4f}) | V-mIoU: {val_iou:.2%} | "
            f"V-ADD-S(5mm): {add_s_5mm:.2%} | Best={self.best_metric_name}={self.best_metric_value:.6f}"
        )
        self.logger.info(msg)


__all__ = ["TrainerSCSeg"]
