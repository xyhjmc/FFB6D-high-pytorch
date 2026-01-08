# lgff/engines/evaluator_sc_seg.py
from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lgff.utils.config_seg import LGFFConfigSeg
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc_seg import LGFF_SC_SEG
from lgff.utils.pose_metrics_seg import (
    fuse_pose_from_outputs,
    compute_batch_pose_metrics,
)

# Optional PnP
try:
    from lgff.utils.pnp_solver import PnPSolver
except ImportError:
    PnPSolver = None


def _np_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x.mean())


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / (den.clamp_min(eps))


class _WarnOnce:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def warn(self, logger: logging.Logger, msg: str) -> None:
        if msg in self._seen:
            return
        self._seen.add(msg)
        logger.warning(msg)


class EvaluatorSCSeg:
    """
    Evaluator for LGFF single-class segmentation variant.

    Enhancements vs your current version:
    - FIX: detect seg logits key automatically (pred_mask_logits is the one in your output)
    - Compute detailed seg metrics: IoU/Dice/Precision/Recall for FULL and VALID(mask_valid)
    - Full per-image CSV similar to EvaluatorSC (ADD, ADD-S, thresholds, t/rot error, cmd, seg metrics)
    - Robust seg->point valid_mask mapping using 'choose' (Top-K safety net)
    - Optional PnP (non-symmetric) + optional ICP refinement
    """

    def __init__(
        self,
        model: LGFF_SC_SEG,
        test_loader: DataLoader,
        cfg: LGFFConfigSeg,
        geometry: GeometryToolkit,
        save_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("lgff.evaluator_seg")
        self.geometry = geometry
        self.warn_once = _WarnOnce()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader

        # save dir
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            out = getattr(cfg, "output_dir", None) or getattr(cfg, "log_dir", ".")
            self.save_dir = Path(out)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ===== Pose meters =====
        self.metrics_meter: Dict[str, List[float]] = {
            "add_s": [],
            "add": [],
            "t_err": [],
            "t_err_x": [],
            "t_err_y": [],
            "t_err_z": [],
            "rot_err": [],
            "cmd_acc": [],
        }

        # ===== Seg meters (NEW) =====
        self.seg_meter: Dict[str, List[float]] = {
            "seg_iou_full": [],
            "seg_dice_full": [],
            "seg_prec_full": [],
            "seg_rec_full": [],
            "seg_iou_valid": [],
            "seg_dice_valid": [],
            "seg_prec_valid": [],
            "seg_rec_valid": [],
        }

        # obj diameter
        self.obj_diameter: Optional[float] = getattr(cfg, "obj_diameter_m", None)
        if self.obj_diameter is None:
            self.obj_diameter = getattr(cfg, "obj_diameter", None)

        # thresholds
        self.acc_abs_thresholds_m: List[float] = getattr(
            cfg, "eval_abs_add_thresholds", [0.005, 0.01, 0.015, 0.02, 0.03]
        )
        self.acc_rel_thresholds_d: List[float] = getattr(
            cfg, "eval_rel_add_thresholds", [0.02, 0.05, 0.10]
        )
        self.cmd_acc_threshold_m: float = getattr(
            cfg, "cmd_threshold_m", getattr(cfg, "eval_cmd_threshold_m", 0.02)
        )

        # per-image records
        self.per_image_records: List[Dict[str, Any]] = []
        self.sample_counter: int = 0

        # ===== PnP =====
        self.use_pnp = bool(getattr(cfg, "eval_use_pnp", True))
        if self.use_pnp:
            if PnPSolver is None:
                self.logger.error("[EvaluatorSCSeg] PnPSolver missing -> fallback to regression.")
                self.use_pnp = False
                self.pnp_solver = None
            else:
                self.pnp_solver = PnPSolver(cfg)
                self.logger.info("[EvaluatorSCSeg] PnP Solver ENABLED.")
        else:
            self.pnp_solver = None
            self.logger.info("[EvaluatorSCSeg] PnP Solver DISABLED.")

        # ===== ICP =====
        self.icp_enable: bool = bool(getattr(cfg, "icp_enable", False))
        self.icp_iters: int = int(getattr(cfg, "icp_iters", 10))
        self.icp_max_corr_dist: float = float(getattr(cfg, "icp_max_corr_dist", 0.02))
        self.icp_trim_ratio: float = float(getattr(cfg, "icp_trim_ratio", 0.7))
        self.icp_sample_model: int = int(getattr(cfg, "icp_sample_model", 512))
        self.icp_sample_obs: int = int(getattr(cfg, "icp_sample_obs", 2048))
        self.icp_min_corr: int = int(getattr(cfg, "icp_min_corr", 50))

        self.icp_z_min: Optional[float] = getattr(cfg, "icp_z_min", None)
        self.icp_z_max: Optional[float] = getattr(cfg, "icp_z_max", None)
        self.icp_obs_mad_k: float = float(getattr(cfg, "icp_obs_mad_k", 0.0))

        self.icp_corr_schedule_m: Optional[List[float]] = getattr(cfg, "icp_corr_schedule_m", None)
        self.icp_iters_schedule: Optional[List[int]] = getattr(cfg, "icp_iters_schedule", None)

        # Robust seg->point valid mask
        self.min_valid_points = int(getattr(cfg, "pose_fusion_min_valid_points", 32))

        # If you want to log keys once
        self._printed_keys_once = False

    # ==========================================================
    # Main
    # ==========================================================
    def run(self) -> Dict[str, float]:
        self.model.eval()
        self.logger.info(f"Start Evaluation on {len(self.test_loader)} batches...")

        for k in self.metrics_meter:
            self.metrics_meter[k] = []
        for k in self.seg_meter:
            self.seg_meter[k] = []
        self.per_image_records = []
        self.sample_counter = 0
        self._printed_keys_once = False

        pnp_trigger_count = 0
        reg_trigger_count = 0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                B = int(batch["rgb"].shape[0])

                outputs = self.model(batch)

                if not self._printed_keys_once:
                    self._printed_keys_once = True
                    self.logger.info(f"[EvalSeg] batch keys  = {list(batch.keys())}")
                    self.logger.info(f"[EvalSeg] output keys = {list(outputs.keys())}")

                # ---- pose regression (with optional valid_mask for fusion) ----
                pred_rt_reg, pose_valid_mask = self._process_predictions(outputs, batch=batch)

                pred_rt_final = pred_rt_reg.clone()

                # ---- PnP for non-symmetric ----
                cls_ids = batch.get("cls_id", None)
                sym_mask = self._sym_mask_from_cls_ids(cls_ids, B, self.device)  # [B]
                nonsym_mask = ~sym_mask

                should_run_pnp = (
                    self.use_pnp
                    and nonsym_mask.any()
                    and ("pred_kp_ofs" in outputs)
                    and (outputs["pred_kp_ofs"] is not None)
                )

                if should_run_pnp:
                    model_kps = batch.get("kp3d_model", None)
                    if model_kps is not None:
                        pred_rt_pnp_raw = self.pnp_solver.solve_batch(
                            points=outputs["points"],
                            pred_kp_ofs=outputs["pred_kp_ofs"],
                            pred_conf=outputs.get("pred_conf", None),
                            model_kps=model_kps,
                        )

                        R_pnp = pred_rt_pnp_raw[:, :3, :3]
                        t_pnp = pred_rt_pnp_raw[:, :3, 3]

                        # Hybrid t: xy from pnp, z from reg
                        if "pred_trans" in outputs:
                            pred_c = outputs.get("pred_conf", None)
                            if pred_c is not None:
                                c = pred_c.squeeze(2)
                                w = c / (c.sum(dim=1, keepdim=True) + 1e-6)
                                t_reg = (outputs["pred_trans"] * w.unsqueeze(-1)).sum(dim=1)
                            else:
                                t_reg = outputs["pred_trans"].mean(dim=1)
                        else:
                            t_reg = pred_rt_reg[:, :3, 3]

                        t_hybrid = torch.stack([t_pnp[:, 0], t_pnp[:, 1], t_reg[:, 2]], dim=1)
                        pred_rt_pnp_hybrid = torch.cat([R_pnp, t_hybrid.unsqueeze(-1)], dim=2)

                        mask_bc = nonsym_mask.view(B, 1, 1).expand(-1, 3, 4)
                        pred_rt_final = torch.where(mask_bc, pred_rt_pnp_hybrid, pred_rt_final)

                        pnp_count = int(nonsym_mask.sum().item())
                        pnp_trigger_count += pnp_count
                        reg_trigger_count += (B - pnp_count)
                    else:
                        reg_trigger_count += B
                else:
                    reg_trigger_count += B

                # ---- ICP refine ----
                if self.icp_enable:
                    pred_rt_final = self._icp_refine_batch(pred_rt_final, batch)

                # ---- metrics ----
                gt_rt = self._process_gt(batch)

                # Pose metrics (ADD/ADD-S/...)
                batch_pose_metrics = self._compute_pose_metrics(pred_rt_final, gt_rt, batch)

                # Seg metrics (IoU/Dice/Prec/Rec) full + valid
                batch_seg_metrics = self._compute_seg_metrics(outputs, batch)

                # record meters
                self._accumulate_meters(batch_pose_metrics, batch_seg_metrics)

                # per-image CSV record
                self._record_per_image(batch, outputs, pred_rt_final, gt_rt, batch_pose_metrics, batch_seg_metrics)

        self.logger.info(
            f"Inference Stats: PnP used for {pnp_trigger_count} samples, Regression used for {reg_trigger_count} samples."
        )

        summary = self._summarize_metrics()
        self._dump_per_image_csv()
        return summary

    # ==========================================================
    # Prediction processing (pose fusion + optional seg->point valid mask)
    # ==========================================================
    def _process_predictions(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          pred_rt: [B,3,4]
          valid_mask_for_pose: Optional[bool tensor] [B,N] used in fusion (if enabled)
        """
        valid_mask: Optional[torch.Tensor] = None

        if bool(getattr(self.cfg, "pose_fusion_use_valid_mask", False)):
            mask_src = str(getattr(self.cfg, "pose_fusion_valid_mask_source", "")).lower().strip()

            if mask_src == "seg":
                valid_mask = self._compute_robust_valid_mask_from_seg(outputs, batch)
                if valid_mask is None:
                    vm = outputs.get("pred_valid_mask_bool", outputs.get("pred_valid_mask", None))
                    if isinstance(vm, torch.Tensor):
                        # vm could be [B,N,1] or [B,N]
                        if vm.dim() == 3 and vm.shape[-1] == 1:
                            valid_mask = vm.squeeze(-1).bool()
                        else:
                            valid_mask = vm.bool()

            elif mask_src == "labels":
                lbl = batch.get("labels", None)
                if isinstance(lbl, torch.Tensor):
                    valid_mask = (lbl > 0)
                    if valid_mask.dim() == 1:
                        valid_mask = valid_mask.unsqueeze(0).expand(outputs["pred_quat"].shape[0], -1)

        pred_rt = fuse_pose_from_outputs(outputs, self.geometry, self.cfg, stage="eval", valid_mask=valid_mask)
        return pred_rt, valid_mask

    def _process_gt(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["pose"].to(self.device)

    # ==========================================================
    # Robust seg->point mapping using choose
    # ==========================================================
    def _find_pred_mask_logits(self, outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Your model outputs 'pred_mask_logits' (confirmed by sanity check).
        Keep fallbacks for future variants.
        Expected shapes:
          - [B,1,H,W] (binary logits)
          - [B,2,H,W] (2-class logits)
        """
        for k in ["pred_mask_logits", "pred_seg", "seg_logits", "mask_logits", "pred_mask"]:
            v = outputs.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _compute_robust_valid_mask_from_seg(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Map 2D seg prob to per-point prob using choose indices [B,N].
        Then threshold with safety Top-K.
        """
        pred_logits = self._find_pred_mask_logits(outputs)
        if pred_logits is None:
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg] No seg logits found in outputs; cannot build pose valid mask.")
            return None

        choose = batch.get("choose", None)
        if not isinstance(choose, torch.Tensor):
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg] 'choose' missing; cannot map 2D seg to 3D points.")
            return None

        # logits -> probs
        if pred_logits.shape[1] == 1:
            probs = torch.sigmoid(pred_logits)  # [B,1,H,W]
        else:
            probs = torch.softmax(pred_logits, dim=1)[:, 1:2]  # [B,1,H,W]

        B, _, H, W = probs.shape
        N = choose.shape[1]
        HW = H * W

        # safety: check choose range once
        ch_max = int(choose.max().item())
        if ch_max >= HW:
            self.warn_once.warn(
                self.logger,
                f"[EvaluatorSCSeg] choose max={ch_max} exceeds H*W-1={HW-1}. Crop/resize mismatch likely. Fallback to None."
            )
            return None

        probs_flat = probs.view(B, -1)  # [B,HW]
        point_probs = torch.gather(probs_flat, 1, choose)  # [B,N]

        valid_mask = point_probs > 0.5
        counts = valid_mask.sum(dim=1)

        for b in range(B):
            if int(counts[b].item()) < self.min_valid_points:
                k = min(self.min_valid_points, N)
                _, top_idx = torch.topk(point_probs[b], k=k, largest=True)
                new_mask = torch.zeros_like(valid_mask[b])
                new_mask[top_idx] = True
                valid_mask[b] = new_mask

        return valid_mask

    # ==========================================================
    # Pose metrics
    # ==========================================================
    def _ensure_obj_diameter(self, model_points: torch.Tensor) -> None:
        if self.obj_diameter is not None:
            return
        if model_points.dim() == 2:
            model_points = model_points.unsqueeze(0)
        B, M, _ = model_points.shape
        if M <= 1:
            self.obj_diameter = 0.0
            return
        mp0 = model_points[0]
        dist_mat = torch.cdist(mp0.unsqueeze(0), mp0.unsqueeze(0)).squeeze(0)
        self.obj_diameter = float(dist_mat.max().item())
        self.logger.info(f"[EvaluatorSCSeg] Estimated obj_diameter from CAD: {self.obj_diameter:.6f} m")

    def _compute_add_and_adds_fallback(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        model_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        add = self.geometry.compute_add(pred_rt, gt_rt, model_points)
        adds = self.geometry.compute_adds(pred_rt, gt_rt, model_points)
        return add, adds

    def _compute_pose_metrics(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model_points = batch["model_points"].to(self.device)
        if model_points.dim() == 2:
            model_points = model_points.unsqueeze(0).expand(pred_rt.shape[0], -1, -1)

        self._ensure_obj_diameter(model_points)
        cls_ids = batch.get("cls_id", None)

        batch_metrics = compute_batch_pose_metrics(
            pred_rt=pred_rt,
            gt_rt=gt_rt,
            model_points=model_points,
            cls_ids=cls_ids,
            geometry=self.geometry,
            cfg=self.cfg,
        )

        if ("add" not in batch_metrics) or ("add_s" not in batch_metrics):
            add_fb, adds_fb = self._compute_add_and_adds_fallback(pred_rt, gt_rt, model_points)
            if "add" not in batch_metrics:
                batch_metrics["add"] = add_fb
            if "add_s" not in batch_metrics:
                batch_metrics["add_s"] = adds_fb

        for k in ["t_err", "t_err_x", "t_err_y", "t_err_z", "rot_err", "cmd_acc"]:
            if k not in batch_metrics:
                batch_metrics[k] = torch.zeros((pred_rt.shape[0],), dtype=torch.float32, device=pred_rt.device)

        return batch_metrics

    # ==========================================================
    # Seg metrics
    # ==========================================================
    def _find_gt_mask(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        for k in ["mask", "mask_visib", "mask_full", "gt_mask"]:
            v = batch.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _find_valid_mask_2d(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        v = batch.get("mask_valid", None)
        if isinstance(v, torch.Tensor):
            return v
        return None

    @staticmethod
    def _seg_binary_metrics_from_logits(
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        valid_mask_2d: Optional[torch.Tensor] = None,
        thr: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        pred_logits: [B,1,H,W] or [B,2,H,W]
        gt_mask:     [B,1,H,W] float/bool
        valid_mask_2d: [B,1,H,W] float/bool (where to evaluate)
        returns per-sample metrics tensors [B]
        """
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

        def compute_on(mask_eval: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if mask_eval is None:
                pe = pred
                ge = gt
            else:
                pe = pred & mask_eval
                ge = gt & mask_eval

            # Confusion components per sample
            tp = (pe & ge).flatten(1).sum(dim=1).float()
            fp = (pe & (~ge)).flatten(1).sum(dim=1).float()
            fn = ((~pe) & ge).flatten(1).sum(dim=1).float()

            # IoU = tp / (tp+fp+fn)
            iou = _safe_div(tp, tp + fp + fn)

            # Dice = 2tp / (2tp+fp+fn)
            dice = _safe_div(2 * tp, 2 * tp + fp + fn)

            # Precision = tp/(tp+fp)
            prec = _safe_div(tp, tp + fp)

            # Recall = tp/(tp+fn)
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

    def _compute_seg_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gt = self._find_gt_mask(batch)
        if gt is None:
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg] No GT mask found in batch; seg metrics disabled.")
            B = int(batch["rgb"].shape[0])
            z = torch.zeros((B,), dtype=torch.float32, device=self.device)
            return {k: z for k in self.seg_meter.keys()}

        pred_logits = self._find_pred_mask_logits(outputs)
        if pred_logits is None:
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg] No pred mask logits found in outputs; seg metrics disabled.")
            B = int(batch["rgb"].shape[0])
            z = torch.zeros((B,), dtype=torch.float32, device=self.device)
            return {k: z for k in self.seg_meter.keys()}

        valid_2d = self._find_valid_mask_2d(batch)

        # Ensure sizes match: pred is [B,*,H,W], gt is [B,1,H,W]
        # If mismatch exists, we try interpolate pred to gt size
        if gt.dim() == 4:
            Ht, Wt = int(gt.shape[2]), int(gt.shape[3])
        else:
            Ht, Wt = int(gt.shape[1]), int(gt.shape[2])

        Hp, Wp = int(pred_logits.shape[2]), int(pred_logits.shape[3])
        if (Hp != Ht) or (Wp != Wt):
            pred_logits = torch.nn.functional.interpolate(
                pred_logits, size=(Ht, Wt), mode="bilinear", align_corners=False
            )

        return self._seg_binary_metrics_from_logits(pred_logits, gt, valid_mask_2d=valid_2d, thr=0.5)

    # ==========================================================
    # Meters + Per-image records
    # ==========================================================
    def _accumulate_meters(self, pose_metrics: Dict[str, torch.Tensor], seg_metrics: Dict[str, torch.Tensor]) -> None:
        for name, t in pose_metrics.items():
            if isinstance(t, torch.Tensor) and t.dim() == 1:
                self.metrics_meter.setdefault(name, []).extend(t.detach().cpu().numpy().tolist())

        for name, t in seg_metrics.items():
            if isinstance(t, torch.Tensor) and t.dim() == 1:
                self.seg_meter.setdefault(name, []).extend(t.detach().cpu().numpy().tolist())

    def _record_per_image(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        pose_metrics: Dict[str, torch.Tensor],
        seg_metrics: Dict[str, torch.Tensor],
    ) -> None:
        B = int(pred_rt.shape[0])

        scene_ids = batch.get("scene_id", torch.full((B,), -1, device=self.device)).detach().cpu().numpy()
        im_ids = batch.get("im_id", torch.full((B,), -1, device=self.device)).detach().cpu().numpy()

        cls_ids = batch.get("cls_id", None)
        cls_id_arr = None
        if isinstance(cls_ids, torch.Tensor):
            cid = cls_ids.view(-1) if cls_ids.dim() > 1 else cls_ids
            cls_id_arr = cid.detach().cpu().numpy()
        else:
            cls_id_arr = np.full((B,), int(getattr(self.cfg, "obj_id", -1)), dtype=np.int64)

        sym_mask = self._sym_mask_from_cls_ids(cls_ids, B=B, device=self.device).detach().cpu().numpy().astype(bool)

        add_np = pose_metrics["add"].detach().cpu().numpy()
        adds_np = pose_metrics["add_s"].detach().cpu().numpy()
        t_err_np = pose_metrics["t_err"].detach().cpu().numpy()
        tdx_np = pose_metrics["t_err_x"].detach().cpu().numpy()
        tdy_np = pose_metrics["t_err_y"].detach().cpu().numpy()
        tdz_np = pose_metrics["t_err_z"].detach().cpu().numpy()
        rot_np = pose_metrics["rot_err"].detach().cpu().numpy()
        cmd_np = pose_metrics["cmd_acc"].detach().cpu().numpy()

        # seg arrays
        seg_arr = {k: seg_metrics[k].detach().cpu().numpy() for k in seg_metrics.keys()}

        # dist_for_cmd
        dist_for_cmd_np = np.where(sym_mask, adds_np, add_np)

        gt_t = gt_rt[:, :3, 3].detach().cpu().numpy()
        pred_t = pred_rt[:, :3, 3].detach().cpu().numpy()

        # threshold flags
        abs_flags_add: Dict[str, np.ndarray] = {}
        abs_flags_adds: Dict[str, np.ndarray] = {}
        for th in self.acc_abs_thresholds_m:
            mm = int(round(th * 1000))
            abs_flags_add[f"succ_add_{mm}mm"] = (add_np < th)
            abs_flags_adds[f"succ_adds_{mm}mm"] = (adds_np < th)

        rel_flags_add: Dict[str, np.ndarray] = {}
        rel_flags_adds: Dict[str, np.ndarray] = {}
        obj_d = float(self.obj_diameter) if (self.obj_diameter is not None) else 0.0
        if obj_d > 0:
            for alpha in self.acc_rel_thresholds_d:
                th = alpha * obj_d
                rel_flags_add[f"succ_add_{alpha:.3f}d"] = (add_np < th)
                rel_flags_adds[f"succ_adds_{alpha:.3f}d"] = (adds_np < th)

        for i in range(B):
            rec: Dict[str, Any] = {
                "index": int(self.sample_counter),
                "scene_id": int(scene_ids[i]),
                "im_id": int(im_ids[i]),
                "cls_id": int(cls_id_arr[i]),
                "is_symmetric": bool(sym_mask[i]),

                "add": float(add_np[i]),
                "add_s": float(adds_np[i]),
                "t_err": float(t_err_np[i]),
                "t_err_x": float(tdx_np[i]),
                "t_err_y": float(tdy_np[i]),
                "t_err_z": float(tdz_np[i]),
                "rot_err_deg": float(rot_np[i]),
                "dist_for_cmd": float(dist_for_cmd_np[i]),
                "cmd_success": bool(cmd_np[i]),

                "gt_tx": float(gt_t[i, 0]),
                "gt_ty": float(gt_t[i, 1]),
                "gt_tz": float(gt_t[i, 2]),
                "pred_tx": float(pred_t[i, 0]),
                "pred_ty": float(pred_t[i, 1]),
                "pred_tz": float(pred_t[i, 2]),
            }

            # seg metrics per image
            for k, arr in seg_arr.items():
                rec[k] = float(arr[i])

            # threshold flags
            for k, arr in abs_flags_add.items():
                rec[k] = bool(arr[i])
            for k, arr in abs_flags_adds.items():
                rec[k] = bool(arr[i])
            for k, arr in rel_flags_add.items():
                rec[k] = bool(arr[i])
            for k, arr in rel_flags_adds.items():
                rec[k] = bool(arr[i])

            self.per_image_records.append(rec)
            self.sample_counter += 1

    # ==========================================================
    # Summary + CSV
    # ==========================================================
    def _summarize_metrics(self) -> Dict[str, float]:
        add_s = np.asarray(self.metrics_meter.get("add_s", []), dtype=np.float64)
        add = np.asarray(self.metrics_meter.get("add", []), dtype=np.float64)
        t_err = np.asarray(self.metrics_meter.get("t_err", []), dtype=np.float64)
        tdx = np.asarray(self.metrics_meter.get("t_err_x", []), dtype=np.float64)
        tdy = np.asarray(self.metrics_meter.get("t_err_y", []), dtype=np.float64)
        tdz = np.asarray(self.metrics_meter.get("t_err_z", []), dtype=np.float64)
        rot = np.asarray(self.metrics_meter.get("rot_err", []), dtype=np.float64)
        cmd = np.asarray(self.metrics_meter.get("cmd_acc", []), dtype=np.float64)

        summary: Dict[str, float] = {}

        # Pose
        summary["mean_add_s"] = _safe_mean(add_s)
        summary["mean_add"] = _safe_mean(add)
        summary["mean_t_err"] = _safe_mean(t_err)
        summary["mean_t_err_x"] = _safe_mean(tdx)
        summary["mean_t_err_y"] = _safe_mean(tdy)
        summary["mean_t_err_z"] = _safe_mean(tdz)
        summary["mean_rot_err"] = _safe_mean(rot)
        summary["mean_cmd_acc"] = _safe_mean(cmd)

        summary["add_s_p50"] = _np_percentile(add_s, 50)
        summary["add_s_p75"] = _np_percentile(add_s, 75)
        summary["add_s_p90"] = _np_percentile(add_s, 90)
        summary["add_s_p95"] = _np_percentile(add_s, 95)

        summary["add_p50"] = _np_percentile(add, 50)
        summary["add_p75"] = _np_percentile(add, 75)
        summary["add_p90"] = _np_percentile(add, 90)
        summary["add_p95"] = _np_percentile(add, 95)

        summary["t_err_p50"] = _np_percentile(t_err, 50)
        summary["t_err_p75"] = _np_percentile(t_err, 75)
        summary["t_err_p90"] = _np_percentile(t_err, 90)
        summary["t_err_p95"] = _np_percentile(t_err, 95)

        summary["rot_err_deg_p50"] = _np_percentile(rot, 50)
        summary["rot_err_deg_p75"] = _np_percentile(rot, 75)
        summary["rot_err_deg_p90"] = _np_percentile(rot, 90)
        summary["rot_err_deg_p95"] = _np_percentile(rot, 95)

        for th in self.acc_abs_thresholds_m:
            mm = int(round(th * 1000))
            summary[f"acc_add<{mm}mm"] = float(np.mean(add < th)) if add.size else 0.0
            summary[f"acc_adds<{mm}mm"] = float(np.mean(add_s < th)) if add_s.size else 0.0

        obj_d = float(self.obj_diameter) if self.obj_diameter is not None else 0.0
        summary["obj_diameter"] = obj_d
        summary["cmd_threshold_m"] = float(self.cmd_acc_threshold_m)

        if obj_d > 0:
            for alpha in self.acc_rel_thresholds_d:
                th = alpha * obj_d
                summary[f"acc_add<{alpha:.3f}d"] = float(np.mean(add < th)) if add.size else 0.0
                summary[f"acc_adds<{alpha:.3f}d"] = float(np.mean(add_s < th)) if add_s.size else 0.0

        # Seg summary (NEW)
        for k in self.seg_meter.keys():
            arr = np.asarray(self.seg_meter.get(k, []), dtype=np.float64)
            summary[f"{k}_mean"] = _safe_mean(arr)
            summary[f"{k}_p50"] = _np_percentile(arr, 50)
            summary[f"{k}_p90"] = _np_percentile(arr, 90)

        self.logger.info(f"Evaluation Summary: {summary}")
        return summary

    def _dump_per_image_csv(self) -> None:
        if not self.per_image_records:
            self.logger.warning("[EvaluatorSCSeg] No per-image records to dump.")
            return
        csv_path = self.save_dir / "per_image_metrics_seg.csv"
        fieldnames = list(self.per_image_records[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.per_image_records)
        self.logger.info(f"[EvaluatorSCSeg] Per-image metrics saved to: {csv_path}")

    # ==========================================================
    # Sym mask
    # ==========================================================
    def _sym_mask_from_cls_ids(self, cls_ids: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        sym_ids = torch.as_tensor(getattr(self.cfg, "sym_class_ids", []), device=device)
        sym_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if isinstance(cls_ids, torch.Tensor) and sym_ids.numel() > 0:
            cid = cls_ids.view(-1) if cls_ids.dim() > 1 else cls_ids
            cid = cid.to(device)
            sym_mask = (cid.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)
        return sym_mask

    # ==========================================================
    # ICP (copied as robust implementation, not placeholder)
    # ==========================================================
    def _filter_obs_points(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.numel() == 0:
            return pts
        valid = torch.isfinite(pts).all(dim=1) & (pts.abs().sum(dim=1) > 1e-9)
        pts = pts[valid]
        if pts.shape[0] == 0:
            return pts

        if self.icp_z_min is not None:
            pts = pts[pts[:, 2] >= float(self.icp_z_min)]
        if self.icp_z_max is not None:
            pts = pts[pts[:, 2] <= float(self.icp_z_max)]
        if pts.shape[0] == 0:
            return pts

        if self.icp_obs_mad_k and self.icp_obs_mad_k > 0:
            center = pts.median(dim=0).values
            r = torch.norm(pts - center.unsqueeze(0), dim=1)
            med = r.median()
            mad = (r - med).abs().median().clamp_min(1e-6)
            keep = (r - med).abs() <= (self.icp_obs_mad_k * mad)
            pts = pts[keep]

        return pts

    @staticmethod
    def _kabsch_umeyama(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve R,t minimizing ||(R P + t) - Q|| (least squares)
        P,Q: [K,3]
        """
        mu_P = P.mean(dim=0)
        mu_Q = Q.mean(dim=0)
        X = P - mu_P
        Y = Q - mu_Q

        H = X.t() @ Y
        U, _, Vt = torch.linalg.svd(H, full_matrices=False)
        V = Vt.transpose(0, 1)

        R = V @ U.transpose(0, 1)
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.transpose(0, 1)

        t = mu_Q - (R @ mu_P)
        return R, t

    def _run_icp_one(
        self,
        rt_init: torch.Tensor,      # [3,4]
        obs_points: torch.Tensor,   # [N,3]
        model_points: torch.Tensor, # [M,3]
        iters: int,
        max_corr_dist: float,
        trim_ratio: float,
        sample_model: int,
        sample_obs: int,
        min_corr: int,
    ) -> torch.Tensor:
        device = rt_init.device
        dtype = rt_init.dtype

        obs = obs_points.to(device=device, dtype=dtype)
        mp = model_points.to(device=device, dtype=dtype)

        if obs.shape[0] < min_corr or mp.shape[0] < 3:
            return rt_init

        if obs.shape[0] > sample_obs:
            idx = torch.randperm(obs.shape[0], device=device)[:sample_obs]
            obs = obs[idx]

        if mp.shape[0] > sample_model:
            idx = torch.randperm(mp.shape[0], device=device)[:sample_model]
            mp = mp[idx]

        R = rt_init[:, :3].clone()
        t = rt_init[:, 3].clone()

        for _ in range(int(iters)):
            P = mp @ R.transpose(0, 1) + t.unsqueeze(0)  # [m,3]
            D = torch.cdist(P.unsqueeze(0), obs.unsqueeze(0), p=2).squeeze(0)  # [m,n]
            nn_dist, nn_idx = torch.min(D, dim=1)
            Q = obs[nn_idx]

            mask = nn_dist < float(max_corr_dist)
            if mask.sum().item() < min_corr:
                break

            P_sel = P[mask]
            Q_sel = Q[mask]
            d_sel = nn_dist[mask]

            if 0.0 < trim_ratio < 1.0 and P_sel.shape[0] > min_corr:
                k = max(min_corr, int(P_sel.shape[0] * trim_ratio))
                topk = torch.topk(d_sel, k=k, largest=False).indices
                P_sel = P_sel[topk]
                Q_sel = Q_sel[topk]

            dR, dt = self._kabsch_umeyama(P_sel, Q_sel)
            R = dR @ R
            t = dR @ t + dt

        return torch.cat([R, t.view(3, 1)], dim=1)

    def _icp_refine_batch(self, pred_rt: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        device = pred_rt.device
        B = int(pred_rt.shape[0])

        obs_points = batch.get("points", None)
        if obs_points is None:
            obs_points = batch.get("point_cloud", None)
        if not isinstance(obs_points, torch.Tensor):
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg][ICP] No obs points in batch; skip ICP.")
            return pred_rt

        if obs_points.dim() == 2:
            obs_points = obs_points.unsqueeze(0).expand(B, -1, -1)
        if obs_points.dim() != 3:
            self.warn_once.warn(self.logger, f"[EvaluatorSCSeg][ICP] obs_points dim={obs_points.dim()} invalid; skip.")
            return pred_rt
        obs_points = obs_points.to(device)

        labels = batch.get("labels", None)
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 1:
                labels = labels.unsqueeze(0).expand(B, -1)
            elif labels.dim() != 2:
                labels = None

        model_points_b = batch.get("model_points", None)
        if not isinstance(model_points_b, torch.Tensor):
            self.warn_once.warn(self.logger, "[EvaluatorSCSeg][ICP] No model_points in batch; skip ICP.")
            return pred_rt
        if model_points_b.dim() == 2:
            model_points_b = model_points_b.unsqueeze(0).expand(B, -1, -1)
        model_points_b = model_points_b.to(device)

        corr_schedule = self.icp_corr_schedule_m
        iters_schedule = self.icp_iters_schedule
        use_schedule = (
            isinstance(corr_schedule, (list, tuple)) and len(corr_schedule) > 0
            and isinstance(iters_schedule, (list, tuple)) and len(iters_schedule) == len(corr_schedule)
        )

        refined: List[torch.Tensor] = []
        for i in range(B):
            rt_i = pred_rt[i]

            pts_i = obs_points[i]
            if labels is not None and labels[i].numel() == pts_i.shape[0]:
                pts_i = pts_i[labels[i] > 0]

            pts_i = self._filter_obs_points(pts_i)
            if pts_i.shape[0] < self.icp_min_corr:
                refined.append(rt_i)
                continue

            mp_i = model_points_b[i].view(-1, 3)

            try:
                if use_schedule:
                    rt_ref = rt_i
                    for corr_d, it_n in zip(corr_schedule, iters_schedule):
                        rt_ref = self._run_icp_one(
                            rt_init=rt_ref,
                            obs_points=pts_i,
                            model_points=mp_i,
                            iters=int(it_n),
                            max_corr_dist=float(corr_d),
                            trim_ratio=float(self.icp_trim_ratio),
                            sample_model=int(self.icp_sample_model),
                            sample_obs=int(self.icp_sample_obs),
                            min_corr=int(self.icp_min_corr),
                        )
                    refined.append(rt_ref)
                else:
                    rt_ref = self._run_icp_one(
                        rt_init=rt_i,
                        obs_points=pts_i,
                        model_points=mp_i,
                        iters=int(self.icp_iters),
                        max_corr_dist=float(self.icp_max_corr_dist),
                        trim_ratio=float(self.icp_trim_ratio),
                        sample_model=int(self.icp_sample_model),
                        sample_obs=int(self.icp_sample_obs),
                        min_corr=int(self.icp_min_corr),
                    )
                    refined.append(rt_ref)
            except Exception as e:
                self.warn_once.warn(self.logger, f"[EvaluatorSCSeg][ICP] refine failed on sample {i}, fallback. err={e}")
                refined.append(rt_i)

        return torch.stack(refined, dim=0).to(device=device, dtype=pred_rt.dtype)


__all__ = ["EvaluatorSCSeg"]
