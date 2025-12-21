"""
Evaluation pipeline for the single-class LGFF model.

Key additions in this version:
- Ensure BOTH ADD and ADD-S are computed and reported.
- Provide accuracy-vs-threshold curves for BOTH add and add_s.
- Per-image CSV includes succ_add_* flags analogous to succ_adds_*.
- Robust fallback if compute_batch_pose_metrics misses 'add'/'add_s'.
- [ICP] Configurable, robust, "refine only bad" by default, bad-metric can be ADD.
"""

from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc import LGFF_SC
from lgff.utils.pose_metrics import (
    fuse_pose_from_outputs,
    compute_batch_pose_metrics,
)


def _np_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x.mean())


class EvaluatorSC:
    def __init__(
        self,
        model: LGFF_SC,
        test_loader: DataLoader,
        cfg: LGFFConfig,
        geometry: GeometryToolkit,
        save_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("lgff.evaluator")
        self.geometry = geometry

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

        # meter
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

        # obj diameter (m)
        self.obj_diameter: Optional[float] = getattr(cfg, "obj_diameter_m", None)
        if self.obj_diameter is None:
            self.obj_diameter = getattr(cfg, "obj_diameter", None)

        # thresholds (absolute in meters)
        self.acc_abs_thresholds_m: List[float] = getattr(
            cfg, "eval_abs_add_thresholds", [0.005, 0.01, 0.015, 0.02, 0.03]
        )
        # relative thresholds (multipliers of diameter)
        self.acc_rel_thresholds_d: List[float] = getattr(
            cfg, "eval_rel_add_thresholds", [0.02, 0.05, 0.10]
        )

        # CMD threshold
        self.cmd_acc_threshold_m: float = getattr(
            cfg, "cmd_threshold_m", getattr(cfg, "eval_cmd_threshold_m", 0.02)
        )

        # per-image
        self.per_image_records: List[Dict[str, Any]] = []
        self.sample_counter: int = 0

        # ===== ICP config (all via cfg) =====
        self.icp_enable: bool = bool(getattr(cfg, "icp_enable", False))
        # strongly recommend True based on your experiments
        self.icp_only_when_bad: bool = bool(getattr(cfg, "icp_only_when_bad", False))
        # 'add' | 'adds' | 'dist_for_cmd'
        self.icp_bad_metric: str = str(getattr(cfg, "icp_bad_metric", "add")).lower()

        # threshold: use abs(m) if provided; else use rel(d) if provided; else fallback to cmd threshold
        self.icp_bad_abs_th_m: Optional[float] = getattr(cfg, "icp_bad_abs_th_m", None)
        self.icp_bad_rel_th_d: Optional[float] = getattr(cfg, "icp_bad_rel_th_d", None)

        # ICP main params
        self.icp_iters: int = int(getattr(cfg, "icp_iters", 10))
        self.icp_max_corr_dist: float = float(getattr(cfg, "icp_max_corr_dist", 0.02))
        self.icp_trim_ratio: float = float(getattr(cfg, "icp_trim_ratio", 0.7))
        self.icp_sample_model: int = int(getattr(cfg, "icp_sample_model", 512))
        self.icp_sample_obs: int = int(getattr(cfg, "icp_sample_obs", 2048))
        self.icp_min_corr: int = int(getattr(cfg, "icp_min_corr", 50))

        # Optional point filtering
        self.icp_z_min: Optional[float] = getattr(cfg, "icp_z_min", None)  # e.g. 0.1
        self.icp_z_max: Optional[float] = getattr(cfg, "icp_z_max", None)  # e.g. 5.0
        # Robust outlier removal by MAD (None/0 disables). Suggested 3.0~4.0
        self.icp_obs_mad_k: float = float(getattr(cfg, "icp_obs_mad_k", 0.0))

        # Optional multi-stage schedule (corr dist + iters)
        self.icp_corr_schedule_m: Optional[List[float]] = getattr(cfg, "icp_corr_schedule_m", None)
        self.icp_iters_schedule: Optional[List[int]] = getattr(cfg, "icp_iters_schedule", None)

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, float]:
        self.model.eval()
        self.logger.info(f"Start Evaluation on {len(self.test_loader)} batches...")

        for k in self.metrics_meter:
            self.metrics_meter[k] = []
        self.per_image_records = []
        self.sample_counter = 0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(batch)
                pred_rt = self._process_predictions(outputs)

                if self.icp_enable:
                    pred_rt = self._icp_refine_batch(pred_rt, batch)

                gt_rt = self._process_gt(batch)
                self._compute_metrics(pred_rt, gt_rt, batch)

        summary = self._summarize_metrics()
        self._dump_per_image_csv()
        return summary

    # ------------------------------------------------------------------ #
    def _process_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return fuse_pose_from_outputs(outputs, self.geometry, self.cfg, stage="eval")

    def _process_gt(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["pose"].to(self.device)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _ensure_obj_diameter(self, model_points: torch.Tensor) -> None:
        if self.obj_diameter is not None:
            return
        B, M, _ = model_points.shape
        if M <= 1:
            self.obj_diameter = 0.0
            return
        mp0 = model_points[0]
        dist_mat = torch.cdist(mp0.unsqueeze(0), mp0.unsqueeze(0)).squeeze(0)
        self.obj_diameter = float(dist_mat.max().item())
        self.logger.info(f"[EvaluatorSC] Estimated obj_diameter from CAD: {self.obj_diameter:.6f} m")

    def _sym_mask_from_cls_ids(self, cls_ids: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        sym_ids = torch.as_tensor(getattr(self.cfg, "sym_class_ids", []), device=device)
        sym_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if isinstance(cls_ids, torch.Tensor) and sym_ids.numel() > 0:
            cid = cls_ids
            if cid.dim() > 1:
                cid = cid.view(-1)
            cid = cid.to(device)
            sym_mask = (cid.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)
        return sym_mask

    def _get_bad_threshold_m(self) -> float:
        # priority: abs_th_m > rel_th_d > cmd_threshold_m
        if self.icp_bad_abs_th_m is not None:
            return float(self.icp_bad_abs_th_m)
        if self.icp_bad_rel_th_d is not None:
            d = float(self.obj_diameter or 0.0)
            if d > 0:
                return float(self.icp_bad_rel_th_d) * d
        return float(self.cmd_acc_threshold_m)

    def _filter_obs_points(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: [N,3] in camera frame
        if pts.numel() == 0:
            return pts

        valid = torch.isfinite(pts).all(dim=1) & (pts.abs().sum(dim=1) > 1e-9)
        pts = pts[valid]
        if pts.shape[0] == 0:
            return pts

        # Z-range filter
        if self.icp_z_min is not None:
            pts = pts[pts[:, 2] >= float(self.icp_z_min)]
        if self.icp_z_max is not None:
            pts = pts[pts[:, 2] <= float(self.icp_z_max)]
        if pts.shape[0] == 0:
            return pts

        # MAD outlier removal (very light & effective for ROI point clouds)
        if self.icp_obs_mad_k and self.icp_obs_mad_k > 0:
            center = pts.median(dim=0).values
            r = torch.norm(pts - center.unsqueeze(0), dim=1)  # [N]
            med = r.median()
            mad = (r - med).abs().median().clamp_min(1e-6)
            keep = (r - med).abs() <= (self.icp_obs_mad_k * mad)
            pts = pts[keep]

        return pts

    # ------------------------------------------------------------------ #
    # ADD / ADD-S fallback (GeometryToolkit)
    # ------------------------------------------------------------------ #
    def _compute_add_and_adds_fallback(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        model_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        add = self.geometry.compute_add(pred_rt, gt_rt, model_points)     # [B]
        adds = self.geometry.compute_adds(pred_rt, gt_rt, model_points)   # [B]
        return add, adds

    # ------------------------------------------------------------------ #
    # ICP core: Kabsch/Umeyama
    # ------------------------------------------------------------------ #
    @staticmethod
    def _kabsch_umeyama(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert P.ndim == 2 and Q.ndim == 2 and P.shape == Q.shape and P.shape[1] == 3

        mu_P = P.mean(dim=0)
        mu_Q = Q.mean(dim=0)
        X = P - mu_P
        Y = Q - mu_Q

        H = X.t() @ Y
        U, S, Vt = torch.linalg.svd(H, full_matrices=False)
        V = Vt.transpose(0, 1)

        R = V @ U.transpose(0, 1)
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.transpose(0, 1)

        t = mu_Q - (R @ mu_P)
        return R, t

    def _run_icp_one(
        self,
        rt_init: torch.Tensor,          # [3,4] model->cam
        obs_points: torch.Tensor,       # [N,3] cam
        model_points: torch.Tensor,     # [M,3] model
        iters: int,
        max_corr_dist: float,
        trim_ratio: float,
        sample_model: int,
        sample_obs: int,
        min_corr: int,
    ) -> torch.Tensor:
        device = rt_init.device
        dtype = rt_init.dtype

        if obs_points is None or model_points is None:
            return rt_init
        if obs_points.ndim != 2 or obs_points.shape[1] != 3:
            return rt_init
        if model_points.ndim != 2 or model_points.shape[1] != 3:
            return rt_init

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

        prev_mean = None

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
                d_sel = d_sel[topk]

            mean_d = float(d_sel.mean().item())
            if prev_mean is not None and abs(prev_mean - mean_d) < 1e-5:
                # converged
                pass
            prev_mean = mean_d

            dR, dt = self._kabsch_umeyama(P_sel, Q_sel)
            R = dR @ R
            t = dR @ t + dt

        return torch.cat([R, t.view(3, 1)], dim=1)

    # ------------------------------------------------------------------ #
    # ICP refinement batch with "refine only bad" gating (BAD metric = ADD by default)
    # ------------------------------------------------------------------ #
    def _icp_refine_batch(self, pred_rt: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        # if not bool(getattr(self.cfg, "icp_enable", False)):
        #     return pred_rt

        device = pred_rt.device
        B = pred_rt.shape[0]

        # obs points
        obs_points = batch.get("points", None)
        if obs_points is None:
            obs_points = batch.get("point_cloud", None)
        if obs_points is None or not isinstance(obs_points, torch.Tensor):
            self.logger.warning("[EvaluatorSC][ICP] No obs points in batch, skip ICP.")
            return pred_rt

        # normalize to [B,N,3]
        if obs_points.dim() == 2:
            obs_points = obs_points.unsqueeze(0).expand(B, -1, -1)
        if obs_points.dim() != 3:
            self.logger.warning(f"[EvaluatorSC][ICP] obs_points dim={obs_points.dim()} invalid, skip ICP.")
            return pred_rt
        obs_points = obs_points.to(device)

        # optional labels (foreground mask)
        labels = batch.get("labels", None)
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 1:
                labels = labels.unsqueeze(0).expand(B, -1)
            elif labels.dim() == 2:
                pass
            else:
                labels = None

        # model_points [B,M,3]
        if "model_points" not in batch:
            self.logger.warning("[EvaluatorSC][ICP] No model_points in batch, skip ICP.")
            return pred_rt
        model_points_b = batch["model_points"]
        if not isinstance(model_points_b, torch.Tensor):
            return pred_rt
        if model_points_b.dim() == 2:
            model_points_b = model_points_b.unsqueeze(0).expand(B, -1, -1)
        model_points_b = model_points_b.to(device)

        # ensure diameter for rel threshold
        self._ensure_obj_diameter(model_points_b)

        # compute coarse add/adds for gating (fast, pure geometry)
        gt_rt = batch.get("pose", None)
        if not isinstance(gt_rt, torch.Tensor):
            # if no GT available, cannot gate by error -> fallback: no ICP (safer)
            self.logger.warning("[EvaluatorSC][ICP] No GT pose in batch, skip ICP gating -> skip ICP.")
            return pred_rt
        gt_rt = gt_rt.to(device)

        add_coarse = self.geometry.compute_add(pred_rt, gt_rt, model_points_b)    # [B]
        adds_coarse = self.geometry.compute_adds(pred_rt, gt_rt, model_points_b) # [B]

        cls_ids = batch.get("cls_id", None)
        sym_mask = self._sym_mask_from_cls_ids(cls_ids, B=B, device=device)  # [B]

        dist_for_cmd = torch.where(sym_mask, adds_coarse, add_coarse)  # [B]

        # choose gating metric
        if self.icp_bad_metric == "adds":
            metric = adds_coarse
        elif self.icp_bad_metric == "dist_for_cmd":
            metric = dist_for_cmd
        else:
            metric = add_coarse  # default 'add'

        bad_th = self._get_bad_threshold_m()

        # ICP schedule
        corr_schedule = self.icp_corr_schedule_m
        iters_schedule = self.icp_iters_schedule
        use_schedule = (
            isinstance(corr_schedule, (list, tuple)) and len(corr_schedule) > 0
            and isinstance(iters_schedule, (list, tuple)) and len(iters_schedule) == len(corr_schedule)
        )

        refined_list: List[torch.Tensor] = []

        for i in range(B):
            rt_i = pred_rt[i]

            # gating
            if self.icp_only_when_bad:
                if metric.dim() == 0:
                    m_i = float(metric.detach().cpu().item())
                else:
                    m_i = float(metric[i].detach().cpu().item())
                if m_i <= bad_th:
                    refined_list.append(rt_i)
                    continue

            pts_i = obs_points[i]  # [N,3]
            if labels is not None:
                lab_i = labels[i]
                if lab_i.numel() == pts_i.shape[0]:
                    pts_i = pts_i[lab_i > 0]

            pts_i = self._filter_obs_points(pts_i)
            if pts_i.shape[0] < self.icp_min_corr:
                refined_list.append(rt_i)
                continue

            mp_i = model_points_b[i]
            if mp_i.dim() != 2:
                mp_i = mp_i.view(-1, 3)

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
                    refined_list.append(rt_ref)
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
                    refined_list.append(rt_ref)
            except Exception as e:
                self.logger.warning(f"[EvaluatorSC][ICP] sample {i} refine failed, fallback. err={e}")
                refined_list.append(rt_i)

        if len(refined_list) != B:
            self.logger.warning(f"[EvaluatorSC][ICP] refined_list size {len(refined_list)} != B {B}. fallback.")
            return pred_rt

        return torch.stack(refined_list, dim=0).to(device=device, dtype=pred_rt.dtype)

    # ------------------------------------------------------------------ #
    def _compute_metrics(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        model_points = batch["model_points"].to(self.device)  # [B,M,3]
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

        # ensure 'add' and 'add_s'
        if ("add" not in batch_metrics) or ("add_s" not in batch_metrics):
            add_fb, adds_fb = self._compute_add_and_adds_fallback(
                pred_rt=pred_rt, gt_rt=gt_rt, model_points=model_points
            )
            if "add" not in batch_metrics:
                batch_metrics["add"] = add_fb
            if "add_s" not in batch_metrics:
                batch_metrics["add_s"] = adds_fb

        # ensure other keys exist
        for k in ["t_err", "t_err_x", "t_err_y", "t_err_z", "rot_err", "cmd_acc"]:
            if k not in batch_metrics:
                batch_metrics[k] = torch.zeros((pred_rt.shape[0],), dtype=torch.float32, device=pred_rt.device)

        # ---------------- global meter (SAFE: to cpu) ----------------
        for name, tensor_1d in batch_metrics.items():
            if not isinstance(tensor_1d, torch.Tensor):
                continue
            arr = tensor_1d.detach().cpu().numpy()
            if name not in self.metrics_meter:
                self.metrics_meter[name] = []
            self.metrics_meter[name].extend(arr.tolist())

        # ---------------- per-image records ----------------
        B = pred_rt.shape[0]
        scene_ids = batch["scene_id"].detach().cpu().numpy() if "scene_id" in batch else None
        im_ids = batch["im_id"].detach().cpu().numpy() if "im_id" in batch else None

        cls_id_arr = None
        if isinstance(cls_ids, torch.Tensor):
            cid = cls_ids
            if cid.dim() > 1:
                cid = cid.view(-1)
            cls_id_arr = cid.detach().cpu().numpy()
        elif cls_ids is not None:
            cls_id_arr = np.asarray(cls_ids)

        add_np = batch_metrics["add"].detach().cpu().numpy()
        adds_np = batch_metrics["add_s"].detach().cpu().numpy()
        t_err_np = batch_metrics["t_err"].detach().cpu().numpy()
        tdx_np = batch_metrics["t_err_x"].detach().cpu().numpy()
        tdy_np = batch_metrics["t_err_y"].detach().cpu().numpy()
        tdz_np = batch_metrics["t_err_z"].detach().cpu().numpy()
        rot_np = batch_metrics["rot_err"].detach().cpu().numpy()
        cmd_np = batch_metrics["cmd_acc"].detach().cpu().numpy()

        sym_mask = self._sym_mask_from_cls_ids(cls_ids, B=B, device=pred_rt.device).detach().cpu().numpy().astype(bool)
        dist_for_cmd_np = np.where(sym_mask, adds_np, add_np)

        gt_t = gt_rt[:, :3, 3].detach().cpu().numpy()
        pred_t = pred_rt[:, :3, 3].detach().cpu().numpy()

        # success flags for BOTH add and adds
        abs_flags_add: Dict[str, List[bool]] = {}
        abs_flags_adds: Dict[str, List[bool]] = {}
        for th in self.acc_abs_thresholds_m:
            mm = int(round(th * 1000))
            abs_flags_add[f"succ_add_{mm}mm"] = (add_np < th).tolist()
            abs_flags_adds[f"succ_adds_{mm}mm"] = (adds_np < th).tolist()

        rel_flags_add: Dict[str, List[bool]] = {}
        rel_flags_adds: Dict[str, List[bool]] = {}
        if self.obj_diameter is not None and self.obj_diameter > 0:
            for alpha in self.acc_rel_thresholds_d:
                th = alpha * float(self.obj_diameter)
                rel_flags_add[f"succ_add_{alpha:.3f}d"] = (add_np < th).tolist()
                rel_flags_adds[f"succ_adds_{alpha:.3f}d"] = (adds_np < th).tolist()

        for i in range(B):
            rec: Dict[str, Any] = {
                "index": int(self.sample_counter),
                "scene_id": int(scene_ids[i]) if scene_ids is not None else -1,
                "im_id": int(im_ids[i]) if im_ids is not None else -1,
                "cls_id": int(cls_id_arr[i]) if cls_id_arr is not None else int(getattr(self.cfg, "obj_id", -1)),
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

    # ------------------------------------------------------------------ #
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

            if 0.10 in self.acc_rel_thresholds_d:
                summary["acc_add_0.1d"] = summary.get("acc_add<0.100d", 0.0)
                summary["acc_adds_0.1d"] = summary.get("acc_adds<0.100d", 0.0)

        self.logger.info(f"Evaluation Summary: {summary}")
        return summary

    # ------------------------------------------------------------------ #
    def _dump_per_image_csv(self) -> None:
        if not self.per_image_records:
            self.logger.warning("[EvaluatorSC] No per-image records to dump.")
            return

        csv_path = self.save_dir / "per_image_metrics.csv"
        fieldnames = list(self.per_image_records[0].keys())

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.per_image_records)

        self.logger.info(f"[EvaluatorSC] Per-image metrics saved to: {csv_path}")


__all__ = ["EvaluatorSC"]
