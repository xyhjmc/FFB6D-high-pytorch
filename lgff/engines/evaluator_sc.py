"""Evaluation pipeline for the single-class LGFF model.

The implementation borrows the dense-to-sparse pose selection logic from
FFB6D's evaluation utilities while adapting it to the lightweight LGFF
setting. It focuses on:

1) Confidence-aware pose fusion instead of picking a single best point,
   mirroring FFB6D's mean-shift style aggregation.
2) CAD-based ADD / ADD-S metrics using canonical model points (BOP
   meshes), consistent with the official FFB6D evaluation.
3) Reporting a simple <2cm accuracy metric for quick sanity checks.
4) [Extended] Rich metrics: accuracy vs. thresholds, percentiles, and
   per-image CSV logging for detailed analysis.
"""
from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    summarize_pose_metrics,
)


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

        # 结果保存路径（用于 per-image CSV）
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            # 如果 cfg 里有 output_dir / log_dir 可以优先用
            out = getattr(cfg, "output_dir", None) or getattr(cfg, "log_dir", ".")
            self.save_dir = Path(out)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 评估指标统计容器（逐样本）
        # 注意：键名要和 summarize_pose_metrics 预期的一致
        self.metrics_meter: Dict[str, List[float]] = {
            "add_s": [],
            "add": [],
            "t_err": [],      # ||t_pred - t_gt|| 的 L2
            "t_err_x": [],
            "t_err_y": [],
            "t_err_z": [],
            "rot_err": [],    # 旋转角误差 (deg)
            "cmd_acc": [],    # <2cm 成功标记（逐样本）
        }

        # BOP 风格：物体直径（优先从 cfg 读取，没有就后面从 CAD 点云算一次）
        # 统一使用单位：米 (m)
        self.obj_diameter: Optional[float] = getattr(cfg, "obj_diameter_m", None)
        if self.obj_diameter is None:
            # 向后兼容：如果旧 cfg 用的是 obj_diameter，也尝试读一下
            self.obj_diameter = getattr(cfg, "obj_diameter", None)

        # Accuracy vs threshold 设置
        # 绝对阈值（单位 m），例如 5mm/10mm/15mm/20mm/30mm
        self.acc_abs_adds_thresholds: List[float] = getattr(
            cfg,
            "eval_abs_adds_thresholds",
            [0.005, 0.01, 0.015, 0.02, 0.03],
        )
        # 相对直径阈值（BOP 风格），例如 0.02d / 0.05d / 0.10d
        self.acc_rel_adds_thresholds: List[float] = getattr(
            cfg,
            "eval_rel_adds_thresholds",
            [0.02, 0.05, 0.10],
        )

        # CMD accuracy 使用的阈值（默认 2cm）
        # 统一和 pose_metrics 中的 compute_batch_pose_metrics 使用同一个字段：
        # cfg.cmd_threshold_m，如未配置则回退 eval_cmd_threshold_m -> 0.02
        self.cmd_acc_threshold_m: float = getattr(
            cfg,
            "cmd_threshold_m",
            getattr(cfg, "eval_cmd_threshold_m", 0.02),
        )

        # 每张图的记录容器
        self.per_image_records: List[Dict[str, Any]] = []
        self.sample_counter: int = 0

    # ------------------------------------------------------------------ #
    # 主评估入口
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, float]:
        """
        主评估循环：遍历测试集 -> 推理 -> 后处理 -> 计算指标。
        """
        self.model.eval()
        self.logger.info(f"Start Evaluation on {len(self.test_loader)} batches...")

        # 重置统计
        for k in self.metrics_meter:
            self.metrics_meter[k] = []
        self.per_image_records = []
        self.sample_counter = 0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # 1. 数据搬运
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 2. 模型推理
                outputs = self.model(batch)

                # 3. 后处理: 置信度加权融合姿态（统一用 fuse_pose_from_outputs）
                pred_rt = self._process_predictions(outputs)
                gt_rt = self._process_gt(batch)

                # 4. 指标计算 + 每张图记录
                self._compute_metrics_external(pred_rt, gt_rt, batch)

        # 5. 汇总结果（所有 mean / percentile / acc 曲线）
        summary = self._summarize_metrics()

        # 6. 写出 per-image CSV
        self._dump_per_image_csv()

        return summary

    # ------------------------------------------------------------------ #
    # 姿态后处理
    # ------------------------------------------------------------------ #
    def _process_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        使用公共工具函数 fuse_pose_from_outputs 得到 [B,3,4] 的 [R|t]。
        配置项 eval_use_best_point / TopK 加权融合逻辑都在该函数内部处理。
        """
        return fuse_pose_from_outputs(outputs, self.geometry, self.cfg)

    # ------------------------------------------------------------------ #
    # GT 预处理
    # ------------------------------------------------------------------ #
    def _process_gt(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将 GT Pose 转为当前设备上的 tensor，保持形状 [B, 3, 4]。
        """
        return batch["pose"].to(self.device)

    # ------------------------------------------------------------------ #
    # 指标计算 + per-image 记录
    # ------------------------------------------------------------------ #
    def _compute_metrics_external(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        """
        使用 CAD 模型点作为参考，计算 ADD / ADD-S，并记录每张图的全部指标。
        正式的 ADD / ADD-S / t_err / rot_err / cmd_acc 计算委托给
        compute_batch_pose_metrics，确保和训练阶段复用同一套定义。
        """
        model_points = batch["model_points"].to(self.device)  # [B, M, 3]
        B, M, _ = model_points.shape

        # 若未显式提供直径，则用第一批的 CAD 点云估算一次（单位 m）
        if self.obj_diameter is None and M > 1:
            with torch.no_grad():
                mp0 = model_points[0]  # [M, 3]
                # 这里是 O(M^2)，单类 + 适量点数情况下是可以接受的
                dist_mat = torch.cdist(mp0.unsqueeze(0), mp0.unsqueeze(0)).squeeze(0)
                self.obj_diameter = float(dist_mat.max().item())
                self.logger.info(
                    f"[EvaluatorSC] Estimated obj_diameter from CAD: "
                    f"{self.obj_diameter:.6f} m"
                )

        # 调用公共 batch 级指标计算函数
        cls_ids = batch.get("cls_id", None)
        batch_metrics = compute_batch_pose_metrics(
            pred_rt=pred_rt,
            gt_rt=gt_rt,
            model_points=model_points,
            cls_ids=cls_ids,
            geometry=self.geometry,
            cfg=self.cfg,
        )

        # ---------------- 逐样本累加全局统计 ---------------- #
        # batch_metrics: key -> 1D tensor on CPU
        for name, tensor_1d in batch_metrics.items():
            if name not in self.metrics_meter:
                self.metrics_meter[name] = []
            self.metrics_meter[name].extend(tensor_1d.numpy().tolist())

        # ---------------- per-image 记录到内存 ---------------- #
        # 尽量带上 scene_id / im_id 等 dataset 信息（如果存在）
        scene_ids = None
        im_ids = None
        if "scene_id" in batch:
            scene_ids = batch["scene_id"].detach().cpu().numpy()
        if "im_id" in batch:
            im_ids = batch["im_id"].detach().cpu().numpy()

        # 取出 numpy 数组
        add_np   = batch_metrics["add"].numpy()
        adds_np  = batch_metrics["add_s"].numpy()
        t_err_np = batch_metrics["t_err"].numpy()
        tdx_np   = batch_metrics["t_err_x"].numpy()
        tdy_np   = batch_metrics["t_err_y"].numpy()
        tdz_np   = batch_metrics["t_err_z"].numpy()
        rot_np   = batch_metrics["rot_err"].numpy()
        succ_cmd_np = batch_metrics["cmd_acc"].numpy()  # 0 / 1

        # 再计算一次「用于 CMD 的距离」dist_for_cmd（与 compute_batch_pose_metrics 一致）
        cls_id_scalar: Optional[int] = None
        if isinstance(cls_ids, torch.Tensor):
            cid = cls_ids
            if cid.dim() > 1:
                cid = cid.view(-1)
            if cid.numel() > 0:
                cls_id_scalar = int(cid[0].item())

        is_symmetric = (
            cls_id_scalar in getattr(self.cfg, "sym_class_ids", [])
        ) if cls_id_scalar is not None else False

        dist_for_cmd_np = adds_np if is_symmetric else add_np  # [B]

        # GT / Pred t 也一起记录下来（方便分析偏差）
        gt_r = gt_rt[:, :3, :3]
        gt_t = gt_rt[:, :3, 3]
        pred_r = pred_rt[:, :3, :3]
        pred_t = pred_rt[:, :3, 3]

        gt_t_np   = gt_t.detach().cpu().numpy()
        pred_t_np = pred_t.detach().cpu().numpy()

        # 阈值下的成功标记 (ADD-S 为主)
        adds_arr = adds_np
        abs_th_success: Dict[str, List[bool]] = {}
        for th in self.acc_abs_adds_thresholds:
            key = f"succ_adds_{int(th * 1000)}mm"  # e.g., succ_adds_10mm
            abs_th_success[key] = (adds_arr < th).tolist()

        rel_th_success: Dict[str, List[bool]] = {}
        if self.obj_diameter is not None and self.obj_diameter > 0:
            for alpha in self.acc_rel_adds_thresholds:
                th = alpha * self.obj_diameter
                key = f"succ_adds_{alpha:.3f}d"
                rel_th_success[key] = (adds_arr < th).tolist()

        # 逐样本写入 per_image_records
        for i in range(B):
            record: Dict[str, Any] = {
                "index": int(self.sample_counter),
                "scene_id": int(scene_ids[i]) if scene_ids is not None else -1,
                "im_id": int(im_ids[i]) if im_ids is not None else -1,
                "cls_id": int(cls_id_scalar) if cls_id_scalar is not None else -1,
                "is_symmetric": bool(is_symmetric),
                "add": float(add_np[i]),
                "add_s": float(adds_np[i]),
                "t_err": float(t_err_np[i]),
                "t_err_x": float(tdx_np[i]),
                "t_err_y": float(tdy_np[i]),
                "t_err_z": float(tdz_np[i]),
                "rot_err_deg": float(rot_np[i]),
                "dist_for_cmd": float(dist_for_cmd_np[i]),
                "cmd_success": bool(succ_cmd_np[i]),
                "gt_tx": float(gt_t_np[i, 0]),
                "gt_ty": float(gt_t_np[i, 1]),
                "gt_tz": float(gt_t_np[i, 2]),
                "pred_tx": float(pred_t_np[i, 0]),
                "pred_ty": float(pred_t_np[i, 1]),
                "pred_tz": float(pred_t_np[i, 2]),
            }

            # 加上各个阈值下的成功标记
            for key, arr in abs_th_success.items():
                record[key] = bool(arr[i])
            for key, arr in rel_th_success.items():
                record[key] = bool(arr[i])

            self.per_image_records.append(record)
            self.sample_counter += 1

        # ---------------- 首批 debug 输出 ---------------- #
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print(">>> debug units")
            print("model_points norm (mean over batch):",
                  model_points.norm(dim=2).mean().item())
            print("gt_t norm (mean over batch):", gt_t.norm(dim=1).mean().item())
        if not hasattr(self, "_debug_printed_samples"):
            self._debug_printed_samples = True
            for i in range(min(5, pred_r.shape[0])):
                print(
                    f"[Sample {i}] add={add_np[i]:.4f}, "
                    f"adds={adds_np[i]:.4f}, "
                    f"t_err={t_err_np[i]:.4f}, "
                    f"rot_err={rot_np[i]:.2f}"
                )

    # ------------------------------------------------------------------ #
    # 指标汇总：均值 + percentile + accuracy 曲线
    # ------------------------------------------------------------------ #
    def _summarize_metrics(self) -> Dict[str, float]:
        """
        汇总所有样本的指标。
        这里直接调用 summarize_pose_metrics，保证和 Trainer 里（如果也调用它）
        完全一致的计算逻辑和字段名称。
        """
        obj_diam = self.obj_diameter if self.obj_diameter is not None else 0.0

        summary = summarize_pose_metrics(
            meter=self.metrics_meter,
            obj_diameter=obj_diam,
            cmd_threshold_m=self.cmd_acc_threshold_m,
        )

        self.logger.info(f"Evaluation Summary: {summary}")
        return summary

    # ------------------------------------------------------------------ #
    # per-image CSV 输出
    # ------------------------------------------------------------------ #
    def _dump_per_image_csv(self) -> None:
        """
        将每张图的所有指标和关键信息写入 CSV。
        """
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
