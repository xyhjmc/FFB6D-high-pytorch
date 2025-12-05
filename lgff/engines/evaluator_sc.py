"""Evaluation pipeline for the single-class LGFF model.

The implementation borrows the dense-to-sparse pose selection logic from
FFB6D's evaluation utilities while adapting it to the lightweight LGFF
setting. It focuses on:

1) Converting dense per-point predictions into a single pose hypothesis
   using confidence scores.
2) Computing ADD / ADD-S metrics directly on the sampled input points,
   mirroring the training loss formulation.
3) Reporting a simple <2cm accuracy metric for quick sanity checks.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.models.lgff_sc import LGFF_SC


class EvaluatorSC:
    def __init__(
        self,
        model: LGFF_SC,
        test_loader: DataLoader,
        cfg: LGFFConfig,
        geometry: GeometryToolkit,
    ) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("lgff.evaluator")
        self.geometry = geometry

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader

        # 评估指标统计容器
        self.metrics_meter = {
            "add_s": [],   # 存储每个样本的 ADD-S 误差
            "add": [],     # 存储每个样本的 ADD 误差
            "cmd_acc": []  # < 2cm 准确率
        }

    def run(self) -> Dict[str, float]:
        """
        主评估循环：遍历测试集 -> 推理 -> 后处理 -> 计算指标。
        """
        self.model.eval()
        self.logger.info(f"Start Evaluation on {len(self.test_loader)} batches...")

        # 重置统计
        for k in self.metrics_meter:
            self.metrics_meter[k] = []

        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # 1. 数据搬运
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 2. 模型推理
                outputs = self.model(batch)

                # 3. 后处理: 选取置信度最高的姿态
                pred_rt = self._process_predictions(outputs)
                gt_rt = self._process_gt(batch)

                # 4. 指标计算
                self._compute_metrics_external(pred_rt, gt_rt, batch)

        # 5. 汇总结果
        summary = self._summarize_metrics()
        return summary

    def _process_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将逐点预测转换为单一姿态 (Best-Confidence 策略)。

        Returns:
            torch.Tensor: [B, 3, 4]，在当前 device 上
        """
        pred_q = outputs["pred_quat"]   # [B, N, 4]
        pred_t = outputs["pred_trans"]  # [B, N, 3]
        pred_c = outputs["pred_conf"]   # [B, N, 1]

        # 置信度压缩到 [B, N]
        conf = pred_c.squeeze(-1)

        # 选出每个样本中置信度最高的点索引
        best_idx = torch.argmax(conf, dim=1)  # [B]
        batch_idx = torch.arange(pred_q.size(0), device=pred_q.device)

        # 按索引 gather 对应的四元数和位移
        best_q = pred_q[batch_idx, best_idx]  # [B, 4]
        best_t = pred_t[batch_idx, best_idx]  # [B, 3]

        # 四元数 -> 旋转矩阵
        rot = self.geometry.quat_to_rot(best_q)  # [B, 3, 3]

        # 拼接 [R|t]
        pred_rt = torch.cat([rot, best_t.unsqueeze(-1)], dim=2)  # [B, 3, 4]
        return pred_rt

    def _process_gt(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将 GT Pose 转为当前设备上的 tensor，保持形状 [B, 3, 4]。
        """
        return batch["pose"].to(self.device)

    def _compute_metrics_external(
        self,
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> None:
        """
        使用输入点云作为参考，直接计算 ADD / ADD-S。

        逻辑与训练阶段的损失保持一致：
        1) 通过 GT 姿态将输入点云反变换回模型坐标系。
        2) 在预测姿态下重新投影到相机坐标系。
        3) 计算 ADD（对应点）和 ADD-S（最近邻）距离。
        """
        points = batch["points"].to(self.device)  # [B, N, 3]

        gt_r = gt_rt[:, :3, :3]   # [B, 3, 3]
        gt_t = gt_rt[:, :3, 3]    # [B, 3]

        # 将观测点云逆变换到模型坐标系: X_model = R_gt^T (p - t_gt)
        points_centered = points - gt_t.unsqueeze(1)
        gt_r_inv = gt_r.transpose(1, 2)
        points_model = torch.matmul(points_centered, gt_r_inv)  # [B, N, 3]

        pred_r = pred_rt[:, :3, :3]  # [B, 3, 3]
        pred_t = pred_rt[:, :3, 3]   # [B, 3]

        # 预测姿态下的点云
        points_pred = torch.matmul(points_model, pred_r.transpose(1, 2))
        points_pred = points_pred + pred_t.unsqueeze(1)

        # Ground truth 点云（原始观测）
        points_gt = points

        # ADD: 对应点之间的平均距离
        add_dist = torch.norm(points_pred - points_gt, dim=2).mean(dim=1)

        # ADD-S: 最近邻距离
        adds_nn = torch.cdist(points_pred, points_gt).min(dim=2).values
        adds_dist = adds_nn.mean(dim=1)

        # 对称性判断（单类场景，直接读取 batch 中的 cls_id）
        cls_id: Optional[int] = None
        if "cls_id" in batch:
            cls_tensor = batch["cls_id"]
            if cls_tensor.dim() > 1:
                cls_tensor = cls_tensor.view(-1)
            if cls_tensor.numel() > 0:
                cls_id = int(cls_tensor[0].item())

        is_symmetric = (
            cls_id in getattr(self.cfg, "sym_class_ids", [])
        ) if cls_id is not None else False

        # 选择用于 <2cm 精度统计的距离（对称物体使用 ADD-S）
        dist_for_acc = adds_dist if is_symmetric else add_dist

        self.metrics_meter["add"].extend(add_dist.detach().cpu().numpy().tolist())
        self.metrics_meter["add_s"].extend(adds_dist.detach().cpu().numpy().tolist())

        cmd_acc = (dist_for_acc < 0.02).float().mean().item()
        self.metrics_meter["cmd_acc"].append(cmd_acc)

    def _summarize_metrics(self) -> Dict[str, float]:
        """
        汇总所有样本的平均指标
        """
        summary = {}
        for k, v_list in self.metrics_meter.items():
            if len(v_list) > 0:
                summary[f"mean_{k}"] = float(np.mean(v_list))
            else:
                summary[f"mean_{k}"] = 0.0

        self.logger.info(f"Evaluation Summary: {summary}")
        return summary


__all__ = ["EvaluatorSC"]
