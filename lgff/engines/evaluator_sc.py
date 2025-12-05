"""Evaluation pipeline for the single-class LGFF model.

The implementation borrows the dense-to-sparse pose selection logic from
FFB6D's evaluation utilities while adapting it to the lightweight LGFF
setting. It focuses on:

1) Confidence-aware pose fusion instead of picking a single best point,
   mirroring FFB6D's mean-shift style aggregation.
2) CAD-based ADD / ADD-S metrics using canonical model points (BOP
   meshes), consistent with the official FFB6D evaluation.
3) Reporting a simple <2cm accuracy metric for quick sanity checks.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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

                # 3. 后处理: 置信度加权融合姿态 (FFB6D 风格)
                pred_rt = self._process_predictions(outputs)
                gt_rt = self._process_gt(batch)

                # 4. 指标计算
                self._compute_metrics_external(pred_rt, gt_rt, batch)

        # 5. 汇总结果
        summary = self._summarize_metrics()
        return summary

    def _process_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将逐点预测转换为单一姿态，使用置信度加权的四元数/平移融合。

        Returns:
            torch.Tensor: [B, 3, 4]，在当前 device 上
        """
        pred_q = outputs["pred_quat"]   # [B, N, 4]
        pred_t = outputs["pred_trans"]  # [B, N, 3]
        pred_c = outputs["pred_conf"]   # [B, N, 1]

        conf = pred_c.squeeze(-1)  # [B, N]

        # Top-K + 置信度归一化，避免极端异常点影响
        k = min(max(32, conf.size(1) // 4), conf.size(1))
        conf_topk, idx = torch.topk(conf, k=k, dim=1)
        conf_topk = conf_topk.clamp(min=1e-4)

        def _gather(t: torch.Tensor) -> torch.Tensor:
            expand_shape = idx.unsqueeze(-1).expand(-1, -1, t.size(-1))
            return torch.gather(t, dim=1, index=expand_shape)

        top_q = _gather(pred_q)  # [B, K, 4]
        top_t = _gather(pred_t)  # [B, K, 3]

        # 对齐四元数符号，随后用特征向量法进行加权平均
        top_q = F.normalize(top_q, dim=-1)
        ref_q = top_q[:, :1, :]
        sign = torch.sign(torch.sum(ref_q * top_q, dim=-1, keepdim=True)).clamp(min=0.0)
        sign[sign == 0] = 1.0  # 处理 dot=0 的边界情形
        top_q = top_q * sign

        weights = conf_topk / conf_topk.sum(dim=1, keepdim=True).clamp(min=1e-6)
        quat_cov = torch.einsum("bki,bkj->bij", weights.unsqueeze(-1) * top_q, top_q)
        eigvals, eigvecs = torch.linalg.eigh(quat_cov)
        fused_q = eigvecs[..., -1]

        fused_t = torch.sum(top_t * weights.unsqueeze(-1), dim=1)

        rot = self.geometry.quat_to_rot(fused_q)  # [B, 3, 3]
        pred_rt = torch.cat([rot, fused_t.unsqueeze(-1)], dim=2)  # [B, 3, 4]
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
        使用 CAD 模型点作为参考，计算 ADD / ADD-S。
        """
        model_points = batch["model_points"].to(self.device)  # [B, M, 3]

        gt_r = gt_rt[:, :3, :3]   # [B, 3, 3]
        gt_t = gt_rt[:, :3, 3]    # [B, 3]

        pred_r = pred_rt[:, :3, :3]  # [B, 3, 3]
        pred_t = pred_rt[:, :3, 3]   # [B, 3]

        # CAD 点云分别投影到预测/GT 姿态
        points_gt = torch.bmm(model_points, gt_r.transpose(1, 2)) + gt_t.unsqueeze(1)
        points_pred = torch.bmm(model_points, pred_r.transpose(1, 2)) + pred_t.unsqueeze(1)

        # ADD: 对应点距离的均值
        add_dist = torch.norm(points_pred - points_gt, dim=2).mean(dim=1)

        # ADD-S: 最近邻距离 (逐 batch 处理以避免显存过大)
        adds_list: List[torch.Tensor] = []
        for b in range(points_pred.size(0)):
            dist_mat = torch.cdist(points_pred[b], points_gt[b])
            adds_list.append(dist_mat.min(dim=1).values.mean())
        adds_dist = torch.stack(adds_list)

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
