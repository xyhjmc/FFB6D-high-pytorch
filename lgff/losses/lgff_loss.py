"""
LGFF Loss Module (DenseFusion Style).
Implements ADD/ADD-S loss with Confidence Regularization for per-point pose estimation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit


class LGFFLoss(nn.Module):
    """
    LGFF 核心损失函数，包含：
    1. Pose Loss: ADD (非对称) 或 ADD-S (对称)
    2. Confidence Loss: 鼓励高置信度，惩罚低置信度
    3. Regularization: 防止置信度全部坍缩到 0
    """

    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry

        # 权重超参 (建议在 config 中可配)
        # w_rate: 置信度正则项的权重。
        # 经验值: 0.015 (DenseFusion default)
        self.w_rate = getattr(cfg, "w_rate", 0.015)

        # 对称物体的 Class ID 列表 (需要从 config 读取)
        self.sym_class_ids = set(getattr(cfg, "sym_class_ids", []))

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs:
                - pred_quat:  [B, N, 4]
                - pred_trans: [B, N, 3]
                - pred_conf:  [B, N, 1]
            batch:
                - pose:   [B, 3, 4] (Ground Truth [R|t] in camera coords)
                - points: [B, N, 3] (Input points in camera coords)
                - cls_id: [B] or [B,1] (optional, 用于判断对称性)

        Returns:
            total_loss: scalar tensor
            metrics: dict[str, float]
        """
        # 1. Unpack Predictions
        pred_q = outputs["pred_quat"]   # [B, N, 4]
        pred_t = outputs["pred_trans"]  # [B, N, 3]
        pred_c = outputs["pred_conf"]   # [B, N, 1]

        # 2. Unpack Ground Truth Pose
        gt_pose = batch["pose"]         # [B, 3, 4]
        gt_r = gt_pose[:, :3, :3]       # [B, 3, 3]
        gt_t = gt_pose[:, :3, 3]        # [B, 3]

        # 输入点云: camera 坐标系
        points_cam = batch["points"]    # [B, N, 3]
        B, N, _ = points_cam.shape

        # ------------------------------------------------------------
        # Step A: 恢复模型坐标系下的点 (Model Space Points)
        # ------------------------------------------------------------
        # 假设 points_cam ≈ R_gt * X_model + t_gt (加噪声后的观测)
        # 逆变换: X_model = R_gt^T * (points_cam - t_gt)
        gt_r_inv = gt_r.transpose(1, 2).unsqueeze(1)        # [B, 1, 3, 3]
        p_centered = points_cam - gt_t.unsqueeze(1)         # [B, N, 3]
        points_model = torch.matmul(
            p_centered.unsqueeze(2), gt_r_inv
        ).squeeze(2)                                        # [B, N, 3]

        # ------------------------------------------------------------
        # Step B: 预测姿态下的点云 (Predicted Points in Camera Space)
        # ------------------------------------------------------------
        # pred_q: [B,N,4] -> [B*N,4]
        pred_q_flat = pred_q.reshape(-1, 4)
        # quat -> R: [B*N,3,3]
        pred_r_flat = self.geometry.quat_to_rot(pred_q_flat)
        # -> [B,N,3,3]
        pred_r = pred_r_flat.view(B, N, 3, 3)

        # X_model: [B,N,3] -> [B,N,3,1]
        p_model_exp = points_model.unsqueeze(3)
        # R_pred * X_model: [B,N,3,3] @ [B,N,3,1] -> [B,N,3,1]
        p_rotated = torch.matmul(pred_r, p_model_exp).squeeze(3)
        # + t_pred: [B,N,3] + [B,N,3] -> [B,N,3]
        points_pred = p_rotated + pred_t

        # ------------------------------------------------------------
        # Step C: 计算距离损失 (ADD 或 ADD-S)
        # ------------------------------------------------------------
        points_target = points_cam  # 视为 "GT" 点云 (camera space)

        # 读取 cls_id（单类 loader 假设一个 batch 只含一种类）
        cls_id = None
        if "cls_id" in batch:
            cls = batch["cls_id"]  # [B] 或 [B,1]
            if cls.dim() == 2:     # [B,1] -> [B]
                cls = cls[:, 0]
            # 取第一个样本的类别 ID
            cls_id = int(cls[0].item())

        # 判定是否对称
        is_symmetric = (cls_id in self.sym_class_ids) if cls_id is not None else False

        if is_symmetric:
            # === 对称物体 Loss (ADD-S) ===
            # 对每个预测点，找 target 点集中最近的点
            # points_pred / points_target: [B,N,3]
            # dist_matrix: [B,N,N]
            dist_matrix = torch.cdist(points_pred, points_target, p=2)
            # 最近邻距离: [B,N]
            min_dist, _ = torch.min(dist_matrix, dim=2)
            loss_dist = min_dist  # [B,N]
        else:
            # === 非对称物体 Loss (ADD) ===
            # 一一对应: || (R_pred X_model + t_pred) - (R_gt X_model + t_gt) ||
            loss_dist = torch.norm(points_pred - points_target, dim=2, p=2)  # [B,N]

        # ------------------------------------------------------------
        # Step D: 结合置信度 (DenseFusion 风格)
        # ------------------------------------------------------------
        # pred_c: [B,N,1] -> [B,N]
        conf = pred_c.squeeze(2)

        # 数值稳健性: 避免 log(0) 或非常极端的梯度
        # 这里 clamp 只影响 loss 的计算，不改变网络真正输出值
        conf_clamped = conf.clamp(min=1e-4, max=1.0)

        # 1) 加权几何损失: conf * dist
        weighted_dist = loss_dist * conf_clamped  # [B,N]

        # 2) 置信度正则项: - w * log(conf)
        reg_term = self.w_rate * torch.log(conf_clamped)  # [B,N]

        # 3) 逐点总损失:
        #    L_i = conf_i * ||Δ_i|| - w * log(conf_i)
        pixel_loss = weighted_dist - reg_term  # [B,N]

        # 4) Batch Mean
        loss = torch.mean(pixel_loss)

        # ------------------------------------------------------------
        # Metrics (for logging only)
        # ------------------------------------------------------------
        with torch.no_grad():
            metrics = {
                "loss_total": loss.item(),
                "dist_mean": loss_dist.mean().item(),   # 平均几何误差
                "conf_mean": conf.mean().item(),        # 平均置信度 (未 clamp)
            }

        return loss, metrics


__all__ = ["LGFFLoss"]
