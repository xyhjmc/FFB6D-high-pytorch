"""
LGFF Loss Module (Optimized, no explicit rotation loss).

1. Translation Loss (Weighted L1) -> Fix Z-axis bias.
2. DenseFusion-style ADD/ADD-S Loss on ROI points.
3. Explicit Confidence Regularization -> Prevent confidence collapse.
4. Optional CAD-level ADD/ADD-S Loss (using canonical model points),
   aligned with the Evaluator.
5. [NEW] Optional Keypoint Offset Loss (FFB6D-style), using pred_kp_ofs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit


class LGFFLoss(nn.Module):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry

        # DenseFusion 部分
        self.w_rate = getattr(cfg, "w_rate", 0.015)
        self.sym_class_ids = set(getattr(cfg, "sym_class_ids", []))

        # ===== 权重配置（全部暴露到 cfg，方便调参） =====
        self.lambda_t       = getattr(cfg, "lambda_t", 0.5)        # 平移权重
        self.lambda_add     = getattr(cfg, "lambda_add", 1.0)      # ROI 几何权重
        self.lambda_conf    = getattr(cfg, "lambda_conf", 0.1)     # conf 显式正则权重
        self.lambda_add_cad = getattr(cfg, "lambda_add_cad", 0.0)  # CAD 级 ADD/ADD-S loss 权重
        # [NEW] 关键点 offset 辅助 loss 权重（FFB6D 风格）
        self.lambda_kp_of   = getattr(cfg, "lambda_kp_of", 0.6)

        # Z 轴额外权重（解决深度轴 bias）
        self.t_z_weight = getattr(cfg, "t_z_weight", 2.0)
        axis_weight = torch.tensor([1.0, 1.0, self.t_z_weight], dtype=torch.float32)
        self.register_buffer("axis_weight", axis_weight)

        # Confidence 正则参数
        self.conf_alpha    = getattr(cfg, "conf_alpha", 10.0)      # 对误差的敏感度
        self.conf_dist_max = getattr(cfg, "conf_dist_max", 0.1)    # 误差裁剪上限

    # ------------------------------------------------------------------
    # [NEW] Keypoint Offset Loss (FFB6D-style of_l1_loss)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_kp_offset_loss(
        pred_kp_ofs: torch.Tensor,
        kp_targ_ofst: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_kp_ofs:  [B, K, N, 3]
        kp_targ_ofst: [B, N, K, 3] 或 [B, K, N, 3]
        labels:       [B, N] 或 [B, N, 1]，>0 表示该点属于物体
        """
        B, K, N, C = pred_kp_ofs.shape

        # 处理 labels -> [B, N]
        if labels.dim() == 3:
            labels = labels.squeeze(-1)
        assert labels.shape == (B, N), f"labels shape expected [B, N], got {labels.shape}"

        # 对齐 gt offset 形状到 [B, K, N, 3]
        if kp_targ_ofst.shape == (B, N, K, C):
            kp_targ = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()  # [B, K, N, 3]
        elif kp_targ_ofst.shape == (B, K, N, C):
            kp_targ = kp_targ_ofst
        else:
            raise ValueError(
                f"kp_targ_ofst shape mismatch: expected [B, N, K, 3] or [B, K, N, 3], "
                f"got {kp_targ_ofst.shape}"
            )

        # 权重 mask: 只对前景点计算 offset
        w = (labels > 1e-8).float()          # [B, N]
        w = w.view(B, 1, N, 1).expand(-1, K, -1, C)  # [B, K, N, 3]

        diff = pred_kp_ofs - kp_targ         # [B, K, N, 3]
        abs_diff = torch.abs(diff) * w       # [B, K, N, 3]

        # FFB6D 的 normalize 方式：对每个 (B,K) 分别按点数归一化
        abs_view = abs_diff.view(B, K, -1)   # [B, K, N*3]
        w_view   = w.reshape(B, K, -1)          # [B, K, N*3]

        per_kp_loss = abs_view.sum(dim=2) / (w_view.sum(dim=2) + 1e-3)  # [B, K]
        loss = per_kp_loss.mean()  # 标量

        return loss

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # -------------------- 解析输出与 GT --------------------
        pred_q = outputs["pred_quat"]   # [B, N, 4]
        pred_t = outputs["pred_trans"]  # [B, N, 3]
        pred_c = outputs["pred_conf"]   # [B, N, 1]

        gt_pose = batch["pose"]         # [B, 3, 4]
        gt_r = gt_pose[:, :3, :3]       # [B, 3, 3]
        gt_t = gt_pose[:, :3, 3]        # [B, 3]

        points_cam = batch["points"]    # [B, N, 3]
        B, N, _ = points_cam.shape

        # ======================================================
        # Part 1: Translation Loss (Weighted L1, detach conf)
        # ======================================================
        conf = pred_c.squeeze(2)                         # [B, N]
        conf_stable = conf.clamp(min=1e-6)
        conf_sum = conf_stable.sum(dim=1, keepdim=True)  # [B, 1]
        # 防止 conf 通过平移 loss 学坏：只用它做加权，不回传梯度
        conf_norm = (conf_stable / conf_sum).detach()    # [B, N]

        # conf 加权平均得到最终平移
        t_pred_final = (pred_t * conf_norm.unsqueeze(-1)).sum(dim=1)  # [B, 3]

        diff_t = t_pred_final - gt_t                     # [B, 3]
        abs_diff_t = torch.abs(diff_t)

        axis_weight = self.axis_weight.to(
            device=diff_t.device, dtype=diff_t.dtype
        )
        loss_t = (abs_diff_t * axis_weight).mean()

        # ======================================================
        # Part 2: DenseFusion-style ROI 几何项 (ADD/ADD-S on ROI)
        # ======================================================
        # 将相机坐标下的点云转换到物体坐标系
        gt_r_inv = gt_r.transpose(1, 2).unsqueeze(1)     # [B, 1, 3, 3]
        p_centered = points_cam - gt_t.unsqueeze(1)      # [B, N, 3]
        points_model = torch.matmul(
            p_centered.unsqueeze(2), gt_r_inv
        ).squeeze(2)                                     # [B, N, 3]

        # quat -> rotation (per-point)
        pred_q_flat = pred_q.reshape(-1, 4)              # [B*N, 4]
        pred_r_flat = self.geometry.quat_to_rot(pred_q_flat)
        pred_r = pred_r_flat.view(B, N, 3, 3)            # [B, N, 3, 3]

        # 物体坐标系点 -> 预测相机坐标系
        p_model_exp = points_model.unsqueeze(3)          # [B, N, 3, 1]
        p_rotated = torch.matmul(pred_r, p_model_exp).squeeze(3)  # [B, N, 3]
        points_pred = p_rotated + pred_t                 # [B, N, 3]
        points_target = points_cam                       # [B, N, 3]

        # 对称类判断
        cls_id_tensor = batch.get("cls_id", None)
        if cls_id_tensor is None:
            cls_id = 0
        else:
            cls_id = int(cls_id_tensor[0].item()) if torch.is_tensor(cls_id_tensor) else int(cls_id_tensor[0])
        is_symmetric = cls_id in self.sym_class_ids

        # ROI 内点的误差
        if is_symmetric:
            dist_matrix = torch.cdist(points_pred, points_target, p=2)  # [B, N, N]
            loss_dist, _ = torch.min(dist_matrix, dim=2)                # [B, N]
        else:
            loss_dist = torch.norm(points_pred - points_target, dim=2, p=2)  # [B, N]

        # ======================================================
        # Part 3: Dense Loss + Confidence Regularization
        # ======================================================
        # 3.1 原始 DenseFusion 风格：
        #     L_dense = E[ dist * c - w * log(c) ]
        conf_clamped = conf.clamp(min=1e-4, max=1.0)
        loss_dense = (
            loss_dist * conf_clamped - self.w_rate * torch.log(conf_clamped)
        ).mean()

        # 3.2 显式 conf 回归：误差小 -> conf 接近 1，误差大 -> conf 接近 0
        with torch.no_grad():
            d_clipped = loss_dist.detach().clamp(
                min=0.0, max=self.conf_dist_max
            )
            target_conf = torch.exp(-self.conf_alpha * d_clipped)  # [B, N]

        loss_conf_reg = F.mse_loss(conf, target_conf)

        # ======================================================
        # Part 4: 可选 CAD 级 ADD Loss（与 Evaluator 对齐）
        # ======================================================
        loss_add_cad = points_cam.new_tensor(0.0)
        if self.lambda_add_cad > 0.0 and ("model_points" in batch):
            # 1) 取每个 batch 中最高 conf 的点作为全局 pose 代表
            idx_max = conf.argmax(dim=1)  # [B]
            idx_exp = idx_max.view(B, 1, 1)

            best_q = torch.gather(pred_q, 1, idx_exp.expand(-1, 1, 4)).squeeze(1)  # [B, 4]
            best_t = torch.gather(pred_t, 1, idx_exp.expand(-1, 1, 3)).squeeze(1)  # [B, 3]

            best_q = F.normalize(best_q, dim=-1)
            best_r = self.geometry.quat_to_rot(best_q)  # [B, 3, 3]

            pred_rt_cad = torch.cat([best_r, best_t.unsqueeze(-1)], dim=2)  # [B, 3, 4]
            gt_rt_cad = gt_pose                                            # [B, 3, 4]

            model_points = batch["model_points"]  # [M, 3] 或 [B, M, 3]
            if model_points.dim() == 2:
                model_points = model_points.unsqueeze(0).expand(B, -1, -1)

            if is_symmetric:
                add_cad = self.geometry.compute_adds(pred_rt_cad, gt_rt_cad, model_points)
            else:
                add_cad = self.geometry.compute_add(pred_rt_cad, gt_rt_cad, model_points)
            loss_add_cad = add_cad.mean()

        # ======================================================
        # Part 5: [NEW] Keypoint Offset Loss（FFB6D-style）
        # ======================================================
        loss_kp_of = points_cam.new_tensor(0.0)
        if (
            self.lambda_kp_of > 0.0
            and ("pred_kp_ofs" in outputs)
            and ("kp_targ_ofst" in batch)
            and ("labels" in batch)
        ):
            loss_kp_of = self._compute_kp_offset_loss(
                pred_kp_ofs=outputs["pred_kp_ofs"],   # [B, K, N, 3]
                kp_targ_ofst=batch["kp_targ_ofst"],   # [B, N, K, 3] or [B, K, N, 3]
                labels=batch["labels"],               # [B, N] 或 [B, N, 1]
            )

        # ======================================================
        # Final Loss
        # ======================================================
        total_loss = (
            self.lambda_add * loss_dense
            + self.lambda_t * loss_t
            + self.lambda_conf * loss_conf_reg
            + self.lambda_add_cad * loss_add_cad
            + self.lambda_kp_of * loss_kp_of
        )

        # ======================================================
        # Metrics（用于日志记录/对比曲线）
        # ======================================================
        with torch.no_grad():
            metrics = {
                "loss_total": total_loss.item(),
                "loss_add": loss_dense.item(),
                "loss_t": loss_t.item(),
                "loss_conf": loss_conf_reg.item(),
                "loss_add_cad": float(loss_add_cad.item()),
                "loss_kp_of": float(loss_kp_of.item()),   # [NEW]
                "dist_mean": loss_dist.mean().item(),
                "conf_mean": conf.mean().item(),
                "t_err_mean": abs_diff_t.mean().item(),
                "t_err_z": abs_diff_t[:, 2].mean().item(),
                "t_bias_z": diff_t[:, 2].mean().item(),
            }

        return total_loss, metrics


__all__ = ["LGFFLoss"]
