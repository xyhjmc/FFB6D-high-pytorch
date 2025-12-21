"""
LGFF Loss Module (v3 naming-fixed + explicit rotation loss).

Base behavior: identical to your current v3 (DenseFusion-style + t + conf_reg + optional cad_add + optional kp_of),
with ONE addition:
- Explicit rotation loss (SO(3) geodesic) on fused rotation, weighted by lambda_rot.

Route-A compatibility:
- Keypoint Offset Loss is STRICTLY gated by cfg.lambda_kp_of (read dynamically each forward).
- If lambda_kp_of <= 0: kp loss is not computed, and does not affect gradients.

Compatibility:
- Exposes lambda_dense (alias of lambda_add) for TrainerSC expectations.
- Metrics expose loss_dense (alias of old loss_add) + keep loss_add for backward plots.
- Exposes lambda_rot and metric loss_rot for TrainerSC curriculum/breakdown.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.pose_metrics import fuse_pose_from_outputs


class LGFFLoss(nn.Module):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry

        # DenseFusion 部分
        self.w_rate = getattr(cfg, "w_rate", 0.015)
        self.sym_class_ids = set(getattr(cfg, "sym_class_ids", []))

        # ===== 权重配置（兼容 lambda_dense / lambda_add 两套命名） =====
        self.lambda_t = float(getattr(cfg, "lambda_t", 0.5))  # 平移权重

        _lambda_dense = getattr(cfg, "lambda_dense", None)
        if _lambda_dense is None:
            _lambda_dense = getattr(cfg, "lambda_add", 1.0)
        self.lambda_dense = float(_lambda_dense)   # 主名：供 trainer 使用
        self.lambda_add = self.lambda_dense        # 别名：兼容旧代码/旧配置

        # explicit rotation loss weight
        self.lambda_rot = float(getattr(cfg, "lambda_rot", 0.0))

        self.lambda_conf = float(getattr(cfg, "lambda_conf", 0.1))
        self.lambda_add_cad = float(getattr(cfg, "lambda_add_cad", 0.2))
        # NOTE: lambda_kp_of 不在 __init__ 固化，forward 动态读取，避免被 checkpoint/CLI 覆盖失效

        # Z 轴额外权重
        self.t_z_weight = float(getattr(cfg, "t_z_weight", 2.0))
        axis_weight = torch.tensor([1.0, 1.0, self.t_z_weight], dtype=torch.float32)
        self.register_buffer("axis_weight", axis_weight)

        # Confidence 正则参数
        self.conf_alpha = float(getattr(cfg, "conf_alpha", 10.0))
        self.conf_dist_max = float(getattr(cfg, "conf_dist_max", 0.1))

        # rot loss 数值稳定（acos clamp）
        self.rot_geodesic_eps = float(getattr(cfg, "rot_geodesic_eps", 1e-6))



    # ------------------------------------------------------------------
    # Keypoint Offset Loss (FFB6D-style of_l1_loss)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_kp_offset_loss(
        pred_kp_ofs: torch.Tensor,
        kp_targ_ofst: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_kp_ofs:  [B, K, N, 3]
        kp_targ_ofst: [B, N, K, 3] or [B, K, N, 3]
        labels:       [B, N] or [B, N, 1], >0 indicates foreground points
        """
        assert pred_kp_ofs.dim() == 4 and pred_kp_ofs.size(-1) == 3, (
            "pred_kp_ofs shape must be [B, K, N, 3]"
        )
        assert kp_targ_ofst.dim() == 4 and kp_targ_ofst.size(-1) == 3, (
            "kp_targ_ofst must be [B, N, K, 3] or [B, K, N, 3]"
        )

        B, K, N, C = pred_kp_ofs.shape

        # labels -> [B, N]
        if labels.dim() == 3:
            labels = labels.squeeze(-1)
        assert labels.shape == (B, N), f"labels shape expected [B, N], got {labels.shape}"

        # align gt offsets to [B, K, N, 3]
        if kp_targ_ofst.shape == (B, N, K, C):
            kp_targ = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
        elif kp_targ_ofst.shape == (B, K, N, C):
            kp_targ = kp_targ_ofst
        else:
            raise ValueError(
                f"kp_targ_ofst shape mismatch: expected [B, N, K, 3] or [B, K, N, 3], "
                f"got {kp_targ_ofst.shape}"
            )

        # foreground mask
        w = (labels > 1e-8).float()  # [B, N]
        w = w.view(B, 1, N, 1).expand(-1, K, -1, C)  # [B, K, N, 3]

        diff = pred_kp_ofs - kp_targ
        abs_diff = torch.abs(diff) * w

        # normalize per (B,K)
        abs_view = abs_diff.view(B, K, -1)  # [B, K, N*3]
        w_view = w.reshape(B, K, -1)        # [B, K, N*3]

        per_kp_loss = abs_view.sum(dim=2) / (w_view.sum(dim=2) + 1e-3)
        return per_kp_loss.mean()

    # ------------------------------------------------------------------
    # SO(3) Geodesic Loss from rotation matrices
    # ------------------------------------------------------------------
    def _so3_geodesic_loss(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        R_pred, R_gt: [B,3,3] float tensors
        Returns: mean geodesic angle in radians
        """
        R_rel = torch.matmul(R_pred, R_gt.transpose(1, 2))  # [B,3,3]
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [B]
        cos_theta = (trace - 1.0) * 0.5

        eps = self.rot_geodesic_eps
        cos_theta = cos_theta.clamp(min=-1.0 + eps, max=1.0 - eps)

        theta = torch.acos(cos_theta)  # [B] in radians
        loss = theta.mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ---- outputs ----
        pred_q = outputs["pred_quat"]   # [B, N, 4]
        pred_t = outputs["pred_trans"]  # [B, N, 3]
        pred_c = outputs["pred_conf"]   # [B, N, 1]

        # ---- GT ----
        gt_pose = batch["pose"]         # [B, 3, 4]
        gt_r = gt_pose[:, :3, :3]       # [B, 3, 3]
        gt_t = gt_pose[:, :3, 3]        # [B, 3]

        points_cam = batch["points"]    # [B, N, 3]
        B, N, _ = points_cam.shape
        device = points_cam.device
        device_type = device.type

        # symmetry mask
        sym_ids = torch.as_tensor(getattr(self.cfg, "sym_class_ids", []), device=device)
        sym_mask = torch.zeros(B, dtype=torch.bool, device=device)
        cls_id_tensor = batch.get("cls_id", None)
        if cls_id_tensor is not None:
            cls_flat = cls_id_tensor.view(-1).to(device=device)
            if sym_ids.numel() > 0:
                sym_mask = (cls_flat.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)

        # ------------------------------------------------------
        # Pose fusion (robust under AMP): force FP32 + disable autocast here
        # ------------------------------------------------------
        with torch.amp.autocast(device_type=device_type, enabled=False):
            fused_rt = fuse_pose_from_outputs(
                outputs=outputs,
                geometry=self.geometry,
                cfg=self.cfg,
                stage="train",
            ).float()
        fused_r = fused_rt[:, :3, :3]
        fused_t = fused_rt[:, :3, 3]

        # ======================================================
        # Part 0: Explicit Rotation Loss (geodesic on fused rotation)
        # ======================================================
        loss_rot = points_cam.new_tensor(0.0)
        if self.lambda_rot > 0.0:
            nonsym_mask = ~sym_mask
            if nonsym_mask.any():
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    loss_rot = self._so3_geodesic_loss(
                        R_pred=fused_r[nonsym_mask].float(),
                        R_gt=gt_r[nonsym_mask].float(),
                    )
            else:
                loss_rot = points_cam.new_tensor(0.0)

        # ======================================================
        # Part 1: Translation Loss (Weighted L1)
        # ======================================================
        conf = pred_c.squeeze(2)  # [B, N]
        conf_stable = conf.clamp(min=1e-6)
        conf_sum = conf_stable.sum(dim=1, keepdim=True)
        conf_norm = (conf_stable / conf_sum).detach()

        use_fused_t = getattr(self.cfg, "loss_use_fused_pose", True)
        if use_fused_t:
            t_pred_final = fused_t
        else:
            t_pred_final = (pred_t * conf_norm.unsqueeze(-1)).sum(dim=1)

        diff_t = t_pred_final - gt_t
        abs_diff_t = torch.abs(diff_t)

        axis_weight = self.axis_weight.to(device=device, dtype=diff_t.dtype)
        loss_t = (abs_diff_t * axis_weight).mean()
        loss_t = torch.nan_to_num(loss_t, nan=0.0, posinf=1e4, neginf=1e4)

        # ======================================================
        # Part 1.5: Optional Z-bias penalty (systematic bias on tz)
        # ======================================================
        lambda_t_bias_z = float(getattr(self.cfg, "lambda_t_bias_z", 0.3))
        t_bias_z_mode = str(getattr(self.cfg, "t_bias_z_mode", "batch_mean")).lower()
        # batch_mean: penalize systematic bias only (recommended)
        # per_sample: penalize per-sample bias (stronger, intrusive)

        loss_t_bias_z = points_cam.new_tensor(0.0)
        if lambda_t_bias_z > 0.0:
            # Use FP32 for numerical stability under AMP
            dz = diff_t[:, 2].float()  # [B], signed

            if t_bias_z_mode == "per_sample":
                # stronger; tends to hurt <5mm metrics
                # SmoothL1 is usually safer than pure abs
                beta = float(getattr(self.cfg, "t_bias_beta", 0.005))  # meters, e.g. 5mm
                loss_t_bias_z = F.smooth_l1_loss(dz, torch.zeros_like(dz), beta=beta, reduction="mean")
            else:
                # recommended: only suppress systematic (batch-level) bias
                mu = dz.mean()  # scalar
                # Option 1 (simplest & smooth): L2 on mean
                loss_t_bias_z = mu * mu
                # Option 2 (robust): SmoothL1 on mean
                # beta = float(getattr(self.cfg, "t_bias_beta", 0.005))
                # loss_t_bias_z = F.smooth_l1_loss(mu, mu.new_zeros(()), beta=beta, reduction="mean")

            loss_t_bias_z = torch.nan_to_num(loss_t_bias_z, nan=0.0, posinf=1e4, neginf=1e4)

        # ======================================================
        # Part 2: DenseFusion-style ROI geometric term (ADD/ADD-S on ROI)
        # ======================================================
        gt_r_inv = gt_r.transpose(1, 2).unsqueeze(1)  # [B, 1, 3, 3]
        p_centered = points_cam - gt_t.unsqueeze(1)   # [B, N, 3]
        points_model = torch.matmul(p_centered.unsqueeze(2), gt_r_inv).squeeze(2)  # [B, N, 3]

        pred_q_flat = pred_q.reshape(-1, 4)
        pred_r_flat = self.geometry.quat_to_rot(pred_q_flat)
        pred_r = pred_r_flat.view(B, N, 3, 3)

        p_model_exp = points_model.unsqueeze(3)  # [B, N, 3, 1]
        p_rotated = torch.matmul(pred_r, p_model_exp).squeeze(3)  # [B, N, 3]
        points_pred = p_rotated + pred_t
        points_target = points_cam

        loss_dist = torch.norm(points_pred - points_target, dim=2, p=2)  # [B, N]

        # ADD-S for symmetric
        if sym_mask.any():
            sym_indices = torch.where(sym_mask)[0]
            for idx in sym_indices:
                dist_matrix = torch.cdist(
                    points_pred[idx: idx + 1],
                    points_target[idx: idx + 1],
                    p=2,
                )
                loss_dist[idx] = torch.min(dist_matrix, dim=2).values.squeeze(0)

        # ======================================================
        # Part 3: Dense loss + explicit confidence regression
        # ======================================================
        conf_clamped = conf.clamp(min=1e-4, max=1.0)
        loss_dense = (loss_dist * conf_clamped - self.w_rate * torch.log(conf_clamped)).mean()
        loss_dense = torch.nan_to_num(loss_dense, nan=0.0, posinf=1e4, neginf=1e4)

        with torch.no_grad():
            d_clipped = loss_dist.detach().clamp(min=0.0, max=self.conf_dist_max)
            target_conf = torch.exp(-self.conf_alpha * d_clipped)

        loss_conf_reg = F.mse_loss(conf, target_conf)
        loss_conf_reg = torch.nan_to_num(loss_conf_reg, nan=0.0, posinf=1e4, neginf=1e4)

        # ======================================================
        # Part 4: Optional CAD-level ADD/ADD-S (aligned with evaluator)
        # ======================================================
        loss_add_cad = points_cam.new_tensor(0.0)
        if self.lambda_add_cad > 0.0 and ("model_points" in batch):
            use_fused_pose = getattr(self.cfg, "loss_use_fused_pose", True)
            if use_fused_pose:
                pred_rt_cad = fused_rt
            else:
                idx_max = conf.argmax(dim=1)
                idx_exp = idx_max.view(B, 1, 1)
                best_q = torch.gather(pred_q, 1, idx_exp.expand(-1, 1, 4)).squeeze(1)
                best_t = torch.gather(pred_t, 1, idx_exp.expand(-1, 1, 3)).squeeze(1)
                best_q = F.normalize(best_q, dim=-1)
                best_r = self.geometry.quat_to_rot(best_q)
                pred_rt_cad = torch.cat([best_r, best_t.unsqueeze(-1)], dim=2)

            gt_rt_cad = gt_pose
            model_points = batch["model_points"]
            if model_points.dim() == 2:
                model_points = model_points.unsqueeze(0).expand(B, -1, -1)

            add_cad = self.geometry.compute_add(pred_rt_cad, gt_rt_cad, model_points)
            adds_cad = self.geometry.compute_adds(pred_rt_cad, gt_rt_cad, model_points)

            loss_add_cad_batch = torch.where(sym_mask, adds_cad, add_cad)
            loss_add_cad = loss_add_cad_batch.mean()
            loss_add_cad = torch.nan_to_num(loss_add_cad, nan=0.0, posinf=1e4, neginf=1e4)

        # ======================================================
        # Part 5: Optional Keypoint Offset Loss (STRICT gating by cfg.lambda_kp_of)
        # ======================================================
        loss_kp_of = points_cam.new_tensor(0.0)
        lambda_kp_of = float(getattr(self.cfg, "lambda_kp_of", 0.0))

        if lambda_kp_of > 0.0:
            if ("pred_kp_ofs" in outputs) and ("kp_targ_ofst" in batch) and ("labels" in batch):
                loss_kp_of = self._compute_kp_offset_loss(
                    pred_kp_ofs=outputs["pred_kp_ofs"],
                    kp_targ_ofst=batch["kp_targ_ofst"],
                    labels=batch["labels"],
                )
                loss_kp_of = torch.nan_to_num(loss_kp_of, nan=0.0, posinf=1e4, neginf=1e4)
            else:
                # missing keys -> treat as disabled (avoid crashes / side effects)
                loss_kp_of = points_cam.new_tensor(0.0)

        # ======================================================
        # Final Loss
        # ======================================================
        total_loss = (
            self.lambda_dense * loss_dense
            + self.lambda_t * loss_t
            + lambda_t_bias_z * loss_t_bias_z
            + self.lambda_rot * loss_rot
            + self.lambda_conf * loss_conf_reg
            + self.lambda_add_cad * loss_add_cad
            + lambda_kp_of * loss_kp_of
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=1e4)

        # ======================================================
        # Metrics (trainer-compatible)
        # ======================================================
        with torch.no_grad():
            metrics = {
                "loss_total": float(total_loss.item()),

                "loss_dense": float(loss_dense.item()),
                "loss_add": float(loss_dense.item()),

                "loss_t": float(loss_t.item()),
                "loss_rot": float(loss_rot.item()),
                "loss_conf": float(loss_conf_reg.item()),
                "loss_add_cad": float(loss_add_cad.item()),
                "loss_kp_of": float(loss_kp_of.item()),

                "lambda_kp_of": float(lambda_kp_of),

                "dist_mean": float(loss_dist.mean().item()),
                "conf_mean": float(conf.mean().item()),
                "t_err_mean": float(abs_diff_t.mean().item()),
                "t_err_z": float(abs_diff_t[:, 2].mean().item()),
                "t_bias_z": float(diff_t[:, 2].mean().item()),

                "loss_t_bias_z": float(loss_t_bias_z.item()),
                "lambda_t_bias_z": float(lambda_t_bias_z),
                "t_bias_z_batch": float(diff_t[:, 2].mean().item()),
                "num_sym_in_batch": float(sym_mask.sum().item()),
                "rot_sym_ignored_count": float(sym_mask.sum().item()) if self.lambda_rot > 0 else 0.0,

            }

        return total_loss, metrics


__all__ = ["LGFFLoss"]
