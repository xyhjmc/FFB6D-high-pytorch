# lgff/losses/lgff_loss_seg.py
# -*- coding: utf-8 -*-
"""
LGFF Loss Module (Seg Enhanced + Robust).

Small improvements (minimal invasive):
1) Seg loss supports optional per-pixel weight map: batch["mask_weight"] (0~1).
   - Different from mask_valid(ignore). mask_weight is a soft supervision weight.
2) Focal branch now actually uses weight if provided.
3) Dice branch supports weight (weighted soft dice) to reduce boundary penalty under misalignment.
4) Keypoint loss uses SmoothL1 (Huber) instead of pure L1 for robustness to outliers/noisy points.
   - Still preserves your labels-based normalization semantics.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lgff.utils.config_seg import LGFFConfigSeg
from lgff.utils.geometry import GeometryToolkit


class SigmoidFocalLoss(nn.Module):
    """
    Binary Sigmoid Focal Loss.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None):
        # inputs: logits, targets: 0/1 (or soft label in [0,1])
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LGFFLoss_SEG(nn.Module):
    def __init__(self, cfg: LGFFConfigSeg, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry

        # --- Symmetry ---
        sym_ids = getattr(cfg, "sym_class_ids", [])
        self.register_buffer("sym_class_ids", torch.tensor(sym_ids, dtype=torch.long))

        # --- Weights (Pose) ---
        self.lambda_t = float(getattr(cfg, "lambda_t", 2.0))
        self.lambda_kp = float(getattr(cfg, "lambda_kp_of", 2.0))

        _lambda_dense = getattr(cfg, "lambda_dense", None)
        if _lambda_dense is None:
            _lambda_dense = getattr(cfg, "lambda_add", 1.0)
        self.lambda_dense = float(_lambda_dense)

        self.lambda_conf = float(getattr(cfg, "lambda_conf", 0.1))

        # --- Weights (Seg) ---
        self.lambda_seg = float(getattr(cfg, "lambda_seg", 0.0))  # Default 0.0 if not set

        # --- Translation axis weighting ---
        self.t_z_weight = float(getattr(cfg, "t_z_weight", 5.0))
        axis_weight = torch.tensor([1.0, 1.0, self.t_z_weight], dtype=torch.float32)
        self.register_buffer("axis_weight", axis_weight)

        # --- DenseFusion constants ---
        self.w_rate = float(getattr(cfg, "w_rate", 0.015))
        self.conf_alpha = float(getattr(cfg, "conf_alpha", 10.0))
        self.conf_dist_max = float(getattr(cfg, "conf_dist_max", 0.1))
        self.dense_sample_num = int(getattr(cfg, "loss_dense_sample_num", 512))

        # --- Seg loss options ---
        self.seg_dice_weight = float(getattr(cfg, "seg_dice_weight", 0.5))
        self.seg_min_foreground = float(getattr(cfg, "seg_min_foreground", 1.0))
        self.seg_loss_type: str = str(getattr(cfg, "seg_loss_type", "bce")).lower().strip()

        self.seg_pos_weight = getattr(cfg, "seg_pos_weight", None)
        if self.seg_pos_weight is not None:
            self.register_buffer("seg_pos_weight_tensor", torch.tensor(float(self.seg_pos_weight), dtype=torch.float32))
        else:
            self.seg_pos_weight_tensor = None

        self.seg_ignore_invalid: bool = bool(getattr(cfg, "seg_ignore_invalid", True))

        # --- KP loss robust beta (Huber) ---
        self.kp_smoothl1_beta = float(getattr(cfg, "kp_smoothl1_beta", 0.01))

        # Initialize Focal Loss if needed
        if "focal" in self.seg_loss_type:
            self.focal_loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="none")

    @staticmethod
    def _safe_isin(x: torch.Tensor, set_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "isin"):
            return torch.isin(x, set_ids)
        return (x.view(-1, 1) == set_ids.view(1, -1)).any(dim=1)

    def _compute_kp_loss(self, pred, target, labels, valid_mask=None):
        """
        Robust KP loss:
        - Replace pure L1 with SmoothL1 (Huber) to reduce sensitivity to outliers/noisy points.
        - Keep your original labels normalization semantics.
        """
        B, K, N, _ = pred.shape
        if labels is None:
            w = torch.ones((B, N), device=pred.device, dtype=pred.dtype)
        else:
            w = (labels.squeeze(-1) if labels.dim() == 3 else labels) > 1e-8
            w = w.float()

        if valid_mask is not None:
            w = w * valid_mask.view(B, 1).float()

        w_exp = w.view(B, 1, N, 1).expand(-1, K, -1, 3)

        # SmoothL1 per-element
        # NOTE: F.smooth_l1_loss supports elementwise reduction
        diff = F.smooth_l1_loss(pred, target, beta=self.kp_smoothl1_beta, reduction="none")
        diff = diff * w_exp

        denom = w_exp.sum() + 1e-6
        return diff.sum() / denom

    def _seg_loss(self, pred_mask_logits, gt_mask, mask_valid=None, mask_weight=None):
        """
        Minimal improvements:
        - mask_valid: ignore mask (0/1 or [0,1] but treated as ignore region when seg_ignore_invalid=True)
        - mask_weight: soft weight map (0~1) to down-weight boundary / halo region; NOT an ignore mask.
        """
        device = pred_mask_logits.device
        dtype = pred_mask_logits.dtype

        # 1) basic fg filter
        with torch.no_grad():
            fg = gt_mask.sum(dim=(1, 2, 3))  # [B]
            valid_fg = fg > self.seg_min_foreground

        # 2) Prepare mv (ignore) and mw (weight)
        mv_all = None
        if mask_valid is not None:
            mv_all = mask_valid
            if mv_all.dim() == 3:
                mv_all = mv_all.unsqueeze(1)
            mv_all = mv_all.float()
            if mv_all.shape[-2:] != pred_mask_logits.shape[-2:]:
                mv_all = F.interpolate(mv_all, size=pred_mask_logits.shape[-2:], mode="nearest")

            if self.seg_ignore_invalid:
                with torch.no_grad():
                    mv_sum = mv_all.flatten(1).sum(dim=1)  # [B]
                    # keep your heuristic threshold
                    valid_fg = valid_fg & (mv_sum > 50)

        mw_all = None
        if mask_weight is not None:
            mw_all = mask_weight
            if mw_all.dim() == 3:
                mw_all = mw_all.unsqueeze(1)
            mw_all = mw_all.float()
            if mw_all.shape[-2:] != pred_mask_logits.shape[-2:]:
                # weight is a soft map, bilinear is fine
                mw_all = F.interpolate(mw_all, size=pred_mask_logits.shape[-2:], mode="bilinear", align_corners=False)
            mw_all = mw_all.clamp(0.0, 1.0)

        if not valid_fg.any():
            return torch.tensor(0.0, device=device, dtype=dtype)

        logits_v = pred_mask_logits[valid_fg]
        gt_v = gt_mask[valid_fg]

        mv = mv_all[valid_fg].clamp(0.0, 1.0) if mv_all is not None else None
        mw = mw_all[valid_fg].clamp(0.0, 1.0) if mw_all is not None else None

        # 3) Combine mv(ignore) and mw(weight) carefully
        #    - mv is applied only if seg_ignore_invalid=True (to ignore invalid pixels)
        #    - mw is always a weight if provided (for soft supervision)
        eff_w = None
        if mw is not None:
            eff_w = mw
        if mv is not None and self.seg_ignore_invalid:
            eff_w = mv if eff_w is None else (eff_w * mv)

        # 4) base loss
        if "focal" in self.seg_loss_type:
            base_loss = self.focal_loss(logits_v, gt_v, weight=eff_w)  # [Bv,1,H,W]
            # focal_loss already applies weight, so reduce consistent with your logic:
            if eff_w is not None:
                denom = eff_w.sum().clamp(min=1.0)
                loss_main = base_loss.sum() / denom
            else:
                loss_main = base_loss.mean()
        else:
            pos_weight = (
                self.seg_pos_weight_tensor.to(device=device, dtype=dtype)
                if self.seg_pos_weight_tensor is not None
                else None
            )
            base_loss = F.binary_cross_entropy_with_logits(
                logits_v, gt_v, reduction="none", pos_weight=pos_weight
            )
            if eff_w is not None:
                base_loss = base_loss * eff_w
                denom = eff_w.sum().clamp(min=1.0)
                loss_main = base_loss.sum() / denom
            else:
                loss_main = base_loss.mean()

        # 5) dice optional (weighted dice)
        if "dice" in self.seg_loss_type:
            prob = torch.sigmoid(logits_v)
            tgt = gt_v

            if eff_w is not None:
                prob = prob * eff_w
                tgt = tgt * eff_w

            prob_flat = prob.flatten(1)
            tgt_flat = tgt.flatten(1)
            inter = (prob_flat * tgt_flat).sum(dim=1)
            union = prob_flat.sum(dim=1) + tgt_flat.sum(dim=1)
            dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
            return loss_main + self.seg_dice_weight * dice.mean()

        return loss_main

    def forward(self, outputs, batch):
        # Unpack outputs
        pred_kp = outputs.get("pred_kp_ofs", None)
        pred_q = outputs.get("pred_quat", None)
        pred_t = outputs.get("pred_trans", None)
        pred_c = outputs.get("pred_conf", None)
        pred_mask_logits = outputs.get("pred_mask_logits", None)

        gt_pose = batch["pose"]
        points = batch["points"]
        labels = batch.get("labels", None)

        device = points.device
        B, N, _ = points.shape

        # 1. Symmetry Masks
        cls_ids = batch.get("cls_id", None)
        sym_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if cls_ids is not None and self.sym_class_ids.numel() > 0:
            sym_mask = self._safe_isin(cls_ids.view(-1).to(device), self.sym_class_ids)
        nonsym_mask = ~sym_mask

        # 2. Translation Loss
        gt_t = gt_pose[:, :3, 3]
        if pred_t is None:
            t_pred_global = gt_t.detach()
        else:
            if pred_t.dim() == 2 and pred_t.shape == (B, 3):
                t_pred_global = pred_t
            else:
                if pred_c is not None:
                    conf = pred_c.squeeze(2).clamp(min=1e-6)
                    conf_norm = conf / (conf.sum(dim=1, keepdim=True) + 1e-6)
                    t_pred_global = (pred_t * conf_norm.unsqueeze(-1)).sum(dim=1)
                else:
                    t_pred_global = pred_t.mean(dim=1)

        diff_t = torch.abs(t_pred_global - gt_t)
        loss_t = (diff_t * self.axis_weight.to(diff_t.dtype)).mean()

        # 3. Keypoint Loss (Asym)
        loss_kp = torch.tensor(0.0, device=device, dtype=points.dtype)
        kp_error_per_point = None

        if self.lambda_kp > 0 and nonsym_mask.any() and pred_kp is not None:
            kp_gt = batch["kp_targ_ofst"]
            if kp_gt.dim() == 4 and kp_gt.shape[1] == N:
                kp_gt = kp_gt.permute(0, 2, 1, 3).contiguous()

            loss_kp = self._compute_kp_loss(pred_kp, kp_gt, labels, valid_mask=nonsym_mask).to(points.dtype)

            with torch.no_grad():
                kp_err = torch.norm(pred_kp - kp_gt, dim=3).mean(dim=1)
                if labels is not None:
                    l_ = labels.squeeze(-1) if labels.dim() == 3 else labels
                    kp_err = kp_err * (l_ > 1e-8).float()
                kp_error_per_point = kp_err

        # 4. Dense Loss (Sym)
        loss_dense = torch.tensor(0.0, device=device, dtype=points.dtype)
        dense_error_per_point = None

        need_dense = sym_mask.any() and (self.lambda_dense > 0 or self.lambda_conf > 0)
        if need_dense and pred_q is not None:
            p_center = points - gt_t.unsqueeze(1)
            gt_R = gt_pose[:, :3, :3]
            p_model = torch.matmul(p_center, gt_R)

            pred_q_use = pred_q.unsqueeze(1).expand(B, N, 4) if pred_q.dim() == 2 else pred_q
            pred_R = self.geometry.quat_to_rot(pred_q_use.reshape(-1, 4)).view(B, N, 3, 3)
            p_pred = (pred_R @ p_model.unsqueeze(-1)).squeeze(-1) + t_pred_global.unsqueeze(1)

            dist_base = torch.norm(p_pred - points, dim=2)
            final_dist = dist_base.clone()

            if sym_mask.any():
                s_idx = torch.where(sym_mask)[0]
                S = min(self.dense_sample_num, N)
                for bi in s_idx.tolist():
                    if S < N:
                        idx = torch.randperm(N, device=device)[:S]
                        tgt = points[bi, idx, :].unsqueeze(0)
                    else:
                        tgt = points[bi].unsqueeze(0)
                    d_mat = torch.cdist(p_pred[bi:bi + 1], tgt)
                    min_d, _ = d_mat.min(dim=2)
                    final_dist[bi] = min_d.squeeze(0)

            dense_error_per_point = final_dist

            if self.lambda_dense > 0 and sym_mask.any():
                c_clamp = pred_c.squeeze(2).clamp(min=1e-4, max=1.0) if pred_c is not None else torch.ones_like(final_dist)
                dense_term = final_dist * c_clamp - self.w_rate * torch.log(c_clamp)
                mask_bc = sym_mask.view(B, 1).float()
                denom = mask_bc.sum() * N + 1e-6
                loss_dense = (dense_term * mask_bc).sum() / denom

        # 5. Confidence Loss
        loss_conf = torch.tensor(0.0, device=device, dtype=points.dtype)
        if self.lambda_conf > 0 and pred_c is not None:
            final_error = torch.zeros((B, N), device=device, dtype=points.dtype)
            valid_pts = torch.zeros((B, N), device=device, dtype=torch.bool)

            if sym_mask.any() and dense_error_per_point is not None:
                m = sym_mask.view(B, 1).expand(B, N)
                final_error = torch.where(m, dense_error_per_point.to(points.dtype), final_error)
                valid_pts = valid_pts | m

            if nonsym_mask.any() and kp_error_per_point is not None:
                m = nonsym_mask.view(B, 1).expand(B, N)
                final_error = torch.where(m, kp_error_per_point.to(points.dtype), final_error)
                valid_pts = valid_pts | m

            if labels is not None:
                l_ = labels.squeeze(-1) if labels.dim() == 3 else labels
                valid_pts = valid_pts & (l_ > 1e-8)

            if valid_pts.any():
                with torch.no_grad():
                    d_clip = final_error.detach().clamp(max=self.conf_dist_max)
                    target_c = torch.exp(-self.conf_alpha * d_clip)
                pred_c_map = pred_c.squeeze(2)
                loss_conf = F.mse_loss(pred_c_map[valid_pts], target_c[valid_pts])

        # 6. Seg Loss (Safe) + optional soft weight
        loss_seg = torch.tensor(0.0, device=device, dtype=points.dtype)
        if self.lambda_seg > 0:
            if pred_mask_logits is not None:
                gt_mask = batch.get("mask", None)
                if gt_mask is not None:
                    if gt_mask.dim() == 3:
                        gt_mask = gt_mask.unsqueeze(1)
                    gt_mask = gt_mask.float()
                    if gt_mask.max() > 1.0:
                        gt_mask = (gt_mask > 0.5).float()

                    if gt_mask.shape[-2:] != pred_mask_logits.shape[-2:]:
                        gt_mask = F.interpolate(gt_mask, size=pred_mask_logits.shape[-2:], mode="nearest")

                    # NEW: optional mask_weight from dataloader
                    mask_weight = batch.get("mask_weight", None)
                    if mask_weight is not None:
                        if mask_weight.dim() == 3:
                            mask_weight = mask_weight.unsqueeze(1)
                        mask_weight = mask_weight.float()
                        if mask_weight.shape[-2:] != pred_mask_logits.shape[-2:]:
                            mask_weight = F.interpolate(mask_weight, size=pred_mask_logits.shape[-2:], mode="bilinear", align_corners=False)
                        mask_weight = mask_weight.clamp(0.0, 1.0)

                    loss_seg = self._seg_loss(
                        pred_mask_logits,
                        gt_mask,
                        mask_valid=batch.get("mask_valid", None),
                        mask_weight=mask_weight,
                    ).to(points.dtype)

        total_loss = (
            self.lambda_t * loss_t +
            self.lambda_kp * loss_kp +
            self.lambda_dense * loss_dense +
            self.lambda_conf * loss_conf +
            self.lambda_seg * loss_seg
        )

        return total_loss, {
            "loss_total": float(total_loss.detach().item()),
            "loss_t": float(loss_t.detach().item()),
            "loss_kp": float(loss_kp.detach().item()),
            "loss_dense": float(loss_dense.detach().item()),
            "loss_conf": float(loss_conf.detach().item()),
            "loss_seg": float(loss_seg.detach().item()),
            "sym_rate": float(sym_mask.float().mean().detach().item()),
        }
