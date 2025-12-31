# -*- coding: utf-8 -*-
"""
LGFF Loss Module (v9.0: +Seg Head Support, Stable).

Based on your v8.2, with additions:
1) [New] Segmentation loss support:
   - outputs["pred_mask_logits"]: [B,1,128,128] (logits)
   - batch["mask"] (preferred) or batch["mask_visib"] as GT: [B,1,128,128] or [B,128,128]
   - loss_seg = BCEWithLogits + 0.5 * Dice
   - total_loss += lambda_seg * loss_seg
2) Keeps your stability fixes:
   - clone() before in-place assignment
   - variables initialized before conditional blocks
   - true decoupling: Asym->KP, Sym->Dense for conf target
3) Minimal assumptions: if GT mask not present, seg loss is skipped automatically.

Required (to actually train Seg Head):
- dataloader must provide GT mask tensor aligned to ROI size (128x128) as batch["mask"] or batch["mask_visib"].

Notes:
- This module does NOT require changing your trainer logic unless you want:
  (a) warmup schedule for lambda_seg, or
  (b) logging loss_seg.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lgff.utils.config import LGFFConfig
from lgff.utils.geometry import GeometryToolkit


class LGFFLoss_SEG(nn.Module):
    def __init__(self, cfg: LGFFConfig, geometry: GeometryToolkit) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry = geometry

        # --- Symmetry ---
        sym_ids = getattr(cfg, "sym_class_ids", [])
        self.register_buffer("sym_class_ids", torch.tensor(sym_ids, dtype=torch.long))

        # --- Weights ---
        self.lambda_t = float(getattr(cfg, "lambda_t", 2.0))
        self.lambda_kp = float(getattr(cfg, "lambda_kp_of", 2.0))

        _lambda_dense = getattr(cfg, "lambda_dense", None)
        if _lambda_dense is None:
            _lambda_dense = getattr(cfg, "lambda_add", 1.0)
        self.lambda_dense = float(_lambda_dense)

        self.lambda_conf = float(getattr(cfg, "lambda_conf", 0.1))

        # --- [New] Seg weight ---
        # keep small by default (auxiliary)
        self.lambda_seg = float(getattr(cfg, "lambda_seg", 0.1))

        # --- Translation axis weighting ---
        self.t_z_weight = float(getattr(cfg, "t_z_weight", 5.0))
        axis_weight = torch.tensor([1.0, 1.0, self.t_z_weight], dtype=torch.float32)
        self.register_buffer("axis_weight", axis_weight)

        # --- DenseFusion constants ---
        self.w_rate = float(getattr(cfg, "w_rate", 0.015))
        self.conf_alpha = float(getattr(cfg, "conf_alpha", 10.0))
        self.conf_dist_max = float(getattr(cfg, "conf_dist_max", 0.1))

        # --- ADD-S target subsample ---
        self.dense_sample_num = int(getattr(cfg, "loss_dense_sample_num", 512))

        # --- Seg loss options ---
        self.seg_dice_weight = float(getattr(cfg, "seg_dice_weight", 0.5))
        self.seg_min_foreground = float(getattr(cfg, "seg_min_foreground", 1.0))  # pixels (in 128x128)
        self.seg_loss_type: str = str(getattr(cfg, "seg_loss_type", "bce")).lower().strip()
        self.seg_pos_weight = getattr(cfg, "seg_pos_weight", None)
        if self.seg_pos_weight is not None:
            self.register_buffer("seg_pos_weight_tensor", torch.tensor(float(self.seg_pos_weight), dtype=torch.float32))
        else:
            self.seg_pos_weight_tensor = None
        self.seg_ignore_invalid: bool = bool(getattr(cfg, "seg_ignore_invalid", True))

    @staticmethod
    def _safe_isin(x: torch.Tensor, set_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "isin"):
            return torch.isin(x, set_ids)
        return (x.view(-1, 1) == set_ids.view(1, -1)).any(dim=1)

    @staticmethod
    def _compute_kp_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pred  : [B, K, N, 3]
        target: [B, K, N, 3]
        labels: [B, N] or [B, N, 1]
        valid_mask: [B] (only asym samples participate)
        """
        B, K, N, _ = pred.shape

        if labels is None:
            w = torch.ones((B, N), device=pred.device, dtype=pred.dtype)
        else:
            if labels.dim() == 3:
                labels = labels.squeeze(-1)
            w = (labels > 1e-8).float()

        if valid_mask is not None:
            w = w * valid_mask.view(B, 1).float()

        w_exp = w.view(B, 1, N, 1).expand(-1, K, -1, 3)
        diff = torch.abs(pred - target) * w_exp

        denom = w_exp.sum() + 1e-6
        return diff.sum() / denom

    @staticmethod
    def _dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        logits/target: [B,1,H,W], target in {0,1}
        """
        prob = torch.sigmoid(logits)
        prob = prob.flatten(1)
        tgt = target.flatten(1)
        inter = (prob * tgt).sum(dim=1)
        union = prob.sum(dim=1) + tgt.sum(dim=1)
        dice = (2.0 * inter + eps) / (union + eps)
        return 1.0 - dice.mean()

    def _seg_loss(
        self,
        pred_mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        mask_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        pred_mask_logits: [B,1,H,W]
        gt_mask:         [B,1,H,W] float in {0,1}
        mask_valid:      [B,1,H,W] float/bool in {0,1}, optional (ignored pixels=0)
        """
        device = pred_mask_logits.device
        dtype = pred_mask_logits.dtype

        with torch.no_grad():
            fg = gt_mask.sum(dim=(1, 2, 3))  # [B]
            valid_fg = fg > self.seg_min_foreground
        if not valid_fg.any():
            return torch.tensor(0.0, device=device, dtype=dtype)

        logits_v = pred_mask_logits[valid_fg]
        gt_v = gt_mask[valid_fg]

        # align mask_valid to logits
        mv = None
        if mask_valid is not None:
            mv = mask_valid
            if mv.dim() == 3:
                mv = mv.unsqueeze(1)
            mv = mv.float()
            if mv.shape[-2:] != logits_v.shape[-2:]:
                mv = F.interpolate(mv, size=logits_v.shape[-2:], mode="nearest")
            mv = mv[valid_fg].clamp(0.0, 1.0)

        # BCE
        pos_weight = None
        if self.seg_pos_weight_tensor is not None:
            pos_weight = self.seg_pos_weight_tensor.to(device=device, dtype=dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits_v,
            gt_v,
            reduction="none",
            pos_weight=pos_weight,
        )
        if mv is not None and self.seg_ignore_invalid:
            bce = bce * mv
            denom = mv.sum().clamp(min=1.0)
            bce = bce.sum() / denom
        else:
            bce = bce.mean()

        if self.seg_loss_type == "bce":
            return bce

        if self.seg_loss_type == "bce_dice":
            prob = torch.sigmoid(logits_v)
            tgt = gt_v
            mask_use = None
            if mv is not None and self.seg_ignore_invalid:
                mask_use = mv
                prob = prob * mask_use
                tgt = tgt * mask_use

            prob_flat = prob.flatten(1)
            tgt_flat = tgt.flatten(1)
            if mask_use is not None:
                mflat = mask_use.flatten(1)
                inter = (prob_flat * tgt_flat).sum(dim=1)
                union = (prob_flat * mflat).sum(dim=1) + (tgt_flat * mflat).sum(dim=1)
            else:
                inter = (prob_flat * tgt_flat).sum(dim=1)
                union = prob_flat.sum(dim=1) + tgt_flat.sum(dim=1)
            dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
            return bce + self.seg_dice_weight * dice.mean()

        raise ValueError(f"Unsupported seg_loss_type={self.seg_loss_type}")

    def forward(self, outputs, batch):
        # -------------------------
        # Model outputs
        # -------------------------
        pred_kp = outputs.get("pred_kp_ofs", None)
        pred_q = outputs.get("pred_quat", None)
        pred_t = outputs.get("pred_trans", None)
        pred_c = outputs.get("pred_conf", None)            # [B, N, 1] expected
        pred_mask_logits = outputs.get("pred_mask_logits", None)  # [B,1,128,128] logits

        # -------------------------
        # Batch data
        # -------------------------
        gt_pose = batch["pose"]                           # [B, 3, 4]
        points = batch["points"]                          # [B, N, 3]
        labels = batch.get("labels", None)                # optional
        device = points.device
        B, N, _ = points.shape

        # -------------------------
        # 1) Sym / Asym masks
        # -------------------------
        cls_ids = batch.get("cls_id", None)
        sym_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if cls_ids is not None and self.sym_class_ids.numel() > 0:
            sym_mask = self._safe_isin(cls_ids.view(-1).to(device), self.sym_class_ids)
        nonsym_mask = ~sym_mask

        # -------------------------
        # 2) Global translation (robust)
        # -------------------------
        gt_t = gt_pose[:, :3, 3]  # [B,3]

        if pred_t is None:
            t_pred_global = gt_t.detach()
        else:
            if pred_t.dim() == 2 and pred_t.shape == (B, 3):
                t_pred_global = pred_t
            else:
                # assume per-point votes [B,N,3]
                if pred_c is not None:
                    conf = pred_c.squeeze(2).clamp(min=1e-6)  # [B,N]
                    conf_norm = conf / (conf.sum(dim=1, keepdim=True) + 1e-6)
                    t_pred_global = (pred_t * conf_norm.unsqueeze(-1)).sum(dim=1)  # [B,3]
                else:
                    t_pred_global = pred_t.mean(dim=1)

        diff_t = torch.abs(t_pred_global - gt_t)
        loss_t = (diff_t * self.axis_weight.to(diff_t.dtype)).mean()

        # -------------------------
        # 3) Keypoint loss (Asym only) + KP error per point for conf target
        # -------------------------
        loss_kp = torch.tensor(0.0, device=device, dtype=points.dtype)
        kp_error_per_point = None  # [B,N]

        if self.lambda_kp > 0 and nonsym_mask.any() and pred_kp is not None:
            kp_gt = batch["kp_targ_ofst"]  # [B,K,N,3] or [B,N,K,3]
            if kp_gt.dim() == 4 and kp_gt.shape[1] == N:
                kp_gt = kp_gt.permute(0, 2, 1, 3).contiguous()  # -> [B,K,N,3]

            loss_kp = self._compute_kp_loss(
                pred=pred_kp, target=kp_gt, labels=labels, valid_mask=nonsym_mask
            ).to(points.dtype)

            with torch.no_grad():
                # [B,K,N,3] -> [B,N] (mean over K)
                kp_err = torch.norm(pred_kp - kp_gt, dim=3).mean(dim=1)

                if labels is not None:
                    labels_ = labels.squeeze(-1) if labels.dim() == 3 else labels
                    kp_err = kp_err * (labels_ > 1e-8).float()

                kp_error_per_point = kp_err

        # -------------------------
        # 4) Dense/ADD-S (Sym only) + Dense error per point for conf target
        # -------------------------
        loss_dense = torch.tensor(0.0, device=device, dtype=points.dtype)
        dense_error_per_point = None  # [B,N]

        need_dense = sym_mask.any() and (self.lambda_dense > 0 or self.lambda_conf > 0)
        if need_dense and pred_q is not None:
            # Prepare model-space points via GT
            p_center = points - gt_t.unsqueeze(1)  # [B,N,3]

            gt_R = gt_pose[:, :3, :3]  # [B,3,3]
            p_model = torch.matmul(p_center, gt_R)

            # Pred rotation (per-point or global)
            if pred_q.dim() == 2 and pred_q.shape == (B, 4):
                pred_q_use = pred_q.unsqueeze(1).expand(B, N, 4)
            else:
                pred_q_use = pred_q

            pred_R = self.geometry.quat_to_rot(pred_q_use.reshape(-1, 4)).view(B, N, 3, 3)

            # Use fused global translation
            p_pred = (pred_R @ p_model.unsqueeze(-1)).squeeze(-1) + t_pred_global.unsqueeze(1)  # [B,N,3]

            # Base dist (no inplace)
            dist_base = torch.norm(p_pred - points, dim=2)  # [B,N]
            final_dist = dist_base.clone()                  # [Fix] clone for modification

            if sym_mask.any():
                s_idx = torch.where(sym_mask)[0]
                S = min(self.dense_sample_num, N)

                for bi in s_idx.tolist():
                    if S < N:
                        idx = torch.randperm(N, device=device)[:S]
                        tgt = points[bi, idx, :].unsqueeze(0)  # [1,S,3]
                    else:
                        tgt = points[bi].unsqueeze(0)          # [1,N,3]

                    d_mat = torch.cdist(p_pred[bi:bi+1], tgt)  # [1,N,S] or [1,N,N]
                    min_d, _ = d_mat.min(dim=2)               # [1,N]
                    final_dist[bi] = min_d.squeeze(0)

            dense_error_per_point = final_dist

            # Dense loss (only symmetric samples)
            if self.lambda_dense > 0 and sym_mask.any():
                if pred_c is not None:
                    c_clamp = pred_c.squeeze(2).clamp(min=1e-4, max=1.0)
                else:
                    c_clamp = torch.ones((B, N), device=device, dtype=final_dist.dtype)

                dense_term = final_dist * c_clamp - self.w_rate * torch.log(c_clamp)
                mask_bc = sym_mask.view(B, 1).float()
                denom = mask_bc.sum() * N + 1e-6
                loss_dense = (dense_term * mask_bc).sum() / denom
                loss_dense = loss_dense.to(points.dtype)

        # -------------------------
        # 5) Confidence loss (true decoupling + valid mask)
        # -------------------------
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
                labels_ = labels.squeeze(-1) if labels.dim() == 3 else labels
                valid_pts = valid_pts & (labels_ > 1e-8)

            if valid_pts.any():
                with torch.no_grad():
                    d_clip = final_error.detach().clamp(max=self.conf_dist_max)
                    target_c = torch.exp(-self.conf_alpha * d_clip)  # [B,N]

                pred_c_map = pred_c.squeeze(2)
                loss_conf = F.mse_loss(pred_c_map[valid_pts], target_c[valid_pts]).to(points.dtype)

        # -------------------------
        # 6) [New] Segmentation loss (optional)
        # -------------------------
        loss_seg = torch.tensor(0.0, device=device, dtype=points.dtype)

        if self.lambda_seg > 0 and pred_mask_logits is not None:
            gt_mask = batch.get("mask", None)
            if gt_mask is None:
                raise ValueError("[LGFFLoss_SEG] Seg supervision requires batch['mask']; got None.")

            # normalize/shape to [B,1,H,W] float in {0,1}
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            gt_mask = gt_mask.float()
            if gt_mask.max() > 1.0:
                gt_mask = (gt_mask > 0).float()

            # Ensure spatial match (robust)
            if gt_mask.shape[-2:] != pred_mask_logits.shape[-2:]:
                gt_mask = F.interpolate(gt_mask, size=pred_mask_logits.shape[-2:], mode="nearest")

            mask_valid = batch.get("mask_valid", None)
            loss_seg = self._seg_loss(pred_mask_logits, gt_mask, mask_valid=mask_valid).to(points.dtype)

        # -------------------------
        # 7) Total
        # -------------------------
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
