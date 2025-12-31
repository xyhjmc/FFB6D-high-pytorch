# lgff/utils/pose_metrics.py
from __future__ import annotations
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def _resolve_use_best_point(cfg, stage: str) -> bool:
    stage_flag = getattr(cfg, f"{stage}_use_best_point", None)
    if stage_flag is not None:
        return bool(stage_flag)
    return getattr(cfg, "eval_use_best_point", True)


def _resolve_topk(cfg, N: int) -> int:
    topk_cfg = getattr(cfg, "pose_fusion_topk", None)
    if topk_cfg is not None and topk_cfg > 0:
        return max(1, min(int(topk_cfg), N))
    return max(1, min(max(32, N // 4), N))


def _as_batched_per_point(x: torch.Tensor, B: int, N: int, last_dim: int) -> torch.Tensor:
    """
    Normalize tensor to [B,N,last_dim] if possible.
    Accepts:
      - [B,N,last_dim]
      - [B,last_dim]  -> expand to [B,N,last_dim]
    """
    if x.dim() == 3 and x.shape[0] == B and x.shape[1] == N and x.shape[2] == last_dim:
        return x
    if x.dim() == 2 and x.shape[0] == B and x.shape[1] == last_dim:
        return x.unsqueeze(1).expand(B, N, last_dim)
    raise ValueError(f"Cannot reshape to [B,N,{last_dim}]. Got {tuple(x.shape)}")


def fuse_pose_from_outputs(
    outputs: Dict[str, torch.Tensor],
    geometry,
    cfg,
    stage: str = "eval",
    valid_mask: Optional[torch.Tensor] = None,   # [B,N] bool/int; optional
) -> torch.Tensor:
    """
    Fuse per-point predictions into per-sample pose [B,3,4].

    Robustness:
    - Supports pred_quat: [B,N,4] or [B,4]
    - Supports pred_trans: [B,N,3] or [B,3]
    - Supports pred_conf: [B,N,1] or [B,N] or missing (fallback to uniform)
    - Optional valid_mask (labels/pred-seg) to ignore invalid points in fusion
    """
    if "pred_quat" not in outputs or outputs["pred_quat"] is None:
        raise KeyError("outputs must contain 'pred_quat'")

    pred_q_raw = outputs["pred_quat"]
    pred_t_raw = outputs.get("pred_trans", None)
    pred_c_raw = outputs.get("pred_conf", None)

    device = pred_q_raw.device
    B = int(pred_q_raw.shape[0])

    # infer N
    if pred_q_raw.dim() == 3:
        N = int(pred_q_raw.shape[1])
    elif pred_q_raw.dim() == 2:
        # global quaternion case -> treat as N=1
        N = 1
    else:
        raise ValueError(f"pred_quat dim invalid: {pred_q_raw.dim()}")

    # normalize shapes to [B,N,*]
    pred_q = _as_batched_per_point(pred_q_raw, B=B, N=N, last_dim=4)

    if pred_t_raw is None:
        # no translation head: return rotation + zero t
        pred_t = torch.zeros((B, N, 3), device=device, dtype=pred_q.dtype)
    else:
        pred_t = _as_batched_per_point(pred_t_raw, B=B, N=N, last_dim=3)

    # conf: [B,N]
    if pred_c_raw is None:
        conf = torch.ones((B, N), device=device, dtype=pred_q.dtype)
    else:
        if pred_c_raw.dim() == 3 and pred_c_raw.shape[-1] == 1:
            conf = pred_c_raw.squeeze(-1)
        elif pred_c_raw.dim() == 2 and pred_c_raw.shape == (B, N):
            conf = pred_c_raw
        else:
            # allow [B,1] or [B] -> expand
            if pred_c_raw.dim() == 2 and pred_c_raw.shape[0] == B and pred_c_raw.shape[1] == 1:
                conf = pred_c_raw.expand(B, N)
            elif pred_c_raw.dim() == 1 and pred_c_raw.shape[0] == B:
                conf = pred_c_raw.view(B, 1).expand(B, N)
            else:
                raise ValueError(f"pred_conf shape invalid: {tuple(pred_c_raw.shape)}")

    conf = conf.to(dtype=pred_q.dtype).clamp(min=0.0)

    # optional valid mask
    if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        vm = valid_mask
        if vm.dim() == 1:
            vm = vm.view(1, -1).expand(B, -1)
        if vm.dim() == 2 and vm.shape[0] == B and vm.shape[1] == N:
            conf = conf * vm.to(device=device, dtype=pred_q.dtype)
        # else: silently ignore to keep backward compatibility

    use_best_point = _resolve_use_best_point(cfg, stage)

    # normalize quats (per-point)
    pred_q = F.normalize(pred_q, dim=-1)

    if use_best_point:
        # if all conf == 0, fallback to argmax on tiny eps (stable)
        conf_eps = conf + 1e-12
        idx = torch.argmax(conf_eps, dim=1)  # [B]

        idx_expand_q = idx.view(B, 1, 1).expand(-1, 1, 4)
        idx_expand_t = idx.view(B, 1, 1).expand(-1, 1, 3)

        best_q = torch.gather(pred_q, 1, idx_expand_q).squeeze(1)  # [B,4]
        best_t = torch.gather(pred_t, 1, idx_expand_t).squeeze(1)  # [B,3]

        best_q = F.normalize(best_q, dim=-1)
        rot = geometry.quat_to_rot(best_q)  # [B,3,3]
        return torch.cat([rot, best_t.unsqueeze(-1)], dim=2)

    # -------- Top-K weighted fusion --------
    k = _resolve_topk(cfg, N)

    # if conf all zeros, use uniform weights
    conf_sum = conf.sum(dim=1, keepdim=True)
    conf_safe = torch.where(conf_sum > 0, conf, torch.ones_like(conf))

    conf_topk, idx = torch.topk(conf_safe, k=k, dim=1)
    conf_topk = conf_topk.clamp(min=1e-6)

    def _gather(t: torch.Tensor) -> torch.Tensor:
        expand_shape = idx.unsqueeze(-1).expand(-1, -1, t.size(-1))
        return torch.gather(t, dim=1, index=expand_shape)

    top_q = _gather(pred_q)  # [B,K,4]
    top_t = _gather(pred_t)  # [B,K,3]

    # quaternion sign alignment to the first quat
    ref_q = top_q[:, :1, :]  # [B,1,4]
    dot = torch.sum(ref_q * top_q, dim=-1, keepdim=True)
    sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
    top_q = top_q * sign

    weights = conf_topk / conf_topk.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,K]

    # Markley-style weighted average via eigenvector of covariance
    quat_cov = torch.einsum("bk,bki,bkj->bij", weights, top_q, top_q)  # [B,4,4]
    eigvals, eigvecs = torch.linalg.eigh(quat_cov)
    fused_q = eigvecs[..., -1]  # [B,4]
    fused_q = F.normalize(fused_q, dim=-1)

    # align fused sign to ref
    ref0 = ref_q.squeeze(1)  # [B,4]
    dot2 = torch.sum(fused_q * ref0, dim=-1, keepdim=True)
    fused_q = fused_q * torch.where(dot2 >= 0, torch.ones_like(dot2), -torch.ones_like(dot2))

    fused_t = torch.sum(top_t * weights.unsqueeze(-1), dim=1)  # [B,3]

    rot = geometry.quat_to_rot(fused_q)
    return torch.cat([rot, fused_t.unsqueeze(-1)], dim=2)


def _maybe_sample_model_points(
    model_points: torch.Tensor, cfg, stage: str = "eval"
) -> torch.Tensor:
    """
    Optional speed knob for ADD/ADD-S: sample model points if too many.
    cfg keys:
      - eval_model_point_sample_num (int, default 0 -> disable)
      - eval_model_point_sample_seed (int, default 0)
    """
    sample_num = int(getattr(cfg, f"{stage}_model_point_sample_num", getattr(cfg, "eval_model_point_sample_num", 0)) or 0)
    if sample_num <= 0:
        return model_points

    B, M, _ = model_points.shape
    if M <= sample_num:
        return model_points

    seed = int(getattr(cfg, "eval_model_point_sample_seed", 0))
    g = torch.Generator(device=model_points.device)
    g.manual_seed(seed)

    idx = torch.randperm(M, generator=g, device=model_points.device)[:sample_num]
    return model_points[:, idx, :]


def compute_batch_pose_metrics(
    pred_rt: torch.Tensor,          # [B,3,4]
    gt_rt: torch.Tensor,            # [B,3,4]
    model_points: torch.Tensor,     # [B,M,3]
    cls_ids: Optional[torch.Tensor],
    geometry,
    cfg,
) -> Dict[str, torch.Tensor]:
    """
    Compute ADD / ADD-S / t_err / rot_err / cmd_acc for a batch.
    Returns 1D tensors on CPU, shape [B].
    """
    device = pred_rt.device

    if model_points.dim() == 2:
        model_points = model_points.unsqueeze(0).expand(pred_rt.shape[0], -1, -1)
    model_points = model_points.to(device)

    # optional sampling for speed
    model_points = _maybe_sample_model_points(model_points, cfg, stage="eval")

    gt_r = gt_rt[:, :3, :3]
    gt_t = gt_rt[:, :3, 3]
    pred_r = pred_rt[:, :3, :3]
    pred_t = pred_rt[:, :3, 3]

    # --- ADD / ADD-S (prefer GeometryToolkit if available) ---
    add_dist: torch.Tensor
    adds_dist: torch.Tensor

    if hasattr(geometry, "compute_add") and hasattr(geometry, "compute_adds"):
        add_dist = geometry.compute_add(pred_rt, gt_rt, model_points)       # [B]
        adds_dist = geometry.compute_adds(pred_rt, gt_rt, model_points)     # [B]
    else:
        # fallback (your original)
        points_gt = torch.bmm(model_points, gt_r.transpose(1, 2)) + gt_t.unsqueeze(1)
        points_pred = torch.bmm(model_points, pred_r.transpose(1, 2)) + pred_t.unsqueeze(1)
        add_dist = torch.norm(points_pred - points_gt, dim=2).mean(dim=1)

        adds_list: List[torch.Tensor] = []
        for b in range(points_pred.size(0)):
            dist_mat = torch.cdist(points_pred[b], points_gt[b])
            adds_list.append(dist_mat.min(dim=1).values.mean())
        adds_dist = torch.stack(adds_list).to(device)

    # --- symmetry mask ---
    sym_ids = torch.as_tensor(getattr(cfg, "sym_class_ids", []), device=device)
    sym_mask = torch.zeros(pred_rt.size(0), dtype=torch.bool, device=device)
    if cls_ids is not None:
        cls_tensor = cls_ids
        if isinstance(cls_tensor, torch.Tensor):
            if cls_tensor.dim() > 1:
                cls_tensor = cls_tensor.view(-1)
            cls_tensor = cls_tensor.to(device)
            if sym_ids.numel() > 0:
                sym_mask = (cls_tensor.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)

    dist_for_acc = torch.where(sym_mask, adds_dist, add_dist)

    # --- translation error ---
    t_diff = pred_t - gt_t
    t_err = torch.norm(t_diff, dim=1)

    # --- rotation error ---
    rot_err = geometry.rotation_error_from_mats(pred_r, gt_r, return_deg=True)

    # --- CMD acc ---
    cmd_threshold_m = float(getattr(cfg, "cmd_threshold_m", 0.02))
    cmd_acc = (dist_for_acc < cmd_threshold_m).float()

    out = {
        "add": add_dist.detach().cpu(),
        "add_s": adds_dist.detach().cpu(),
        "t_err": t_err.detach().cpu(),
        "t_err_x": t_diff[:, 0].detach().cpu(),
        "t_err_y": t_diff[:, 1].detach().cpu(),
        "t_err_z": t_diff[:, 2].detach().cpu(),
        "rot_err": rot_err.detach().cpu(),
        "cmd_acc": cmd_acc.detach().cpu(),
    }
    return out


def summarize_pose_metrics(
    meter: Dict[str, List[float]],
    obj_diameter: float,
    cmd_threshold_m: float,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}

    for k, values in meter.items():
        summary[f"mean_{k}"] = float(np.mean(values)) if len(values) > 0 else 0.0

    for name in ["add_s", "add", "t_err", "rot_err"]:
        if name in meter and len(meter[name]) > 0:
            arr = np.array(meter[name], dtype=np.float32)
            if name != "rot_err":
                summary[f"{name}_p50"] = float(np.percentile(arr, 50))
                summary[f"{name}_p75"] = float(np.percentile(arr, 75))
                summary[f"{name}_p90"] = float(np.percentile(arr, 90))
                summary[f"{name}_p95"] = float(np.percentile(arr, 95))
            else:
                summary["rot_err_deg_p50"] = float(np.percentile(arr, 50))
                summary["rot_err_deg_p75"] = float(np.percentile(arr, 75))
                summary["rot_err_deg_p90"] = float(np.percentile(arr, 90))
                summary["rot_err_deg_p95"] = float(np.percentile(arr, 95))

    if "add_s" in meter and len(meter["add_s"]) > 0:
        adds = np.array(meter["add_s"], dtype=np.float32)
        summary["acc_adds<5mm"] = float((adds < 0.005).mean())
        summary["acc_adds<10mm"] = float((adds < 0.010).mean())
        summary["acc_adds<15mm"] = float((adds < 0.015).mean())
        summary["acc_adds<20mm"] = float((adds < 0.020).mean())
        summary["acc_adds<30mm"] = float((adds < 0.030).mean())

        if obj_diameter > 0:
            summary["acc_adds<0.020d"] = float((adds < 0.020 * obj_diameter).mean())
            summary["acc_adds<0.050d"] = float((adds < 0.050 * obj_diameter).mean())
            acc_010d = float((adds < 0.100 * obj_diameter).mean())
            summary["acc_adds<0.100d"] = acc_010d
            summary["acc_adds_0.1d"] = acc_010d

    if "t_err" in meter and len(meter["t_err"]) > 0:
        te = np.array(meter["t_err"], dtype=np.float32)
        summary["acc_t<10mm"] = float((te < 0.010).mean())
        summary["acc_t<20mm"] = float((te < 0.020).mean())
        summary["acc_t<30mm"] = float((te < 0.030).mean())

    summary["obj_diameter"] = float(obj_diameter)
    summary["cmd_threshold_m"] = float(cmd_threshold_m)
    return summary
