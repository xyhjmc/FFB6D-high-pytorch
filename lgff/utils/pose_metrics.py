# lgff/utils/pose_metrics.py
from __future__ import annotations
from typing import Dict, Optional, List

import torch
import torch.nn.functional as F
import numpy as np


def _resolve_use_best_point(cfg, stage: str) -> bool:
    """Resolve whether to use best-point or confidence fusion for a stage."""
    # stage-aware override, e.g., train_use_best_point / eval_use_best_point / viz_use_best_point
    stage_flag = getattr(cfg, f"{stage}_use_best_point", None)
    if stage_flag is not None:
        return bool(stage_flag)
    # backward compatibility
    return getattr(cfg, "eval_use_best_point", True)


def _resolve_topk(cfg, N: int, stage: str) -> int:
    """Resolve Top-K used for weighted fusion (meters, camera frame)."""
    stage_topk = getattr(cfg, f"{stage}_topk", None)
    if stage_topk is not None and stage_topk > 0:
        return min(int(stage_topk), N)

    topk_cfg = getattr(cfg, "pose_fusion_topk", None)
    if topk_cfg is not None and topk_cfg > 0:
        return min(int(topk_cfg), N)
    return min(max(32, N // 4), N)


def fuse_pose_from_outputs(
    outputs: Dict[str, torch.Tensor],
    geometry,
    cfg,
    stage: str = "eval",
) -> torch.Tensor:
    """
    从网络输出的 dense 点级结果中，得到每个样本的 [R|t] 姿态矩阵 [B,3,4]。
    逻辑直接复用 EvaluatorSC._process_predictions。
    """
    pred_q = outputs["pred_quat"]   # [B, N, 4]
    pred_t = outputs["pred_trans"]  # [B, N, 3]
    pred_c = outputs["pred_conf"]   # [B, N, 1]

    B, N, _ = pred_q.shape
    conf = pred_c.squeeze(-1)  # [B, N]

    use_best_point = _resolve_use_best_point(cfg, stage)

    if use_best_point:
        idx = torch.argmax(conf, dim=1)  # [B]
        idx_expand_q = idx.view(B, 1, 1).expand(-1, 1, 4)
        idx_expand_t = idx.view(B, 1, 1).expand(-1, 1, 3)

        best_q = torch.gather(pred_q, 1, idx_expand_q).squeeze(1)  # [B, 4]
        best_t = torch.gather(pred_t, 1, idx_expand_t).squeeze(1)  # [B, 3]

        best_q = F.normalize(best_q, dim=-1)
        rot = geometry.quat_to_rot(best_q)  # [B, 3, 3]
        pred_rt = torch.cat([rot, best_t.unsqueeze(-1)], dim=2)  # [B, 3, 4]
        return pred_rt

    # -------- Top-K + 置信度加权融合 --------
    k = _resolve_topk(cfg, N, stage)
    conf_topk, idx = torch.topk(conf, k=k, dim=1)
    conf_topk = conf_topk.clamp(min=1e-4)

    def _gather(t: torch.Tensor) -> torch.Tensor:
        expand_shape = idx.unsqueeze(-1).expand(-1, -1, t.size(-1))
        return torch.gather(t, dim=1, index=expand_shape)

    top_q = _gather(pred_q)  # [B, K, 4]
    top_t = _gather(pred_t)  # [B, K, 3]

    top_q = F.normalize(top_q, dim=-1)
    ref_q = top_q[:, :1, :]
    dot = torch.sum(ref_q * top_q, dim=-1, keepdim=True)
    sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
    top_q = top_q * sign

    weights = conf_topk / conf_topk.sum(dim=1, keepdim=True).clamp(min=1e-6)
    quat_cov = torch.einsum("bki,bkj->bij", weights.unsqueeze(-1) * top_q, top_q)
    eigvals, eigvecs = torch.linalg.eigh(quat_cov)
    fused_q = eigvecs[..., -1]

    fused_t = torch.sum(top_t * weights.unsqueeze(-1), dim=1)

    rot = geometry.quat_to_rot(fused_q)
    pred_rt = torch.cat([rot, fused_t.unsqueeze(-1)], dim=2)
    return pred_rt


def compute_batch_pose_metrics(
    pred_rt: torch.Tensor,  # [B,3,4]
    gt_rt: torch.Tensor,  # [B,3,4]
    model_points: torch.Tensor,  # [B,M,3]
    cls_ids: Optional[torch.Tensor],
    geometry,
    cfg,
) -> Dict[str, torch.Tensor]:
    """
    一次性对一个 batch 计算 ADD / ADD-S / t_err / rot_err 等指标。
    返回 shape=[B] 的 tensor（在 CPU 上），方便后面自己决定怎么汇总。
    """
    device = pred_rt.device

    model_points = model_points.to(device)  # [B,M,3]

    gt_r = gt_rt[:, :3, :3]
    gt_t = gt_rt[:, :3, 3]

    pred_r = pred_rt[:, :3, :3]
    pred_t = pred_rt[:, :3, 3]

    # 1) 投影 CAD 点云
    points_gt   = torch.bmm(model_points, gt_r.transpose(1, 2))   + gt_t.unsqueeze(1)
    points_pred = torch.bmm(model_points, pred_r.transpose(1, 2)) + pred_t.unsqueeze(1)

    # ADD
    add_dist = torch.norm(points_pred - points_gt, dim=2).mean(dim=1)  # [B]

    # ADD-S: 最近邻 (per-sample 以支持多物体扩展)
    adds_list: List[torch.Tensor] = []
    for b in range(points_pred.size(0)):
        dist_mat = torch.cdist(points_pred[b], points_gt[b])  # [M,M]
        adds_list.append(dist_mat.min(dim=1).values.mean())
    adds_dist = torch.stack(adds_list).to(device)  # [B]

    # 是否对称（逐样本判断，兼容未来多物体/混合 batch）
    sym_ids = torch.as_tensor(getattr(cfg, "sym_class_ids", []), device=device)
    sym_mask = torch.zeros(pred_rt.size(0), dtype=torch.bool, device=device)
    if cls_ids is not None:
        cls_tensor = cls_ids
        if cls_tensor.dim() > 1:
            cls_tensor = cls_tensor.view(-1)
        cls_tensor = cls_tensor.to(device)
        if sym_ids.numel() > 0:
            sym_mask = (cls_tensor.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)

    dist_for_acc = torch.where(sym_mask, adds_dist, add_dist)

    # 2) translation error
    t_diff = pred_t - gt_t  # [B,3]
    t_err = torch.norm(t_diff, dim=1)  # [B]

    # 3) rotation error (deg)
    rot_err_deg = geometry.rotation_error_from_mats(
        pred_r, gt_r, return_deg=True
    )  # [B]
    # sym objects: ignore rotation by zeroing and tracking count
    rot_sym_ignored = sym_mask.detach().cpu()
    rot_err_deg = torch.where(sym_mask, torch.zeros_like(rot_err_deg), rot_err_deg)

    # 4) CMD / <2cm 准确率（按 BOP 常用）
    cmd_threshold_m = getattr(cfg, "cmd_threshold_m", 0.02)
    cmd_acc = (dist_for_acc < cmd_threshold_m).float()  # [B]

    # 整理成 dict，并搬到 CPU（方便 numpy 处理）
    out = {
        "add": add_dist.detach().cpu(),
        "add_s": adds_dist.detach().cpu(),
        "t_err": t_err.detach().cpu(),
        "t_err_x": t_diff[:, 0].detach().cpu(),
        "t_err_y": t_diff[:, 1].detach().cpu(),
        "t_err_z": t_diff[:, 2].detach().cpu(),
        "rot_err_deg": rot_err_deg.detach().cpu(),
        # backward compatibility (deg)
        "rot_err": rot_err_deg.detach().cpu(),
        "cmd_acc": cmd_acc.detach().cpu(),
        "sym_rot_ignored_mask": rot_sym_ignored,
    }
    out["rot_err_deg_valid"] = rot_err_deg[~sym_mask].detach().cpu()
    return out


def summarize_pose_metrics(
    meter: Dict[str, List[float]],
    obj_diameter: float,
    cmd_threshold_m: float,
) -> Dict[str, float]:
    """
    给定一个 epoch / 全测试集累积的指标列表，计算 mean, 各分位数，acc<阈值 等。
    这部分逻辑可以直接照你 Evaluator 里现有的 _summarize_metrics / 新增指标来填。
    """
    summary: Dict[str, float] = {}

    # 均值
    for k, values in meter.items():
        if len(values) == 0:
            summary[f"mean_{k}"] = 0.0
        else:
            summary[f"mean_{k}"] = float(np.mean(values))

    # 分位数 / 阈值准确率，示意：
    for name in ["add_s", "add", "t_err", "rot_err_deg", "rot_err", "rot_err_deg_valid"]:
        if name in meter and len(meter[name]) > 0:
            arr = np.array(meter[name], dtype=np.float32)
            if name not in ["rot_err", "rot_err_deg", "rot_err_deg_valid"]:
                summary[f"{name}_p50"] = float(np.percentile(arr, 50))
                summary[f"{name}_p75"] = float(np.percentile(arr, 75))
                summary[f"{name}_p90"] = float(np.percentile(arr, 90))
                summary[f"{name}_p95"] = float(np.percentile(arr, 95))
            else:
                summary["rot_err_deg_p50"] = float(np.percentile(arr, 50))
                summary["rot_err_deg_p75"] = float(np.percentile(arr, 75))
                summary["rot_err_deg_p90"] = float(np.percentile(arr, 90))
                summary["rot_err_deg_p95"] = float(np.percentile(arr, 95))

    # 准确率阈值示意（你可以直接搬你 Evaluator 现有实现）
    if "add_s" in meter and len(meter["add_s"]) > 0:
        adds = np.array(meter["add_s"], dtype=np.float32)
        summary["acc_adds<5mm"]  = float((adds < 0.005).mean())
        summary["acc_adds<10mm"] = float((adds < 0.010).mean())
        summary["acc_adds<15mm"] = float((adds < 0.015).mean())
        summary["acc_adds<20mm"] = float((adds < 0.020).mean())
        summary["acc_adds<30mm"] = float((adds < 0.030).mean())

        if obj_diameter > 0:
            summary["acc_adds<0.020d"] = float((adds < 0.020 * obj_diameter).mean())
            summary["acc_adds<0.050d"] = float((adds < 0.050 * obj_diameter).mean())
            acc_010d = float((adds < 0.100 * obj_diameter).mean())
            summary["acc_adds<0.100d"] = acc_010d
            summary["acc_adds_0.1d"]   = acc_010d

    if "t_err" in meter and len(meter["t_err"]) > 0:
        te = np.array(meter["t_err"], dtype=np.float32)
        summary["acc_t<10mm"] = float((te < 0.010).mean())
        summary["acc_t<20mm"] = float((te < 0.020).mean())
        summary["acc_t<30mm"] = float((te < 0.030).mean())

    summary["obj_diameter"]  = obj_diameter
    summary["cmd_threshold_m"] = cmd_threshold_m

    return summary
