# lgff/utils/pose_metrics_seg.py
from __future__ import annotations
from typing import Dict, Optional, List

import torch
import torch.nn.functional as F
import numpy as np


def _resolve_use_best_point(cfg, stage: str) -> bool:
    """Resolve whether to use best-point or confidence fusion for a stage."""
    # stage-aware override
    stage_flag = getattr(cfg, f"{stage}_use_best_point", None)
    if stage_flag is not None:
        return bool(stage_flag)
    return getattr(cfg, "eval_use_best_point", True)


def _resolve_topk(cfg, N: int) -> int:
    """Resolve Top-K used for weighted fusion."""
    topk_cfg = getattr(cfg, "pose_fusion_topk", None)
    if topk_cfg is not None and topk_cfg > 0:
        return min(int(topk_cfg), N)
    return min(max(32, N // 4), N)


def fuse_pose_from_outputs(
        outputs: Dict[str, torch.Tensor],
        geometry,
        cfg,
        stage: str = "eval",
        valid_mask: Optional[torch.Tensor] = None,  # [NEW] 支持掩码过滤
) -> torch.Tensor:
    """
    从网络输出中融合得到姿态 [B,3,4]。
    支持 valid_mask：被掩码屏蔽的点置信度将被强制设为 0。
    """
    pred_q = outputs["pred_quat"]  # [B, N, 4]
    pred_t = outputs["pred_trans"]  # [B, N, 3]
    pred_c = outputs["pred_conf"]  # [B, N, 1]

    B, N, _ = pred_q.shape
    conf = pred_c.squeeze(-1)  # [B, N]

    # [CRITICAL] 应用掩码：将无效点的置信度抹零
    if valid_mask is not None:
        # 确保类型匹配
        mask_float = valid_mask.to(dtype=conf.dtype, device=conf.device)
        conf = conf * mask_float

    use_best_point = _resolve_use_best_point(cfg, stage)

    if use_best_point:
        # 如果全被 mask 掉，conf 全 0，argmax 仍会返回索引 0 (此时结果不可靠，但程序不会崩)
        idx = torch.argmax(conf, dim=1)  # [B]
        idx_expand_q = idx.view(B, 1, 1).expand(-1, 1, 4)
        idx_expand_t = idx.view(B, 1, 1).expand(-1, 1, 3)

        best_q = torch.gather(pred_q, 1, idx_expand_q).squeeze(1)
        best_t = torch.gather(pred_t, 1, idx_expand_t).squeeze(1)

        best_q = F.normalize(best_q, dim=-1)
        rot = geometry.quat_to_rot(best_q)
        pred_rt = torch.cat([rot, best_t.unsqueeze(-1)], dim=2)
        return pred_rt

    # -------- Top-K 融合 --------
    k = _resolve_topk(cfg, N)

    # 选取置信度最高的 K 个点
    conf_topk, idx = torch.topk(conf, k=k, dim=1)

    # 防止除以零
    conf_topk = conf_topk.clamp(min=1e-6)

    def _gather(t: torch.Tensor) -> torch.Tensor:
        expand_shape = idx.unsqueeze(-1).expand(-1, -1, t.size(-1))
        return torch.gather(t, dim=1, index=expand_shape)

    top_q = _gather(pred_q)
    top_t = _gather(pred_t)

    top_q = F.normalize(top_q, dim=-1)
    # 解决四元数双倍覆盖问题 (q == -q)
    ref_q = top_q[:, :1, :]
    dot = torch.sum(ref_q * top_q, dim=-1, keepdim=True)
    sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
    top_q = top_q * sign

    # 加权平均
    weights = conf_topk / conf_topk.sum(dim=1, keepdim=True)

    # 旋转：特征向量法求平均
    quat_cov = torch.einsum("bki,bkj->bij", weights.unsqueeze(-1) * top_q, top_q)
    eigvals, eigvecs = torch.linalg.eigh(quat_cov)
    fused_q = eigvecs[..., -1]

    # 平移：加权求和
    fused_t = torch.sum(top_t * weights.unsqueeze(-1), dim=1)

    rot = geometry.quat_to_rot(fused_q)
    pred_rt = torch.cat([rot, fused_t.unsqueeze(-1)], dim=2)
    return pred_rt


def compute_batch_pose_metrics(
        pred_rt: torch.Tensor,
        gt_rt: torch.Tensor,
        model_points: torch.Tensor,
        cls_ids: Optional[torch.Tensor],
        geometry,
        cfg,
) -> Dict[str, torch.Tensor]:
    """
    计算 Batch 指标 (ADD/ADD-S/Err)，直接复制自原版以保持独立性。
    """
    device = pred_rt.device
    model_points = model_points.to(device)

    gt_r = gt_rt[:, :3, :3]
    gt_t = gt_rt[:, :3, 3]
    pred_r = pred_rt[:, :3, :3]
    pred_t = pred_rt[:, :3, 3]

    points_gt = torch.bmm(model_points, gt_r.transpose(1, 2)) + gt_t.unsqueeze(1)
    points_pred = torch.bmm(model_points, pred_r.transpose(1, 2)) + pred_t.unsqueeze(1)

    # ADD
    add_dist = torch.norm(points_pred - points_gt, dim=2).mean(dim=1)

    # ADD-S
    adds_list: List[torch.Tensor] = []
    for b in range(points_pred.size(0)):
        dist_mat = torch.cdist(points_pred[b], points_gt[b])
        adds_list.append(dist_mat.min(dim=1).values.mean())
    adds_dist = torch.stack(adds_list).to(device)

    # Symmetry handling
    sym_ids = torch.as_tensor(getattr(cfg, "sym_class_ids", []), device=device)
    sym_mask = torch.zeros(pred_rt.size(0), dtype=torch.bool, device=device)
    if cls_ids is not None:
        cls_tensor = cls_ids.view(-1).to(device)
        if sym_ids.numel() > 0:
            sym_mask = (cls_tensor.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)

    dist_for_acc = torch.where(sym_mask, adds_dist, add_dist)

    t_diff = pred_t - gt_t
    t_err = torch.norm(t_diff, dim=1)
    rot_err = geometry.rotation_error_from_mats(pred_r, gt_r, return_deg=True)

    cmd_threshold_m = getattr(cfg, "cmd_threshold_m", 0.02)
    cmd_acc = (dist_for_acc < cmd_threshold_m).float()

    return {
        "add": add_dist.detach().cpu(),
        "add_s": adds_dist.detach().cpu(),
        "t_err": t_err.detach().cpu(),
        "t_err_x": t_diff[:, 0].detach().cpu(),
        "t_err_y": t_diff[:, 1].detach().cpu(),
        "t_err_z": t_diff[:, 2].detach().cpu(),
        "rot_err": rot_err.detach().cpu(),
        "cmd_acc": cmd_acc.detach().cpu(),
    }


def summarize_pose_metrics(
        meter: Dict[str, List[float]],
        obj_diameter: float,
        cmd_threshold_m: float,
) -> Dict[str, float]:
    """汇总指标统计"""
    summary: Dict[str, float] = {}
    for k, values in meter.items():
        if len(values) > 0:
            summary[f"mean_{k}"] = float(np.mean(values))
        else:
            summary[f"mean_{k}"] = 0.0

    if "add_s" in meter and len(meter["add_s"]) > 0:
        adds = np.array(meter["add_s"], dtype=np.float32)
        summary["acc_adds<1cm"] = float((adds < 0.010).mean())
        if obj_diameter > 0:
            summary["acc_adds<0.1d"] = float((adds < 0.1 * obj_diameter).mean())

    summary["obj_diameter"] = obj_diameter
    return summary