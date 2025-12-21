from __future__ import annotations

import numpy as np
import torch
from typing import Any, Iterable, Optional, Tuple

from lgff.utils.geometry import GeometryToolkit


def safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x.mean())


def ensure_obj_diameter(
    model_points: torch.Tensor,
    current: Optional[float],
    logger: Optional[Any] = None,
) -> Optional[float]:
    """Compute object diameter if missing. Returns updated diameter."""
    if current is not None:
        return current
    if model_points.ndim != 3 or model_points.shape[1] <= 1:
        return 0.0

    mp0 = model_points[0]
    dist_mat = torch.cdist(mp0.unsqueeze(0), mp0.unsqueeze(0)).squeeze(0)
    diameter = float(dist_mat.max().item())
    if logger is not None:
        logger.info(f"[EvalUtils] Estimated obj_diameter from CAD: {diameter:.6f} m")
    return diameter


def sym_mask_from_cls_ids(
    cls_ids: Optional[torch.Tensor],
    sym_class_ids: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    sym_ids = torch.as_tensor(list(sym_class_ids), device=device)
    sym_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if isinstance(cls_ids, torch.Tensor) and sym_ids.numel() > 0:
        cid = cls_ids
        if cid.dim() > 1:
            cid = cid.view(-1)
        cid = cid.to(device)
        sym_mask = (cid.unsqueeze(1) == sym_ids.view(1, -1)).any(dim=1)
    return sym_mask


def filter_obs_points(
    pts: torch.Tensor,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    mad_k: float = 0.0,
) -> torch.Tensor:
    """Filter observed points for ICP."""
    if pts.numel() == 0:
        return pts

    valid = torch.isfinite(pts).all(dim=1) & (pts.abs().sum(dim=1) > 1e-9)
    pts = pts[valid]
    if pts.shape[0] == 0:
        return pts

    if z_min is not None:
        pts = pts[pts[:, 2] >= float(z_min)]
    if z_max is not None:
        pts = pts[pts[:, 2] <= float(z_max)]
    if pts.shape[0] == 0:
        return pts

    if mad_k and mad_k > 0:
        center = pts.median(dim=0).values
        r = torch.norm(pts - center.unsqueeze(0), dim=1)
        med = r.median()
        mad = (r - med).abs().median().clamp_min(1e-6)
        keep = (r - med).abs() <= (mad_k * mad)
        pts = pts[keep]

    return pts


def kabsch_umeyama(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert P.ndim == 2 and Q.ndim == 2 and P.shape == Q.shape and P.shape[1] == 3

    mu_P = P.mean(dim=0)
    mu_Q = Q.mean(dim=0)
    X = P - mu_P
    Y = Q - mu_Q

    H = X.t() @ Y
    U, _, Vt = torch.linalg.svd(H, full_matrices=False)
    V = Vt.transpose(0, 1)

    R = V @ U.transpose(0, 1)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.transpose(0, 1)

    t = mu_Q - (R @ mu_P)
    return R, t


def run_icp_once(
    rt_init: torch.Tensor,          # [3,4] model->cam
    obs_points: torch.Tensor,       # [N,3] cam
    model_points: torch.Tensor,     # [M,3] model
    iters: int,
    max_corr_dist: float,
    trim_ratio: float,
    sample_model: int,
    sample_obs: int,
    min_corr: int,
) -> torch.Tensor:
    device = rt_init.device
    dtype = rt_init.dtype

    if obs_points is None or model_points is None:
        return rt_init
    if obs_points.ndim != 2 or obs_points.shape[1] != 3:
        return rt_init
    if model_points.ndim != 2 or model_points.shape[1] != 3:
        return rt_init

    obs = obs_points.to(device=device, dtype=dtype)
    mp = model_points.to(device=device, dtype=dtype)

    if obs.shape[0] < min_corr or mp.shape[0] < 3:
        return rt_init

    if obs.shape[0] > sample_obs:
        idx = torch.randperm(obs.shape[0], device=device)[:sample_obs]
        obs = obs[idx]

    if mp.shape[0] > sample_model:
        idx = torch.randperm(mp.shape[0], device=device)[:sample_model]
        mp = mp[idx]

    R = rt_init[:, :3].clone()
    t = rt_init[:, 3].clone()

    prev_mean = None

    for _ in range(int(iters)):
        P = mp @ R.transpose(0, 1) + t.unsqueeze(0)  # [m,3]

        D = torch.cdist(P.unsqueeze(0), obs.unsqueeze(0), p=2).squeeze(0)  # [m,n]
        nn_dist, nn_idx = torch.min(D, dim=1)
        Q = obs[nn_idx]

        mask = nn_dist < float(max_corr_dist)
        if mask.sum().item() < min_corr:
            break

        P_sel = P[mask]
        Q_sel = Q[mask]
        d_sel = nn_dist[mask]

        if 0.0 < trim_ratio < 1.0 and P_sel.shape[0] > min_corr:
            k = max(min_corr, int(P_sel.shape[0] * trim_ratio))
            topk = torch.topk(d_sel, k=k, largest=False).indices
            P_sel = P_sel[topk]
            Q_sel = Q_sel[topk]
            d_sel = d_sel[topk]

        mean_d = float(d_sel.mean().item())
        if prev_mean is not None and abs(prev_mean - mean_d) < 1e-5:
            pass
        prev_mean = mean_d

        dR, dt = kabsch_umeyama(P_sel, Q_sel)
        R = dR @ R
        t = dR @ t + dt

    return torch.cat([R, t.view(3, 1)], dim=1)


def compute_add_and_adds(
    geometry: GeometryToolkit,
    pred_rt: torch.Tensor,
    gt_rt: torch.Tensor,
    model_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    add = geometry.compute_add(pred_rt, gt_rt, model_points)     # [B]
    adds = geometry.compute_adds(pred_rt, gt_rt, model_points)   # [B]
    return add, adds


__all__ = [
    "safe_percentile",
    "safe_mean",
    "ensure_obj_diameter",
    "sym_mask_from_cls_ids",
    "filter_obs_points",
    "kabsch_umeyama",
    "run_icp_once",
    "compute_add_and_adds",
]
