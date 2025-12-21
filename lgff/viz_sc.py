"""
Visualization script for Single-Class LGFF inference.

This tool loads a trained checkpoint, runs forward passes on a chosen split,
selects the best-confidence / fused pose per sample (same logic as evaluator),
and renders:
  * RGB image with cube edges, axes, and predicted pose matrix
  * [NEW] Also overlays a GT cube (wireframe) for direct comparison
  * 3D scatter overlay of:
        - CAD model points under GT pose (blue)
        - CAD model points under predicted pose (orange)
        - Raw ROI depth points in camera frame (gray)

Usage examples:
    python lgff/viz_sc.py --checkpoint output/helmet_sc/checkpoint_best.pth
    python lgff/viz_sc.py --checkpoint output/helmet_sc/checkpoint_best.pth \
        --num-samples 8 --split val --show
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.getcwd())

from common.ffb6d_utils.model_complexity import ModelComplexityLogger
from lgff.utils.config import LGFFConfig, load_config, merge_cfg_from_checkpoint
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.engines.evaluator_sc import EvaluatorSC
from lgff.eval_sc import load_model_weights, resolve_checkpoint_path
from lgff.utils.pose_metrics import compute_batch_pose_metrics


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LGFF Single-Class Predictions")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to checkpoint (.pth) containing model weights",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to visualize (default: cfg.val_split or 'test')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (visualization saves per-sample)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers (default: cfg.num_workers)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override output directory for logs and figures",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to store visualization images (default: <work_dir>/viz)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to display figures interactively while saving",
    )
    parser.add_argument(
        "--per-image-csv",
        type=str,
        default=None,
        help="Optional path to per_image_metrics.csv for cross-checking metrics",
    )
    args, _ = parser.parse_known_args()
    return args


def ensure_dirs(work_dir: str, save_dir: str) -> None:
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


def load_per_image_metrics(path: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load per-image metrics CSV into a dict keyed by (scene_id, im_id)."""
    records: Dict[Tuple[int, int], Dict[str, float]] = {}
    if path is None:
        return records

    if not os.path.isfile(path):
        print(f"[viz_sc] per-image CSV not found: {path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scene = int(row.get("scene_id", -1))
                im = int(row.get("im_id", -1))
            except ValueError:
                continue
            key = (scene, im)
            # 将可解析的字段转成 float，保留原始字符串以防缺失
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            records[key] = parsed
    print(f"[viz_sc] Loaded {len(records)} per-image rows from {path}")
    return records


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def denormalize_image(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized RGB tensor [3,H,W] back to uint8 HxWx3 (RGB)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = rgb_tensor.detach().cpu().numpy()
    img = (img * std[:, None, None]) + mean[:, None, None]
    img = np.clip(img * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return img


def build_cube_primitives(
    points_model: torch.Tensor,
    scale: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """Create cube vertices/edges/axes from canonical model-frame points."""
    pts = points_model.detach().cpu().numpy()
    # 用 2% / 98% 百分位，把极少数离群点滤掉
    mins = np.percentile(pts, 2, axis=0)
    maxs = np.percentile(pts, 98, axis=0)
    center = 0.5 * (mins + maxs)
    half_extent = 0.5 * (maxs - mins)
    scaled_half = half_extent * scale

    signs = [-1.0, 1.0]
    verts = np.array(
        [
            center + scaled_half * np.array([sx, sy, sz])
            for sx in signs
            for sy in signs
            for sz in signs
        ],
        dtype=np.float32,
    )

    # 12 条边的索引
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )

    # 坐标轴：从立方体中心出发
    axis_len = float(scaled_half.max() * 1.5 + 1e-6)
    axes = np.array(
        [
            center,
            center + np.array([axis_len, 0.0, 0.0]),
            center + np.array([0.0, axis_len, 0.0]),
            center + np.array([0.0, 0.0, axis_len]),
        ],
        dtype=np.float32,
    )
    return verts, axes, edges


def project_primitives(
    verts_model: np.ndarray,
    axes_model: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    bs_utils,
) -> Tuple[np.ndarray, np.ndarray]:
    # cam_scale 固定为 1.0：所有点、位姿均在相机坐标系且单位为米
    verts_cam = np.dot(verts_model, rotation.T) + translation
    axes_cam = np.dot(axes_model, rotation.T) + translation
    verts_2d = bs_utils.project_p3d(verts_cam, 1.0, K)
    axes_2d = bs_utils.project_p3d(axes_cam, 1.0, K)
    return verts_2d, axes_2d


def draw_cube_overlay(
    rgb_np: np.ndarray,
    verts_2d: np.ndarray,
    axes_2d: np.ndarray,
    edges,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """保留原来的单 cube 画法（仅预测用），兼容旧代码."""
    img = rgb_np.copy()

    # cube edges
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_2d[i0]).astype(int))
        p1 = tuple(np.round(verts_2d[i1]).astype(int))
        img = cv2.line(img, p0, p1, color=color, thickness=2)

    # axes (X=red, Y=green, Z=blue)
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx, c in enumerate(axis_colors, start=1):
        p0 = tuple(np.round(axes_2d[0]).astype(int))
        p1 = tuple(np.round(axes_2d[idx]).astype(int))
        img = cv2.arrowedLine(img, p0, p1, color=c, thickness=2, tipLength=0.08)

    return img


def draw_two_cubes_overlay(
    rgb_np: np.ndarray,
    verts_2d_pred: np.ndarray,
    axes_2d_pred: np.ndarray,
    verts_2d_gt: np.ndarray,
    edges,
    color_pred: Tuple[int, int, int] = (0, 255, 0),
    color_gt: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    在同一张图上同时画：
      - GT cube: 仅边框，用 color_gt
      - Pred cube: 边框 + 坐标轴，用 color_pred

    方便直接比较“预测框线 vs 正确框线”。
    """
    img = rgb_np.copy()

    # 1) 先画 GT cube（避免被预测覆盖，看得更清楚）
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_2d_gt[i0]).astype(int))
        p1 = tuple(np.round(verts_2d_gt[i1]).astype(int))
        img = cv2.line(img, p0, p1, color=color_gt, thickness=2, lineType=cv2.LINE_AA)

    # 2) 再画 Pred cube
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_2d_pred[i0]).astype(int))
        p1 = tuple(np.round(verts_2d_pred[i1]).astype(int))
        img = cv2.line(img, p0, p1, color=color_pred, thickness=2, lineType=cv2.LINE_AA)

    # 3) Pred 坐标轴：X=red, Y=green, Z=blue
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx, c in enumerate(axis_colors, start=1):
        p0 = tuple(np.round(axes_2d_pred[0]).astype(int))
        p1 = tuple(np.round(axes_2d_pred[idx]).astype(int))
        img = cv2.arrowedLine(img, p0, p1, color=c, thickness=2, tipLength=0.08)

    # 4) 简单 legend（文字标注，避免线宽太乱）
    h, w = img.shape[:2]
    cv2.putText(
        img,
        "Pred Cube",
        (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color_pred,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "GT Cube",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color_gt,
        1,
        lineType=cv2.LINE_AA,
    )

    return img


def build_pose_matrix(pred_rt: torch.Tensor) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = pred_rt.detach().cpu().numpy()
    return pose


def render_sample(
    image_overlay: np.ndarray,
    points_pred_cam: torch.Tensor,
    points_gt_cam: torch.Tensor,
    points_raw_cam: torch.Tensor,
    pose_matrix: np.ndarray,
    title: str,
    save_path: str,
    show: bool = False,
) -> None:
    """
    points_pred_cam / points_gt_cam / points_raw_cam: [M,3] / [M,3] / [N,3] in camera frame.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])

    # RGB + cube
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_overlay)
    ax_img.axis("off")
    ax_img.set_title("RGB + Pred (cube+axes) & GT (cube)")

    # 3D scatter: Raw ROI (gray) + CAD-GT (blue) + CAD-Pred (orange)
    ax_scatter = fig.add_subplot(gs[0, 1], projection="3d")

    raw_np = points_raw_cam.detach().cpu().numpy()
    gt_np = points_gt_cam.detach().cpu().numpy()
    pred_np = points_pred_cam.detach().cpu().numpy()

    # raw ROI points
    ax_scatter.scatter(
        raw_np[:, 0],
        raw_np[:, 1],
        raw_np[:, 2],
        c="0.7",
        s=4,
        alpha=0.4,
        label="Raw ROI (cam)",
    )
    # CAD under GT pose
    ax_scatter.scatter(
        gt_np[:, 0],
        gt_np[:, 1],
        gt_np[:, 2],
        c="tab:blue",
        s=6,
        alpha=0.8,
        label="CAD @ GT",
    )
    # CAD under predicted pose
    ax_scatter.scatter(
        pred_np[:, 0],
        pred_np[:, 1],
        pred_np[:, 2],
        c="tab:orange",
        s=6,
        alpha=0.8,
        label="CAD @ Pred",
    )

    ax_scatter.set_xlabel("X (m)")
    ax_scatter.set_ylabel("Y (m)")
    ax_scatter.set_zlabel("Z (m)")
    ax_scatter.legend(loc="upper right")
    ax_scatter.set_title("3D Points (camera frame)")

    # Pose matrix table
    ax_pose = fig.add_subplot(gs[1, :])
    ax_pose.axis("off")
    table = ax_pose.table(
        cellText=np.round(pose_matrix, 4),
        loc="center",
        cellLoc="center",
    )
    table.scale(1.1, 1.4)
    ax_pose.set_title("Predicted homogeneous transform [R|t]")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show(block=False)
        plt.pause(0.5)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 1) 先用当前工程的配置系统载入基础 cfg（会打印 [Config] Final config saved ...）
    cfg_cli: LGFFConfig = load_config()

    # 2) 再从 checkpoint 中恢复训练时的 config（保证 backbone / 维度完全一致）
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))
    print(
        f"[viz_sc] Loaded config from checkpoint: "
        f"obj_id={getattr(cfg, 'obj_id', None)}, "
        f"dataset={getattr(cfg, 'dataset', None)}"
    )

    # 3) 工作目录和保存目录
    work_dir = args.work_dir or getattr(cfg, "work_dir", None) or getattr(cfg, "log_dir", "output")
    save_dir = args.save_dir or os.path.join(work_dir, "viz")
    ensure_dirs(work_dir, save_dir)

    # 4) 日志
    log_file = os.path.join(work_dir, "viz.log")
    setup_logger(log_file, name="lgff.viz")
    logger = get_logger("lgff.viz")
    logger.setLevel(logging.INFO)

    split = args.split or getattr(cfg, "val_split", "test")
    num_workers = args.num_workers or getattr(cfg, "num_workers", 4)
    per_image_metrics = load_per_image_metrics(args.per_image_csv)

    logger.info(
        f"Visualizing split={split} | batch_size={args.batch_size} | "
        f"num_workers={num_workers} | num_samples={args.num_samples} | "
        f"checkpoint={ckpt_path}"
    )

    if per_image_metrics:
        logger.info(
            f"Loaded per-image metrics for cross-checking: {len(per_image_metrics)} rows"
        )

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) Dataset & loader
    dataset = SingleObjectDataset(cfg, split=split)
    if len(dataset) == 0:
        logger.warning(f"[viz_sc] Dataset for split={split} is empty. Nothing to visualize.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 6) Model
    model = LGFF_SC(cfg, geometry)
    load_model_weights(model, ckpt_path, device, checkpoint=ckpt)
    model = model.to(device)
    model.eval()

    # 7) Complexity (params/FLOPs)
    complexity_logger = ModelComplexityLogger()
    try:
        example_batch = next(iter(loader))
        complexity_info = complexity_logger.maybe_log(model, example_batch, stage="viz")
        if complexity_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=viz",
                        f"Params: {complexity_info['params']:,} "
                        f"({complexity_info['param_mb']:.2f} MB)",
                        (
                            f"GFLOPs: {complexity_info['gflops']:.3f}"
                            if complexity_info.get("gflops") is not None
                            else "GFLOPs: N/A"
                        ),
                    ]
                )
            )
    except StopIteration:
        logger.warning("Visualization loader is empty; skip complexity logging.")
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")

    # 8) 重新创建 loader，避免跳过第一个 batch
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 9) 复用 EvaluatorSC 的姿态融合逻辑（只用到 _process_predictions / _process_gt）
    helper = EvaluatorSC(model=model, test_loader=loader, cfg=cfg, geometry=geometry)
    sym_class_ids = set(getattr(cfg, "sym_class_ids", []))

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            B = batch["rgb"].shape[0]

            # 模型前向 + pose 融合
            outputs = model(batch)
            pred_rt = helper._process_predictions(outputs)  # [B, 3, 4]
            gt_rt = helper._process_gt(batch)               # [B, 3, 4]

            # CAD 模型点（优先从 batch["model_points"]）
            if "model_points" in batch:
                model_points = batch["model_points"]  # [M,3] or [B,M,3]
                if model_points.dim() == 2:
                    model_points = model_points.unsqueeze(0).expand(B, -1, -1)
            else:
                # 兜底：用当前 ROI 点 + GT pose 反推到 object frame
                points_cam_all = batch["points"]  # [B, N, 3] in camera frame
                gt_r_all = gt_rt[:, :3, :3]
                gt_t_all = gt_rt[:, :3, 3]
                points_centered = points_cam_all - gt_t_all.unsqueeze(1)   # [B, N, 3]
                gt_r_inv_all = gt_r_all.transpose(1, 2)                    # [B, 3, 3]
                model_points = torch.matmul(points_centered, gt_r_inv_all)  # [B, N, 3]

            for i in range(B):
                if saved >= args.num_samples:
                    break

                # ------- 取当前样本的关键数据 ------- #
                mp_obj = model_points[i]           # [M, 3] in object frame
                pred_rt_i = pred_rt[i]             # [3, 4]
                gt_rt_i = gt_rt[i]                 # [3, 4]

                R_pred = pred_rt_i[:, :3]          # [3, 3]
                t_pred = pred_rt_i[:, 3]           # [3]
                R_gt = gt_rt_i[:, :3]              # [3, 3]
                t_gt = gt_rt_i[:, 3]               # [3]

                K = batch["intrinsic"][i].detach().cpu().numpy()

                cls_id = int(batch["cls_id"][i].item()) if "cls_id" in batch else -1
                scene_id = int(batch["scene_id"][i].item()) if "scene_id" in batch else -1
                im_id = int(batch["im_id"][i].item()) if "im_id" in batch else -1

                # ------- 用 CAD 点生成立方体 primitive ------- #
                verts_model, axes_model, edges = build_cube_primitives(mp_obj)

                # Pred cube 投影
                verts_2d_pred, axes_2d_pred = project_primitives(
                    verts_model,
                    axes_model,
                    rotation=R_pred.detach().cpu().numpy(),
                    translation=t_pred.detach().cpu().numpy(),
                    K=K,
                    bs_utils=geometry.bs_utils,
                )

                # GT cube 投影（只画边，不画轴）
                verts_2d_gt, _ = project_primitives(
                    verts_model,
                    axes_model,
                    rotation=R_gt.detach().cpu().numpy(),
                    translation=t_gt.detach().cpu().numpy(),
                    K=K,
                    bs_utils=geometry.bs_utils,
                )

                # ------- RGB + 叠加 Pred / GT cube ------- #
                rgb_np = denormalize_image(batch["rgb"][i].detach().cpu())
                color_pred = geometry.bs_utils.get_label_color(
                    cls_id if cls_id >= 0 else 0,
                    n_obj=22,
                    mode=2,
                )
                # GT 用固定红色
                color_gt = (0, 0, 255)

                overlay = draw_two_cubes_overlay(
                    rgb_np,
                    verts_2d_pred,
                    axes_2d_pred,
                    verts_2d_gt,
                    edges,
                    color_pred=color_pred,
                    color_gt=color_gt,
                )

                # ------- 计算三种点云（相机坐标系） ------- #
                # CAD points under GT / Pred
                points_gt_cam = torch.matmul(mp_obj, R_gt.transpose(0, 1)) + t_gt  # [M,3]
                points_pred_cam = torch.matmul(mp_obj, R_pred.transpose(0, 1)) + t_pred  # [M,3]

                # Raw ROI depth points (already in camera frame)
                points_raw_cam = batch["points"][i]  # [N,3]

                # ------- Alignment debug: how far is Raw ROI from CAD@GT? ------- #
                dist_raw_to_gt = torch.cdist(
                    points_raw_cam.unsqueeze(0),  # [1,N,3]
                    points_gt_cam.unsqueeze(0),   # [1,M,3]
                ).min(dim=2).values.mean().item()  # scalar

                dist_gt_to_raw = torch.cdist(
                    points_gt_cam.unsqueeze(0),
                    points_raw_cam.unsqueeze(0),
                ).min(dim=2).values.mean().item()

                print(
                    f"[DEBUG] sample {saved:03d} | "
                    f"mean nn(Raw->GT)={dist_raw_to_gt:.4f} m, "
                    f"mean nn(GT->Raw)={dist_gt_to_raw:.4f} m"
                )

                # ------- 指标计算：与 EvaluatorSC 完全一致的路径 ------- #
                cls_ids_single: Optional[torch.Tensor] = None
                if "cls_id" in batch:
                    cls_ids_single = batch["cls_id"][i : i + 1]

                batch_metrics = compute_batch_pose_metrics(
                    pred_rt=pred_rt_i.unsqueeze(0),
                    gt_rt=gt_rt_i.unsqueeze(0),
                    model_points=mp_obj.unsqueeze(0),
                    cls_ids=cls_ids_single,
                    geometry=geometry,
                    cfg=cfg,
                )

                add = float(batch_metrics["add"][0])
                add_s = float(batch_metrics["add_s"][0])
                t_err = float(batch_metrics["t_err"][0])
                rot_err = float(batch_metrics.get("rot_err_deg", batch_metrics.get("rot_err", [0]))[0])
                is_sym = cls_id in sym_class_ids

                csv_values = None
                if per_image_metrics:
                    key = (scene_id, im_id)
                    csv_values = per_image_metrics.get(key)

                def _fmt_pair(
                    name: str,
                    pred_val: float,
                    csv_key: str,
                    unit: str,
                    scale: float = 1.0,
                    tol: float = 1e-6,
                ) -> Tuple[str, bool]:
                    pred_scaled = pred_val * scale
                    warn = False
                    if csv_values is not None and csv_key in csv_values:
                        csv_val = float(csv_values[csv_key]) * scale
                        if abs(pred_scaled - csv_val) > tol:
                            warn = True
                        return (
                            f"{name}(pred)={pred_scaled:.4f}{unit} / CSV={csv_val:.4f}{unit}",
                            warn,
                        )
                    return (f"{name}(pred)={pred_scaled:.4f}{unit} / CSV=N/A", warn)

                title_lines = [
                    f"sample_{saved:03d} | scene={scene_id} im={im_id} cls={cls_id} | sym={is_sym}",
                    "Pred cube: colored + axes | GT cube: red wireframe",
                ]

                warn_flags = []
                pair, w = _fmt_pair("ADD", add, "add", unit="m")
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("ADD-S", add_s, "add_s", unit="m")
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("t_err", t_err, "t_err", unit="mm", scale=1000.0)
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("rot", rot_err, "rot_err_deg", unit="deg")
                warn_flags.append(w)
                title_lines.append(pair)

                title_lines.append(
                    f"NN Raw->GT={dist_raw_to_gt:.4f}m | GT->Raw={dist_gt_to_raw:.4f}m"
                )

                if any(warn_flags):
                    logger.warning(
                        f"Metric mismatch over tolerance for scene={scene_id}, im={im_id}"
                    )

                # ------- Pose matrix for table ------- #
                pose_matrix = build_pose_matrix(pred_rt_i)
                title = "\n".join(title_lines)
                save_path = os.path.join(save_dir, f"sample_{saved:03d}.png")

                render_sample(
                    overlay,
                    points_pred_cam,
                    points_gt_cam,
                    points_raw_cam,
                    pose_matrix,
                    title,
                    save_path,
                    show=args.show,
                )
                logger.info(f"Saved visualization to {save_path}")
                saved += 1

            if saved >= args.num_samples:
                break


if __name__ == "__main__":
    main()
