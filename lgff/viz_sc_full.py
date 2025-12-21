"""
Visualization script for Single-Class LGFF inference (full-image overlay).

This tool loads a trained checkpoint, runs forward passes on a chosen split,
selects the fused pose per sample (same logic as evaluator), and renders:

  * RGB image (either cropped ROI or original full image, if provided)
    with:
        - Predicted cube edges & axes
        - GT cube edges & axes
        - Small text labels "Pred cube" / "GT cube"

  * 4x4 homogeneous pose matrix table (Pred [R|t]).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, Optional, Tuple, Any

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
    parser = argparse.ArgumentParser(
        description="Visualize LGFF Single-Class Predictions (full-image overlay)"
    )
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
        default=100,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
    # 你说这部分不用动，我保持原样（default 写死）
    parser.add_argument(
        "--orig-root",
        type=str,
        default="/home/xyh/datasets/LM(BOP)/lm",
        help=(
            "Path to original full-res BOP dataset root. "
            "If provided, cubes will be rendered on original images instead of cropped ROI."
        ),
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
# Image / primitive helpers
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
    """
    Project cube vertices & axes from object frame to image plane.

    rotation: [3,3], translation: [3], both in camera frame (meters).
    K: 3x3 intrinsic matrix for the target image.
    """
    verts_cam = np.dot(verts_model, rotation.T) + translation
    axes_cam = np.dot(axes_model, rotation.T) + translation
    verts_2d = bs_utils.project_p3d(verts_cam, 1.0, K)
    axes_2d = bs_utils.project_p3d(axes_cam, 1.0, K)
    return verts_2d, axes_2d


def draw_two_cubes_overlay(
    rgb_np: np.ndarray,
    verts_pred_2d: np.ndarray,
    axes_pred_2d: np.ndarray,
    verts_gt_2d: np.ndarray,
    edges,
    color_pred: Tuple[int, int, int] = (0, 255, 0),  # BGR
    color_gt: Tuple[int, int, int] = (0, 0, 255),    # BGR (red)
    font_scale: float = 0.45,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    在一张图上同时画 Pred cube 和 GT cube（线色不同），并标注小文字。

    注意：输入 rgb_np 是 RGB（matplotlib 风格），但 cv2 的颜色是 BGR。
    这里内部做 RGB<->BGR 转换，确保颜色显示正确。
    """
    # RGB -> BGR for cv2 drawing
    img_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # Pred cube
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_pred_2d[i0]).astype(int))
        p1 = tuple(np.round(verts_pred_2d[i1]).astype(int))
        img_bgr = cv2.line(img_bgr, p0, p1, color_pred, thickness=2)

    # Axis colors (BGR)
    axis_colors_pred = [
        (0, 255, 255),  # Z: yellow-ish
        (0, 255, 0),    # Y: green
        (255, 0, 0),    # X: blue
    ]
    for idx, c in enumerate(axis_colors_pred, start=1):
        p0 = tuple(np.round(axes_pred_2d[0]).astype(int))
        p1 = tuple(np.round(axes_pred_2d[idx]).astype(int))
        img_bgr = cv2.arrowedLine(img_bgr, p0, p1, color=c, thickness=2, tipLength=0.08)

    # GT cube
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_gt_2d[i0]).astype(int))
        p1 = tuple(np.round(verts_gt_2d[i1]).astype(int))
        img_bgr = cv2.line(img_bgr, p0, p1, color_gt, thickness=2)

    # Labels
    h, w = img_bgr.shape[:2]
    x0, y0 = int(0.02 * w), int(0.05 * h)
    cv2.putText(
        img_bgr,
        "Pred cube",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_pred,
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr,
        "GT cube",
        (x0, y0 + int(1.6 * 20 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_gt,
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )

    # BGR -> RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def build_pose_matrix(pred_rt: torch.Tensor) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = pred_rt.detach().cpu().numpy()
    return pose


def render_sample(
    image_overlay: np.ndarray,
    pose_matrix: np.ndarray,
    title: str,
    save_path: str,
    show: bool = False,
) -> None:
    """
    渲染一个样本：

      - 上：RGB + Pred/GT cube 叠加图
      - 下：4x4 Homogeneous pose matrix
    """
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_overlay)
    ax_img.axis("off")
    ax_img.set_title("RGB with Pred & GT cubes")

    ax_pose = fig.add_subplot(gs[1, 0])
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


def load_full_bop_rgb_K(
    orig_root: str,
    scene_id: int,
    im_id: int,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    从原始 BOP 数据集中读取 full-res RGB 和相机内参 K_full。
    """
    scene_dir = os.path.join(orig_root, split, f"{scene_id:06d}")
    rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
    cam_path = os.path.join(scene_dir, "scene_camera.json")

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB not found: {rgb_path}")
    if not os.path.isfile(cam_path):
        raise FileNotFoundError(f"scene_camera.json not found: {cam_path}")

    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Failed to load image: {rgb_path}")
    rgb_full = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    with open(cam_path, "r") as f:
        scene_cam = json.load(f)

    entry = scene_cam.get(str(im_id))
    if entry is None:
        raise KeyError(f"No camera entry for im_id={im_id} in {cam_path}")

    K_list = entry["cam_K"]
    K_full = np.array(K_list, dtype=np.float32).reshape(3, 3)

    return rgb_full, K_full


def _compute_metrics_safe(
    pred_rt_1: torch.Tensor,  # [1,3,4]
    gt_rt_1: torch.Tensor,    # [1,3,4]
    mp_obj_1: torch.Tensor,   # [1,M,3]
    cls_ids_single: Optional[torch.Tensor],
    geometry: GeometryToolkit,
    cfg: LGFFConfig,
) -> Dict[str, Any]:
    """Robust metrics wrapper: ensure add/add_s exist."""
    m = compute_batch_pose_metrics(
        pred_rt=pred_rt_1,
        gt_rt=gt_rt_1,
        model_points=mp_obj_1,
        cls_ids=cls_ids_single,
        geometry=geometry,
        cfg=cfg,
    )
    # Fallback: if add or add_s missing
    if "add" not in m or "add_s" not in m:
        add_fb = geometry.compute_add(pred_rt_1.to(gt_rt_1.device), gt_rt_1, mp_obj_1)    # [1]
        adds_fb = geometry.compute_adds(pred_rt_1.to(gt_rt_1.device), gt_rt_1, mp_obj_1)  # [1]
        if "add" not in m:
            m["add"] = add_fb.detach().cpu()
        if "add_s" not in m:
            m["add_s"] = adds_fb.detach().cpu()

    # Ensure common keys exist to avoid KeyError in visualization
    for k in ["t_err", "rot_err"]:
        if k not in m:
            m[k] = torch.zeros((1,), dtype=torch.float32)
    return m


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 1) 用当前工程的配置系统载入基础 cfg
    cfg_cli: LGFFConfig = load_config()

    # 2) checkpoint config merge（保证 backbone / 维度一致）
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))
    print(
        f"[viz_sc] Loaded config from checkpoint: "
        f"obj_id={getattr(cfg, 'obj_id', None)}, "
        f"dataset={getattr(cfg, 'dataset_name', None)}"
    )

    # 3) work/save dirs
    work_dir = args.work_dir or getattr(cfg, "work_dir", None) or getattr(cfg, "log_dir", "output")
    save_dir = args.save_dir or os.path.join(work_dir, "viz_full")
    ensure_dirs(work_dir, save_dir)

    # 4) logging
    log_file = os.path.join(work_dir, "viz_full.log")
    setup_logger(log_file, name="lgff.viz")
    logger = get_logger("lgff.viz")
    logger.setLevel(logging.INFO)

    split = args.split or getattr(cfg, "val_split", "test")
    num_workers = args.num_workers or getattr(cfg, "num_workers", 4)
    per_image_metrics = load_per_image_metrics(args.per_image_csv)

    # 原始数据集 root（命令行优先；你说 orig-root 不动，我就保持这里逻辑不变）
    orig_root = args.orig_root or getattr(cfg, "orig_dataset_root", None)
    if orig_root is not None:
        logger.info(f"[viz_sc] Will overlay cubes on original dataset at: {orig_root}")

    logger.info(
        f"Visualizing split={split} | batch_size={args.batch_size} | "
        f"num_workers={num_workers} | num_samples={args.num_samples} | "
        f"checkpoint={ckpt_path}"
    )
    if per_image_metrics:
        logger.info(f"Loaded per-image metrics for cross-checking: {len(per_image_metrics)} rows")

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) dataset/loader
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

    # 6) model
    model = LGFF_SC(cfg, geometry)
    load_model_weights(model, ckpt_path, device, checkpoint=ckpt)
    model = model.to(device)
    model.eval()

    # 7) complexity logging (optional)
    complexity_logger = ModelComplexityLogger()
    try:
        example_batch = next(iter(loader))
        complexity_info = complexity_logger.maybe_log(model, example_batch, stage="viz_full")
        if complexity_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=viz_full",
                        f"Params: {complexity_info['params']:,} ({complexity_info['param_mb']:.2f} MB)",
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

    # 8) re-create loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 9) reuse evaluator helper (pose fusion + icp refine)
    helper = EvaluatorSC(model=model, test_loader=loader, cfg=cfg, geometry=geometry)
    sym_class_ids = set(getattr(cfg, "sym_class_ids", []))

    # config-driven switches
    use_icp = bool(getattr(cfg, "viz_use_icp", getattr(cfg, "icp_enable", True)))
    show_coarse_metrics = bool(getattr(cfg, "viz_show_coarse_metrics", True)) and use_icp
    metric_tol = float(getattr(cfg, "viz_metric_tol", 1e-6))

    if use_icp:
        logger.info("[viz_sc] ICP refine ENABLED for visualization.")
    else:
        logger.info("[viz_sc] ICP refine DISABLED for visualization.")

    saved = 0
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            B = batch["rgb"].shape[0]

            outputs = model(batch)

            # 1) coarse fused pose
            pred_rt_coarse = helper._process_predictions(outputs)  # [B,3,4]

            # 2) refine pose (config-driven)
            if use_icp:
                pred_rt = helper._icp_refine_batch(pred_rt_coarse, batch)  # [B,3,4]
            else:
                pred_rt = pred_rt_coarse

            gt_rt = helper._process_gt(batch)

            # model points
            if "model_points" in batch:
                model_points = batch["model_points"]
                if model_points.dim() == 2:
                    model_points = model_points.unsqueeze(0).expand(B, -1, -1)
            else:
                # fallback: back-project ROI points to object frame via GT pose
                points_cam_all = batch["points"]  # [B,N,3]
                gt_r_all = gt_rt[:, :3, :3]
                gt_t_all = gt_rt[:, :3, 3]
                points_centered = points_cam_all - gt_t_all.unsqueeze(1)
                gt_r_inv_all = gt_r_all.transpose(1, 2)
                model_points = torch.matmul(points_centered, gt_r_inv_all)

            for i in range(B):
                if saved >= args.num_samples:
                    break

                mp_obj = model_points[i]  # [M,3]
                pred_rt_i = pred_rt[i]
                gt_rt_i = gt_rt[i]

                R_pred = pred_rt_i[:, :3]
                t_pred = pred_rt_i[:, 3]
                R_gt = gt_rt_i[:, :3]
                t_gt = gt_rt_i[:, 3]

                cls_id = int(batch["cls_id"][i].item()) if "cls_id" in batch else -1
                scene_id = int(batch["scene_id"][i].item()) if "scene_id" in batch else -1
                im_id = int(batch["im_id"][i].item()) if "im_id" in batch else -1
                is_sym = cls_id in sym_class_ids  # <-- 修复：你原脚本这里漏了

                # choose image & K
                if orig_root is not None and scene_id >= 0 and im_id >= 0:
                    try:
                        rgb_np, K_vis = load_full_bop_rgb_K(
                            orig_root=orig_root,
                            scene_id=scene_id,
                            im_id=im_id,
                            split=split if split in ["train", "test", "val"] else "train",
                        )
                    except Exception as e:
                        logger.warning(
                            f"[viz_sc] Failed to load full image for "
                            f"scene={scene_id}, im={im_id}: {e}. "
                            f"Falling back to cropped ROI visualization."
                        )
                        rgb_np = denormalize_image(batch["rgb"][i].detach().cpu())
                        K_vis = batch["intrinsic"][i].detach().cpu().numpy()
                else:
                    rgb_np = denormalize_image(batch["rgb"][i].detach().cpu())
                    K_vis = batch["intrinsic"][i].detach().cpu().numpy()

                # project cubes
                verts_model, axes_model, edges = build_cube_primitives(mp_obj)
                verts_2d_pred, axes_2d_pred = project_primitives(
                    verts_model,
                    axes_model,
                    rotation=R_pred.detach().cpu().numpy(),
                    translation=t_pred.detach().cpu().numpy(),
                    K=K_vis,
                    bs_utils=geometry.bs_utils,
                )
                verts_2d_gt, axes_2d_gt = project_primitives(
                    verts_model,
                    axes_model,
                    rotation=R_gt.detach().cpu().numpy(),
                    translation=t_gt.detach().cpu().numpy(),
                    K=K_vis,
                    bs_utils=geometry.bs_utils,
                )

                overlay = draw_two_cubes_overlay(
                    rgb_np,
                    verts_2d_pred,
                    axes_2d_pred,
                    verts_2d_gt,
                    edges,
                    color_pred=(0, 255, 0),
                    color_gt=(0, 0, 255),
                    font_scale=0.45,
                    font_thickness=1,
                )

                # metrics
                cls_ids_single: Optional[torch.Tensor] = None
                if "cls_id" in batch:
                    cls_ids_single = batch["cls_id"][i : i + 1]

                mp_obj_1 = mp_obj.unsqueeze(0)
                pred_rt_1 = pred_rt_i.unsqueeze(0)
                gt_rt_1 = gt_rt_i.unsqueeze(0)

                metrics_ref = _compute_metrics_safe(
                    pred_rt_1=pred_rt_1,
                    gt_rt_1=gt_rt_1,
                    mp_obj_1=mp_obj_1,
                    cls_ids_single=cls_ids_single,
                    geometry=geometry,
                    cfg=cfg,
                )

                add = float(metrics_ref["add"][0])
                add_s = float(metrics_ref["add_s"][0])
                t_err = float(metrics_ref["t_err"][0])
                rot_err = float(metrics_ref["rot_err"][0])

                # optional coarse metrics for debug
                add0 = add_s0 = t_err0 = rot0 = None
                if show_coarse_metrics:
                    pred_rt0_i = pred_rt_coarse[i]
                    metrics_coarse = _compute_metrics_safe(
                        pred_rt_1=pred_rt0_i.unsqueeze(0),
                        gt_rt_1=gt_rt_1,
                        mp_obj_1=mp_obj_1,
                        cls_ids_single=cls_ids_single,
                        geometry=geometry,
                        cfg=cfg,
                    )
                    add0 = float(metrics_coarse["add"][0])
                    add_s0 = float(metrics_coarse["add_s"][0])
                    t_err0 = float(metrics_coarse["t_err"][0])
                    rot0 = float(metrics_coarse["rot_err"][0])

                # csv compare
                csv_values = None
                if per_image_metrics:
                    csv_values = per_image_metrics.get((scene_id, im_id), None)

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
                        try:
                            csv_val = float(csv_values[csv_key]) * scale
                            if abs(pred_scaled - csv_val) > tol:
                                warn = True
                            return (f"{name}(pred)={pred_scaled:.4f}{unit} / CSV={csv_val:.4f}{unit}", warn)
                        except Exception:
                            pass
                    return (f"{name}(pred)={pred_scaled:.4f}{unit} / CSV=N/A", warn)

                title_lines = [
                    f"sample_{saved:03d} | scene={scene_id} im={im_id} cls={cls_id} | sym={is_sym} | icp={use_icp}",
                ]

                warn_flags = []
                pair, w = _fmt_pair("ADD", add, "add", unit="m", tol=metric_tol)
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("ADD-S", add_s, "add_s", unit="m", tol=metric_tol)
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("t_err", t_err, "t_err", unit="mm", scale=1000.0, tol=max(metric_tol * 1000.0, 1e-3))
                warn_flags.append(w)
                title_lines.append(pair)

                pair, w = _fmt_pair("rot", rot_err, "rot_err_deg", unit="deg", tol=max(metric_tol, 1e-3))
                warn_flags.append(w)
                title_lines.append(pair)

                # show coarse->refined delta in title (very useful to confirm refine is working)
                if show_coarse_metrics and add0 is not None:
                    title_lines.append(
                        f"COARSE->REF | ADD: {add0:.4f} -> {add:.4f} m | "
                        f"ADD-S: {add_s0:.4f} -> {add_s:.4f} m"
                    )
                    title_lines.append(
                        f"COARSE->REF | t_err: {t_err0*1000.0:.2f} -> {t_err*1000.0:.2f} mm | "
                        f"rot: {rot0:.2f} -> {rot_err:.2f} deg"
                    )

                if any(warn_flags):
                    logger.warning(f"Metric mismatch over tolerance for scene={scene_id}, im={im_id}")

                pose_matrix = build_pose_matrix(pred_rt_i)  # refined (or coarse if icp disabled)
                title = "\n".join(title_lines)
                save_path = os.path.join(save_dir, f"sample_{saved:03d}.png")

                render_sample(
                    overlay,
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
