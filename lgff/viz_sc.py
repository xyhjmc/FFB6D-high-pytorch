"""
Visualization script for Single-Class LGFF inference.

This tool loads a trained checkpoint, runs forward passes on a chosen split,
selects the best-confidence pose per sample (same logic as evaluator), and
renders:
  * RGB image with cube edges, axes, and predicted pose matrix
  * 3D scatter overlay of predicted vs. GT point clouds

Usage examples:
    python lgff/viz_sc.py --config configs/helmet_sc.yaml --checkpoint output/helmet_sc/checkpoint_best.pth
    python lgff/viz_sc.py --config configs/helmet_sc.yaml --checkpoint output/helmet_sc/checkpoint_best.pth --num-samples 8 --split val --show
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.getcwd())

from common.ffb6d_utils.model_complexity import ModelComplexityLogger
from lgff.utils.config import LGFFConfig, load_config
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.models.lgff_sc import LGFF_SC
from lgff.engines.evaluator_sc import EvaluatorSC
from lgff.eval_sc import load_model_weights


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
        help="Dataset split to visualize (default: cfg.val_split)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
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
    args, _ = parser.parse_known_args()
    return args


def ensure_dirs(work_dir: str, save_dir: str) -> None:
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


def denormalize_image(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized RGB tensor [3,H,W] back to uint8 HxWx3 (RGB)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = rgb_tensor.detach().cpu().numpy()
    img = (img * std[:, None, None]) + mean[:, None, None]
    img = np.clip(img * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return img


def build_cube_primitives(points_model: torch.Tensor, scale: float = 1.05) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
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
    img = rgb_np.copy()
    for i0, i1 in edges:
        p0 = tuple(np.round(verts_2d[i0]).astype(int))
        p1 = tuple(np.round(verts_2d[i1]).astype(int))
        img = cv2.line(img, p0, p1, color=color, thickness=2)
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx, c in enumerate(axis_colors, start=1):
        p0 = tuple(np.round(axes_2d[0]).astype(int))
        p1 = tuple(np.round(axes_2d[idx]).astype(int))
        img = cv2.arrowedLine(img, p0, p1, color=c, thickness=2, tipLength=0.08)
    return img


def build_pose_matrix(pred_rt: torch.Tensor) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = pred_rt.detach().cpu().numpy()
    return pose


def render_sample(
    image_overlay: np.ndarray,
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    pose_matrix: np.ndarray,
    title: str,
    save_path: str,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_overlay)
    ax_img.axis("off")
    ax_img.set_title("RGB + cube & axes")

    ax_scatter = fig.add_subplot(gs[0, 1], projection="3d")
    pred_np = points_pred.detach().cpu().numpy()
    gt_np = points_gt.detach().cpu().numpy()
    ax_scatter.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], c="tab:orange", s=6, alpha=0.7, label="Pred")
    ax_scatter.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], c="tab:blue", s=6, alpha=0.5, label="GT")
    ax_scatter.set_xlabel("X (m)")
    ax_scatter.set_ylabel("Y (m)")
    ax_scatter.set_zlabel("Z (m)")
    ax_scatter.legend(loc="upper right")
    ax_scatter.set_title("Point Clouds")

    ax_pose = fig.add_subplot(gs[1, :])
    ax_pose.axis("off")
    table = ax_pose.table(cellText=np.round(pose_matrix, 4), loc="center", cellLoc="center")
    table.scale(1.1, 1.4)
    ax_pose.set_title("Predicted homogeneous transform [R|t]")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show(block=False)
        plt.pause(0.5)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg: LGFFConfig = load_config()

    work_dir = args.work_dir or cfg.work_dir or cfg.log_dir
    save_dir = args.save_dir or os.path.join(work_dir, "viz")
    ensure_dirs(work_dir, save_dir)

    log_file = os.path.join(work_dir, "viz.log")
    setup_logger(log_file, name="lgff.viz")
    logger = get_logger("lgff.viz")
    logger.setLevel(logging.INFO)

    split = args.split or getattr(cfg, "val_split", "test")
    num_workers = args.num_workers or getattr(cfg, "num_workers", 4)

    logger.info(
        f"Visualizing split={split} | batch_size={args.batch_size} | num_workers={num_workers} | num_samples={args.num_samples}"
    )

    geometry = GeometryToolkit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    dataset = SingleObjectDataset(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = LGFF_SC(cfg, geometry)
    load_model_weights(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Complexity (params/FLOPs)
    complexity_logger = ModelComplexityLogger()
    try:
        example_batch = next(iter(loader))
        complexity_info = complexity_logger.maybe_log(model, example_batch, stage="viz")
        if complexity_info:
            logger.info(
                " | ".join(
                    [
                        "[ModelComplexity] Stage=viz",
                        f"Params: {complexity_info['params']:,} ({complexity_info['param_mb']:.2f} MB)",
                        f"GFLOPs: {complexity_info['gflops']:.3f}" if complexity_info.get("gflops") is not None else "GFLOPs: N/A",
                    ]
                )
            )
    except StopIteration:
        logger.warning("Visualization loader is empty; skip complexity logging.")
    except Exception as exc:
        logger.warning(f"Model complexity logging failed: {exc}")

    # Recreate loader to avoid skipping the first batch
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Reuse evaluator helpers for pose selection
    helper = EvaluatorSC(model=model, test_loader=loader, cfg=cfg, geometry=geometry)

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)
            pred_rt = helper._process_predictions(outputs)
            gt_rt = helper._process_gt(batch)

            points = batch["points"]  # [B, N, 3]

            # Transform points to model frame using GT
            gt_r = gt_rt[:, :3, :3]
            gt_t = gt_rt[:, :3, 3]
            points_centered = points - gt_t.unsqueeze(1)
            gt_r_inv = gt_r.transpose(1, 2)
            points_model = torch.matmul(points_centered, gt_r_inv)  # [B, N, 3]

            # Project using predicted pose
            pred_r = pred_rt[:, :3, :3]
            pred_t = pred_rt[:, :3, 3]
            points_pred = torch.matmul(points_model, pred_r.transpose(1, 2)) + pred_t.unsqueeze(1)

            # points_gt is just original observation
            points_gt = points

            for i in range(points.shape[0]):
                if saved >= args.num_samples:
                    break

                K = batch["intrinsic"][i].detach().cpu().numpy()
                verts_model, axes_model, edges = build_cube_primitives(points_model[i])
                verts_2d, axes_2d = project_primitives(
                    verts_model,
                    axes_model,
                    pred_r[i].detach().cpu().numpy(),
                    pred_t[i].detach().cpu().numpy(),
                    K,
                    geometry.bs_utils,
                )

                rgb_np = denormalize_image(batch["rgb"][i].detach().cpu())
                color = geometry.bs_utils.get_label_color(int(batch["cls_id"][i].item()), n_obj=22, mode=2)
                overlay = draw_cube_overlay(rgb_np, verts_2d, axes_2d, edges, color=color)

                pose_matrix = build_pose_matrix(pred_rt[i])

                title = f"sample_{saved:03d} | scene={int(batch['cls_id'][i].item())}"
                save_path = os.path.join(save_dir, f"sample_{saved:03d}.png")

                render_sample(overlay, points_pred[i], points_gt[i], pose_matrix, title, save_path, show=args.show)
                logger.info(f"Saved visualization to {save_path}")
                saved += 1

            if saved >= args.num_samples:
                break


if __name__ == "__main__":
    main()
