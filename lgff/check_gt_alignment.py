"""
GT–RGB/深度端到端对齐自检工具。

针对 SingleObjectDataset 的几何链路做可视化自检：
    * 将 CAD 模型点 (object frame, 单位 m) 通过 GT 姿态投影到 RGB 图像。
    * 将深度点云 (camera frame, 单位 m) 在同一内参下投影，并与 CAD 投影对比。
    * 计算 Raw->CAD / CAD->Raw 的最近邻距离（均值/中位数，单位 m）。

使用示例：

python lgff/check_gt_alignment.py \
  --config configs/linemod_ape_sc.yaml \
  --split test \
  --num-samples 8 \
  --save-dir output/check_align_ape
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List

import cv2
import numpy as np
import torch

from lgff.utils.config import LGFFConfig, load_config
from lgff.datasets.single_loader import SingleObjectDataset
from lgff.utils.geometry import GeometryToolkit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GT alignment for LGFF datasets")
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--obj-id", type=int, default=None, help="Override obj_id in config")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of random samples")
    parser.add_argument(
        "--idx",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit sample indices (overrides random sampling)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output/check_align",
        help="Directory to save alignment visualizations",
    )
    parser.add_argument(
        "--max-raw-points",
        type=int,
        default=2000,
        help="Subsample raw depth points for clearer visualization",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    args, _ = parser.parse_known_args()
    return args


def denormalize_image(rgb_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] tensor -> uint8 [H,W,3] RGB。"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = rgb_tensor.detach().cpu().numpy()
    img = (img * std[:, None, None]) + mean[:, None, None]
    img = np.clip(img * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return img


def select_indices(total: int, idx_list: Iterable[int] | None, k: int, seed: int) -> List[int]:
    if idx_list is not None and len(idx_list) > 0:
        return [i for i in idx_list if 0 <= i < total]

    rng = np.random.default_rng(seed)
    k = min(k, total)
    return rng.choice(total, size=k, replace=False).tolist()


def project_points_cam(points_cam: np.ndarray, K: np.ndarray, geometry: GeometryToolkit) -> np.ndarray:
    """Project camera-frame points (meters) to pixel coordinates."""
    return geometry.project_points(points_cam, depth_scale=1.0, camera_matrix=K)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 加载配置并根据需要覆盖 obj_id
    cfg: LGFFConfig = load_config()
    if args.obj_id is not None:
        cfg.obj_id = args.obj_id

    geometry = GeometryToolkit()

    # 2) 构建数据集（单位均为米）
    dataset = SingleObjectDataset(cfg, split=args.split)
    if len(dataset) == 0:
        print(f"[check_gt_alignment] split={args.split} is empty, abort.")
        return

    indices = select_indices(len(dataset), args.idx, args.num_samples, args.seed)
    print(f"[check_gt_alignment] Using indices: {indices}")

    for idx in indices:
        sample = dataset[idx]
        rgb = denormalize_image(sample["rgb"])
        K = sample["intrinsic"].numpy()

        model_pts = sample["model_points"].numpy()  # [M,3], object frame, meters
        pose = sample["pose"].numpy()  # [3,4], [R|t], meters
        R = pose[:, :3]
        t = pose[:, 3]

        # CAD 点 -> 相机坐标系（单位米）
        cad_cam = model_pts @ R.T + t
        cad_uv = project_points_cam(cad_cam, K, geometry)

        # Raw 深度点云（相机坐标系，单位米）；用 labels 过滤占位零点
        raw_pts = sample["points"].numpy()
        labels = sample.get("labels")
        if labels is not None:
            labels_np = labels.numpy()
            if labels_np.ndim > 1:
                labels_np = labels_np.squeeze()
            mask_valid = labels_np > 0
            raw_pts = raw_pts[mask_valid]

        if raw_pts.shape[0] > args.max_raw_points:
            rng = np.random.default_rng(args.seed + idx)
            choice = rng.choice(raw_pts.shape[0], args.max_raw_points, replace=False)
            raw_pts = raw_pts[choice]

        raw_uv = project_points_cam(raw_pts, K, geometry)

        # 最近邻距离（双向），单位米
        cad_tensor = torch.from_numpy(cad_cam).float()
        raw_tensor = torch.from_numpy(raw_pts).float()
        nn_raw_to_cad = torch.cdist(raw_tensor.unsqueeze(0), cad_tensor.unsqueeze(0)).min(dim=2).values
        nn_cad_to_raw = torch.cdist(cad_tensor.unsqueeze(0), raw_tensor.unsqueeze(0)).min(dim=2).values

        mean_r2c = float(nn_raw_to_cad.mean())
        median_r2c = float(nn_raw_to_cad.median())
        mean_c2r = float(nn_cad_to_raw.mean())
        median_c2r = float(nn_cad_to_raw.median())

        # 绘制：CAD 投影为绿色，深度点投影为红色
        vis = rgb.copy()
        for u, v in cad_uv:
            cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 255, 0), -1)
        for u, v in raw_uv:
            cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)

        scene_id = int(sample.get("scene_id", -1))
        im_id = int(sample.get("im_id", -1))
        title = (
            f"idx={idx} scene={scene_id} im={im_id} | nn Raw->CAD: mean={mean_r2c:.4f}m, "
            f"median={median_r2c:.4f}m | nn CAD->Raw: mean={mean_c2r:.4f}m, median={median_c2r:.4f}m"
        )
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        save_path = os.path.join(args.save_dir, f"check_align_sample_{idx:03d}.png")
        cv2.imwrite(save_path, vis)

        print(f"[check_gt_alignment] Saved {save_path}")
        print(
            f"  Raw->CAD nn mean={mean_r2c:.6f} m, median={median_r2c:.6f} m | "
            f"CAD->Raw nn mean={mean_c2r:.6f} m, median={median_c2r:.6f} m"
        )


if __name__ == "__main__":
    main()
