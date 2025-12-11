import torch
import cv2
import numpy as np
import os
from lgff.utils.config import load_config
from lgff.datasets.single_loader import SingleObjectDataset


def verify_data():
    # 1. 加载配置 (请指向你正在用的 config)
    cfg = load_config()
    cfg.dataset_root = "/home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/linemod_ape_crop128"  # 确保路径对
    cfg.resize_h = 128
    cfg.resize_w = 128

    print(f"Checking dataset: {cfg.dataset_root}")
    ds = SingleObjectDataset(cfg, split="train")

    # 2. 随机取一个样本
    idx = np.random.randint(0, len(ds))
    sample = ds[idx]

    rgb = sample["rgb"].numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    # 反归一化 RGB (ImageNet mean/std) 用于显示
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = (rgb * std + mean) * 255.0
    rgb = rgb.astype(np.uint8).copy()
    # RGB -> BGR for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    points = sample["point_cloud"].numpy()  # [N, 3]
    K = sample["intrinsic"].numpy()  # [3, 3]

    print(f"Sample {idx}: Points {points.shape}, K shape {K.shape}")
    print(f"Z-range: {points[:, 2].min():.3f} ~ {points[:, 2].max():.3f} m")

    # 3. 投影 3D -> 2D
    # u = fx * x / z + cx
    # v = fy * y / z + cy
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = (x * K[0, 0] / z) + K[0, 2]
    v = (y * K[1, 1] / z) + K[1, 2]

    # 4. 画图
    H, W = rgb.shape[:2]
    for i in range(len(u)):
        ui, vi = int(u[i]), int(v[i])
        if 0 <= ui < W and 0 <= vi < H:
            # 画绿色小点
            cv2.circle(rgb, (ui, vi), 1, (0, 255, 0), -1)

    # 5. 保存检查
    os.makedirs("debug_vis", exist_ok=True)
    save_path = "debug_vis/alignment_check.png"
    cv2.imwrite(save_path, rgb)
    print(f"Saved visualization to {save_path}. Please check it immediately!")


if __name__ == "__main__":
    verify_data()