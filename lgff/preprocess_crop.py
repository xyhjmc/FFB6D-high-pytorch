"""
Offline Preprocessing Tool for LGFF (JSON-based).
Fast cropping using 'scene_gt_info.json' bounding boxes.
AND
Automatic Keypoint Generation (Corners for Phone, Random/FPS for others).

- 对每个子数据集（train_pbr/train_real/...）显示 scene 级进度条
- 对每个 scene 内部的 im_id：
    - 若在终端(TTY)里：用 tqdm 进度条
    - 若在非 TTY 环境（如 PyCharm 默认 console）：用简洁的 print 进度
- 只有在当前 scene 产生了至少 1 个 crop 时，才在 dst 下创建对应 scene 目录并写 JSON
- 额外打印统计信息：每个子数据集的 crop 数、整体 crop 总数
- [NEW] 最后自动生成 keypoints/obj_xxxxxx.npy，优先使用 corners.txt
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData  # 确保你的环境安装了 plyfile (pip install plyfile)

IS_TTY = sys.stdout.isatty()  # 终端才用完整 tqdm 条


def process_scene(scene_path, out_path, obj_id, crop_size=128, pad_ratio=0.15) -> int:
    """
    处理单个 scene：
        - 返回该 scene 实际生成的 crop 数量（0 表示这个 scene 没有该物体）
    """
    scene_path = Path(scene_path)
    out_path = Path(out_path)

    # 1. 读取 JSON（不创建输出目录，先看看这 scene 到底有没有东西）
    gt_path   = scene_path / "scene_gt.json"
    info_path = scene_path / "scene_gt_info.json"
    cam_path  = scene_path / "scene_camera.json"

    if not gt_path.exists() or not info_path.exists() or not cam_path.exists():
        tqdm.write(f"[Skip] Missing GT / Info / Camera: {scene_path}")
        return 0

    with open(gt_path, "r") as f:
        scene_gt = json.load(f)
    with open(info_path, "r") as f:
        scene_info = json.load(f)
    with open(cam_path, "r") as f:
        scene_cam = json.load(f)

    new_scene_gt  = {}
    new_scene_cam = {}
    out_im_id = 0

    # 延迟创建输出目录：只有真正有 crop 时才建
    dirs_created = False

    def ensure_out_dirs():
        nonlocal dirs_created
        if not dirs_created:
            (out_path / "rgb").mkdir(parents=True, exist_ok=True)
            (out_path / "depth").mkdir(parents=True, exist_ok=True)
            (out_path / "mask_visib").mkdir(parents=True, exist_ok=True)
            dirs_created = True

    # im_id list
    im_ids = sorted(int(x) for x in scene_gt.keys())
    n_frames = len(im_ids)

    # -------- 帧级“进度条”选择：TTY 用 tqdm，非 TTY 用普通 for + print --------
    if IS_TTY:
        frame_iter = tqdm(
            im_ids,
            desc=f"[{scene_path.parent.name}/{scene_path.name}] frames",
            leave=False,
            dynamic_ncols=True,
        )
    else:
        frame_iter = im_ids

    for frame_idx, im_id in enumerate(frame_iter):
        im_str = str(im_id)
        gts   = scene_gt[im_str]
        infos = scene_info[im_str]

        # 非 TTY 环境下，适度打印进度（比如每 50 帧一条）
        if not IS_TTY and (frame_idx % 50 == 0 or frame_idx == n_frames - 1):
            print(
                f"[{scene_path.parent.name}/{scene_path.name}] "
                f"frames {frame_idx+1}/{n_frames}",
                end="\r",
                flush=True,
            )

        # 找这一帧里所有目标实例
        target_indices = []
        for idx, gt_obj in enumerate(gts):
            if gt_obj["obj_id"] != obj_id:
                continue
            bbox = infos[idx].get("bbox_visib", [-1, -1, -1, -1])
            if bbox[2] > 5 and bbox[3] > 5:  # w / h > 5 像素
                target_indices.append(idx)

        if not target_indices:
            continue

        # 确认图片存在
        rgb_f = scene_path / "rgb" / f"{im_id:06d}.png"
        if not rgb_f.exists():
            rgb_f = scene_path / "rgb" / f"{im_id:06d}.jpg"
        depth_f = scene_path / "depth" / f"{im_id:06d}.png"

        if not rgb_f.exists() or not depth_f.exists():
            # 某些 BOP 子集可能缺深度，直接跳过
            continue

        rgb_full    = cv2.imread(str(rgb_f))
        depth_full = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
        if rgb_full is None or depth_full is None:
            continue

        K_raw = np.array(scene_cam[im_str]["cam_K"], dtype=np.float32).reshape(3, 3)
        depth_scale = scene_cam[im_str].get("depth_scale", 1000.0)
        img_h, img_w = rgb_full.shape[:2]

        # 针对每个实例单独生成一张裁剪图
        for idx in target_indices:
            gt_obj   = gts[idx]
            info_obj = infos[idx]

            # bbox_visib: [x, y, w, h]
            x, y, w, h = info_obj["bbox_visib"]

            cx, cy = x + w / 2.0, y + h / 2.0
            size   = max(w, h) * (1.0 + pad_ratio)

            x1 = int(max(0, cx - size / 2.0))
            y1 = int(max(0, cy - size / 2.0))
            x2 = int(min(img_w, cx + size / 2.0))
            y2 = int(min(img_h, cy + size / 2.0))

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            # 第一次真要写文件时再建目录
            ensure_out_dirs()

            # --- Crop ---
            rgb_crop    = rgb_full[y1:y2, x1:x2]
            depth_crop = depth_full[y1:y2, x1:x2]

            # Mask
            mask_name = f"{im_id:06d}_{idx:06d}.png"
            mask_f = scene_path / "mask_visib" / mask_name
            if not mask_f.exists():
                mask_f = scene_path / "mask" / mask_name

            if mask_f.exists():
                mask_full = cv2.imread(str(mask_f), cv2.IMREAD_GRAYSCALE)
                mask_crop = mask_full[y1:y2, x1:x2]
            else:
                # fallback: bbox 区内全 1
                mask_crop = np.ones(rgb_crop.shape[:2], dtype=np.uint8) * 255

            # --- Resize ---
            rgb_fin    = cv2.resize(rgb_crop,    (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            depth_fin = cv2.resize(depth_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            mask_fin  = cv2.resize(mask_crop,  (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

            # 强制二值
            mask_fin = (mask_fin > 0).astype(np.uint8) * 255

            # --- Recompute K ---
            real_h, real_w = rgb_crop.shape[:2]
            scale_x = crop_size / float(real_w)
            scale_y = crop_size / float(real_h)

            K_new = K_raw.copy()
            K_new[0, 0] *= scale_x
            K_new[1, 1] *= scale_y
            K_new[0, 2] = (K_new[0, 2] - x1) * scale_x
            K_new[1, 2] = (K_new[1, 2] - y1) * scale_y

            out_id_str = f"{out_im_id:06d}"

            cv2.imwrite(str(out_path / "rgb"        / f"{out_id_str}.png"), rgb_fin)
            cv2.imwrite(str(out_path / "depth"      / f"{out_id_str}.png"), depth_fin)
            cv2.imwrite(str(out_path / "mask_visib" / f"{out_id_str}_000000.png"), mask_fin)

            # 保存 GT / Camera
            new_scene_gt[out_id_str] = [gt_obj]
            new_scene_cam[out_id_str] = {
                "cam_K": K_new.reshape(-1).tolist(),
                "depth_scale": float(depth_scale),
            }

            out_im_id += 1

    # 如果整个 scene 没有生成任何 crop，就完全不创建这个 scene 的输出目录
    if out_im_id == 0:
        tqdm.write(f"[Skip] Scene {scene_path.name}: no crops for obj_id={obj_id}")
        return 0

    # 有 crop 的 scene，才写 JSON
    with open(out_path / "scene_gt.json", "w") as f:
        json.dump(new_scene_gt, f, indent=2)
    with open(out_path / "scene_camera.json", "w") as f:
        json.dump(new_scene_cam, f, indent=2)

    tqdm.write(
        f"[Done] Scene {scene_path.name} -> {out_im_id} crops saved at {out_path}"
    )
    return out_im_id


import numpy as np
from pathlib import Path
from plyfile import PlyData  # 确保安装了 plyfile: pip install plyfile


# =========================================================
#  核心算法：最远点采样 (FPS)
# =========================================================
def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    使用最远点采样 (FPS) 从点云中选取关键点。

    Args:
        points: [N, 3] 输入点云
        n_samples: 需要采样的点数 (K)
    Returns:
        selected_points: [K, 3] 采样后的点
    """
    N, D = points.shape
    if N <= n_samples:
        return points

    # 初始化索引列表
    centroids = np.zeros(n_samples, dtype=int)

    # 1. 随机选择第一个点 (为了确定性，这里固定选 index=0，也可以随机)
    # rng = np.random.default_rng(0)
    # centroids[0] = rng.choice(N)
    centroids[0] = 0

    # 2. 初始化距离矩阵 (到已选点集的最小距离)
    # 初始状态下，所有点到“已选点集”的距离都是无穷大
    distance = np.full(N, np.inf)

    # 3. 迭代选择剩余 K-1 个点
    for i in range(1, n_samples):
        # 获取上一个选中的点的坐标 [1, 3]
        last_centroid = points[centroids[i - 1]]

        # 计算所有点到上一个选中点的欧氏距离平方 [N]
        # 使用广播机制: (N, 3) - (1, 3)
        dist = np.sum((points - last_centroid) ** 2, axis=1)

        # 更新 distance: 每个点记录它到“所有已选点”的最小距离
        distance = np.minimum(distance, dist)

        # 选择距离最远的点作为下一个关键点
        centroids[i] = np.argmax(distance)

    return points[centroids]


# =========================================================
#  工具函数：关键点生成 (基于模型 + FPS)
# =========================================================
def gen_keypoints(src_root: Path, dst_root: Path, obj_id: int, num_kps: int = 8):
    """
    生成并保存关键点文件 (obj_{id}.npy)。

    策略：
    1. 读取 .ply 模型。
    2. 处理单位 (mm -> m)。
    3. 使用 FPS 算法在物体表面均匀采样 num_kps 个点。
    """
    # 输出目录
    kp_out_dir = dst_root / "keypoints"
    kp_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = kp_out_dir / f"obj_{obj_id:06d}.npy"

    print(f"\n[Keypoints] Generating keypoints for obj_{obj_id} using FPS...")

    # 1. 读取模型文件
    # 尝试标准路径和 eval 路径
    model_path = src_root / "models" / f"obj_{obj_id:06d}.ply"
    if not model_path.exists():
        model_path = src_root / "models_eval" / f"obj_{obj_id:06d}.ply"

    if not model_path.exists():
        print(f"  -> [Error] Cannot find model .ply at {model_path}. Skipping.")
        return

    try:
        # 读取 PLY
        ply = PlyData.read(str(model_path))
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        # 2. 单位检查与转换 (LineMod 的 ply 通常是 mm，我们需要 m)
        # 如果最大值超过 10 (通常物体不会超过 10米)，则认为是毫米
        if np.abs(pts).max() > 10.0:
            print(f"  -> Detected mm unit (max={np.abs(pts).max():.2f}), converting to meters...")
            pts = pts / 1000.0

        # 3. 执行 FPS 采样
        # 如果模型点数少于需要的关键点数 (极少见)，直接重复填充
        if pts.shape[0] < num_kps:
            print(f"  -> [Warning] Mesh has fewer points ({pts.shape[0]}) than num_kps ({num_kps}).")
            indices = np.random.choice(pts.shape[0], num_kps, replace=True)
            kps = pts[indices]
        else:
            kps = farthest_point_sampling(pts, num_kps)
            print(f"  -> FPS Sampling done. Selected {num_kps} points.")

        # 4. 保存
        np.save(str(out_path), kps.astype(np.float32))
        print(f"  -> Saved to: {out_path}")

    except Exception as e:
        print(f"  -> [Error] Failed to process obj_{obj_id}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Crop BOP dataset based on scene_gt_info.json"
    )
    parser.add_argument(
        "--src",
        required=False,
        default="/home/xyh/datasets/LM(BOP)/lm",
        help="Root of raw dataset (e.g. datasets/bop/lm)",
    )
    parser.add_argument(
        "--dst",
        required=False,
        default="/home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/linemod_phone_crop128_all",
        help="Output root for cropped dataset",
    )
    parser.add_argument(
        "--obj_id", type=int, required=False, default=15, help="Object ID to crop (e.g. 15 for phone)"
    )
    parser.add_argument(
        "--size", type=int, default=128, help="Target resize size (e.g. 128)"
    )
    parser.add_argument(
        "--pad", type=float, default=0.15, help="Padding ratio (e.g. 0.15)"
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    sub_dirs = ["train_pbr", "train_real", "train_synt", "test", "val"]

    total_crops = 0
    total_scenes_with_data = 0

    for sub in sub_dirs:
        src_sub = src_root / sub
        if not src_sub.exists():
            continue

        tqdm.write(f"[Info] Found sub-dataset: {sub}")
        dst_sub = dst_root / sub
        dst_sub.mkdir(parents=True, exist_ok=True)

        scene_dirs = sorted(
            d for d in src_sub.iterdir() if d.is_dir() and d.name.isdigit()
        )

        sub_crops = 0
        sub_scenes_with_data = 0

        # scene 级 tqdm（非 TTY 就只是一层 for + print）
        scene_iter = tqdm(
            scene_dirs,
            desc=f"[{sub}] scenes",
            leave=True,
            dynamic_ncols=True,
            disable=not IS_TTY,
        )

        for scene_dir in scene_iter:
            out_scene_dir = dst_sub / scene_dir.name
            num_crops = process_scene(
                scene_dir,
                out_scene_dir,
                obj_id=args.obj_id,
                crop_size=args.size,
                pad_ratio=args.pad,
            )
            if num_crops > 0:
                sub_crops += num_crops
                sub_scenes_with_data += 1

        total_crops += sub_crops
        total_scenes_with_data += sub_scenes_with_data

        # 子数据集统计信息
        print(
            f"[Summary] Sub-dataset '{sub}': "
            f"{len(scene_dirs)} scenes scanned, "
            f"{sub_scenes_with_data} scenes with target, "
            f"{sub_crops} crops saved."
        )

    # 总体统计
    print(
        f"[Overall Summary] Total crops: {total_crops}, "
        f"scenes with target across all subsets: {total_scenes_with_data}."
    )

    # >>>>>>>> 自动生成关键点 <<<<<<<<
    gen_keypoints(src_root, dst_root, args.obj_id)


if __name__ == "__main__":
    main()