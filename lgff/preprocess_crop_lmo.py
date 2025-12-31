# -*- coding: utf-8 -*-
"""
Offline Preprocessing Tool for LGFF (BOP JSON-based) — Safe LMO Test Crop into an existing LM crop root.

What this script does
- Reads a BOP dataset subset (default: test) under --src (e.g., .../bop/lmo/test/000002/...)
- Crops instances of a given obj_id using bbox_visib from scene_gt_info.json
- Resizes to --size x --size, recomputes intrinsics K for the crop
- Writes results under:  <dst>/<dst_subset>/<scene_id>/{rgb,depth,mask_visib,scene_gt.json,scene_camera.json}
- DOES NOT delete anything.
- By default, it will NOT overwrite existing per-scene outputs; it will skip those scenes.
- Keypoints are written under: <dst>/<dst_subset>/keypoints/obj_0000xx.npy (no collision with existing LM keypoints)

Example (your case)
python crop_bop_test_only_safe.py \
  --src /home/xyh/datasets/LMO(BOP)/lmo \
  --dst /home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/linemod_lamp_crop128_all \
  --src_subset test \
  --dst_subset test_lmo \
  --obj_id 1 \
  --size 128 \
  --pad 0.15

If you really want to overwrite existing outputs under test_lmo:
  add --overwrite

Dependencies:
  pip install opencv-python tqdm plyfile
"""

import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData

IS_TTY = sys.stdout.isatty()


# =========================================================
#  Core: process one scene (lazy output dir creation)
# =========================================================
def process_scene(
    scene_path: Path,
    out_path: Path,
    obj_id: int,
    crop_size: int = 128,
    pad_ratio: float = 0.15,
    overwrite_scene: bool = False,
) -> int:
    """
    Process one BOP scene directory.

    Returns:
        number of generated crops in this scene (0 if none)
    """
    scene_path = Path(scene_path)
    out_path = Path(out_path)

    # If out_path already has content and overwrite is disabled, skip to avoid clobbering.
    if out_path.exists() and any(out_path.iterdir()) and not overwrite_scene:
        tqdm.write(f"[Skip] Output scene exists (no overwrite): {out_path}")
        return 0

    gt_path = scene_path / "scene_gt.json"
    info_path = scene_path / "scene_gt_info.json"
    cam_path = scene_path / "scene_camera.json"

    if not gt_path.exists() or not info_path.exists() or not cam_path.exists():
        tqdm.write(f"[Skip] Missing GT / Info / Camera: {scene_path}")
        return 0

    with open(gt_path, "r") as f:
        scene_gt = json.load(f)
    with open(info_path, "r") as f:
        scene_info = json.load(f)
    with open(cam_path, "r") as f:
        scene_cam = json.load(f)

    new_scene_gt = {}
    new_scene_cam = {}
    out_im_id = 0

    dirs_created = False

    def ensure_out_dirs():
        nonlocal dirs_created
        if not dirs_created:
            (out_path / "rgb").mkdir(parents=True, exist_ok=True)
            (out_path / "depth").mkdir(parents=True, exist_ok=True)
            (out_path / "mask_visib").mkdir(parents=True, exist_ok=True)
            dirs_created = True

    im_ids = sorted(int(x) for x in scene_gt.keys())
    n_frames = len(im_ids)

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
        gts = scene_gt.get(im_str, [])
        infos = scene_info.get(im_str, [])

        if not IS_TTY and (frame_idx % 50 == 0 or frame_idx == n_frames - 1):
            print(
                f"[{scene_path.parent.name}/{scene_path.name}] frames {frame_idx+1}/{n_frames}",
                end="\r",
                flush=True,
            )

        if len(gts) == 0 or len(infos) == 0:
            continue

        # Find target instances for this frame
        target_indices = []
        for idx, gt_obj in enumerate(gts):
            if gt_obj.get("obj_id", -1) != obj_id:
                continue
            if idx >= len(infos):
                continue
            bbox = infos[idx].get("bbox_visib", [-1, -1, -1, -1])  # [x,y,w,h]
            if bbox[2] > 5 and bbox[3] > 5:
                target_indices.append(idx)

        if not target_indices:
            continue

        # load rgb/depth
        rgb_f = scene_path / "rgb" / f"{im_id:06d}.png"
        if not rgb_f.exists():
            rgb_f = scene_path / "rgb" / f"{im_id:06d}.jpg"
        depth_f = scene_path / "depth" / f"{im_id:06d}.png"

        if not rgb_f.exists() or not depth_f.exists():
            continue

        rgb_full = cv2.imread(str(rgb_f))
        depth_full = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
        if rgb_full is None or depth_full is None:
            continue

        cam_entry = scene_cam.get(im_str, {})
        if "cam_K" not in cam_entry:
            continue

        K_raw = np.array(cam_entry["cam_K"], dtype=np.float32).reshape(3, 3)
        depth_scale = float(cam_entry.get("depth_scale", 1000.0))
        img_h, img_w = rgb_full.shape[:2]

        for idx in target_indices:
            gt_obj = gts[idx]
            info_obj = infos[idx]

            x, y, w, h = info_obj.get("bbox_visib", [-1, -1, -1, -1])
            if w <= 0 or h <= 0:
                continue

            cx, cy = x + w / 2.0, y + h / 2.0
            size = max(w, h) * (1.0 + pad_ratio)

            x1 = int(max(0, cx - size / 2.0))
            y1 = int(max(0, cy - size / 2.0))
            x2 = int(min(img_w, cx + size / 2.0))
            y2 = int(min(img_h, cy + size / 2.0))

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            ensure_out_dirs()

            rgb_crop = rgb_full[y1:y2, x1:x2]
            depth_crop = depth_full[y1:y2, x1:x2]

            # BOP mask naming: {im_id:06d}_{gt_id:06d}.png  where gt_id == idx
            mask_name = f"{im_id:06d}_{idx:06d}.png"
            mask_f = scene_path / "mask_visib" / mask_name
            if not mask_f.exists():
                mask_f = scene_path / "mask" / mask_name

            if mask_f.exists():
                mask_full = cv2.imread(str(mask_f), cv2.IMREAD_GRAYSCALE)
                if mask_full is None:
                    mask_crop = np.ones(rgb_crop.shape[:2], dtype=np.uint8) * 255
                else:
                    mask_crop = mask_full[y1:y2, x1:x2]
            else:
                mask_crop = np.ones(rgb_crop.shape[:2], dtype=np.uint8) * 255

            rgb_fin = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            depth_fin = cv2.resize(depth_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            mask_fin = cv2.resize(mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            mask_fin = (mask_fin > 0).astype(np.uint8) * 255

            real_h, real_w = rgb_crop.shape[:2]
            if real_w <= 0 or real_h <= 0:
                continue

            scale_x = crop_size / float(real_w)
            scale_y = crop_size / float(real_h)

            K_new = K_raw.copy()
            K_new[0, 0] *= scale_x
            K_new[1, 1] *= scale_y
            K_new[0, 2] = (K_new[0, 2] - x1) * scale_x
            K_new[1, 2] = (K_new[1, 2] - y1) * scale_y

            out_id_str = f"{out_im_id:06d}"

            cv2.imwrite(str(out_path / "rgb" / f"{out_id_str}.png"), rgb_fin)
            cv2.imwrite(str(out_path / "depth" / f"{out_id_str}.png"), depth_fin)
            cv2.imwrite(str(out_path / "mask_visib" / f"{out_id_str}_000000.png"), mask_fin)

            # Save single-instance GT/cam for this crop
            new_scene_gt[out_id_str] = [gt_obj]
            new_scene_cam[out_id_str] = {
                "cam_K": K_new.reshape(-1).tolist(),
                "depth_scale": depth_scale,
            }

            out_im_id += 1

    if out_im_id == 0:
        tqdm.write(f"[Skip] Scene {scene_path.name}: no crops for obj_id={obj_id}")
        return 0

    with open(out_path / "scene_gt.json", "w") as f:
        json.dump(new_scene_gt, f, indent=2)
    with open(out_path / "scene_camera.json", "w") as f:
        json.dump(new_scene_cam, f, indent=2)

    tqdm.write(f"[Done] Scene {scene_path.name} -> {out_im_id} crops saved at {out_path}")
    return out_im_id


# =========================================================
#  FPS keypoints
# =========================================================
def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS).

    Args:
        points: [N, 3]
        n_samples: K
    Returns:
        [K, 3]
    """
    N = points.shape[0]
    if N <= n_samples:
        return points

    centroids = np.zeros(n_samples, dtype=np.int64)
    centroids[0] = 0
    distance = np.full(N, np.inf, dtype=np.float32)

    for i in range(1, n_samples):
        last = points[centroids[i - 1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        centroids[i] = int(np.argmax(distance))

    return points[centroids]


def gen_keypoints(
    src_root: Path,
    kp_root: Path,
    obj_id: int,
    num_kps: int = 8,
    overwrite: bool = False,
):
    """
    Generate keypoints for a given obj_id and save to:
      kp_root/keypoints/obj_0000xx.npy

    - No collision with your existing LM crop roots if kp_root is <dst>/<dst_subset>.
    """
    kp_out_dir = kp_root / "keypoints"
    kp_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = kp_out_dir / f"obj_{obj_id:06d}.npy"

    if out_path.exists() and not overwrite:
        print(f"[Keypoints] Exists (no overwrite): {out_path}")
        return

    print(f"\n[Keypoints] Generating keypoints for obj_{obj_id:06d} using FPS...")

    model_path = src_root / "models" / f"obj_{obj_id:06d}.ply"
    if not model_path.exists():
        model_path = src_root / "models_eval" / f"obj_{obj_id:06d}.ply"

    if not model_path.exists():
        print(f"  -> [Error] Cannot find model .ply: {model_path}")
        return

    try:
        ply = PlyData.read(str(model_path))
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        mx = float(np.abs(pts).max())
        if mx > 10.0:
            print(f"  -> Detected mm unit (max={mx:.2f}), converting to meters...")
            pts = pts / 1000.0

        if pts.shape[0] < num_kps:
            print(f"  -> [Warn] Mesh has fewer points ({pts.shape[0]}) than num_kps ({num_kps}). Sampling with replacement.")
            idx = np.random.default_rng(0).choice(pts.shape[0], num_kps, replace=True)
            kps = pts[idx]
        else:
            kps = farthest_point_sampling(pts, num_kps)

        np.save(str(out_path), kps.astype(np.float32))
        print(f"  -> Saved: {out_path}")

    except Exception as e:
        print(f"  -> [Error] Failed to generate keypoints: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
#  Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Safe crop BOP subset into an existing LM crop root (no deletion).")

    parser.add_argument(
        "--src",
        default="/home/xyh/datasets/LM(BOP)/lmo",
        help="Root of raw BOP dataset (e.g., .../bop/lmo)",
    )
    parser.add_argument(
        "--dst",
        default="/home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/linemod_dog_crop128_all",
        help="Existing crop root to append into (will create <dst_subset> under it).",
    )
    parser.add_argument("--src_subset", type=str, default="test", help="Subset in src to read (default: test).")
    parser.add_argument("--dst_subset", type=str, default="test_lmo", help="Subset folder name in dst (default: test_lmo).")

    parser.add_argument("--obj_id", type=int, default=6, help="Target obj_id to crop 1,5,6,8,9,10,11,12.")
    parser.add_argument("--size", type=int, default=128, help="Crop resize size.")
    parser.add_argument("--pad", type=float, default=0.15, help="Padding ratio around bbox_visib.")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing outputs under <dst>/<dst_subset> (and per-scene dirs).",
    )
    parser.add_argument(
        "--skip_keypoints",
        action="store_true",
        help="If set, do not generate keypoints.",
    )
    parser.add_argument("--num_kps", type=int, default=8, help="Number of FPS keypoints if generated.")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    src_sub = src_root / args.src_subset
    if not src_sub.exists():
        raise FileNotFoundError(f"[Error] Cannot find src subset: {src_sub}")

    dst_sub = dst_root / args.dst_subset
    dst_sub.mkdir(parents=True, exist_ok=True)

    # Safety behavior: if dst_sub is non-empty and overwrite is not set, we will *not* fail;
    # instead we skip per-scene dirs that already exist to avoid clobbering.
    if any(dst_sub.iterdir()) and not args.overwrite:
        print(f"[Info] Output subset already has content: {dst_sub}")
        print("[Info] Overwrite disabled; existing per-scene outputs will be skipped (safe append mode).")

    print(f"[Info] Source subset : {src_sub}")
    print(f"[Info] Output subset : {dst_sub}")
    print(f"[Info] Target obj_id : {args.obj_id}")
    print(f"[Info] Overwrite     : {args.overwrite}")

    scene_dirs = sorted(d for d in src_sub.iterdir() if d.is_dir() and d.name.isdigit())

    total_crops = 0
    scenes_with_data = 0
    scenes_skipped_existing = 0

    scene_iter = tqdm(
        scene_dirs,
        desc=f"[{args.src_subset}] scenes",
        leave=True,
        dynamic_ncols=True,
        disable=not IS_TTY,
    )

    for scene_dir in scene_iter:
        out_scene_dir = dst_sub / scene_dir.name

        # If already exists and overwrite not allowed, skip.
        if out_scene_dir.exists() and any(out_scene_dir.iterdir()) and not args.overwrite:
            scenes_skipped_existing += 1
            continue

        num_crops = process_scene(
            scene_dir,
            out_scene_dir,
            obj_id=args.obj_id,
            crop_size=args.size,
            pad_ratio=args.pad,
            overwrite_scene=args.overwrite,
        )

        if num_crops > 0:
            total_crops += num_crops
            scenes_with_data += 1

    print(
        f"\n[Summary] Subset '{args.src_subset}': "
        f"{len(scene_dirs)} scenes scanned, "
        f"{scenes_with_data} scenes produced crops, "
        f"{scenes_skipped_existing} scenes skipped (existing output), "
        f"{total_crops} crops saved."
    )

    # Keypoints saved under <dst>/<dst_subset>/keypoints to avoid any collision with existing LM keypoints.
    if not args.skip_keypoints:
        gen_keypoints(
            src_root=src_root,
            kp_root=dst_sub,
            obj_id=args.obj_id,
            num_kps=args.num_kps,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
