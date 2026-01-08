# -*- coding: utf-8 -*-
"""
Offline Preprocessing Tool for LGFF (JSON-based, BOP-style).

Fast cropping using 'scene_gt_info.json' bounding boxes,
with optional automatic filtering of low-quality samples (e.g., very low/zero mask_visib).

Key behaviors (kept from your original):
- Per sub-dataset (train_pbr/train_real/...) scene-level tqdm
- Per scene frame-level tqdm only in TTY; otherwise lightweight printing
- Create output scene dir ONLY if at least 1 crop is actually saved
- Print summary stats per subset + overall
- Auto-generate keypoints/obj_xxxxxx.npy using FPS (models/models_eval)

New in this version:
- Robust low-quality filtering:
  * min bbox size
  * min visib_fract (from scene_gt_info.json if present, else derived)
  * min visible mask pixels (from mask_visib/mask)
  * min mask fill ratio in bbox (mask_visib area / bbox area)
  * optional depth validity check inside mask
  * optional strict mode for missing/empty masks

Usage example:
python crop_bop_filter.py \
  --src /path/to/your_bop_dataset \
  --dst /path/to/output_crop_root \
  --obj_id 15 --size 128 --pad 0.15 \
  --min_visib_fract 0.05 \
  --min_mask_visib_px 50 \
  --min_mask_fill_ratio 0.02 \
  --min_bbox_side 10

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
#  Helper: safely get visibility fraction from gt_info
# =========================================================
def get_visib_fract(info_obj: dict) -> float | None:
    """
    Return visibility fraction if available or derivable.
    BOP gt_info often provides one of:
      - visib_fract
      - px_count_visib / px_count_all
    """
    if "visib_fract" in info_obj and info_obj["visib_fract"] is not None:
        try:
            return float(info_obj["visib_fract"])
        except Exception:
            pass

    if "px_count_visib" in info_obj and "px_count_all" in info_obj:
        try:
            pv = float(info_obj["px_count_visib"])
            pa = float(info_obj["px_count_all"])
            if pa > 0:
                return pv / pa
        except Exception:
            pass

    return None


# =========================================================
#  Scene processor
# =========================================================
def process_scene(
    scene_path: Path,
    out_path: Path,
    obj_id: int,
    crop_size: int = 128,
    pad_ratio: float = 0.15,
    # filtering thresholds
    min_bbox_side: int = 10,
    min_visib_fract: float = 0.0,
    min_mask_visib_px: int = 0,
    min_mask_fill_ratio: float = 0.0,
    min_depth_valid_px: int = 0,
    # behavior toggles
    require_mask: bool = True,
    prefer_mask_visib: bool = True,
    strict_empty_mask: bool = True,
    strict_zero_visib_fract: bool = True,
) -> tuple[int, dict]:
    """
    Process one scene and return:
      - num_crops_saved
      - stats dict

    Notes on filtering:
      - If require_mask=True: missing mask -> skip instance
      - If strict_empty_mask=True: mask exists but all-zero -> skip
      - If strict_zero_visib_fract=True and visib_fract==0 -> skip
    """
    scene_path = Path(scene_path)
    out_path = Path(out_path)

    stats = {
        "frames_scanned": 0,
        "instances_found": 0,
        "instances_kept": 0,
        "skip_bbox_small": 0,
        "skip_visib_fract": 0,
        "skip_mask_missing": 0,
        "skip_mask_empty": 0,
        "skip_mask_visib_px": 0,
        "skip_mask_fill_ratio": 0,
        "skip_depth_valid_px": 0,
        "skip_image_missing": 0,
        "skip_read_fail": 0,
    }

    gt_path = scene_path / "scene_gt.json"
    info_path = scene_path / "scene_gt_info.json"
    cam_path = scene_path / "scene_camera.json"

    if not gt_path.exists() or not info_path.exists() or not cam_path.exists():
        tqdm.write(f"[Skip] Missing GT / Info / Camera: {scene_path}")
        return 0, stats

    with open(gt_path, "r") as f:
        scene_gt = json.load(f)
    with open(info_path, "r") as f:
        scene_info = json.load(f)
    with open(cam_path, "r") as f:
        scene_cam = json.load(f)

    new_scene_gt = {}
    new_scene_cam = {}
    out_im_id = 0

    # lazy output dir creation
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
        stats["frames_scanned"] += 1

        im_str = str(im_id)
        if im_str not in scene_gt or im_str not in scene_info or im_str not in scene_cam:
            continue

        gts = scene_gt[im_str]
        infos = scene_info[im_str]

        if not IS_TTY and (frame_idx % 50 == 0 or frame_idx == n_frames - 1):
            print(
                f"[{scene_path.parent.name}/{scene_path.name}] frames {frame_idx+1}/{n_frames}",
                end="\r",
                flush=True,
            )

        # find target instances in this frame
        target_indices = []
        for idx, gt_obj in enumerate(gts):
            if gt_obj.get("obj_id", -1) != obj_id:
                continue
            if idx >= len(infos):
                continue
            bbox = infos[idx].get("bbox_visib", [-1, -1, -1, -1])  # [x,y,w,h]
            if bbox[2] > 1 and bbox[3] > 1:
                target_indices.append(idx)

        if not target_indices:
            continue

        # read rgb/depth (once per frame)
        rgb_f = scene_path / "rgb" / f"{im_id:06d}.png"
        if not rgb_f.exists():
            rgb_f = scene_path / "rgb" / f"{im_id:06d}.jpg"
        depth_f = scene_path / "depth" / f"{im_id:06d}.png"

        if not rgb_f.exists() or not depth_f.exists():
            stats["skip_image_missing"] += len(target_indices)
            continue

        rgb_full = cv2.imread(str(rgb_f))
        depth_full = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
        if rgb_full is None or depth_full is None:
            stats["skip_read_fail"] += len(target_indices)
            continue

        cam_entry = scene_cam[im_str]
        if "cam_K" not in cam_entry:
            stats["skip_read_fail"] += len(target_indices)
            continue

        K_raw = np.array(cam_entry["cam_K"], dtype=np.float32).reshape(3, 3)
        depth_scale = float(cam_entry.get("depth_scale", 1000.0))
        img_h, img_w = rgb_full.shape[:2]

        # process each instance
        for idx in target_indices:
            stats["instances_found"] += 1

            gt_obj = gts[idx]
            info_obj = infos[idx]

            # bbox_visib
            x, y, w, h = info_obj.get("bbox_visib", [-1, -1, -1, -1])
            if w < min_bbox_side or h < min_bbox_side:
                stats["skip_bbox_small"] += 1
                continue

            # visib_fract filter (if available/derivable)
            vf = get_visib_fract(info_obj)
            if vf is not None:
                if strict_zero_visib_fract and vf <= 0.0:
                    stats["skip_visib_fract"] += 1
                    continue
                if vf < float(min_visib_fract):
                    stats["skip_visib_fract"] += 1
                    continue

            # load mask (preferred: mask_visib; fallback: mask)
            # for filtering we evaluate mask FULL IMAGE area (fast + stable)
            mask_name = f"{im_id:06d}_{idx:06d}.png"
            mask_full = None

            if prefer_mask_visib:
                mask_path = scene_path / "mask_visib" / mask_name
                if not mask_path.exists():
                    mask_path = scene_path / "mask" / mask_name
            else:
                mask_path = scene_path / "mask" / mask_name
                if not mask_path.exists():
                    mask_path = scene_path / "mask_visib" / mask_name

            if mask_path.exists():
                mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask_full is None:
                if require_mask:
                    stats["skip_mask_missing"] += 1
                    continue
                # allow missing mask: treat bbox as mask
                mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
                x1m = max(0, int(x))
                y1m = max(0, int(y))
                x2m = min(img_w, int(x + w))
                y2m = min(img_h, int(y + h))
                if x2m > x1m and y2m > y1m:
                    mask_full[y1m:y2m, x1m:x2m] = 255

            # mask visibility px filter in bbox area (focus on the labeled target bbox)
            x1b = max(0, int(x))
            y1b = max(0, int(y))
            x2b = min(img_w, int(x + w))
            y2b = min(img_h, int(y + h))
            if x2b <= x1b or y2b <= y1b:
                stats["skip_bbox_small"] += 1
                continue

            mask_bbox = mask_full[y1b:y2b, x1b:x2b]
            visib_px = int(np.count_nonzero(mask_bbox))

            if strict_empty_mask and visib_px == 0:
                stats["skip_mask_empty"] += 1
                continue
            if visib_px < int(min_mask_visib_px):
                stats["skip_mask_visib_px"] += 1
                continue

            bbox_area = int((x2b - x1b) * (y2b - y1b))
            fill_ratio = (visib_px / float(bbox_area)) if bbox_area > 0 else 0.0
            if fill_ratio < float(min_mask_fill_ratio):
                stats["skip_mask_fill_ratio"] += 1
                continue

            # crop region (square-ish with padding around bbox center)
            cx, cy = x + w / 2.0, y + h / 2.0
            size = max(w, h) * (1.0 + pad_ratio)

            x1 = int(max(0, cx - size / 2.0))
            y1 = int(max(0, cy - size / 2.0))
            x2 = int(min(img_w, cx + size / 2.0))
            y2 = int(min(img_h, cy + size / 2.0))

            if (x2 - x1) < min_bbox_side or (y2 - y1) < min_bbox_side:
                stats["skip_bbox_small"] += 1
                continue

            # crop
            rgb_crop = rgb_full[y1:y2, x1:x2]
            depth_crop = depth_full[y1:y2, x1:x2]
            mask_crop = mask_full[y1:y2, x1:x2]

            # optional depth validity filter (inside mask)
            if min_depth_valid_px > 0:
                mask_bin = (mask_crop > 0)
                if mask_bin.any():
                    depth_valid = (depth_crop > 0) & mask_bin
                    valid_px = int(np.count_nonzero(depth_valid))
                else:
                    valid_px = 0
                if valid_px < int(min_depth_valid_px):
                    stats["skip_depth_valid_px"] += 1
                    continue

            # first time writing => create dirs
            ensure_out_dirs()

            # resize
            rgb_fin = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            depth_fin = cv2.resize(depth_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            mask_fin = cv2.resize(mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            mask_fin = (mask_fin > 0).astype(np.uint8) * 255

            # recompute K for crop+resize
            real_h, real_w = rgb_crop.shape[:2]
            if real_w <= 0 or real_h <= 0:
                stats["skip_read_fail"] += 1
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

            new_scene_gt[out_id_str] = [gt_obj]
            new_scene_cam[out_id_str] = {
                "cam_K": K_new.reshape(-1).tolist(),
                "depth_scale": float(depth_scale),
            }

            out_im_id += 1
            stats["instances_kept"] += 1

    if out_im_id == 0:
        tqdm.write(f"[Skip] Scene {scene_path.name}: no valid crops for obj_id={obj_id}")
        return 0, stats

    with open(out_path / "scene_gt.json", "w") as f:
        json.dump(new_scene_gt, f, indent=2)
    with open(out_path / "scene_camera.json", "w") as f:
        json.dump(new_scene_cam, f, indent=2)

    tqdm.write(f"[Done] Scene {scene_path.name} -> {out_im_id} crops saved at {out_path}")
    return out_im_id, stats


# =========================================================
#  FPS keypoints
# =========================================================
def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
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


def gen_keypoints(src_root: Path, dst_root: Path, obj_id: int, num_kps: int = 8):
    kp_out_dir = dst_root / "keypoints"
    kp_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = kp_out_dir / f"obj_{obj_id:06d}.npy"

    print(f"\n[Keypoints] Generating keypoints for obj_{obj_id} using FPS...")

    model_path = src_root / "models" / f"obj_{obj_id:06d}.ply"
    if not model_path.exists():
        model_path = src_root / "models_eval" / f"obj_{obj_id:06d}.ply"

    if not model_path.exists():
        print(f"  -> [Error] Cannot find model .ply at {model_path}. Skipping.")
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
            print(f"  -> [Warning] Mesh has fewer points ({pts.shape[0]}) than num_kps ({num_kps}).")
            indices = np.random.choice(pts.shape[0], num_kps, replace=True)
            kps = pts[indices]
        else:
            kps = farthest_point_sampling(pts, num_kps)
            print(f"  -> FPS Sampling done. Selected {num_kps} points.")

        np.save(str(out_path), kps.astype(np.float32))
        print(f"  -> Saved to: {out_path}")

    except Exception as e:
        print(f"  -> [Error] Failed to process obj_{obj_id}: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
#  Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Crop BOP dataset based on scene_gt_info.json (with quality filtering).")

    parser.add_argument("--src", default="/home/xyh/datasets/minepose",
                        help="Root of raw dataset (e.g. datasets/bop/lm)")
    parser.add_argument("--dst", default="/home/xyh/PycharmProjects/FFB6D-high-pytorch/lgff/data/minepose_shovel_crop128_all",
                        help="Output root for cropped dataset")
    parser.add_argument("--obj_id", type=int, default=2, help="Object ID to crop")
    parser.add_argument("--size", type=int, default=128, help="Target resize size")
    parser.add_argument("--pad", type=float, default=0.15, help="Padding ratio")

    # --- filtering args ---
    parser.add_argument("--min_bbox_side", type=int, default=30,
                        help="Skip instances whose bbox_visib w/h < this value (default: 10)")
    parser.add_argument("--min_visib_fract", type=float, default=0.45,
                        help="Minimum visibility fraction (visib_fract or derived). (default: 0.0)")
    parser.add_argument("--min_mask_visib_px", type=int, default=300,
                        help="Minimum non-zero pixels of mask (inside bbox_visib). (default: 0)")
    parser.add_argument("--min_mask_fill_ratio", type=float, default=0.1,
                        help="Minimum mask fill ratio inside bbox: mask_px / bbox_area. (default: 0.0)")
    parser.add_argument("--min_depth_valid_px", type=int, default=200,
                        help="Minimum valid depth pixels (>0) inside mask within crop. (default: 0)")

    parser.add_argument("--require_mask", action="store_true",
                        help="If set, missing mask_visib/mask => skip instance (recommended for noisy datasets).")
    parser.add_argument("--prefer_mask_visib", action="store_true", default=True,
                        help="If set, prefer mask_visib first, else mask first. (default: False)")
    parser.add_argument("--strict_empty_mask", action="store_true",
                        help="If set, empty (all-zero) mask inside bbox => skip instance.")
    parser.add_argument("--strict_zero_visib_fract", action="store_true",
                        help="If set, visib_fract==0 => skip instance (when visib_fract is available).")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    sub_dirs = ["train_pbr", "train_real", "train_synt", "test", "val"]

    total_crops = 0
    total_scenes_with_data = 0

    overall_stats = {
        "frames_scanned": 0,
        "instances_found": 0,
        "instances_kept": 0,
        "skip_bbox_small": 0,
        "skip_visib_fract": 0,
        "skip_mask_missing": 0,
        "skip_mask_empty": 0,
        "skip_mask_visib_px": 0,
        "skip_mask_fill_ratio": 0,
        "skip_depth_valid_px": 0,
        "skip_image_missing": 0,
        "skip_read_fail": 0,
    }

    for sub in sub_dirs:
        src_sub = src_root / sub
        if not src_sub.exists():
            continue

        tqdm.write(f"[Info] Found sub-dataset: {sub}")
        dst_sub = dst_root / sub
        dst_sub.mkdir(parents=True, exist_ok=True)

        scene_dirs = sorted(d for d in src_sub.iterdir() if d.is_dir() and d.name.isdigit())

        sub_crops = 0
        sub_scenes_with_data = 0

        sub_stats = {k: 0 for k in overall_stats.keys()}

        scene_iter = tqdm(
            scene_dirs,
            desc=f"[{sub}] scenes",
            leave=True,
            dynamic_ncols=True,
            disable=not IS_TTY,
        )

        for scene_dir in scene_iter:
            out_scene_dir = dst_sub / scene_dir.name
            num_crops, st = process_scene(
                scene_dir,
                out_scene_dir,
                obj_id=args.obj_id,
                crop_size=args.size,
                pad_ratio=args.pad,
                min_bbox_side=args.min_bbox_side,
                min_visib_fract=args.min_visib_fract,
                min_mask_visib_px=args.min_mask_visib_px,
                min_mask_fill_ratio=args.min_mask_fill_ratio,
                min_depth_valid_px=args.min_depth_valid_px,
                require_mask=args.require_mask,
                prefer_mask_visib=args.prefer_mask_visib,
                strict_empty_mask=args.strict_empty_mask,
                strict_zero_visib_fract=args.strict_zero_visib_fract,
            )

            for k in sub_stats.keys():
                sub_stats[k] += int(st.get(k, 0))

            if num_crops > 0:
                sub_crops += num_crops
                sub_scenes_with_data += 1

        total_crops += sub_crops
        total_scenes_with_data += sub_scenes_with_data

        for k in overall_stats.keys():
            overall_stats[k] += sub_stats[k]

        print(
            f"\n[Summary] Sub-dataset '{sub}': "
            f"{len(scene_dirs)} scenes scanned, "
            f"{sub_scenes_with_data} scenes with valid crops, "
            f"{sub_crops} crops saved."
        )
        print(
            f"[Filter Stats] found={sub_stats['instances_found']}, kept={sub_stats['instances_kept']}, "
            f"skip_bbox_small={sub_stats['skip_bbox_small']}, "
            f"skip_visib_fract={sub_stats['skip_visib_fract']}, "
            f"skip_mask_missing={sub_stats['skip_mask_missing']}, "
            f"skip_mask_empty={sub_stats['skip_mask_empty']}, "
            f"skip_mask_visib_px={sub_stats['skip_mask_visib_px']}, "
            f"skip_mask_fill_ratio={sub_stats['skip_mask_fill_ratio']}, "
            f"skip_depth_valid_px={sub_stats['skip_depth_valid_px']}, "
            f"skip_image_missing={sub_stats['skip_image_missing']}, "
            f"skip_read_fail={sub_stats['skip_read_fail']}"
        )

    print(
        f"\n[Overall Summary] Total crops: {total_crops}, "
        f"scenes with valid crops across all subsets: {total_scenes_with_data}."
    )
    print(
        f"[Overall Filter Stats] found={overall_stats['instances_found']}, kept={overall_stats['instances_kept']}, "
        f"skip_bbox_small={overall_stats['skip_bbox_small']}, "
        f"skip_visib_fract={overall_stats['skip_visib_fract']}, "
        f"skip_mask_missing={overall_stats['skip_mask_missing']}, "
        f"skip_mask_empty={overall_stats['skip_mask_empty']}, "
        f"skip_mask_visib_px={overall_stats['skip_mask_visib_px']}, "
        f"skip_mask_fill_ratio={overall_stats['skip_mask_fill_ratio']}, "
        f"skip_depth_valid_px={overall_stats['skip_depth_valid_px']}, "
        f"skip_image_missing={overall_stats['skip_image_missing']}, "
        f"skip_read_fail={overall_stats['skip_read_fail']}"
    )

    # keypoints
    gen_keypoints(src_root, dst_root, args.obj_id)


if __name__ == "__main__":
    main()
