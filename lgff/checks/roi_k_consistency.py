"""
ROI intrinsic consistency checker.

For N random samples:
- Loads depth & mask using SingleObjectDataset helpers.
- Applies the same intrinsic resize logic as the dataloader.
- Projects CAD keypoints with GT pose into the ROI.
- Samples the depth-backprojected point nearest to each keypoint and reprojects it.
- Reports per-sample mean pixel error + directional (y/x) bias.
"""

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from lgff.datasets.single_loader import SingleObjectDataset
from lgff.utils.config import load_config


def _project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points to pixel coordinates."""
    z = np.clip(points_cam[:, 2], a_min=1e-8, a_max=None)
    u = points_cam[:, 0] * K[0, 0] / z + K[0, 2]
    v = points_cam[:, 1] * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def _evaluate_sample(ds: SingleObjectDataset, idx: int) -> Optional[Dict[str, float]]:
    rec = ds.samples[idx]

    depth_scale_scene = float(rec.get("depth_scale", 1.0))
    depth_m, orig_h, orig_w = ds._load_depth(rec["depth_path"], depth_scale_scene)

    roi_xywh: Tuple[int, int, int, int] = rec.get("roi_xywh", (0, 0, orig_w, orig_h))
    pad_xy = rec.get("pad_xy", (0, 0))
    K_scaled = ds.compute_roi_intrinsic(rec["K"], roi_xywh, (ds.resize_w, ds.resize_h), pad_xy)
    pts_img = ds._depth_to_points(depth_m, K_scaled)
    H, W, _ = pts_img.shape

    pose = rec["pose"]
    R = pose[:, :3]
    t = pose[:, 3]
    kp_cam = (R @ ds.kp3d_model.T).T + t  # [K,3]

    uv_kp = _project(kp_cam, K_scaled)

    sampled_uv_from_depth: List[np.ndarray] = []
    uv_valid: List[np.ndarray] = []
    for uv in uv_kp:
        u_int, v_int = int(round(float(uv[0]))), int(round(float(uv[1])))
        if u_int < 0 or v_int < 0 or u_int >= W or v_int >= H:
            continue
        pt = pts_img[v_int, u_int]
        if pt[2] <= 1e-8:
            continue
        uv_depth = _project(pt[None, :], K_scaled)[0]
        sampled_uv_from_depth.append(uv_depth)
        uv_valid.append(uv)

    if not uv_valid:
        return None

    uv_valid_np = np.stack(uv_valid)
    uv_depth_np = np.stack(sampled_uv_from_depth)
    diffs = uv_depth_np - uv_valid_np
    pix_err = np.linalg.norm(diffs, axis=1)

    return {
        "mean_pixel_error": float(np.mean(pix_err)),
        "y_bias": float(np.mean(diffs[:, 1])),
        "x_bias": float(np.mean(diffs[:, 0])),
        "num_keypoints": int(len(pix_err)),
        "roi_xywh": roi_xywh,
        "pad_xy": pad_xy,
        "fx": float(K_scaled[0, 0]),
        "fy": float(K_scaled[1, 1]),
        "cx": float(K_scaled[0, 2]),
        "cy": float(K_scaled[1, 2]),
        "pixel_errors": pix_err.tolist(),
        "x_diffs": diffs[:, 0].tolist(),
        "y_diffs": diffs[:, 1].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI intrinsic consistency checker", add_help=True)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--th-mean", type=float, default=0.5, help="Mean pixel error threshold (px)")
    parser.add_argument("--th-bias", type=float, default=0.2, help="Absolute bias threshold (px)")
    parser.add_argument("--th-p90", type=float, default=0.8, help="p90 pixel error threshold (px)")
    parser.add_argument("--exit-on-fail", action="store_true", help="Exit with code 1 when thresholds fail")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    cfg = load_config()

    ds = SingleObjectDataset(cfg, split=args.split)
    if len(ds) == 0:
        print(f"[Check] Dataset split '{args.split}' is empty. Nothing to evaluate.")
        return

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(ds), size=min(len(ds), args.num_samples), replace=False)

    results: List[Dict[str, float]] = []
    roi_x0_list: List[float] = []
    roi_y0_list: List[float] = []
    x_bias_list: List[float] = []
    y_bias_list: List[float] = []
    for idx in sample_indices:
        res = _evaluate_sample(ds, int(idx))
        if res is None:
            print(f"[Check] Sample idx={idx}: no valid keypoints after ROI clipping, skipped.")
            continue
        results.append(res)
        roi_x0, roi_y0, roi_w, roi_h = res["roi_xywh"]
        roi_x0_list.append(float(roi_x0))
        roi_y0_list.append(float(roi_y0))
        x_bias_list.append(res["x_bias"])
        y_bias_list.append(res["y_bias"])
        print(
            f"[Check] idx={idx} | kp={res['num_keypoints']:02d} | "
            f"roi_xywh={res['roi_xywh']} pad_xy={res['pad_xy']} "
            f"K(fx,fy,cx,cy)=({res['fx']:.3f},{res['fy']:.3f},{res['cx']:.3f},{res['cy']:.3f}) | "
            f"mean_pixel_error={res['mean_pixel_error']:.4f} | "
            f"y_bias={res['y_bias']:.4f} | x_bias={res['x_bias']:.4f}"
        )

    if not results:
        print("[Check] No valid samples to summarize.")
        return

    pixel_err_all = np.concatenate([np.asarray(r["pixel_errors"], dtype=np.float32) for r in results])
    x_err_all = np.concatenate([np.asarray(r["x_diffs"], dtype=np.float32) for r in results])
    y_err_all = np.concatenate([np.asarray(r["y_diffs"], dtype=np.float32) for r in results])

    mean_err = float(np.mean(pixel_err_all))
    mean_y_bias = float(np.mean(y_err_all))
    mean_x_bias = float(np.mean(x_err_all))
    p50_err = float(np.percentile(pixel_err_all, 50))
    p90_err = float(np.percentile(pixel_err_all, 90))
    p50_y = float(np.percentile(y_err_all, 50))
    p90_y = float(np.percentile(y_err_all, 90))
    p50_x = float(np.percentile(x_err_all, 50))
    p90_x = float(np.percentile(x_err_all, 90))

    resize_norm = float(max(ds.resize_w, ds.resize_h))
    norm_mean_err = mean_err / resize_norm
    norm_p90_err = p90_err / resize_norm

    def _safe_corr(a: List[float], b: List[float]) -> str:
        if len(a) < 2 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return "N/A"
        return f"{float(np.corrcoef(a, b)[0,1]):.4f}"

    corr_x = _safe_corr(roi_x0_list, x_bias_list)
    corr_y = _safe_corr(roi_y0_list, y_bias_list)

    print("========== ROI Intrinsic Consistency Summary ==========")
    print(f"Samples evaluated: {len(results)}")
    print(f"Avg mean_pixel_error: {mean_err:.4f} (norm={norm_mean_err:.6f}) | p50={p50_err:.4f} | p90={p90_err:.4f} (norm={norm_p90_err:.6f})")
    print(f"Avg y_bias: {mean_y_bias:.4f} | p50={p50_y:.4f} | p90={p90_y:.4f}")
    print(f"Avg x_bias: {mean_x_bias:.4f} | p50={p50_x:.4f} | p90={p90_x:.4f}")
    print(f"Corr(x_bias, roi_x0)={corr_x} | Corr(y_bias, roi_y0)={corr_y}")
    fail_mean = mean_err > args.th_mean
    fail_bias = (abs(mean_x_bias) > args.th_bias) or (abs(mean_y_bias) > args.th_bias)
    fail_p90 = p90_err > args.th_p90
    status = "PASS" if not (fail_mean or fail_bias or fail_p90) else "FAIL"
    print(f"Thresholds: th_mean={args.th_mean} th_bias={args.th_bias} th_p90={args.th_p90} => {status}")
    print("=======================================================")
    if status == "FAIL" and args.exit_on_fail:
        raise SystemExit(1)
    print("=======================================================")


if __name__ == "__main__":
    main()
