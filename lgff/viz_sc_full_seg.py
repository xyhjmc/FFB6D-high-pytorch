"""
Visualization script for LGFF-Seg (Side-by-Side Layout).
Left: Full Image with Pose Box (Clean).
Right: Cropped Region with Predicted Mask Overlay.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader


from typing import Dict, Tuple, Optional, List
import torch
sys.path.append(os.getcwd())

from lgff.utils.config_seg import load_config, merge_cfg_from_checkpoint
from lgff.utils.geometry import GeometryToolkit
from lgff.utils.logger import setup_logger, get_logger
from lgff.datasets.single_loader_seg import SingleObjectDatasetSeg
from lgff.models.lgff_sc_seg import LGFF_SC_SEG
from lgff.eval_sc import load_model_weights, resolve_checkpoint_path
# Inherit for logic consistency
from lgff.engines.evaluator_sc_seg import EvaluatorSCSeg
from lgff.utils.pose_metrics_seg import fuse_pose_from_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LGFF-Seg Side-by-Side")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint")
    parser.add_argument("--orig-root", required=False,default="/home/xyh/datasets/minepose", type=str, help="Path to ORIGINAL BOP dataset root")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples to visualize")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Interactive mode")

    # Logic flags
    parser.add_argument("--no-pnp", action="store_true", help="Disable PnP")
    parser.add_argument("--no-mask", action="store_true", help="Disable Mask Fusion")

    args, _ = parser.parse_known_args()
    return args


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Helper: Reconstruct Crop (Logic must match Dataloader)
# ----------------------------------------------------------------------
def get_crop_coordinates(pose, K, model_points, img_h, img_w, resize_hw=128):
    """
    Re-calculate the bounding box used for cropping.
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_3d = model_points @ R.T + t

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = pts_3d[:, 0]
    Y = pts_3d[:, 1]
    Z = pts_3d[:, 2].clip(min=1e-3)

    u = (X * fx) / Z + cx
    v = (Y * fy) / Z + cy

    rmin, rmax = np.min(v), np.max(v)
    cmin, cmax = np.min(u), np.max(u)

    # Dataloader padding logic
    pad = 10
    rmin = max(0, int(rmin) - pad)
    rmax = min(img_h - 1, int(rmax) + pad)
    cmin = max(0, int(cmin) - pad)
    cmax = min(img_w - 1, int(cmax) + pad)

    # Make Square (matches SingleObjectDataset logic)
    h_box, w_box = rmax - rmin, cmax - cmin
    box_size = max(h_box, w_box)
    center_r, center_c = (rmin + rmax) / 2, (cmin + cmax) / 2

    # Scale (Testing usually uses 1.0 or 1.1 depending on implementation, assuming 1.0 here for tightness)
    scale = 1.0
    box_size = int(box_size * scale)

    rmin = int(max(0, center_r - box_size // 2))
    cmin = int(max(0, center_c - box_size // 2))
    rmax = int(min(img_h, center_r + box_size // 2))
    cmax = int(min(img_w, center_c + box_size // 2))

    return rmin, rmax, cmin, cmax


# ----------------------------------------------------------------------
# Helper: Load Full BOP Image
# ----------------------------------------------------------------------
def load_full_bop_data(orig_root, scene_id, im_id, split="test"):
    scene_str = f"{scene_id:06d}"
    candidates = [
        os.path.join(orig_root, split, scene_str),
        os.path.join(orig_root, "test", scene_str),
        os.path.join(orig_root, "test_all", scene_str),
    ]
    scene_dir = None
    for c in candidates:
        if os.path.isdir(c):
            scene_dir = c
            break
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {scene_str} not found in {orig_root}")

    rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.jpg")
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    with open(os.path.join(scene_dir, "scene_camera.json"), "r") as f:
        cam_dict = json.load(f)
    K = np.array(cam_dict[str(im_id)]["cam_K"], dtype=np.float32).reshape(3, 3)

    return rgb, K


def build_cube_primitives(points_model, scale=1.0):
    pts = points_model.detach().cpu().numpy()
    mins, maxs = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    center, half = 0.5 * (mins + maxs), 0.5 * (maxs - mins) * scale
    verts = np.array([center + half * np.array([x, y, z]) for x in (-1,1) for y in (-1,1) for z in (-1,1)], dtype=np.float32)
    edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    return verts, edges


def project_points(verts, R, t, K):
    verts_c = np.dot(verts, R.T) + t
    u = (verts_c[:,0] * K[0,0]) / verts_c[:,2] + K[0,2]
    v = (verts_c[:,1] * K[1,1]) / verts_c[:,2] + K[1,2]
    return np.stack([u, v], axis=1)


# ----------------------------------------------------------------------
# Visualizer Class (Fixed Signature)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Visualizer Class (Fixed Method Name)
# ----------------------------------------------------------------------
class VisualizerSCSeg(EvaluatorSCSeg):
    def __init__(self, model, loader, cfg, geometry, save_dir, orig_root, show=False):
        super().__init__(model, loader, cfg, geometry, save_dir)
        self.orig_root = orig_root
        self.show = show
        self.viz_count = 0
        self.max_samples = 50
        self._viz_outputs = {}
        self.complexity_logger = None

    # 1. 拦截预测结果，保存 Mask 信息
    def _process_predictions(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        valid_mask = None
        # 计算 mask 逻辑 (保持不变)
        if bool(getattr(self.cfg, "pose_fusion_use_valid_mask", False)):
            mask_src = str(getattr(self.cfg, "pose_fusion_valid_mask_source", "")).lower()
            if mask_src == "seg":
                valid_mask = self._compute_robust_valid_mask_from_seg(outputs, batch)
            elif mask_src == "labels":
                lbl = batch.get("labels", None)
                if isinstance(lbl, torch.Tensor):
                    valid_mask = lbl > 0
                    if valid_mask.dim() == 1:
                        valid_mask = valid_mask.expand(outputs["pred_quat"].shape[0], -1)

        # 缓存供画图使用
        self._viz_outputs = {
            "pred_mask_logits": outputs.get("pred_mask_logits", None),
            "valid_mask": valid_mask
        }

        # 执行融合
        pred_rt = fuse_pose_from_outputs(outputs, self.geometry, self.cfg, stage="eval", valid_mask=valid_mask)
        return pred_rt, valid_mask

    # 2. [CRITICAL FIX] 重写正确的父类方法名: _compute_pose_metrics
    def _compute_pose_metrics(self, pred_rt, gt_rt, batch):
        # 先让父类把指标算好
        metrics = super()._compute_pose_metrics(pred_rt, gt_rt, batch)

        # 如果已经画够了数量，直接返回
        if self.viz_count >= self.max_samples:
            return metrics

        # 开始画图逻辑
        try:
            scene_id = int(batch["scene_id"][0].item())
            im_id = int(batch["im_id"][0].item())

            # 1. 加载原始大图
            rgb_full, K_full = load_full_bop_data(self.orig_root, scene_id, im_id, self.cfg.val_split)
            H_full, W_full = rgb_full.shape[:2]

            # --- 左图：全图 Pose ---
            mp_cpu = batch["model_points"][0].cpu().numpy()
            gt_pose_cpu = batch["pose"][0].cpu().numpy()
            pred_rt_cpu = pred_rt[0].cpu().numpy()

            verts, edges = build_cube_primitives(batch["model_points"][0])
            v_pred = project_points(verts, pred_rt_cpu[:3, :3], pred_rt_cpu[:3, 3], K_full)
            v_gt = project_points(verts, gt_pose_cpu[:3, :3], gt_pose_cpu[:3, 3], K_full)

            img_left = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
            # 画 GT (红)
            for i, j in edges:
                cv2.line(img_left, tuple(v_gt[i].astype(int)), tuple(v_gt[j].astype(int)), (0, 0, 255), 2)
            # 画 Pred (绿)
            for i, j in edges:
                cv2.line(img_left, tuple(v_pred[i].astype(int)), tuple(v_pred[j].astype(int)), (0, 255, 0), 2)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

            # --- 右图：局部 Mask ---
            # 还原 Crop 区域
            rmin, rmax, cmin, cmax = get_crop_coordinates(gt_pose_cpu, K_full, mp_cpu, H_full, W_full)

            # 边界安全检查
            rmin, cmin = max(0, rmin), max(0, cmin)
            rmax, cmax = min(H_full, rmax), min(W_full, cmax)

            if rmax <= rmin or cmax <= cmin:
                raise ValueError(f"Invalid crop coords: {rmin},{rmax},{cmin},{cmax}")

            img_crop = rgb_full[rmin:rmax, cmin:cmax].copy()
            crop_h, crop_w = img_crop.shape[:2]

            mask_overlay_crop = img_crop.copy()

            # 叠加 Mask
            if self._viz_outputs.get("pred_mask_logits") is not None:
                logits = self._viz_outputs["pred_mask_logits"]
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # [128, 128]

                if crop_h > 0 and crop_w > 0:
                    probs_resized = cv2.resize(probs, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (probs_resized > 0.5).astype(np.uint8)

                    color_mask = np.zeros_like(img_crop)
                    color_mask[mask_bin == 1] = [255, 255, 0]  # 黄色

                    mask_overlay_crop = cv2.addWeighted(img_crop, 0.7, color_mask, 0.3, 0)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_overlay_crop, contours, -1, (255, 255, 255), 1)

            # --- 拼图 & 保存 ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_left)
            axes[0].set_title("Full Image (GT:Red / Pred:Green)")
            axes[0].axis("off")
            axes[1].imshow(mask_overlay_crop)
            axes[1].set_title("Predicted Seg Mask (ROI)")
            axes[1].axis("off")

            # 从刚刚算好的 metrics 里取值
            add_val = metrics["add"][0].item() * 1000
            plt.suptitle(f"Scene {scene_id} Img {im_id} | ADD: {add_val:.1f}mm", fontsize=14, y=0.05)

            save_name = f"viz_{self.viz_count:03d}_{scene_id}_{im_id}.png"
            out_path = os.path.join(str(self.save_dir), save_name)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            self.logger.info(f"Saved {out_path} | ADD={add_val:.1f}mm")
            self.viz_count += 1

        except Exception as e:
            # 打印详细错误，不再静默
            import traceback
            traceback.print_exc()
            self.logger.error(f"Viz Failed for {scene_id}/{im_id}: {e}")

        return metrics

    def run(self):
        if hasattr(self.cfg, "num_viz_samples"):
            self.max_samples = self.cfg.num_viz_samples
        return super().run()

def main():
    args = parse_args()
    cfg_cli = load_config()
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = merge_cfg_from_checkpoint(cfg_cli, ckpt.get("config"))

    # FORCE CORRECT SETTINGS
    cfg.eval_use_pnp = not args.no_pnp
    cfg.pose_fusion_use_valid_mask = not args.no_mask
    if cfg.pose_fusion_use_valid_mask:
        cfg.pose_fusion_valid_mask_source = "seg"

    cfg.num_viz_samples = args.num_samples
    cfg.val_split = args.split

    log_dir = args.save_dir or os.path.join(getattr(cfg, "log_dir", "output"), "viz_side_by_side")
    ensure_dirs(log_dir)
    setup_logger(os.path.join(log_dir, "viz.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = GeometryToolkit()
    dataset = SingleObjectDatasetSeg(cfg, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = LGFF_SC_SEG(cfg, geometry).to(device)
    load_model_weights(model, ckpt_path, device, checkpoint=ckpt)

    visualizer = VisualizerSCSeg(model, loader, cfg, geometry, log_dir, args.orig_root, args.show)
    visualizer.run()

if __name__ == "__main__":
    main()