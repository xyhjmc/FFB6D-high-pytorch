# lgff/datasets/single_loader_seg.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.append(os.getcwd())

from lgff.utils.config_seg import LGFFConfigSeg


def _binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)

    mask = mask.astype(bool)
    k = 2 * radius + 1
    pad = radius
    m = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)

    try:
        win = np.lib.stride_tricks.sliding_window_view(m, (k, k))
        return np.all(win, axis=(-1, -2))
    except Exception:
        out = mask.copy()
        for _ in range(radius):
            up = np.pad(out, ((1, 0), (0, 0)), mode="constant", constant_values=False)[:-1, :]
            dn = np.pad(out, ((0, 1), (0, 0)), mode="constant", constant_values=False)[1:, :]
            lf = np.pad(out, ((0, 0), (1, 0)), mode="constant", constant_values=False)[:, :-1]
            rt = np.pad(out, ((0, 0), (0, 1)), mode="constant", constant_values=False)[:, 1:]
            out = out & up & dn & lf & rt
        return out


def _depth_edge_mask(depth_m: np.ndarray, thresh_m: float) -> np.ndarray:
    """
    True = "non-edge".
    """
    if thresh_m <= 0:
        return np.ones_like(depth_m, dtype=bool)

    d = depth_m.astype(np.float32)
    dx = np.abs(d[:, 1:] - d[:, :-1])
    dy = np.abs(d[1:, :] - d[:-1, :])

    edge = np.zeros_like(d, dtype=bool)
    edge[:, 1:] |= dx > thresh_m
    edge[:, :-1] |= dx > thresh_m
    edge[1:, :] |= dy > thresh_m
    edge[:-1, :] |= dy > thresh_m

    return ~edge


class SingleObjectDatasetSeg(Dataset):
    """
    Single-object BOP-format loader (Seg Enhanced).

    Mask semantics (important):
    - raw_mask: GT object mask (foreground=1, background=0)
    - depth_valid_pts: global depth reliability mask for POINT sampling (can include edge suppression)
    - valid_pix_roi: ROI pixels for POINT sampling (raw_mask & depth_valid_pts & optional erosion)

    Returns (optional):
    - mask: GT object mask (for seg supervision)
    - mask_valid: seg loss valid region (NOT necessarily depth_valid_pts; now decoupled)
    - choose: flattened pixel indices of sampled points (0..H*W-1)
    """

    def __init__(self, cfg: LGFFConfigSeg, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.dataset_root)
        self.num_points = int(cfg.num_points)

        self.depth_scale = float(getattr(cfg, "depth_scale", 1000.0))

        self.resize_h = int(getattr(cfg, "resize_h", 480))
        self.resize_w = int(getattr(cfg, "resize_w", 640))

        self.obj_id = int(getattr(cfg, "obj_id", 1))

        self.return_mask = bool(getattr(cfg, "return_mask", False))
        self.return_valid_mask = bool(getattr(cfg, "return_valid_mask", False))

        # Robust sampling controls (POINTS)
        self.depth_z_min_m = float(getattr(cfg, "depth_z_min_m", 0.10))
        self.depth_z_max_m = float(getattr(cfg, "depth_z_max_m", 5.00))
        self.mask_erosion = int(getattr(cfg, "mask_erosion", 1))  # for point sampling only
        self.depth_edge_thresh_m = float(getattr(cfg, "depth_edge_thresh_m", 0.05))  # for point sampling only

        # Seg loss valid region policy (NEW / decoupled)
        # Options: "all" (recommended), "depth_z", "depth_valid"
        self.seg_loss_valid_source = str(getattr(cfg, "seg_loss_valid_source", "all")).lower().strip()
        # If you still want depth constraints for seg loss, DO NOT use edge suppression by default
        self.seg_loss_use_edge = bool(getattr(cfg, "seg_loss_use_edge", False))
        self.seg_loss_z_min_m = float(getattr(cfg, "seg_loss_z_min_m", self.depth_z_min_m))
        self.seg_loss_z_max_m = float(getattr(cfg, "seg_loss_z_max_m", self.depth_z_max_m))
        self.seg_loss_edge_thresh_m = float(getattr(cfg, "seg_loss_edge_thresh_m", self.depth_edge_thresh_m))

        # CAD points
        self.num_model_points = int(getattr(cfg, "num_model_points", self.num_points))
        self.model_points = self._load_model_points()
        self.model_points_torch = torch.from_numpy(self.model_points.astype(np.float32))

        # keypoints
        self.num_keypoints = int(getattr(cfg, "num_keypoints", getattr(cfg, "n_keypoints", 8)))
        self.kp3d_model = self._load_or_sample_keypoints()
        self.kp3d_model_torch = torch.from_numpy(self.kp3d_model.astype(np.float32))

        # color aug
        if split == "train":
            self.color_aug = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
        else:
            self.color_aug = None

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.samples: List[Dict[str, Any]] = []
        self._build_index()

        if len(self.samples) == 0:
            if self.split == "train":
                raise RuntimeError(
                    f"[SingleObjectDatasetSeg] No samples found for split={split}, obj_id={self.obj_id} under {self.root}"
                )
            else:
                print(f"[Warning] No samples found for split={split}, obj_id={self.obj_id} under {self.root}")
        else:
            print(
                f"[SingleObjectDatasetSeg] split={split}, obj_id={self.obj_id}, num_samples={len(self.samples)} | "
                f"pts_z_range=[{self.depth_z_min_m},{self.depth_z_max_m}]m | "
                f"pts_erosion={self.mask_erosion} | pts_edge_th={self.depth_edge_thresh_m} | "
                f"seg_valid_source={self.seg_loss_valid_source} | seg_use_edge={self.seg_loss_use_edge} | "
                f"return_mask={self.return_mask} | return_valid_mask={self.return_valid_mask}"
            )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _get_scene_dirs(self) -> List[Path]:
        candidates: List[Path] = []
        if self.split == "train":
            for sub in ["train_pbr", "train_synt", "train_real", "train"]:
                d = self.root / sub
                if d.is_dir():
                    candidates.append(d)
        elif self.split in ("val", "test"):
            for sub in ["test", "val", "test_all"]:
                d = self.root / sub
                if d.is_dir():
                    candidates.append(d)
        else:
            for sub in ["test_lmo", self.split]:
                d = self.root / sub
                if d.is_dir():
                    candidates.append(d)

        scene_dirs: List[Path] = []
        for split_dir in candidates:
            for sd in sorted(split_dir.iterdir()):
                if sd.is_dir() and sd.name.isdigit():
                    scene_dirs.append(sd)
        return scene_dirs

    def _build_index(self) -> None:
        scene_dirs = self._get_scene_dirs()
        for scene_dir in scene_dirs:
            gt_path = scene_dir / "scene_gt.json"
            cam_path = scene_dir / "scene_camera.json"
            if not gt_path.exists() or not cam_path.exists():
                continue

            with open(gt_path, "r") as f:
                scene_gt = json.load(f)
            with open(cam_path, "r") as f:
                scene_cam = json.load(f)

            for im_id_str, gt_list in scene_gt.items():
                im_id = int(im_id_str)
                cam_info = scene_cam[im_id_str]

                K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_scale_scene = float(cam_info.get("depth_scale", 1.0))

                for gt_idx, gt in enumerate(gt_list):
                    if int(gt["obj_id"]) != self.obj_id:
                        continue

                    R = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                    t = np.array(gt["cam_t_m2c"], dtype=np.float32).reshape(3, 1) / 1000.0
                    pose = np.concatenate([R, t], axis=1)

                    im_name = f"{im_id:06d}.png"
                    rgb_path = scene_dir / "rgb" / im_name
                    if not rgb_path.exists():
                        rgb_path = scene_dir / "rgb" / f"{im_id:06d}.jpg"
                    depth_path = scene_dir / "depth" / im_name

                    inst_name = f"{im_id:06d}_{gt_idx:06d}.png"
                    mask_visib = scene_dir / "mask_visib" / inst_name
                    mask_full = scene_dir / "mask" / inst_name
                    mask_visib_path = mask_visib if mask_visib.exists() else None
                    mask_full_path = mask_full if mask_full.exists() else None

                    if mask_visib_path is None and mask_full_path is None and self.split == "train":
                        continue
                    if not rgb_path.exists() or not depth_path.exists():
                        continue

                    self.samples.append(
                        {
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_visib_path": mask_visib_path,
                            "mask_full_path": mask_full_path,
                            "K": K,
                            "depth_scale": depth_scale_scene,
                            "pose": pose,
                            "scene_id": int(scene_dir.name),
                            "im_id": im_id,
                        }
                    )

    # ------------------------------------------------------------------
    # CAD model points
    # ------------------------------------------------------------------
    def _load_model_points(self) -> np.ndarray:
        model_dir_candidates = [self.root / "models_eval", self.root / "models"]
        ply_path: Optional[Path] = None
        for d in model_dir_candidates:
            candidate = d / f"obj_{self.obj_id:06d}.ply"
            if candidate.exists():
                ply_path = candidate
                break

        if ply_path is None:
            print(f"[Warning] CAD model for obj_id={self.obj_id} not found under {self.root}.")
            return np.zeros((self.num_model_points, 3), dtype=np.float32)

        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32) / 1000.0

        if pts.shape[0] > self.num_model_points:
            rng = np.random.default_rng(seed=0)
            choice = rng.choice(pts.shape[0], self.num_model_points, replace=False)
            pts = pts[choice]

        return pts.astype(np.float32)

    # ------------------------------------------------------------------
    # Keypoints
    # ------------------------------------------------------------------
    def _load_or_sample_keypoints(self) -> np.ndarray:
        kp_dir = self.root / "keypoints"
        if kp_dir.is_dir():
            kp_path = kp_dir / f"obj_{self.obj_id:06d}.npy"
            if kp_path.exists():
                try:
                    kp = np.asarray(np.load(kp_path), dtype=np.float32)
                    if kp.ndim == 2 and kp.shape[1] == 3:
                        if kp.shape[0] > self.num_keypoints:
                            rng = np.random.default_rng(seed=0)
                            kp = kp[rng.choice(kp.shape[0], self.num_keypoints, replace=False)]
                        elif kp.shape[0] < self.num_keypoints:
                            rng = np.random.default_rng(seed=0)
                            extra = rng.choice(kp.shape[0], self.num_keypoints - kp.shape[0], replace=True)
                            kp = np.concatenate([kp, kp[extra]], axis=0)
                        print(f"[SingleObjectDatasetSeg] load keypoints from {kp_path}, K={self.num_keypoints}")
                        return kp.astype(np.float32)
                    else:
                        print(f"[Warning] keypoints file {kp_path} has shape {kp.shape}, expected [*,3].")
                except Exception as e:
                    print(f"[Warning] failed to load keypoints from {kp_path}: {e}")

        pts = self.model_points
        rng = np.random.default_rng(seed=1)
        if pts.shape[0] >= self.num_keypoints:
            kp = pts[rng.choice(pts.shape[0], self.num_keypoints, replace=False)]
        else:
            extra = rng.choice(pts.shape[0], self.num_keypoints - pts.shape[0], replace=True)
            kp = np.concatenate([pts, pts[extra]], axis=0)

        print(f"[SingleObjectDatasetSeg] sample keypoints from model_points, K={self.num_keypoints}")
        return kp.astype(np.float32)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def _load_rgb(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if self.resize_h and self.resize_w:
            img = img.resize((self.resize_w, self.resize_h), resample=Image.Resampling.BILINEAR)
        if self.color_aug is not None:
            img = self.color_aug(img)
        return np.array(img)

    def _load_depth(self, path: Path, depth_scale_scene: float) -> Tuple[np.ndarray, int, int]:
        d_img = Image.open(path)
        orig_w, orig_h = d_img.size

        if self.resize_h and self.resize_w:
            d_img = d_img.resize((self.resize_w, self.resize_h), resample=Image.Resampling.NEAREST)

        d = np.array(d_img).astype(np.float32)

        depth_m = d * float(depth_scale_scene) / float(self.depth_scale)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0

        return depth_m.astype(np.float32), orig_h, orig_w

    def _load_mask(self, path: Optional[Path], depth_m: np.ndarray) -> np.ndarray:
        if path is not None and path.exists():
            m_img = Image.open(path)
            if self.resize_h and self.resize_w:
                m_img = m_img.resize((self.resize_w, self.resize_h), resample=Image.Resampling.NEAREST)
            m = np.array(m_img).astype(np.uint8)
            if m.ndim == 3:
                m = m[..., 0]
            mask = (m > 0)
        else:
            mask = (depth_m > 1e-8)
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def _build_depth_valid_mask_for_points(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Depth validity for POINT sampling (can include edge suppression).
        """
        depth_valid = (
            np.isfinite(depth_m)
            & (depth_m > self.depth_z_min_m)
            & (depth_m < self.depth_z_max_m)
        )
        if self.depth_edge_thresh_m > 0:
            depth_valid &= _depth_edge_mask(depth_m, self.depth_edge_thresh_m)
        return depth_valid.astype(bool)

    def _build_seg_loss_valid_mask(self, depth_m: np.ndarray, depth_valid_pts: np.ndarray) -> np.ndarray:
        """
        Seg loss valid region (decoupled from point-depth validity).

        Recommended default: "all" -> supervise full image regardless of depth holes/edges.
        Optional:
        - "depth_z": use only z-range (NO edge suppression by default)
        - "depth_valid": same as point validity (not recommended for seg)
        """
        src = self.seg_loss_valid_source

        if src == "all":
            return np.ones_like(depth_m, dtype=bool)

        if src == "depth_valid":
            # if you insist, optionally allow edge suppression
            if self.seg_loss_use_edge:
                dv = depth_valid_pts
            else:
                dv = (
                    np.isfinite(depth_m)
                    & (depth_m > self.seg_loss_z_min_m)
                    & (depth_m < self.seg_loss_z_max_m)
                )
            return dv.astype(bool)

        if src == "depth_z":
            dv = (
                np.isfinite(depth_m)
                & (depth_m > self.seg_loss_z_min_m)
                & (depth_m < self.seg_loss_z_max_m)
            )
            if self.seg_loss_use_edge and self.seg_loss_edge_thresh_m > 0:
                dv &= _depth_edge_mask(depth_m, self.seg_loss_edge_thresh_m)
            return dv.astype(bool)

        # fallback
        return np.ones_like(depth_m, dtype=bool)

    def _build_roi_sampling_mask(self, raw_mask: np.ndarray, depth_valid_pts: np.ndarray) -> np.ndarray:
        """
        ROI pixels for POINT sampling.
        """
        valid_pix_roi = raw_mask.astype(bool) & depth_valid_pts.astype(bool)
        if self.mask_erosion > 0:
            valid_pix_roi = _binary_erosion(valid_pix_roi, radius=self.mask_erosion)
        return valid_pix_roi.astype(bool)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _depth_to_points(self, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
        H, W = depth_m.shape
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        Z = depth_m
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        pts[~np.isfinite(pts)] = 0.0
        return pts

    def _sample_points(
        self, pts: np.ndarray, valid_pix: np.ndarray, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts_flat = pts.reshape(-1, 3)
        valid_flat = valid_pix.reshape(-1)
        valid_idxs = np.where(valid_flat)[0]

        if valid_idxs.size == 0:
            sampled = np.zeros((num_points, 3), dtype=np.float32)
            valid_mask = np.zeros((num_points,), dtype=bool)
            choose = np.zeros((num_points,), dtype=np.int64)
            return sampled, valid_mask, choose

        if valid_idxs.size >= num_points:
            choice = np.random.choice(valid_idxs, size=num_points, replace=False)
        else:
            choice = np.random.choice(valid_idxs, size=num_points, replace=True)

        sampled_pts = pts_flat[choice].astype(np.float32)
        valid_mask = np.ones((num_points,), dtype=bool)
        return sampled_pts, valid_mask, choice.astype(np.int64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]

        depth_scale_scene = float(rec.get("depth_scale", 1.0))
        depth_m, orig_h, orig_w = self._load_depth(rec["depth_path"], depth_scale_scene)

        # Update K if resized
        K = rec["K"].astype(np.float32).copy()
        if self.resize_h and self.resize_w and (orig_h != self.resize_h or orig_w != self.resize_w):
            scale_x = self.resize_w / float(orig_w)
            scale_y = self.resize_h / float(orig_h)
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y

        rgb_np = self._load_rgb(rec["rgb_path"])

        seg_src = str(getattr(self.cfg, "seg_supervision_source", "mask_visib")).lower().strip()
        seg_enabled = bool(getattr(self.cfg, "use_seg_head", False)) or str(getattr(self.cfg, "model_variant", "sc")).endswith("seg")

        if seg_src == "mask_full":
            mask_path = rec.get("mask_full_path", None)
        else:
            mask_path = rec.get("mask_visib_path", None)

        if mask_path is None:
            if seg_enabled or self.return_mask or self.return_valid_mask:
                raise RuntimeError(
                    f"[SingleObjectDatasetSeg] seg_supervision_source={seg_src} but mask is missing "
                    f"for scene {rec.get('scene_id', '?')} image {rec.get('im_id', '?')}."
                )
            raw_mask = self._load_mask(None, depth_m)
        else:
            raw_mask = self._load_mask(mask_path, depth_m)

        # 1) depth validity for POINTS
        depth_valid_pts = self._build_depth_valid_mask_for_points(depth_m)

        # 2) ROI pixels for POINT sampling
        valid_pix_roi = self._build_roi_sampling_mask(raw_mask, depth_valid_pts)

        # 3) points from depth
        pts_img = self._depth_to_points(depth_m, K)
        pcld, pcld_valid_mask, choose_idx = self._sample_points(pts_img, valid_pix_roi, self.num_points)

        # 4) seg loss valid region (DECOUPLED)
        #    This mask is only used if you return mask_valid and seg_ignore_invalid is True in loss.
        if bool(getattr(self.cfg, "seg_ignore_invalid", True)):
            loss_valid_mask = self._build_seg_loss_valid_mask(depth_m, depth_valid_pts)
        else:
            loss_valid_mask = np.ones_like(depth_valid_pts, dtype=bool)

        # tensors
        rgb_t = self.normalize(self.to_tensor(rgb_np))
        points_t = torch.from_numpy(pcld.astype(np.float32))
        pose_t = torch.from_numpy(rec["pose"].astype(np.float32))
        K_t = torch.from_numpy(K.astype(np.float32))
        choose_t = torch.from_numpy(choose_idx.astype(np.int64))

        # keypoint offsets
        R = pose_t[:, :3]
        t = pose_t[:, 3]
        kp_model = self.kp3d_model_torch.to(dtype=torch.float32)
        kp_cam = (R @ kp_model.T).T + t.unsqueeze(0)
        kp_targ_ofst = kp_cam.unsqueeze(0) - points_t.unsqueeze(1)

        labels = torch.from_numpy(pcld_valid_mask.astype(np.int64))

        out: Dict[str, torch.Tensor] = {
            "rgb": rgb_t,
            "point_cloud": points_t,
            "points": points_t,
            "pose": pose_t,
            "intrinsic": K_t,
            "cls_id": torch.tensor(0, dtype=torch.long),
            "model_points": self.model_points_torch,
            "scene_id": torch.tensor(int(rec.get("scene_id", -1)), dtype=torch.long),
            "im_id": torch.tensor(int(rec.get("im_id", -1)), dtype=torch.long),
            "kp_targ_ofst": kp_targ_ofst,
            "labels": labels,
            "kp3d_model": self.kp3d_model_torch,
            "kp3d_cam": kp_cam,
            "pcld_valid_mask": torch.from_numpy(pcld_valid_mask.astype(np.bool_)),
            "choose": choose_t,
        }

        if self.return_mask:
            out["mask"] = torch.from_numpy(raw_mask.astype(np.float32)).unsqueeze(0)

        if self.return_valid_mask:
            out["mask_valid"] = torch.from_numpy(loss_valid_mask.astype(np.float32)).unsqueeze(0)

        return out


__all__ = ["SingleObjectDatasetSeg"]
