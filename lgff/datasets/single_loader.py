# lgff/datasets/single_loader.py
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

from lgff.utils.config import LGFFConfig


def _binary_erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Pure-numpy binary erosion with a square kernel of size (2*radius+1).
    radius=0 -> identity.
    """
    if radius <= 0:
        return mask.astype(bool)

    mask = mask.astype(bool)
    k = 2 * radius + 1
    pad = radius
    # pad with False so erosion shrinks mask at borders
    m = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)

    # sliding window view (H, W, k, k)
    try:
        win = np.lib.stride_tricks.sliding_window_view(m, (k, k))
        return np.all(win, axis=(-1, -2))
    except Exception:
        # fallback: very small radius only
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
    Return a boolean mask where True indicates "non-edge" pixels.
    Pixels near strong depth discontinuities are marked False.
    """
    if thresh_m <= 0:
        return np.ones_like(depth_m, dtype=bool)

    d = depth_m.astype(np.float32)
    # invalid depth treated as 0; gradient around zeros will be large -> filtered out
    dx = np.abs(d[:, 1:] - d[:, :-1])
    dy = np.abs(d[1:, :] - d[:-1, :])

    edge = np.zeros_like(d, dtype=bool)
    edge[:, 1:] |= dx > thresh_m
    edge[:, :-1] |= dx > thresh_m
    edge[1:, :] |= dy > thresh_m
    edge[:-1, :] |= dy > thresh_m

    return ~edge


class SingleObjectDataset(Dataset):
    """
    SingleObjectDataset: single-object BOP-format loader.

    Outputs:
        rgb:          [3, H, W] float32
        point_cloud:  [N, 3]    float32 (camera frame, meters)
        pose:         [3, 4]    float32 (GT [R|t], meters)
        intrinsic:    [3, 3]    float32 (scaled to resized resolution)
        cls_id:       []        long (single-class = 0)
        points:       [N, 3]    float32 (alias of point_cloud)
        model_points: [M, 3]    float32 (meters)

        kp_targ_ofst: [N, K, 3] float32
        labels:       [N]       long (1=foreground-valid, 0=invalid/placeholder)
        kp3d_model:   [K, 3]    float32
        kp3d_cam:     [K, 3]    float32
        pcld_valid_mask: [N]    bool (extra debug/robustness hook)
    """

    # global registry for overlap checking
    _split_scene_registry: Dict[str, set] = {}

    def __init__(self, cfg: LGFFConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.dataset_root)
        self.num_points = int(cfg.num_points)

        # mm -> m divisor (usually 1000.0)
        self.depth_scale = float(getattr(cfg, "depth_scale", 1000.0))

        self.resize_h = int(getattr(cfg, "resize_h", 480))
        self.resize_w = int(getattr(cfg, "resize_w", 640))

        self.obj_id = int(getattr(cfg, "obj_id", 1))

        # Robust sampling controls (B-1)
        self.depth_z_min_m = float(getattr(cfg, "depth_z_min_m", 0.10))
        self.depth_z_max_m = float(getattr(cfg, "depth_z_max_m", 5.00))
        self.mask_erosion = int(getattr(cfg, "mask_erosion", 1))          # 0 to disable
        self.depth_edge_thresh_m = float(getattr(cfg, "depth_edge_thresh_m", 0.0))  # <=0 to disable

        self.mask_invalid_policy = str(getattr(cfg, "mask_invalid_policy", "skip")).lower()
        self.allow_mask_fallback = bool(getattr(cfg, "allow_mask_fallback", False))
        self.scene_overlap_policy = str(getattr(cfg, "scene_overlap_policy", "warn")).lower()
        self.forbid_scene_overlap = bool(getattr(cfg, "forbid_scene_overlap", False))

        # ICP point cloud controls
        self.icp_num_points = int(getattr(cfg, "icp_num_points", 8192))
        self.icp_use_full_depth = bool(getattr(cfg, "icp_use_full_depth", True))

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
            self.color_aug = T.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
            )
        else:
            self.color_aug = None

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.samples: List[Dict[str, Any]] = []
        self._missing_mask = 0
        self._invalid_mask = 0
        self._fallback_mask = 0
        self._mask_fallback_used_count = 0
        self._fallback_warned = False
        self._build_index()

        if len(self.samples) == 0:
            if self.split == "train":
                raise RuntimeError(
                    f"[SingleObjectDataset] No samples found for split={split}, "
                    f"obj_id={self.obj_id} under {self.root}"
                )
            else:
                print(
                    f"[Warning] No samples found for split={split}, "
                    f"obj_id={self.obj_id} under {self.root}"
                )
        else:
            print(
                f"[SingleObjectDataset] split={split}, obj_id={self.obj_id}, "
                f"num_samples={len(self.samples)} | "
                f"z_range=[{self.depth_z_min_m},{self.depth_z_max_m}]m | "
                f"erosion={self.mask_erosion} | edge_th={self.depth_edge_thresh_m}"
            )
            if self._missing_mask > 0:
                print(f"[SingleObjectDataset] skipped {self._missing_mask} samples due to missing masks.")
            if self._invalid_mask > 0:
                print(f"[SingleObjectDataset] skipped {self._invalid_mask} samples due to invalid masks.")
            if self._fallback_mask > 0:
                print(f"[SingleObjectDataset] fallback-to-depth for {self._fallback_mask} samples (allow_mask_fallback=True).")

        # optional scene overlap guard
        if self.forbid_scene_overlap:
            reg = SingleObjectDataset._split_scene_registry.setdefault(self.split, set())
            current_scenes = {s["scene_id"] for s in self.samples}
            reg.update(current_scenes)
            for other_split, scenes in SingleObjectDataset._split_scene_registry.items():
                if other_split == self.split:
                    continue
                overlap = current_scenes & scenes
                if overlap:
                    msg = (
                        f"[SingleObjectDataset][SceneOverlap] overlap scenes {sorted(overlap)} "
                        f"between splits '{self.split}' and '{other_split}'"
                    )
                    if self.scene_overlap_policy == "raise":
                        raise RuntimeError(msg)
                    else:
                        print(msg)

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
        else:
            for sub in ["test", "val", "test_all"]:
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
                    t = np.array(gt["cam_t_m2c"], dtype=np.float32).reshape(3, 1) / 1000.0  # mm->m
                    pose = np.concatenate([R, t], axis=1)  # [3,4]

                    im_name = f"{im_id:06d}.png"
                    rgb_path = scene_dir / "rgb" / im_name
                    if not rgb_path.exists():
                        rgb_path = scene_dir / "rgb" / f"{im_id:06d}.jpg"
                    depth_path = scene_dir / "depth" / im_name

                    inst_name = f"{im_id:06d}_{gt_idx:06d}.png"
                    mask_visib = scene_dir / "mask_visib" / inst_name
                    mask_full = scene_dir / "mask" / inst_name
                    mask_path: Optional[Path] = None
                    if mask_visib.exists():
                        mask_path = mask_visib
                    elif mask_full.exists():
                        mask_path = mask_full

                    mask_status = "ok"
                    if not rgb_path.exists() or not depth_path.exists():
                        continue

                    if mask_path is None:
                        self._missing_mask += 1
                        if self.allow_mask_fallback:
                            mask_status = "fallback_depth"
                            self._fallback_mask += 1
                        elif self.mask_invalid_policy == "raise":
                            raise RuntimeError(
                                f"[SingleObjectDataset] Mask missing for scene={scene_dir.name}, im_id={im_id}, gt_idx={gt_idx}"
                            )
                        else:
                            print(
                                f"[SingleObjectDataset][Warning] missing mask for scene={scene_dir.name} "
                                f"im_id={im_id} gt_idx={gt_idx}, sample skipped."
                            )
                            continue
                    else:
                        try:
                            m_img = Image.open(mask_path)
                            m_arr = np.array(m_img)
                            if m_arr.ndim == 3:
                                m_arr = m_arr[..., 0]
                            nonzero = np.count_nonzero(m_arr)
                            d_img = Image.open(depth_path)
                            dw, dh = d_img.size
                            if m_arr.shape[0] != dh or m_arr.shape[1] != dw:
                                raise ValueError("mask shape mismatch depth")
                            if nonzero == 0:
                                raise ValueError("mask all zero")
                        except Exception as e:
                            self._invalid_mask += 1
                            if self.allow_mask_fallback:
                                mask_status = "fallback_depth"
                                self._fallback_mask += 1
                                print(
                                    f"[SingleObjectDataset][Warning] invalid mask ({e}) for scene={scene_dir.name} "
                                    f"im_id={im_id} gt_idx={gt_idx}, using depth fallback."
                                )
                            elif self.mask_invalid_policy == "raise":
                                raise RuntimeError(
                                    f"[SingleObjectDataset] Invalid mask for scene={scene_dir.name}, im_id={im_id}, gt_idx={gt_idx}: {e}"
                                )
                            else:
                                print(
                                    f"[SingleObjectDataset][Warning] invalid mask for scene={scene_dir.name} "
                                    f"im_id={im_id} gt_idx={gt_idx}, sample skipped. err={e}"
                                )
                                continue

                    self.samples.append(
                        {
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path if mask_status == "ok" else None,
                            "K": K,
                            "depth_scale": depth_scale_scene,
                            "pose": pose,
                            "scene_id": int(scene_dir.name),
                            "im_id": im_id,
                            "mask_status": mask_status,
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
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32) / 1000.0  # mm->m

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
                        print(f"[SingleObjectDataset] load keypoints from {kp_path}, K={self.num_keypoints}")
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

        print(f"[SingleObjectDataset] sample keypoints from model_points, K={self.num_keypoints}")
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

        # raw -> mm -> m
        depth_m = d * float(depth_scale_scene) / float(self.depth_scale)
        # sanitize
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0] = 0.0

        return depth_m.astype(np.float32), orig_h, orig_w

    def _load_mask(self, path: Optional[Path], depth_m: np.ndarray) -> Optional[np.ndarray]:
        if path is None:
            return None
        try:
            m_img = Image.open(path)
            if self.resize_h and self.resize_w:
                m_img = m_img.resize((self.resize_w, self.resize_h), resample=Image.Resampling.NEAREST)
            m = np.array(m_img).astype(np.uint8)
            if m.ndim == 3:
                m = m[..., 0]
            mask = (m > 0)
            return mask.astype(bool)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Robust valid mask (B-1)
    # ------------------------------------------------------------------
    def _build_valid_pixel_mask(self, mask: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """
        valid = mask & depth_valid & (optional erosion) & (optional edge_filter)
        """
        # depth validity
        depth_valid = (
            np.isfinite(depth_m)
            & (depth_m > self.depth_z_min_m)
            & (depth_m < self.depth_z_max_m)
        )

        valid = mask.astype(bool) & depth_valid

        # remove strong depth discontinuities (edge noise)
        if self.depth_edge_thresh_m > 0:
            non_edge = _depth_edge_mask(depth_m, self.depth_edge_thresh_m)
            valid &= non_edge

        # erode to avoid boundary pixels
        if self.mask_erosion > 0:
            valid = _binary_erosion(valid, radius=self.mask_erosion)

        return valid.astype(bool)

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

    def _sample_points(self, pts: np.ndarray, valid_pix: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points only from valid pixels. If none, return zeros and valid_mask=0.
        """
        pts_flat = pts.reshape(-1, 3)
        valid_flat = valid_pix.reshape(-1)

        valid_idxs = np.where(valid_flat)[0]
        if valid_idxs.size == 0:
            sampled = np.zeros((num_points, 3), dtype=np.float32)
            valid_mask = np.zeros((num_points,), dtype=bool)
            return sampled, valid_mask

        # sample with/without replacement
        if valid_idxs.size >= num_points:
            choice = np.random.choice(valid_idxs, size=num_points, replace=False)
        else:
            choice = np.random.choice(valid_idxs, size=num_points, replace=True)

        sampled_pts = pts_flat[choice].astype(np.float32)
        # sampled points are valid by construction
        valid_mask = np.ones((num_points,), dtype=bool)
        return sampled_pts, valid_mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]

        depth_scale_scene = float(rec.get("depth_scale", 1.0))
        depth_m, orig_h, orig_w = self._load_depth(rec["depth_path"], depth_scale_scene)

        K = rec["K"].astype(np.float32).copy()
        if self.resize_h and self.resize_w and (orig_h != self.resize_h or orig_w != self.resize_w):
            scale_x = self.resize_w / float(orig_w)
            scale_y = self.resize_h / float(orig_h)
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y

        rgb_np = self._load_rgb(rec["rgb_path"])
        raw_mask = self._load_mask(rec.get("mask_path"), depth_m)
        mask_status = rec.get("mask_status", "ok")

        # runtime mask validation
        mask_invalid = (
            raw_mask is None
            or raw_mask.shape != depth_m.shape
            or raw_mask.dtype != np.bool_
            or np.count_nonzero(raw_mask) == 0
        )

        if mask_invalid:
            if self.allow_mask_fallback:
                raw_mask = (depth_m > 1e-8)
                mask_status = "fallback_depth"
                self._mask_fallback_used_count += 1
                if not self._fallback_warned:
                    print("[SingleObjectDataset][Warning] using depth>0 fallback for invalid mask; enable allow_mask_fallback=False to block.")
                    self._fallback_warned = True
            elif self.mask_invalid_policy == "raise":
                raise RuntimeError(
                    f"[SingleObjectDataset] invalid mask encountered at runtime for idx={idx} (status={mask_status})"
                )
            else:
                raise RuntimeError(
                    f"[SingleObjectDataset] invalid mask encountered (skip policy) for idx={idx}"
                )

        # B-1: build robust valid pixel mask
        valid_pix = self._build_valid_pixel_mask(raw_mask, depth_m)

        pts_img = self._depth_to_points(depth_m, K)  # [H,W,3]
        pcld, pcld_valid_mask = self._sample_points(pts_img, valid_pix, self.num_points)  # [N,3]

        # ICP point cloud (independent density / source)
        icp_source = pts_img[valid_pix] if self.icp_use_full_depth else pcld
        if icp_source.shape[0] == 0:
            icp_sampled = np.zeros((self.icp_num_points, 3), dtype=np.float32)
        else:
            replace_icp = icp_source.shape[0] < self.icp_num_points
            choice_icp = np.random.choice(
                icp_source.shape[0],
                size=self.icp_num_points,
                replace=replace_icp,
            )
            icp_sampled = icp_source[choice_icp].astype(np.float32)

        rgb_t = self.normalize(self.to_tensor(rgb_np))
        points_t = torch.from_numpy(pcld.astype(np.float32))
        points_icp_t = torch.from_numpy(icp_sampled.astype(np.float32))
        pose_t = torch.from_numpy(rec["pose"].astype(np.float32))
        K_t = torch.from_numpy(K.astype(np.float32))

        # --- keypoint offsets ---
        R = pose_t[:, :3]     # [3,3]
        t = pose_t[:, 3]      # [3]
        kp_model = self.kp3d_model_torch.to(dtype=torch.float32)  # [K,3]
        kp_cam = (R @ kp_model.T).T + t.unsqueeze(0)              # [K,3]

        kp_targ_ofst = kp_cam.unsqueeze(0) - points_t.unsqueeze(1)  # [N,K,3]

        # labels: 1 for valid sampled points, 0 for placeholders (only happens when no valid pixels)
        labels = torch.from_numpy(pcld_valid_mask.astype(np.int64))  # [N]

        return {
            "rgb": rgb_t,
            "point_cloud": points_t,
            "points": points_t,  # alias for loss/eval
            "points_icp": points_icp_t,
            "pose": pose_t,
            "intrinsic": K_t,
            "cls_id": torch.tensor(self.obj_id, dtype=torch.long),

            "model_points": self.model_points_torch,  # [M,3]
            "scene_id": torch.tensor(int(rec.get("scene_id", -1)), dtype=torch.long),
            "im_id": torch.tensor(int(rec.get("im_id", -1)), dtype=torch.long),
            "mask_status": mask_status,

            # kp supervision
            "kp_targ_ofst": kp_targ_ofst,           # [N,K,3]
            "labels": labels,                       # [N]
            "kp3d_model": self.kp3d_model_torch,     # [K,3]
            "kp3d_cam": kp_cam,                     # [K,3]

            # extra robustness hook
            "pcld_valid_mask": torch.from_numpy(pcld_valid_mask.astype(np.bool_)),  # [N]
        }


__all__ = ["SingleObjectDataset"]
