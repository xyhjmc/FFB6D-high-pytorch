"""Single-class dataset loader for LGFF.

This loader follows a BOP-style layout and reuses camera/geometry
utilities from the original FFB6D codebase.  Each sample dictionary is
expected to contain RGB/depth paths along with pose/keypoint metadata.
When annotation files are not provided, synthetic placeholders are
returned so the training loop can still exercise the pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.config import LGFFConfig
from common.geometry import GeometryToolkit
from lgff.utils.geometry import build_geometry


class SingleObjectDataset(Dataset):
    def __init__(self, cfg: LGFFConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.geometry: GeometryToolkit = build_geometry(cfg.camera_intrinsic)
        self.annotations = self._load_annotations(cfg.annotation_file, split)

    def _load_annotations(self, annotation_file: Optional[str], split: str) -> List[Dict[str, Any]]:
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and split in data:
                return data[split]
            if isinstance(data, list):
                return data

        # Fallback: build a synthetic annotation to keep the pipeline alive
        dummy = dict(
            rgb=None,
            depth=None,
            intrinsic=self.cfg.camera_intrinsic,
            pose=np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1).tolist(),
            kps_3d=np.zeros((self.cfg.num_keypoints, 3)).tolist(),
            cls_id=1,
        )
        return [dummy]

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_rgb(self, path: Optional[str]) -> np.ndarray:
        if path and os.path.exists(path):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _load_depth(self, path: Optional[str]) -> np.ndarray:
        if path and os.path.exists(path):
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return np.zeros((480, 640), dtype=np.float32)
            return depth.astype(np.float32)
        return np.zeros((480, 640), dtype=np.float32)

    def _sample_points(self, cloud: np.ndarray, choose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if choose is None or cloud is None:
            cloud = np.zeros((self.cfg.num_points, 3), dtype=np.float32)
            choose = np.arange(self.cfg.num_points, dtype=np.int64)
            return cloud, choose

        if cloud.shape[0] < self.cfg.num_points:
            repeat = int(np.ceil(self.cfg.num_points / cloud.shape[0]))
            cloud = np.tile(cloud, (repeat, 1))
            choose = np.tile(choose, repeat)
        idx = np.random.choice(cloud.shape[0], self.cfg.num_points, replace=False)
        return cloud[idx], choose[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        rgb = self._load_rgb(ann.get("rgb"))
        depth = self._load_depth(ann.get("depth"))
        intrinsic = np.asarray(ann.get("intrinsic", self.cfg.camera_intrinsic))

        point_cloud, choose = self.geometry.depth_to_point_cloud(depth, self.cfg.depth_scale, intrinsic)
        point_cloud, choose = self._sample_points(point_cloud, choose)

        pose = np.asarray(ann.get("pose"))
        if pose.shape == (4, 4):
            pose = pose[:3, :]
        kps_3d = np.asarray(ann.get("kps_3d", np.zeros((self.cfg.num_keypoints, 3))))

        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0,
            "depth": torch.from_numpy(depth).unsqueeze(0),
            "point_cloud": torch.from_numpy(point_cloud).float(),
            "choose": torch.from_numpy(choose.astype(np.int64)),
            "pose": torch.from_numpy(pose).float(),
            "kps_3d": torch.from_numpy(kps_3d).float(),
            "intrinsic": torch.from_numpy(intrinsic).float(),
            "cls_id": torch.tensor(int(ann.get("cls_id", 1)), dtype=torch.long),
        }


__all__ = ["SingleObjectDataset"]
