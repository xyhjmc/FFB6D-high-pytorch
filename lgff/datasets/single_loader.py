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

# 确保能导入 common 模块 (如果直接运行此脚本)
sys.path.append(os.getcwd())

from lgff.utils.config import LGFFConfig


class SingleObjectDataset(Dataset):
    """
    SingleObjectDataset: 针对单一物体的 BOP 格式数据集加载器。

    输出字段：
        - rgb:         [3, H, W] float32 (归一化后)
        - point_cloud: [N, 3]    float32 (相机坐标系下的点云)
        - pose:        [3, 4]    float32 (GT [R|t]，单位 m)
        - intrinsic:   [3, 3]    float32 (相机内参，对应当前网络输入分辨率)
        - cls_id:      []        long (单类固定为 0)
        - points:      [N, 3]    float32 (= point_cloud, 便于 loss 复用)
        - model_points:[M, 3]    float32 (CAD 模型点，单位 m)
    """

    def __init__(self, cfg: LGFFConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.dataset_root)
        self.num_points = cfg.num_points
        self.depth_scale = cfg.depth_scale

        # [优化] 从 Config 读取 Resize 尺寸，默认 480x640 (BOP标准)
        # 如果显存不够，可以在 yaml 中设置 resize_h=240, resize_w=320
        self.resize_h = getattr(cfg, "resize_h", 480)
        self.resize_w = getattr(cfg, "resize_w", 640)

        # BOP 中的物体 ID（与模型 ID 对应），需要在 YAML 中指定
        self.obj_id = getattr(cfg, "obj_id", 1)

        # CAD 模型点，供 ADD / ADD-S 计算使用
        self.num_model_points = getattr(cfg, "num_model_points", cfg.num_points)
        self.model_points = self._load_model_points()
        self.model_points_torch = torch.from_numpy(self.model_points)

        # 颜色增强 (仅训练开启)
        if split == "train":
            self.color_aug = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self.color_aug = None

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.samples: List[Dict[str, Any]] = []
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
                f"[SingleObjectDataset] split={split}, "
                f"obj_id={self.obj_id}, num_samples={len(self.samples)}"
            )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _get_scene_dirs(self) -> List[Path]:
        """根据 split 查找场景目录"""
        candidates: List[Path] = []
        if self.split == "train":
            # 优先 PBR / 合成 / 实拍
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
                # 简单过滤非数字文件夹 (e.g. 000000)
                if sd.is_dir() and sd.name.isdigit():
                    scene_dirs.append(sd)
        return scene_dirs

    def _build_index(self) -> None:
        """从 BOP 的 scene_gt / scene_camera 中建立样本列表"""
        scene_dirs = self._get_scene_dirs()
        for scene_dir in scene_dirs:
            gt_path = scene_dir / "scene_gt.json"
            cam_path = scene_dir / "scene_camera.json"

            # 没有 gt 的场景无法用于监督训练
            if not gt_path.exists() or not cam_path.exists():
                continue

            with open(gt_path, "r") as f:
                scene_gt = json.load(f)
            with open(cam_path, "r") as f:
                scene_cam = json.load(f)

            for im_id_str, gt_list in scene_gt.items():
                im_id = int(im_id_str)
                cam_info = scene_cam[im_id_str]
                # BOP 内参是 3x3 展平列表
                K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)

                for gt_idx, gt in enumerate(gt_list):
                    if gt["obj_id"] != self.obj_id:
                        continue

                    # Pose：BOP 中 cam_t_m2c 单位为 mm，这里统一转为 m
                    R = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                    t = np.array(gt["cam_t_m2c"], dtype=np.float32).reshape(3, 1)

                    # 单位统一为 mm

                    t = t / 1000.0  # mm -> m

                    pose = np.concatenate([R, t], axis=1)  # [3,4]

                    # 构建路径
                    im_name = f"{im_id:06d}.png"
                    rgb_path = scene_dir / "rgb" / im_name
                    if not rgb_path.exists():
                        rgb_path = scene_dir / "rgb" / f"{im_id:06d}.jpg"

                    depth_path = scene_dir / "depth" / im_name

                    # Mask 路径 (优先 visible mask)
                    inst_name = f"{im_id:06d}_{gt_idx:06d}.png"
                    mask_visib = scene_dir / "mask_visib" / inst_name
                    mask_full = scene_dir / "mask" / inst_name

                    mask_path: Optional[Path] = None
                    if mask_visib.exists():
                        mask_path = mask_visib
                    elif mask_full.exists():
                        mask_path = mask_full

                    # [关键] 训练阶段无法缺少 mask（不然采不到干净物体点云），直接跳过
                    if mask_path is None and self.split == "train":
                        continue

                    if not rgb_path.exists() or not depth_path.exists():
                        continue

                    self.samples.append(
                        {
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "K": K,
                            "pose": pose,
                            "scene_id": int(scene_dir.name),
                            "im_id": im_id,
                        }
                    )

    def _load_model_points(self) -> np.ndarray:
        """读取 CAD 模型点 (单位 m)。

        优先使用 ``models_eval``，若不存在则回退到 ``models``。默认下采样到
        ``num_model_points`` 以兼顾精度和计算量。
        """

        model_dir_candidates = [self.root / "models_eval", self.root / "models"]
        ply_path: Optional[Path] = None
        for d in model_dir_candidates:
            candidate = d / f"obj_{self.obj_id:06d}.ply"
            if candidate.exists():
                ply_path = candidate
                break

        if ply_path is None:
            print(
                f"[Warning] CAD model for obj_id={self.obj_id} not found under {self.root}."
            )
            return np.zeros((self.num_model_points, 3), dtype=np.float32)

        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        # BOP 模型单位通常为毫米，转为米以匹配深度/位姿单位
        pts = pts / 1000.0

        if pts.shape[0] > self.num_model_points:
            rng = np.random.default_rng(seed=0)
            choice = rng.choice(pts.shape[0], self.num_model_points, replace=False)
            pts = pts[choice]

        return pts

    # ------------------------------------------------------------------
    # IO & geometry helpers
    # ------------------------------------------------------------------
    def _load_rgb(self, path: Path) -> np.ndarray:
        """读取 RGB 并 resize（如果指定）"""
        img = Image.open(path).convert("RGB")
        # Resize: 使用双线性插值
        if self.resize_h and self.resize_w:
            img = img.resize((self.resize_w, self.resize_h), resample=Image.BILINEAR)

        if self.color_aug is not None:
            img = self.color_aug(img)
        return np.array(img)  # [H,W,3], uint8

    def _load_depth(
        self, path: Path
    ) -> Tuple[np.ndarray, int, int]:
        """
        读取深度图，返回:
            depth_m: [H,W] float32, 单位 m
            orig_h, orig_w: resize 前的原始分辨率
        """
        d_img = Image.open(path)
        orig_w, orig_h = d_img.size  # PIL: (width, height)

        # Resize: 深度图必须使用最近邻插值，保持像素值原义
        if self.resize_h and self.resize_w:
            d_img = d_img.resize((self.resize_w, self.resize_h), resample=Image.NEAREST)

        d = np.array(d_img).astype(np.float32)
        depth_m = d / float(self.depth_scale)  # 通常 depth_scale=1000

        return depth_m, orig_h, orig_w

    def _load_mask(self, path: Optional[Path], depth_m: np.ndarray) -> np.ndarray:
        if path is not None and path.exists():
            m_img = Image.open(path)
            # Mask Resize: 最近邻插值
            if self.resize_h and self.resize_w:
                m_img = m_img.resize((self.resize_w, self.resize_h), resample=Image.NEAREST)
            
            m = np.array(m_img).astype(np.uint8)
            # 有些 mask 是 3 通道 (255, 255, 255)，转单通道
            if m.ndim == 3:
                m = m[..., 0]
            return m > 0
        else:
            # 仅在验证/测试且确实没 mask 时 fallback：用 depth>0 作为前景
            return depth_m > 1e-8

    def _depth_to_points(self, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        depth_m: [H,W] in meters
        K: [3,3] 内参矩阵（此处假定已经针对当前分辨率调整过）
        """
        H, W = depth_m.shape
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        Z = depth_m
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        return np.stack([X, Y, Z], axis=-1)  # [H,W,3]

    def _sample_points(self, pts: np.ndarray, mask: np.ndarray, num_points: int) -> np.ndarray:
        """
        从点云中根据 mask 采样 num_points 个点。
        """
        pts_flat = pts.reshape(-1, 3)
        mask_flat = mask.reshape(-1)

        valid_idxs = np.where(mask_flat)[0]

        if valid_idxs.size == 0:
            # 极端情况：mask 为空，返回全 0 点云
            return np.zeros((num_points, 3), dtype=np.float32)

        if valid_idxs.size >= num_points:
            choice = np.random.choice(valid_idxs, size=num_points, replace=False)
        else:
            choice = np.random.choice(valid_idxs, size=num_points, replace=True)

        return pts_flat[choice, :].astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]

        # 1. 读 depth，拿到原始分辨率 (orig_h, orig_w)
        depth_m, orig_h, orig_w = self._load_depth(rec["depth_path"])

        # 2. 根据 orig_h, orig_w 与目标 resize 尺寸，动态缩放内参 K
        K = rec["K"].astype(np.float32).copy()
        if self.resize_h and self.resize_w and (orig_h != self.resize_h or orig_w != self.resize_w):
            scale_x = self.resize_w / float(orig_w)
            scale_y = self.resize_h / float(orig_h)
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y

        # 3. 读 RGB（已经 resize 到同一分辨率）
        rgb_np = self._load_rgb(rec["rgb_path"])

        # 4. 读 Mask（已经与 depth / rgb 同分辨率）
        mask = self._load_mask(rec["mask_path"], depth_m)

        # 5. 用调整后的 K 进行深度反投影
        pts = self._depth_to_points(depth_m, K)        # [H,W,3]
        pcld = self._sample_points(pts, mask, self.num_points)  # [N,3]

        # 6. RGB -> tensor + normalize
        rgb_t = self.to_tensor(rgb_np)                # [3,H,W], 0~1
        rgb_t = self.normalize(rgb_t)                 # ImageNet 标准化

        points_t = torch.from_numpy(pcld)             # [N,3]
        pose_t = torch.from_numpy(rec["pose"].astype(np.float32))  # [3,4]

        return {
            "rgb": rgb_t,
            "point_cloud": points_t,
            "pose": pose_t,
            "intrinsic": torch.from_numpy(K),
            "cls_id": torch.tensor(0, dtype=torch.long),  # 单类固定 0
            "points": points_t,
            "model_points": self.model_points_torch,
        }


# ======================================================================
#  Debug Block: 直接运行此脚本可测试 DataLoader 是否正常
# ======================================================================
if __name__ == "__main__":
    import traceback
    from torch.utils.data import DataLoader

    # 模拟一个 Config
    class MockConfig:
        # 请根据你的实际路径修改这里
        dataset_root = "/home/xyh/datasets/LM(BOP)/lm"
        obj_id = 1
        num_points = 1024
        depth_scale = 1000.0
        resize_h = 480
        resize_w = 640

    cfg = MockConfig()

    try:
        print(f"Testing SingleObjectDataset with root: {cfg.dataset_root}")
        # 如果你没有对应路径，这一步会报 Warning 或 Error
        ds = SingleObjectDataset(cfg, split="train")

        if len(ds) > 0:
            print(f"Dataset Size: {len(ds)}")
            
            # 1. Test __getitem__
            sample = ds[0]
            print("\n[Sample 0 Info]")
            print(f"RGB Shape: {sample['rgb'].shape}")
            print(f"PC Shape:  {sample['point_cloud'].shape}")
            print(f"Pose:      \n{sample['pose']}")
            print(f"K (Resized):\n{sample['intrinsic']}")

            # 2. Test DataLoader Batching
            dl = DataLoader(ds, batch_size=4, shuffle=True)
            batch = next(iter(dl))
            print("\n[Batch Info]")
            print(f"Batch RGB: {batch['rgb'].shape}")
            print(f"Batch PC:  {batch['point_cloud'].shape}")
            print("Test Passed! ✅")
        else:
            print("Dataset is empty. Please verify your `dataset_root` and `obj_id`. ⚠️")

    except Exception as e:
        print(f"Test Failed with error: {e} ❌")
        traceback.print_exc()