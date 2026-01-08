#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from common import Config
from common.ffb6d_utils.basic_utils import Basic_Utils
from common.ffb6d_utils.torch_normals import depth_normal
from ffb6d.models.RandLA.helper_tool import DataProcessing as DP


class BOPDataset:
    """
    Minimal-intrusive BOP dataloader for FFB6D.
    Output dict keys match original lm_dataset.py Dataset.get_item().
    """

    def __init__(
        self,
        dataset_name: str,
        cls_type: str,
        cfg_path: str | None = None,
        DEBUG: bool = False,
    ):
        """
        dataset_name: 'train' / 'test' (or you define)
        cls_type: single class name, used for logging and bs_utils interface
        cfg_path: optional yaml for Config(ds_name='bop')
        """
        self.DEBUG = DEBUG
        self.dataset_name = dataset_name
        self.add_noise = (dataset_name == "train")

        # [CHANGED] Use new config branch: ds_name='bop'
        self.config = Config(ds_name="bop", cls_type=cls_type, cfg_path=cfg_path)
        self.bs_utils = Basic_Utils(self.config)

        self.cls_type = cls_type
        self.bop_root = self.config.bop_root
        self.bop_obj_id = int(self.config.bop_obj_id)
        self.use_mask_visib = bool(self.config.bop_use_mask_visib)

        # [CHANGED] Load keypoints/center from config npy (still monkey-patch bs_utils)
        kps_npy = self.config.bop_kps_npy
        ctr_npy = self.config.bop_ctr_npy
        assert kps_npy and os.path.exists(kps_npy), f"Missing kps_npy: {kps_npy}"
        assert ctr_npy and os.path.exists(ctr_npy), f"Missing ctr_npy: {ctr_npy}"

        self._kps_obj = np.load(kps_npy).astype(np.float32)         # [K,3] in object frame
        self._ctr_obj = np.load(ctr_npy).astype(np.float32).reshape(-1)  # [3] or [1,3]

        # unit normalize: if mm -> m
        if self._kps_obj.max() > 10:
            self._kps_obj = self._kps_obj / 1000.0
        if self._ctr_obj.max() > 10:
            self._ctr_obj = self._ctr_obj / 1000.0
        self._ctr_obj = self._ctr_obj.reshape(3)

        self.bs_utils.get_kps = lambda *args, **kwargs: self._kps_obj.copy()
        self.bs_utils.get_ctr = lambda *args, **kwargs: self._ctr_obj.copy()

        # [CHANGED] split list from config
        split_txt = self.config.bop_train_list if dataset_name == "train" else self.config.bop_test_list
        assert split_txt and os.path.exists(split_txt), f"Missing split_txt: {split_txt}"
        with open(split_txt, "r", encoding="utf-8") as f:
            self.all_lst = [ln.strip() for ln in f.readlines() if ln.strip()]

        # color aug
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        # xmap/ymap dynamic
        self._xmap = None
        self._ymap = None

        print(f"[BOPDataset] {dataset_name}_dataset_size:", len(self.all_lst))

    # ------------------------- utils -------------------------
    def _ensure_xy_maps(self, h: int, w: int):
        if self._xmap is None or self._xmap.shape != (h, w):
            xs = np.tile(np.arange(w)[None, :], (h, 1))
            ys = np.tile(np.arange(h)[:, None], (1, w))
            self._xmap = xs
            self._ymap = ys

    def dpt_2_pcld(self, dpt_m: np.ndarray, K: np.ndarray):
        """dpt_m: meters, shape [H,W]"""
        h, w = dpt_m.shape
        self._ensure_xy_maps(h, w)
        msk = (dpt_m > 1e-8).astype(np.float32)
        row = (self._ymap - K[0][2]) * dpt_m / K[0][0]
        col = (self._xmap - K[1][2]) * dpt_m / K[1][1]
        dpt_3d = np.stack([row, col, dpt_m], axis=2) * msk[:, :, None]
        return dpt_3d

    def _load_scene_ann(self, scene_dir: str):
        cam_path = os.path.join(scene_dir, "scene_camera.json")
        gt_path = os.path.join(scene_dir, "scene_gt.json")
        with open(cam_path, "r", encoding="utf-8") as f:
            scene_cam = json.load(f)
        with open(gt_path, "r", encoding="utf-8") as f:
            scene_gt = json.load(f)
        return scene_cam, scene_gt

    def _parse_item_name(self, item_name: str):
        """
        support:
          - "scene_id/im_id"  -> subset = self.dataset_name
          - "subset/scene_id/im_id"
        """
        parts = item_name.split("/")
        if len(parts) == 2:
            subset = self.dataset_name
            scene_id, im_id = parts
        elif len(parts) == 3:
            subset, scene_id, im_id = parts
        else:
            raise ValueError(f"Bad item_name format: {item_name}")
        return subset, scene_id, im_id

    def _pick_instance(self, gt_list):
        """
        return (inst_dict, inst_idx) for the first matching obj_id.
        BOP mask usually uses inst_idx in filename: {im_id}_{inst_idx}.png
        """
        for idx, inst in enumerate(gt_list):
            if int(inst["obj_id"]) == self.bop_obj_id:
                return inst, idx
        return None, None

    def _read_rgb(self, scene_dir: str, im_id: int):
        # try png then jpg
        p_png = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        p_jpg = os.path.join(scene_dir, "rgb", f"{im_id:06d}.jpg")
        p = p_png if os.path.exists(p_png) else p_jpg
        if not os.path.exists(p):
            raise FileNotFoundError(f"RGB not found: {p_png} or {p_jpg}")
        with Image.open(p) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
        return rgb

    def _read_depth(self, scene_dir: str, im_id: int):
        p = os.path.join(scene_dir, "depth", f"{im_id:06d}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Depth not found: {p}")
        with Image.open(p) as di:
            dpt_raw = np.array(di)  # uint16
        return dpt_raw

    def _read_mask(self, scene_dir: str, im_id: int, inst_idx: int):
        """
        Robust:
          1) try {im_id}_{obj_id}.png  (your custom format)
          2) try {im_id}_{inst_idx}.png (standard BOP)
          3) fallback: glob {im_id}_*.png and pick one if single
        """
        mask_dir = "mask_visib" if self.use_mask_visib else "mask"
        base = os.path.join(scene_dir, mask_dir)

        p_obj = os.path.join(base, f"{im_id:06d}_{self.bop_obj_id:06d}.png")
        if os.path.exists(p_obj):
            p = p_obj
        else:
            p_inst = os.path.join(base, f"{im_id:06d}_{inst_idx:06d}.png")
            if os.path.exists(p_inst):
                p = p_inst
            else:
                cands = sorted(glob.glob(os.path.join(base, f"{im_id:06d}_*.png")))
                if len(cands) == 1:
                    p = cands[0]
                else:
                    raise FileNotFoundError(
                        f"Mask not found for im_id={im_id}, obj_id={self.bop_obj_id}, inst_idx={inst_idx}. "
                        f"Tried: {p_obj}, {p_inst}, glob={len(cands)}"
                    )

        with Image.open(p) as mi:
            labels = np.array(mi)
        if labels.ndim > 2:
            labels = labels[:, :, 0]
        labels = (labels > 0).astype("uint8")  # single class {0,1}
        return labels

    # ------------------------- GT builder -------------------------
    def get_pose_gt_info(self, cld, labels, RT):
        RTs = np.zeros((self.config.n_objects, 3, 4), dtype=np.float32)
        kp3ds = np.zeros((self.config.n_objects, self.config.n_keypoints, 3), dtype=np.float32)
        ctr3ds = np.zeros((self.config.n_objects, 3), dtype=np.float32)
        cls_ids = np.zeros((self.config.n_objects, 1), dtype=np.float32)
        kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3), dtype=np.float32)
        ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3), dtype=np.float32)

        # single-class: foreground id = 1 in labels_pt
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.bs_utils.get_ctr(self.cls_type)[:, None]
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0 * ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1], dtype=np.float32)

            if self.config.n_keypoints == 8:
                kp_type = "farthest"
            else:
                kp_type = f"farthest{self.config.n_keypoints}"
            kps = self.bs_utils.get_kps(self.cls_type, kp_type=kp_type)
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0 * kp))
            target_offset = np.array(target).transpose(1, 0, 2)
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    # ------------------------- main fetch -------------------------
    def get_item(self, item_name: str):
        subset, scene_id, im_id = self._parse_item_name(item_name)
        scene_dir = os.path.join(self.bop_root, subset, scene_id)

        scene_cam, scene_gt = self._load_scene_ann(scene_dir)
        im_int = int(im_id)

        cam_info = scene_cam[str(im_int)]
        gt_list = scene_gt[str(im_int)]

        inst, inst_idx = self._pick_instance(gt_list)
        if inst is None:
            return None

        # K
        K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)

        # depth_scale: depth_m = depth_raw * depth_scale
        depth_scale = float(cam_info.get("depth_scale", 0.001))

        # RT (m2c)
        R = np.array(inst["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        t_mm = np.array(inst["cam_t_m2c"], dtype=np.float32)  # mm
        t_m = t_mm / 1000.0
        RT = np.concatenate([R, t_m[:, None]], axis=1).astype(np.float32)

        # read rgb/depth/mask
        rgb = self._read_rgb(scene_dir, im_int)
        dpt_raw = self._read_depth(scene_dir, im_int)
        labels = self._read_mask(scene_dir, im_int, inst_idx)
        rgb_labels = labels.copy()

        # depth to meters, and to mm for normal estimation
        dpt_m = dpt_raw.astype(np.float32) * depth_scale
        dpt_mm = (dpt_m * 1000.0).astype(np.uint16)

        # normals
        nrm_map = depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, True)

        # point cloud (meters)
        dpt_xyz = self.dpt_2_pcld(dpt_m, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        # sample points: follow original (depth-valid points)
        msk_dp = dpt_raw > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < self.config.n_min_points:
            return None

        choose_2 = np.arange(len(choose))
        if len(choose_2) < self.config.n_min_points:
            return None

        if len(choose_2) > self.config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[: self.config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(
                choose_2,
                (0, self.config.n_sample_points - len(choose_2)),
                "wrap",
            )
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]  # {0,1}

        choose = np.array([choose], dtype=np.int32)
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)  # [9, npts]

        # GT target
        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, RT
        )

        # ---------- [CHANGED] Build sr2dptxyz pyramid + RandLA indices (copy original) ----------
        h, w = rgb_labels.shape

        # xyz pyramid: 1,2,4,8
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # [3,h,w]
        for i in range(3):
            scale = 2 ** (i + 1)
            nh, nw = h // scale, w // scale
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])

        sr2dptxyz = {
            2 ** ii: item.reshape(3, -1).transpose(1, 0)  # [N,3]
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}

        cld_l = cld  # [npts,3]
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(cld_l[None, ...], cld_l[None, ...], 16).astype(np.int32).squeeze(0)
            sub_n = cld_l.shape[0] // pcld_sub_s_r[i]
            sub_pts = cld_l[:sub_n, :]
            pool_i = nei_idx[:sub_n, :]
            up_i = DP.knn_search(sub_pts[None, ...], cld_l[None, ...], 1).astype(np.int32).squeeze(0)

            inputs[f"cld_xyz{i}"] = cld_l.astype(np.float32).copy()
            inputs[f"cld_nei_idx{i}"] = nei_idx.astype(np.int32).copy()
            inputs[f"cld_sub_idx{i}"] = pool_i.astype(np.int32).copy()
            inputs[f"cld_interp_idx{i}"] = up_i.astype(np.int32).copy()

            # rgb->point neighbor on downsampled scale
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs[f"r2p_ds_nei_idx{i}"] = nei_r2p.copy()

            # point->rgb neighbor
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs[f"p2r_ds_nei_idx{i}"] = nei_p2r.copy()

            cld_l = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            src_cld = inputs[f"cld_xyz{n_ds_layers - i - 1}"]
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                src_cld[None, ...],
                16,
            ).astype(np.int32).squeeze(0)
            inputs[f"r2p_up_nei_idx{i}"] = r2p_nei.copy()

            p2r_nei = DP.knn_search(
                src_cld[None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                1,
            ).astype(np.int32).squeeze(0)
            inputs[f"p2r_up_nei_idx{i}"] = p2r_nei.copy()

        # cam_scale: keep original convention
        cam_scale = 1000.0

        rgb_chw = np.transpose(rgb, (2, 0, 1))

        item_dict = dict(
            rgb=rgb_chw.astype(np.uint8),
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),
            choose=choose.astype(np.int32),
            labels=labels_pt.astype(np.int32),
            rgb_labels=rgb_labels.astype(np.int32),
            dpt_map_m=dpt_m.astype(np.float32),
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
        )
        item_dict.update(inputs)

        # if you want debug extras compatible with original DEBUG mode:
        if self.DEBUG:
            item_dict.update(
                dict(
                    cam_scale=np.array([cam_scale], dtype=np.float32),
                    K=K.astype(np.float32),
                    normal_map=nrm_map[:, :, :3].astype(np.float32),
                )
            )

        return item_dict

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        item_name = self.all_lst[idx]
        data = self.get_item(item_name)
        while data is None:
            idx = np.random.randint(0, len(self.all_lst))
            item_name = self.all_lst[idx]
            data = self.get_item(item_name)
        return data
