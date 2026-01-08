#!/usr/bin/env python3
import os
import yaml
import numpy as np


def ensure_fd(fd: str) -> None:
    """Create directory ``fd`` if it does not exist."""
    os.makedirs(fd, exist_ok=True)


def _read_yaml_if_exists(p: str) -> dict:
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class ConfigRandLA:
    # NOTE: 保持原默认，尽量不动。你若改分辨率/采点数，可在 Config 里同步设置 n_sample_points。
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 480 * 640 // 24  # Number of input points
    num_classes = 22  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter
    use_checkpoint = True

    batch_size = 3  # batch_size during training
    val_batch_size = 3  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 9

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]


class Config:
    """
    Compatible Config:
    - ds_name: 'ycb' / 'linemod' / 'bop'
    - cls_type: 类别名称（单类训练/评估场景）
    - cfg_path: 可选，外部 YAML 覆盖（推荐用于自建 BOP）
    """

    def __init__(self, ds_name: str = "ycb", cls_type: str = "", cfg_path: str | None = None):
        self.dataset_name = ds_name
        self.cls_type = cls_type

        # project root
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)

        # Optional user cfg (for custom BOP etc.)
        # Default path: datasets/bop/dataset_config/custom_bop.yaml
        if cfg_path is None and self.dataset_name == "bop":
            cfg_path = os.path.join(self.exp_dir, "datasets", "bop", "dataset_config", "custom_bop.yaml")
        self.user_cfg_path = cfg_path
        self.user_cfg = _read_yaml_if_exists(cfg_path) if cfg_path else {}

        # pretrained
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(self.exp_dir, "models/cnn/ResNet_pretrained_mdl")
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folders
        self.log_dir = os.path.abspath(os.path.join(self.exp_dir, "train_log", self.dataset_name))
        ensure_fd(self.log_dir)
        self.log_model_dir = os.path.join(self.log_dir, "checkpoints", self.cls_type)
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, "eval_results", self.cls_type)
        ensure_fd(self.log_eval_dir)
        self.log_traininfo_dir = os.path.join(self.log_dir, "train_info", self.cls_type)
        ensure_fd(self.log_traininfo_dir)

        # -------- training defaults (can be overridden by YAML) --------
        self.n_total_epoch = int(self.user_cfg.get("n_total_epoch", 25))
        self.mini_batch_size = int(self.user_cfg.get("mini_batch_size", 2))
        self.val_mini_batch_size = int(self.user_cfg.get("val_mini_batch_size", 2))
        self.test_mini_batch_size = int(self.user_cfg.get("test_mini_batch_size", 1))

        self.n_sample_points = int(self.user_cfg.get("n_sample_points", 480 * 640 // 24))
        self.n_keypoints = int(self.user_cfg.get("n_keypoints", 8))
        self.n_min_points = int(self.user_cfg.get("n_min_points", 400))
        self.noise_trans = float(self.user_cfg.get("noise_trans", 0.05))

        self.preprocessed_testset_pth = str(self.user_cfg.get("preprocessed_testset_pth", ""))

        # -------- camera intrinsic presets (legacy; BOP usually uses per-image K) --------
        self.intrinsic_matrix = {
            "linemod": np.array(
                [[572.4114, 0.0, 325.2611],
                 [0.0, 573.57043, 242.04899],
                 [0.0, 0.0, 1.0]]
            ),
            "blender": np.array(
                [[700.0, 0.0, 320.0],
                 [0.0, 700.0, 240.0],
                 [0.0, 0.0, 1.0]]
            ),
            "ycb_K1": np.array(
                [[1066.778, 0.0, 312.9869],
                 [0.0, 1067.487, 241.3109],
                 [0.0, 0.0, 1.0]], np.float32
            ),
            "ycb_K2": np.array(
                [[1077.836, 0.0, 323.7872],
                 [0.0, 1078.189, 279.6921],
                 [0.0, 0.0, 1.0]], np.float32
            ),
        }

        # -------- dataset-specific --------
        if self.dataset_name == "ycb":
            self._init_ycb()
        elif self.dataset_name == "linemod":
            self._init_linemod()
        elif self.dataset_name == "bop":
            self._init_bop()
        else:
            raise ValueError(f"Unknown ds_name: {self.dataset_name}")

    # -------------------------- YCB --------------------------
    def _init_ycb(self):
        self.n_objects = 21 + 1
        self.n_classes = self.n_objects

        self.use_orbfps = True
        self.kp_orbfps_dir = "datasets/ycb/ycb_kps/"
        self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, "%s_%d_kps.txt")

        self.ycb_cls_lst_p = os.path.abspath(os.path.join(self.exp_dir, "datasets/ycb/dataset_config/classes.txt"))
        self.ycb_root = os.path.abspath(os.path.join(self.exp_dir, "datasets/ycb/YCB_Video_Dataset"))
        self.ycb_kps_dir = os.path.abspath(os.path.join(self.exp_dir, "datasets/ycb/ycb_kps/"))

        ycb_r_lst_p = os.path.abspath(os.path.join(self.exp_dir, "datasets/ycb/dataset_config/radius.txt"))
        self.ycb_r_lst = list(np.loadtxt(ycb_r_lst_p))
        self.ycb_cls_lst = self.read_lines(self.ycb_cls_lst_p)
        self.ycb_sym_cls_ids = [13, 16, 19, 20, 21]

    # -------------------------- LINEMOD --------------------------
    def _init_linemod(self):
        self.n_objects = 1 + 1
        self.n_classes = self.n_objects

        self.lm_cls_lst = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.lm_sym_cls_ids = [10, 11]
        self.lm_obj_dict = {
            "ape": 1, "benchvise": 2, "cam": 4, "can": 5, "cat": 6,
            "driller": 8, "duck": 9, "eggbox": 10, "glue": 11,
            "holepuncher": 12, "iron": 13, "lamp": 14, "phone": 15,
        }
        try:
            self.cls_id = self.lm_obj_dict[self.cls_type]
        except Exception:
            self.cls_id = None

        self.lm_id2obj_dict = dict(zip(self.lm_obj_dict.values(), self.lm_obj_dict.keys()))
        self.lm_root = os.path.abspath(os.path.join(self.exp_dir, "datasets/linemod/"))

        self.use_orbfps = True
        self.kp_orbfps_dir = "datasets/linemod/kps_orb9_fps/"
        self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, "%s_%d_kps.txt")

        self.lm_fps_kps_dir = os.path.abspath(os.path.join(self.exp_dir, "datasets/linemod/lm_obj_kps/"))

        lm_r_pth = os.path.join(self.lm_root, "dataset_config/models_info.yml")
        with open(lm_r_pth, "r", encoding="utf-8") as f:
            self.lm_r_lst = yaml.safe_load(f)

        self.val_nid_ptn = "/data/6D_Pose_Data/datasets/LINEMOD/pose_nori_lists/{}_real_val.nori.list"

    # -------------------------- BOP (Custom) --------------------------
    def _init_bop(self):
        """
        New config fields for custom BOP dataset.

        Recommended YAML (datasets/bop/dataset_config/custom_bop.yaml):

        bop_root: /abs/path/to/your_bop
        bop_train_list: /abs/path/or/relpath/train.txt
        bop_test_list: /abs/path/or/relpath/test.txt
        bop_obj_id: 1
        bop_use_mask_visib: true
        bop_kps_npy: /abs/path/or/relpath/kps.npy
        bop_ctr_npy: /abs/path/or/relpath/ctr.npy
        """

        # FFB6D 单类场景：仍然保持 n_objects=2（foreground + background）
        self.n_objects = int(self.user_cfg.get("n_objects", 2))
        self.n_classes = self.n_objects

        # BOP dataset root
        bop_root = self.user_cfg.get("bop_root", None)
        if bop_root is None:
            # fallback: datasets/bop/your_dataset
            bop_root = os.path.join(self.exp_dir, "datasets", "bop")
        self.bop_root = os.path.abspath(os.path.join(self.exp_dir, bop_root)) if not os.path.isabs(bop_root) else bop_root

        # object id & class id mapping (single-class)
        self.bop_obj_id = int(self.user_cfg.get("bop_obj_id", 1))
        self.cls_id = 1  # For labels_pt (0/1) training convention in the provided dataset code.

        # split lists (allow relative path)
        def _abs_or_join(p: str | None) -> str:
            if not p:
                return ""
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(self.exp_dir, p))

        self.bop_train_list = _abs_or_join(self.user_cfg.get("bop_train_list", ""))
        self.bop_test_list = _abs_or_join(self.user_cfg.get("bop_test_list", ""))

        # masks
        self.bop_use_mask_visib = bool(self.user_cfg.get("bop_use_mask_visib", True))

        # keypoints (npy) - provide either absolute or relative
        self.bop_kps_npy = _abs_or_join(self.user_cfg.get("bop_kps_npy", ""))
        self.bop_ctr_npy = _abs_or_join(self.user_cfg.get("bop_ctr_npy", ""))

        # Optional: symmetry ids, if you need ADD-S etc.
        self.bop_sym_obj_ids = list(self.user_cfg.get("bop_sym_obj_ids", []))

        # Optional: a place to store cached preprocess if you do later
        self.bop_cache_dir = _abs_or_join(self.user_cfg.get("bop_cache_dir", ""))

        # Keep compatibility switches
        self.use_orbfps = False  # 默认 BOP 走你自己的 npy keypoints
        self.kp_orbfps_dir = ""
        self.kp_orbfps_ptn = ""

    def read_lines(self, p: str):
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]


# default instance (legacy)
config = Config()
# vim: ts=4 sw=4 sts=4 expandtab
