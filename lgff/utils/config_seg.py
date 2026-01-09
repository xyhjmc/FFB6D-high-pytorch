# lgff/utils/config_seg.py
import argparse
import os
import yaml
import ast
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

import torch


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _parse_value(value: str) -> Any:
    """Enhanced value parser for --opt key=value."""
    v = value.strip()
    if v.lower() == "true": return True
    if v.lower() == "false": return False
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    if "," in v:
        try:
            return [ast.literal_eval(i.strip()) for i in v.split(",")]
        except (ValueError, SyntaxError):
            pass
    return v


@dataclass
class LGFFConfigSeg:
    """
    Central Configuration for LGFF (Segmentation Enhanced Version).
    Final Fix: Added seg_dice_weight to silence warnings.
    """

    # ----------------- Dataset / BOP -----------------
    dataset_name: str = "bop-single"
    dataset_root: str = "datasets/bop"
    annotation_file: Optional[str] = None

    obj_id: int = 1
    resize_h: int = 480
    resize_w: int = 640
    depth_scale: float = 1000.0

    num_workers: int = 4
    batch_size: int = 8
    num_classes: int = 1
    num_points: int = 1024
    num_keypoints: int = 8
    val_split: str = "test"

    # [SEG-NEW] Dataset Flags
    return_mask: bool = False
    return_valid_mask: bool = False

    camera_intrinsic: List[List[float]] = field(
        default_factory=lambda: [
            [1066.778, 0.0, 312.9869],
            [0.0, 1067.487, 241.3109],
            [0.0, 0.0, 1.0],
        ]
    )

    # ----------------- Training Hyper-params -----------------
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    log_interval: int = 10
    max_grad_norm: float = 2.0

    scheduler: str = "plateau"
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_step_size: int = 20
    lr_min: float = 1e-6

    seed: int = 42
    deterministic: bool = False

    # ----------------- Model / Feature Hyper-params -----------------
    backbone_name: str = "mobilenet_v3_large"
    backbone_arch: str = "small"
    backbone_output_stride: int = 8
    backbone_pretrained: bool = True
    backbone_freeze_bn: bool = True
    backbone_return_intermediate: bool = False
    backbone_low_level_index: int = 2

    rgb_feat_dim: int = 128
    geo_feat_dim: int = 128

    point_input_dim: int = 3
    point_hidden_dims: tuple = (64, 128)
    point_norm: str = "bn"
    point_use_se: bool = True
    point_dropout: float = 0.0

    head_hidden_dim: int = 128
    head_feat_dim: int = 64
    head_dropout: float = 0.0

    # [SEG-NEW] Model Variants & Seg Head
    model_variant: str = "sc"
    loss_variant: str = "base"

    use_seg_head: bool = False
    lambda_seg: float = 0.0
    seg_head_in: str = "rgb"
    seg_head_channels: int = 64
    seg_output_stride: int = 4
    seg_detach_trunk: bool = False

    seg_loss_type: str = "bce"
    seg_pos_weight: Optional[float] = None
    seg_ignore_invalid: bool = False
    seg_supervision: str = "mask"
    seg_supervision_source: str = "mask_visib"

    # [FIX] Added missing Seg params (Dice Weight, Head Dim, etc.)
    seg_dice_weight: float = 0.5  # [NEW] Weight for Dice Loss in BCE+Dice
    seg_min_foreground: float = 1.0  # [NEW] Min pixels to compute Seg Loss (avoid NaN)
    seg_head_dim: Optional[int] = None  # Alias
    seg_use_gn: bool = True
    seg_point_thresh: float = 0.5

    # ----------------- Loss / Geometry Hyper-params -----------------
    w_rate: float = 0.015
    sym_class_ids: List[int] = field(default_factory=list)

    lambda_dense: float = 1.0
    lambda_t: float = 0.5
    lambda_rot: float = 0.5
    lambda_conf: float = 0.1
    lambda_add_cad: float = 0.1
    lambda_kp_of: float = 0.3

    lambda_t_bias_z: float = 0.0
    t_bias_z_use_abs: bool = True

    t_z_weight: float = 2.0
    conf_alpha: float = 10.0
    conf_dist_max: float = 0.05

    use_geodesic_rot: bool = True

    use_curriculum_loss: bool = False
    curriculum_warmup_frac: float = 0.4
    curriculum_final_factor_t: float = 0.3
    curriculum_final_factor_rot: float = 0.3

    use_uncertainty_weighting: bool = False

    # ----------------- Pose Fusion Control -----------------
    train_use_best_point: Optional[bool] = None
    eval_use_best_point: bool = True
    viz_use_best_point: Optional[bool] = None

    pose_fusion_topk: int = 64
    eval_topk: Optional[int] = None
    train_topk: Optional[int] = None
    viz_topk: Optional[int] = None

    loss_use_fused_pose: bool = True

    # [SEG-NEW] Fusion Masks
    pose_fusion_use_valid_mask: bool = False
    pose_fusion_valid_mask_source: str = "seg"
    pose_fusion_conf_floor: float = 1e-4
    pose_fusion_mask_conf_in_model: bool = False

    # ----------------- 评估 / 阈值相关 -----------------
    cmd_threshold_m: float = 0.02
    obj_diameter_m: Optional[float] = None

    eval_use_pnp: bool = True
    eval_abs_add_thresholds: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.015, 0.02, 0.03])
    eval_rel_add_thresholds: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    best_metric_use_eval: bool = False
    best_metric_eval_save_csv: bool = False

    # ICP
    icp_enable: bool = False
    icp_iters: int = 10
    icp_max_corr_dist: float = 0.02
    icp_trim_ratio: float = 0.7
    icp_sample_model: int = 512
    icp_sample_obs: int = 2048
    icp_min_corr: int = 50
    icp_z_min: Optional[float] = None
    icp_z_max: Optional[float] = None
    icp_obs_mad_k: float = 0.0
    icp_corr_schedule_m: Optional[List[float]] = None
    icp_iters_schedule: Optional[List[int]] = None

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[ConfigSeg] Warning: Unknown config key: {key} (ignored)")

    def validate(self) -> None:
        if self.resize_h <= 0 or self.resize_w <= 0:
            raise ValueError("resize_h/resize_w must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.camera_intrinsic) != 3 or any(len(row) != 3 for row in self.camera_intrinsic):
            raise ValueError("camera_intrinsic must be a 3x3 matrix")

        if self.seg_head_dim is not None and self.seg_head_channels == 64:
            self.seg_head_channels = self.seg_head_dim

        for name in ["lambda_dense", "lambda_t", "lambda_rot", "lambda_conf"]:
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        if self.use_seg_head and self.lambda_seg < 0:
            raise ValueError("lambda_seg cannot be negative")

    def resolve_paths(self) -> None:
        self.dataset_root = os.path.expanduser(self.dataset_root)
        self.log_dir = os.path.expanduser(self.log_dir)
        if self.work_dir is not None:
            self.work_dir = os.path.expanduser(self.work_dir)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "LGFFConfigSeg":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = checkpoint.get("config")
        if cfg_dict is None:
            raise ValueError(f"No config stored in checkpoint: {checkpoint_path}")
        cfg = cls()
        cfg.update(cfg_dict)
        if cfg.work_dir is None and cfg.log_dir:
            cfg.work_dir = cfg.log_dir
        cfg.validate()
        cfg.resolve_paths()
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "LGFFConfigSeg":
        yaml_data = _read_yaml(path)
        cfg = cls()
        cfg.update(yaml_data)
        if not getattr(cfg, "log_dir", None):
            exp_name = os.path.splitext(os.path.basename(path))[0]
            cfg.log_dir = os.path.join("output", exp_name)
        if cfg.work_dir is None:
            cfg.work_dir = cfg.log_dir
        os.makedirs(cfg.work_dir, exist_ok=True)
        return cfg


def _collect_overrides(cli_cfg: LGFFConfigSeg) -> Dict[str, Any]:
    base = LGFFConfigSeg()
    overrides: Dict[str, Any] = {}
    for k, v in asdict(cli_cfg).items():
        if getattr(base, k) != v:
            overrides[k] = v
    return overrides


def merge_cfg_from_checkpoint(cli_cfg: LGFFConfigSeg, ckpt_cfg: Optional[Dict[str, Any]]) -> LGFFConfigSeg:
    if ckpt_cfg is None:
        return cli_cfg

    cfg_dict = dict(ckpt_cfg)
    merged = LGFFConfigSeg()
    merged.update(cfg_dict)

    overrides = _collect_overrides(cli_cfg)
    merged.update(overrides)
    merged.validate()
    merged.resolve_paths()
    return merged


def load_config() -> LGFFConfigSeg:
    parser = argparse.ArgumentParser(description="LGFF Seg Config Loader", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--checkpoint-cfg", type=str, default=None)
    parser.add_argument("--prefer-checkpoint", action="store_true")
    parser.add_argument("--opt", type=str, nargs="*", default=[], help="Override config")
    args, _ = parser.parse_known_args()

    cfg = LGFFConfigSeg()

    yaml_data: Dict[str, Any] = {}
    ckpt_data: Optional[Dict[str, Any]] = None

    if args.config:
        print(f"[ConfigSeg] Loading from {args.config}")
        yaml_data = _read_yaml(args.config)
        cfg.update(yaml_data)
        if "log_dir" not in yaml_data or not yaml_data.get("log_dir"):
            exp_name = os.path.splitext(os.path.basename(args.config))[0]
            cfg.log_dir = os.path.join("output", exp_name)

    if args.checkpoint_cfg:
        print(f"[ConfigSeg] Loading config from checkpoint {args.checkpoint_cfg}")
        checkpoint = torch.load(args.checkpoint_cfg, map_location="cpu")
        ckpt_data = checkpoint.get("config") or {}
        if not ckpt_data:
            print("[ConfigSeg] Warning: checkpoint has no stored config; ignoring")
        else:
            if args.prefer_checkpoint:
                cfg = LGFFConfigSeg()
                cfg.update(ckpt_data)
            else:
                cfg.update(ckpt_data)

    for opt in args.opt:
        if "=" not in opt:
            continue
        key, value_str = opt.split("=", 1)
        value = _parse_value(value_str)
        cfg.update({key: value})
        print(f"[ConfigSeg] Override: {key} = {value}")

    if cfg.work_dir is None:
        cfg.work_dir = cfg.log_dir

    cfg.validate()
    cfg.resolve_paths()

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_path = os.path.join(cfg.work_dir, "config_used.yaml")
    cfg.save(save_path)
    print(f"[ConfigSeg] Final config saved to {save_path}")

    return cfg


__all__ = ["LGFFConfigSeg", "load_config", "merge_cfg_from_checkpoint"]
