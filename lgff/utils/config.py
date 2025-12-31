# lgff/utils/config.py
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

    # 1) Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # 2) Python literal (int/float/list/dict/tuple/None)
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass

    # 3) Comma-separated list without brackets: "1,2,3"
    if "," in v:
        try:
            return [ast.literal_eval(i.strip()) for i in v.split(",")]
        except (ValueError, SyntaxError):
            pass

    # 4) Fallback string
    return v


@dataclass
class LGFFConfig:
    """Central Configuration for LGFF."""

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

    camera_intrinsic: List[List[float]] = field(
        default_factory=lambda: [
            [1066.778, 0.0, 312.9869],
            [0.0, 1067.487, 241.3109],
            [0.0, 0.0, 1.0],
        ]
    )

    # ---- Dataset optional outputs (important for seg) ----
    # SingleObjectDataset uses these to decide whether to return 2D masks.
    return_mask: bool = False         # returns sample["mask"] : [1,H,W] float in {0,1}
    return_valid_mask: bool = False   # returns sample["mask_valid"] : [1,H,W] float in {0,1}

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

    # ----------------- Model / Variant Switch -----------------
    model_variant: str = "sc"     # sc / sc_seg
    loss_variant: str = "base"    # base / seg

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

    # ----------------- SEG Head (optional) -----------------
    use_seg_head: bool = False
    lambda_seg: float = 0.0

    # where seg head taps features from: rgb / fused / geo (project-specific)
    seg_head_in: str = "rgb"
    # keep both for compatibility with different model versions
    seg_head_channels: int = 64
    seg_head_dim: int = 64

    seg_output_stride: int = 4  # meaningful if you design stride-aware head

    # seg loss type: bce / bce_dice / focal (project-specific)
    seg_loss_type: str = "bce"
    seg_pos_weight: Optional[float] = None
    seg_ignore_invalid: bool = True
    seg_detach_trunk: bool = False

    # norm choice for seg head
    seg_use_gn: bool = True

    # point-level threshold after sampling pred_mask_logits -> per-point valid mask
    seg_point_thresh: float = 0.5

    # supervision mode: "point" or "mask"
    # - "point": build per-point GT valid mask, then supervise sampled logits
    # - "mask" : supervise dense 2D logits with dense 2D GT mask
    seg_supervision: str = "point"

    # source of supervision:
    # point: labels / pcld_valid_mask / mask / mask_valid / mask_visib / mask_full
    # mask : mask / mask_valid / mask_visib / mask_full
    #
    # NOTE:
    # - Your current SingleObjectDataset provides: labels, pcld_valid_mask, mask, mask_valid (if enabled)
    # - It does NOT provide "mask_visib"/"mask_full" unless you extend the dataset.
    seg_supervision_source: str = "labels"

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

    # (optional) valid-mask fusion controls (if you implemented them)
    pose_fusion_use_valid_mask: bool = False
    pose_fusion_valid_mask_source: str = "labels"   # labels / seg
    pose_fusion_conf_floor: float = 1.0e-4

    # ----------------- Eval -----------------
    cmd_threshold_m: float = 0.02
    obj_diameter_m: Optional[float] = None

    # Absolute/relative ADD thresholds for evaluator (optional but recommended)
    eval_abs_add_thresholds: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.015, 0.02, 0.03]
    )
    eval_rel_add_thresholds: List[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10]
    )

    # PnP / ICP flags for evaluator (optional)
    eval_use_pnp: bool = False
    icp_enable: bool = False

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: Unknown config key: {key} (ignored)")

    def validate(self) -> None:
        # basic checks
        if self.resize_h <= 0 or self.resize_w <= 0:
            raise ValueError("resize_h/resize_w must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.camera_intrinsic) != 3 or any(len(row) != 3 for row in self.camera_intrinsic):
            raise ValueError("camera_intrinsic must be a 3x3 matrix")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be > 0")

        # lambdas non-negative
        for name in [
            "lambda_dense", "lambda_t", "lambda_rot",
            "lambda_conf", "lambda_add_cad", "lambda_kp_of",
            "lambda_t_bias_z", "lambda_seg",
        ]:
            val = float(getattr(self, name, 0.0))
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        # curriculum bounds
        if not (0.0 <= self.curriculum_warmup_frac <= 1.0):
            raise ValueError(f"curriculum_warmup_frac must be in [0,1], got {self.curriculum_warmup_frac}")
        if not (0.0 <= self.curriculum_final_factor_t <= 1.0):
            raise ValueError(f"curriculum_final_factor_t must be in [0,1], got {self.curriculum_final_factor_t}")
        if not (0.0 <= self.curriculum_final_factor_rot <= 1.0):
            raise ValueError(f"curriculum_final_factor_rot must be in [0,1], got {self.curriculum_final_factor_rot}")

        # topk validity
        for kname in ["pose_fusion_topk", "train_topk", "eval_topk", "viz_topk"]:
            kval = getattr(self, kname, None)
            if kval is None:
                continue
            if not isinstance(kval, int) or kval <= 0:
                raise ValueError(f"{kname} must be a positive int or None, got {kval}")

        # ----------------- SEG validation (dataset-key aligned) -----------------
        if bool(getattr(self, "use_seg_head", False)) or str(getattr(self, "model_variant", "sc")).endswith("seg"):
            sup = str(getattr(self, "seg_supervision", "point")).lower().strip()
            src = str(getattr(self, "seg_supervision_source", "labels")).lower().strip()

            allowed_sup = {"point", "mask"}
            if sup not in allowed_sup:
                raise ValueError(f"seg_supervision must be one of {allowed_sup}, got {sup}")

            # IMPORTANT:
            # - "mask" / "mask_valid" are the keys returned by your SingleObjectDataset (when enabled).
            # - "mask_visib" / "mask_full" are only valid if you extend dataset to return them explicitly.
            allowed_src_mask = {"mask", "mask_valid", "mask_visib", "mask_full"}
            allowed_src_point = {"labels", "pcld_valid_mask", "mask", "mask_valid", "mask_visib", "mask_full"}

            if sup == "mask":
                if src not in allowed_src_mask:
                    raise ValueError(
                        f"seg_supervision_source must be one of {allowed_src_mask} when seg_supervision='mask', got {src}"
                    )
            else:
                if src not in allowed_src_point:
                    raise ValueError(
                        f"seg_supervision_source must be one of {allowed_src_point} when seg_supervision='point', got {src}"
                    )

            # helpful warnings (do not hard fail)
            if float(getattr(self, "lambda_seg", 0.0)) <= 0:
                print("[Config] Warning: seg head is enabled but lambda_seg<=0; seg head will not be trained.")

            # if user wants mask supervision but dataset likely won't return it unless return_mask/return_valid_mask enabled
            if sup == "mask" and src in {"mask", "mask_valid"}:
                if not bool(getattr(self, "return_mask", False)) and not bool(getattr(self, "return_valid_mask", False)):
                    print(
                        "[Config] Warning: seg_supervision='mask' expects dataset to return 2D mask, "
                        "but return_mask/return_valid_mask are both False. "
                        "Set return_mask: true (and/or return_valid_mask: true) in your YAML."
                    )

        # pose_fusion_valid_mask_source sanity
        vsrc = str(getattr(self, "pose_fusion_valid_mask_source", "labels")).lower().strip()
        if vsrc not in {"labels", "seg"}:
            raise ValueError("pose_fusion_valid_mask_source must be one of {'labels','seg'}")

        # seg_point_thresh sanity
        thr = float(getattr(self, "seg_point_thresh", 0.5))
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"seg_point_thresh must be in [0,1], got {thr}")

    def resolve_paths(self) -> None:
        self.dataset_root = os.path.expanduser(self.dataset_root)
        self.log_dir = os.path.expanduser(self.log_dir)
        if self.work_dir is not None:
            self.work_dir = os.path.expanduser(self.work_dir)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "LGFFConfig":
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
    def from_yaml(cls, path: str) -> "LGFFConfig":
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


def _collect_overrides(cli_cfg: LGFFConfig) -> Dict[str, Any]:
    base = LGFFConfig()
    overrides: Dict[str, Any] = {}
    for k, v in asdict(cli_cfg).items():
        if getattr(base, k) != v:
            overrides[k] = v
    return overrides


def merge_cfg_from_checkpoint(cli_cfg: LGFFConfig, ckpt_cfg: Optional[Dict[str, Any]]) -> LGFFConfig:
    if ckpt_cfg is None:
        return cli_cfg

    cfg_dict = dict(ckpt_cfg)
    merged = LGFFConfig()
    merged.update(cfg_dict)

    overrides = _collect_overrides(cli_cfg)
    merged.update(overrides)
    merged.validate()
    merged.resolve_paths()
    return merged


def load_config() -> LGFFConfig:
    parser = argparse.ArgumentParser(description="LGFF Config Loader", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint-cfg",
        type=str,
        default=None,
        help="Optionally pull config from a checkpoint (state['config'])",
    )
    parser.add_argument(
        "--prefer-checkpoint",
        action="store_true",
        help="If both YAML and checkpoint are given, let checkpoint config take precedence",
    )
    parser.add_argument("--opt", type=str, nargs="*", default=[], help="Override config")
    args, _ = parser.parse_known_args()

    cfg = LGFFConfig()

    yaml_data: Dict[str, Any] = {}
    ckpt_data: Optional[Dict[str, Any]] = None

    if args.config:
        print(f"[Config] Loading from {args.config}")
        yaml_data = _read_yaml(args.config)
        cfg.update(yaml_data)
        if "log_dir" not in yaml_data or not yaml_data.get("log_dir"):
            exp_name = os.path.splitext(os.path.basename(args.config))[0]
            cfg.log_dir = os.path.join("output", exp_name)

    if args.checkpoint_cfg:
        print(f"[Config] Loading config from checkpoint {args.checkpoint_cfg}")
        checkpoint = torch.load(args.checkpoint_cfg, map_location="cpu")
        ckpt_data = checkpoint.get("config") or {}
        if not ckpt_data:
            print("[Config] Warning: checkpoint has no stored config; ignoring")
        else:
            if args.prefer_checkpoint:
                cfg = LGFFConfig()
                cfg.update(ckpt_data)
            else:
                cfg.update(ckpt_data)

    for opt in args.opt:
        if "=" not in opt:
            continue
        key, value_str = opt.split("=", 1)
        value = _parse_value(value_str)
        cfg.update({key: value})
        print(f"[Config] Override: {key} = {value}")

    if cfg.work_dir is None:
        cfg.work_dir = cfg.log_dir

    cfg.validate()
    cfg.resolve_paths()

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_path = os.path.join(cfg.work_dir, "config_used.yaml")
    cfg.save(save_path)
    print(f"[Config] Final config saved to {save_path}")

    return cfg


__all__ = ["LGFFConfig", "load_config", "merge_cfg_from_checkpoint"]
