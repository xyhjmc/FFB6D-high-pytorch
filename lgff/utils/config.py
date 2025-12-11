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

    # 1. Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # 2. Try parsing as Python literal (covers int, float, list, dict, tuple)
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass

    # 3. Handle comma-separated lists without brackets (e.g., "1,2,3")
    if "," in v:
        try:
            return [ast.literal_eval(i.strip()) for i in v.split(",")]
        except (ValueError, SyntaxError):
            pass

    # 4. Fallback to string
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

    # ----------------- Training Hyper-params -----------------
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    log_interval: int = 10

    # 梯度裁剪阈值
    max_grad_norm: float = 2.0

    # 调度器配置
    scheduler: str = "plateau"
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_step_size: int = 20
    lr_min: float = 1e-6

    # 随机性 / 复现控制
    seed: int = 42
    deterministic: bool = False

    # ----------------- Model / Loss Hyper-params -----------------

    # [新增] 骨干网络名称，用于切换 ResNet / MobileNet
    backbone_name: str = "mobilenet_v3_large"

    backbone_arch: str = "small"  # 仅 MobileNet 使用
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

    w_rate: float = 0.015
    sym_class_ids: List[int] = field(default_factory=list)

    # 姿态融合控制：train/val/viz 阶段可独立切换 best-point vs Top-K 加权
    train_use_best_point: Optional[bool] = None
    eval_use_best_point: bool = True
    viz_use_best_point: Optional[bool] = None
    pose_fusion_topk: int = 64
    loss_use_fused_pose: bool = True

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: Unknown config key: {key} (ignored)")

    # 增强：基础合法性校验与路径展开，避免静默错误
    def validate(self) -> None:
        if self.resize_h <= 0 or self.resize_w <= 0:
            raise ValueError("resize_h/resize_w must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.camera_intrinsic) != 3 or any(len(row) != 3 for row in self.camera_intrinsic):
            raise ValueError("camera_intrinsic must be a 3x3 matrix")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be > 0")

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
        """Load config dict directly from a TrainerSC checkpoint if available."""
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
    """Collect fields overridden via CLI (difference vs. default LGFFConfig)."""
    base = LGFFConfig()
    overrides: Dict[str, Any] = {}
    for k, v in asdict(cli_cfg).items():
        if getattr(base, k) != v:
            overrides[k] = v
    return overrides


def merge_cfg_from_checkpoint(cli_cfg: LGFFConfig, ckpt_cfg: Optional[Dict[str, Any]]) -> LGFFConfig:
    """
    Prefer checkpoint config (model/backbone dimensions) while keeping CLI overrides.

    Args:
        cli_cfg: config after CLI parsing (may include --opt overrides).
        ckpt_cfg: ``state["config"]`` dict loaded from checkpoint.
    """
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
