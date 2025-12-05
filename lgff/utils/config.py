"""
LGFF Configuration System.
Supports YAML loading, CLI overrides, type conversion, and config persistence.
"""
from __future__ import annotations

import argparse
import os
import yaml
import ast
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _parse_value(value: str) -> Any:
    """
    Enhanced value parser:
    - "true"/"false" -> bool
    - "1", "1.5" -> int, float
    - "[1, 2]" or "1,2" -> list
    - "{'a': 1}" -> dict
    """
    v = value.strip()

    # 1. Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # 2. Try parsing as Python literal (covers int, float, list, dict, tuple, etc.)
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
    """
    Central Configuration for LGFF.
    """

    # ----------------- Dataset -----------------
    dataset_name: str = "bop-single"
    dataset_root: str = "datasets/bop"
    annotation_file: Optional[str] = None

    # BOP 默认内参 (可在 YAML 中覆盖为实际数据集内参)
    camera_intrinsic: List[List[float]] = field(
        default_factory=lambda: [
            [1066.778, 0.0, 312.9869],
            [0.0, 1067.487, 241.3109],
            [0.0, 0.0, 1.0],
        ]
    )

    depth_scale: float = 1000.0    # 深度图单位: depth / depth_scale = meters
    num_workers: int = 4
    batch_size: int = 8            # 可在 YAML 调整
    num_classes: int = 1           # 单类模式

    num_points: int = 1024         # 采样点数
    num_keypoints: int = 8
    val_split: str = "test"        # BOP 通常是 train/test

    # BOP 里目标物体的 obj_id
    obj_id: int = 1

    # 输入分辨率 (SingleObjectDataset 使用)
    resize_h: int = 480
    resize_w: int = 640

    # ----------------- Training Hyper-params -----------------
    epochs: int = 50
    lr: float = 1e-3               # MobileNet 相对 ResNet 可以略大些
    weight_decay: float = 1e-4
    use_amp: bool = True
    log_interval: int = 10

    # ----------------- Backbone (RGB) Hyper-params -----------------
    backbone_arch: str = "small"           # 'small' or 'large'
    backbone_output_stride: int = 8        # 8 / 16 / 32
    backbone_pretrained: bool = True
    backbone_freeze_bn: bool = True
    backbone_return_intermediate: bool = False
    backbone_low_level_index: int = 2

    # ----------------- Model / Feature Dimensions -----------------
    rgb_feat_dim: int = 128
    geo_feat_dim: int = 128

    # ----------------- PointNet (Geometry) Hyper-params -----------------
    point_input_dim: int = 3
    point_hidden_dims: List[int] = field(default_factory=lambda: [64, 128])
    point_norm: str = "bn"
    point_use_se: bool = True
    point_dropout: float = 0.0

    # ----------------- Head Hyper-params -----------------
    head_hidden_dim: int = 128
    head_feat_dim: int = 64
    head_dropout: float = 0.0

    # ----------------- Loss Hyper-params -----------------
    w_rate: float = 0.015          # Confidence regularization weight
    sym_class_ids: List[int] = field(default_factory=list)

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def update(self, data: Dict[str, Any]) -> None:
        """Update attributes with dictionary data."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 可选: 打印未知字段，方便 debug
                print(f"[Warning] Unknown config key: {key}")

    def save(self, path: str) -> None:
        """Dump current config to a YAML file for reproducibility."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def load_config() -> LGFFConfig:
    """
    Load configuration from CLI args and YAML files.
    Priority: CLI --opt > CLI --config > Defaults
    """
    parser = argparse.ArgumentParser(description="LGFF Config Loader", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--opt", type=str, nargs="*", default=[], help="Key=Value overrides")

    # 我们用 parse_known_args 以便主程序还能继续加自己的参数
    args, _ = parser.parse_known_args()

    cfg = LGFFConfig()

    # 1. Load from YAML
    if args.config:
        print(f"[Config] Loading from {args.config}")
        yaml_data = _read_yaml(args.config)
        cfg.update(yaml_data)

        # Auto-set log_dir based on config filename if not specified
        if "log_dir" not in yaml_data:
            exp_name = os.path.splitext(os.path.basename(args.config))[0]
            cfg.log_dir = os.path.join("output", exp_name)

    # 2. Override from CLI --opt
    for opt in args.opt:
        if "=" not in opt:
            continue
        key, value_str = opt.split("=", 1)
        value = _parse_value(value_str)
        cfg.update({key: value})
        print(f"[Config] Override: {key} = {value}")

    # 3. work_dir 默认跟 log_dir 一致
    if cfg.work_dir is None:
        cfg.work_dir = cfg.log_dir

    os.makedirs(cfg.work_dir, exist_ok=True)

    # 4. Save the final config for records
    save_path = os.path.join(cfg.work_dir, "config_used.yaml")
    cfg.save(save_path)

    return cfg


__all__ = ["LGFFConfig", "load_config"]