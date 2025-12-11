# lgff/utils/config.py
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
    Enhanced value parser for --opt key=value
    """
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
    """
    Central Configuration for LGFF.
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

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 这里的警告就是你刚才看到的
                print(f"[Config] Warning: Unknown config key: {key} (ignored)")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

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


def load_config() -> LGFFConfig:
    parser = argparse.ArgumentParser(description="LGFF Config Loader", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--opt", type=str, nargs="*", default=[], help="Override config")
    args, unknown = parser.parse_known_args()

    cfg = LGFFConfig()

    if args.config:
        print(f"[Config] Loading from {args.config}")
        yaml_data = _read_yaml(args.config)
        cfg.update(yaml_data)
        if "log_dir" not in yaml_data or not yaml_data.get("log_dir"):
            exp_name = os.path.splitext(os.path.basename(args.config))[0]
            cfg.log_dir = os.path.join("output", exp_name)

    for opt in args.opt:
        if "=" not in opt: continue
        key, value_str = opt.split("=", 1)
        value = _parse_value(value_str)
        cfg.update({key: value})
        print(f"[Config] Override: {key} = {value}")

    if cfg.work_dir is None:
        cfg.work_dir = cfg.log_dir

    os.makedirs(cfg.work_dir, exist_ok=True)
    save_path = os.path.join(cfg.work_dir, "config_used.yaml")
    cfg.save(save_path)
    print(f"[Config] Final config saved to {save_path}")

    return cfg


__all__ = ["LGFFConfig", "load_config"]