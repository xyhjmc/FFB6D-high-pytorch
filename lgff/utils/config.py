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
    Enhanced value parser for --opt key=value:
    - "true"/"false" -> bool
    - "1", "1.5"     -> int, float
    - "[1, 2]" or "1,2" -> list
    - "{'a': 1}"     -> dict
    - fallback       -> str
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

    目前主要服务于：
    - 数据集加载 (SingleObjectDataset)
    - 模型构建 (LGFF_SC)
    - 训练引擎 (TrainerSC)
    """

    # ----------------- Dataset / BOP -----------------
    dataset_name: str = "bop-single"
    dataset_root: str = "datasets/bop"  # BOP 数据集根目录，如 datasets/bop/lm
    annotation_file: Optional[str] = None  # 预留字段（当前 BOP 不用）

    # 单类 BOP 物体 ID (如 1=ape, 2=benchvise 等，需在 yaml 中指定)
    obj_id: int = 1

    # 目标网络输入分辨率 (H,W)，由 SingleObjectDataset 使用
    # 如果与原始 BOP 分辨率不同，SingleObjectDataset 会自动缩放内参 K
    resize_h: int = 480
    resize_w: int = 640

    # 深度图缩放因子：depth_m = raw_depth / depth_scale
    depth_scale: float = 1000.0  # 通常 BOP 深度为 mm，因此 scale=1000 -> m

    num_workers: int = 4
    batch_size: int = 8          # Nano/桌面通用的默认 batch
    num_classes: int = 1         # 当前 LGFF_SC 只做单类
    num_points: int = 1024       # 点云采样数，越大越精，越小越省算力
    num_keypoints: int = 8       # 预留字段，目前 LGFF 不直接用
    val_split: str = "test"      # 验证集的 split 名称（与 BOP 文件夹对应）

    # 保留 camera_intrinsic 作为通用几何工具的默认值
    # 注意：BOP 下我们主要使用 per-image 的 cam_K，这个字段更多是备用
    camera_intrinsic: List[List[float]] = field(
        default_factory=lambda: [
            [1066.778, 0.0, 312.9869],
            [0.0, 1067.487, 241.3109],
            [0.0, 0.0, 1.0],
        ]
    )

    # ----------------- Training Hyper-params -----------------
    epochs: int = 50
    lr: float = 1e-3             # MobileNet 通常比 ResNet 适合略大的 lr
    weight_decay: float = 1e-4
    use_amp: bool = True         # 混合精度开关
    log_interval: int = 10       # 每多少个 Iter 打一行 log

    # ----------------- Model / Loss Hyper-params -----------------
    rgb_feat_dim: int = 128      # LGFF_SC RGB 分支输出维度
    geo_feat_dim: int = 128      # LGFF_SC Geometry 分支输出维度
    w_rate: float = 0.015        # LGFFLoss 中置信度正则项权重
    sym_class_ids: List[int] = field(default_factory=list)  # 对称物体 ID 列表（多类时用）

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"  # 日志/模型输出根目录
    work_dir: Optional[str] = None # 具体实验目录（不设则等于 log_dir）

    def update(self, data: Dict[str, Any]) -> None:
        """Update attributes with dictionary data."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Optional: warn about unknown keys
                print(f"[Config] Warning: Unknown config key: {key} (ignored)")

    def save(self, path: str) -> None:
        """Dump current config to a YAML file for reproducibility."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "LGFFConfig":
        """
        允许在代码中直接通过路径构建 Config，例如：

            cfg = LGFFConfig.from_yaml("configs/helmet_sc.yaml")
        """
        yaml_data = _read_yaml(path)
        cfg = cls()
        cfg.update(yaml_data)

        # 如果 yaml 中没写 log_dir，则用 config 文件名自动生成
        if not getattr(cfg, "log_dir", None):
            exp_name = os.path.splitext(os.path.basename(path))[0]
            cfg.log_dir = os.path.join("output", exp_name)

        # work_dir 默认等于 log_dir
        if cfg.work_dir is None:
            cfg.work_dir = cfg.log_dir

        os.makedirs(cfg.work_dir, exist_ok=True)
        return cfg


def load_config() -> LGFFConfig:
    """
    Load configuration from CLI args and YAML files.
    Priority: CLI --opt > CLI --config > Defaults

    典型用法（在 train_lgff_sc.py 里）：
        from common.config import load_config
        cfg = load_config()
    """
    parser = argparse.ArgumentParser(description="LGFF Config Loader", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--opt",
        type=str,
        nargs="*",
        default=[],
        help="Override config entries with key=value pairs.",
    )

    args, unknown = parser.parse_known_args()

    cfg = LGFFConfig()

    # 1. Load from YAML
    if args.config:
        print(f"[Config] Loading from {args.config}")
        yaml_data = _read_yaml(args.config)
        cfg.update(yaml_data)

        # Auto-set log_dir based on config filename if not specified
        if "log_dir" not in yaml_data or not yaml_data.get("log_dir"):
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

    # 3. Path setup
    if cfg.work_dir is None:
        cfg.work_dir = cfg.log_dir

    os.makedirs(cfg.work_dir, exist_ok=True)

    # 4. Save the final config for records
    save_path = os.path.join(cfg.work_dir, "config_used.yaml")
    cfg.save(save_path)
    print(f"[Config] Final config saved to {save_path}")

    return cfg


__all__ = ["LGFFConfig", "load_config"]
