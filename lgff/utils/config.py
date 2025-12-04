"""Configuration utilities for the LGFF sub-project."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class LGFFConfig:
    """Minimal configuration container for LGFF."""

    dataset_name: str = "bop-single"
    dataset_root: str = "datasets/bop"
    annotation_file: Optional[str] = None
    camera_intrinsic: Any = field(default_factory=list)
    depth_scale: float = 1000.0
    num_workers: int = 4
    batch_size: int = 2
    num_points: int = 20000
    num_keypoints: int = 8
    log_dir: str = "train_log/lgff"
    epochs: int = 1
    lr: float = 1e-4

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config() -> LGFFConfig:
    """Load an LGFF configuration from command line / YAML."""

    parser = argparse.ArgumentParser(description="LGFF configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--opt",
        type=str,
        nargs="*",
        default=[],
        help="Override configuration entries with key=value pairs.",
    )
    args, _ = parser.parse_known_args()

    cfg = LGFFConfig()
    if args.config:
        yaml_cfg = _read_yaml(args.config)
        cfg.update(yaml_cfg)
    for override in args.opt:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        # Try to cast to numeric types when appropriate.
        if value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        cfg.update({key: value})

    # Default camera intrinsics fall back to the original FFB6D values so
    # that imported geometry utilities can operate without further adaptation.
    if not cfg.camera_intrinsic:
        from ffb6d.utils.basic_utils import intrinsic_matrix

        cfg.camera_intrinsic = intrinsic_matrix.get("ycb_K1")

    # Ensure directories exist for logging output.
    os.makedirs(cfg.log_dir, exist_ok=True)
    return cfg


__all__ = ["LGFFConfig", "load_config"]

