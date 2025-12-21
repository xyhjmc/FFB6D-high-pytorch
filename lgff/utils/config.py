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
    depth_z_min_m: float = 0.10
    depth_z_max_m: float = 5.00
    mask_erosion: int = 1
    depth_edge_thresh_m: float = 0.0

    num_workers: int = 4
    batch_size: int = 8
    num_classes: int = 1
    num_points: int = 1024
    num_model_points: int = 1024
    num_keypoints: int = 8
    val_split: str = "test"
    # mask handling
    mask_invalid_policy: str = "skip"  # "skip" | "raise"
    allow_mask_fallback: bool = False

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

    # ----------------- Model / Feature Hyper-params -----------------

    # 骨干网络名称，用于切换 ResNet / MobileNet
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
    pc_centering: bool = True
    pc_scale_norm: bool = True

    gate_mode: str = "channel"
    gate_hidden: int = 128
    split_fusion_heads: bool = False
    gate_hidden_rot: int = 128
    gate_hidden_tc: int = 128

    conf_detach_trunk: bool = False
    kp_of_detach_trunk: bool = False
    init_z_bias: float = 0.7

    head_hidden_dim: int = 128
    head_feat_dim: int = 64
    head_dropout: float = 0.0

    # ----------------- Loss / Geometry Hyper-params -----------------

    # DenseFusion 中 conf 正则项的系数（如果还在用旧实现可保留）
    w_rate: float = 0.015

    # 对称物体的 class id 列表（用于 ADD-S / CMD 等）
    sym_class_ids: List[int] = field(default_factory=list)

    # 主几何 Loss 权重（LGFFLoss 使用）
    lambda_dense: float = 1.0       # ROI Dense 几何项（点到点 / 点到最近点）
    lambda_t: float = 0.5           # 平移 L1（含 Z 轴加权）
    lambda_rot: float = 0.5         # SO(3) 测地距离
    lambda_conf: float = 0.1        # 显式置信度回归
    lambda_add_cad: float = 0.1     # CAD 级 ADD/ADD-S
    lambda_kp_of: float = 0.3       # 关键点 offset 辅助分支

    # [NEW] 可选：专门抑制 z 轴 bias 的附加分支（你做消融时用）
    # - 默认 0.0：完全不启用，不影响旧实验复现
    lambda_t_bias_z: float = 0.0
    t_bias_z_mode: str = "batch_mean"
    t_bias_beta: float = 0.005
    rot_geodesic_eps: float = 1e-6

    # [NEW] z-bias 的形式开关：True -> 用 |bias_z|；False -> 用 bias_z（带符号）
    # - 建议默认 True（数值更稳定、方向不敏感），但你可做消融
    t_bias_z_use_abs: bool = True

    # 平移 / 置信度细节
    t_z_weight: float = 2.0         # Z 轴额外权重
    conf_alpha: float = 10.0        # 误差 -> 置信度 的敏感度
    conf_dist_max: float = 0.05     # 误差截断上限（m）

    # 是否使用测地旋转 Loss（方便开关对比实验）
    use_geodesic_rot: bool = True

    # 课程式 Loss 权重调度
    use_curriculum_loss: bool = False
    curriculum_warmup_frac: float = 0.4        # 前多少比例 epoch 保持原 lambda
    curriculum_final_factor_t: float = 0.3     # t 的最终衰减倍数
    curriculum_final_factor_rot: float = 0.3   # rot 的最终衰减倍数

    # 多任务同方差不确定性加权（Kendall-style）
    use_uncertainty_weighting: bool = False

    # ----------------- Pose Fusion Control -----------------

    # 姿态融合控制：train/val/viz 阶段可独立切换 best-point vs Top-K 加权
    train_use_best_point: Optional[bool] = None
    eval_use_best_point: bool = True
    viz_use_best_point: Optional[bool] = None

    # 通用的 top-k 融合上限（旧字段，默认仍保留）
    pose_fusion_topk: int = 64

    # eval 专用 topk（如果你在 pose_metrics.py 做了 stage-aware topk，建议保留该字段）
    eval_topk: Optional[int] = None

    # [NEW] 可选：train/viz 专用 topk（用于你后续“训练用 topk、评估用 topk”的消融）
    train_topk: Optional[int] = None
    viz_topk: Optional[int] = None

    # loss 中是否使用融合后的姿态（与 Evaluator 对齐）
    loss_use_fused_pose: bool = True

    # ----------------- 评估 / 阈值相关 -----------------

    # CMD 阈值（单位 m），Trainer / Evaluator / pose_metrics 统一使用
    cmd_threshold_m: float = 0.02

    # 物体直径（单位 m），如果不在 config 中写死，Trainer 会在 val 期间估算一次
    obj_diameter_m: Optional[float] = None

    # ICP 独立配置
    icp_num_points: int = 8192
    icp_use_full_depth: bool = True
    icp_point_source: Optional[str] = None  # optional override for logging clarity

    # ----------------- Logging / Output -----------------
    log_dir: str = "output/debug"
    work_dir: Optional[str] = None
    # Optional scene overlap guard
    forbid_scene_overlap: bool = False
    scene_overlap_policy: str = "warn"  # "warn" | "raise"

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Config] Warning: Unknown config key: {key} (ignored)")

    # 基础合法性校验，避免静默错误
    def validate(self) -> None:
        if self.resize_h <= 0 or self.resize_w <= 0:
            raise ValueError("resize_h/resize_w must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.camera_intrinsic) != 3 or any(len(row) != 3 for row in self.camera_intrinsic):
            raise ValueError("camera_intrinsic must be a 3x3 matrix")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be > 0")

        # lambdas 非负简单检查（防止手滑写成负数）
        for name in [
            "lambda_dense", "lambda_t", "lambda_rot",
            "lambda_conf", "lambda_add_cad", "lambda_kp_of",
            "lambda_t_bias_z",  # [NEW]
        ]:
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        # curriculum 参数基本范围检查
        if not (0.0 <= self.curriculum_warmup_frac <= 1.0):
            raise ValueError(f"curriculum_warmup_frac must be in [0,1], got {self.curriculum_warmup_frac}")
        if not (0.0 <= self.curriculum_final_factor_t <= 1.0):
            raise ValueError(f"curriculum_final_factor_t must be in [0,1], got {self.curriculum_final_factor_t}")
        if not (0.0 <= self.curriculum_final_factor_rot <= 1.0):
            raise ValueError(f"curriculum_final_factor_rot must be in [0,1], got {self.curriculum_final_factor_rot}")

        # topk 合法性（None 表示不覆盖）
        for kname in ["pose_fusion_topk", "train_topk", "eval_topk", "viz_topk"]:
            kval = getattr(self, kname, None)
            if kval is None:
                continue
            if not isinstance(kval, int) or kval <= 0:
                raise ValueError(f"{kname} must be a positive int or None, got {kval}")
        if self.mask_invalid_policy not in ("skip", "raise"):
            raise ValueError(f"mask_invalid_policy must be 'skip' or 'raise', got {self.mask_invalid_policy}")
        if self.scene_overlap_policy not in ("warn", "raise"):
            raise ValueError(f"scene_overlap_policy must be 'warn' or 'raise', got {self.scene_overlap_policy}")

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
