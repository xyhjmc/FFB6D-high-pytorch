"""Lightweight Geometric-Feature Fusion (LGFF) package."""

# Models / engines
from lgff.models import LGFF_SC, LGFFBase
from lgff.datasets import SingleObjectDataset
from lgff.losses import LGFFLoss
from lgff.engines import TrainerSC

# Configuration & utilities
from lgff.utils import (
    GeometryToolkit,
    LGFFConfig,
    add_metric,
    adds_metric,
    best_fit_transform,
    build_geometry,
    dense_backproject,
    depth_to_point_cloud,
    load_config,
    project_points,
    quaternion_to_matrix,
)

__all__ = [
    # Core components
    "LGFF_SC",
    "LGFFBase",
    "SingleObjectDataset",
    "LGFFLoss",
    "TrainerSC",
    # Config & geometry helpers
    "GeometryToolkit",
    "LGFFConfig",
    "add_metric",
    "adds_metric",
    "best_fit_transform",
    "build_geometry",
    "dense_backproject",
    "depth_to_point_cloud",
    "load_config",
    "project_points",
    "quaternion_to_matrix",
]
