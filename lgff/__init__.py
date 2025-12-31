"""Lightweight Geometric-Feature Fusion (LGFF) package."""

# Models / engines
from lgff.models import LGFF_SC, LGFF_SC_SEG, LGFFBase
from lgff.datasets import SingleObjectDataset
from lgff.datasets.single_loader_seg import SingleObjectDataset as SingleObjectDatasetSeg
from lgff.losses import LGFFLoss, LGFFLoss_SEG
from lgff.engines import TrainerSC
from lgff.engines.trainer_sc_seg import TrainerSC as TrainerSCSeg
from lgff.engines.evaluator_sc_seg import EvaluatorSC as EvaluatorSCSeg

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
    "LGFF_SC_SEG",
    "LGFFBase",
    "SingleObjectDataset",
    "SingleObjectDatasetSeg",
    "LGFFLoss",
    "LGFFLoss_SEG",
    "TrainerSC",
    "TrainerSCSeg",
    "EvaluatorSCSeg",
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
