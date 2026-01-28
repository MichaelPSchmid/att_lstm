"""
Shared library for evaluation scripts.

This module provides centralized utilities for:
- Model configurations (Single Source of Truth)
- Path management
- Checkpoint discovery
- Metrics calculation

Usage:
    from scripts.lib import MODELS, MODEL_BY_ID, find_best_checkpoint, calculate_metrics
"""

from .checkpoints import (
    CheckpointInfo,
    find_all_checkpoints,
    find_best_checkpoint,
    parse_checkpoint_name,
)
from .metrics import (
    MetricResult,
    calculate_metrics,
    calculate_metrics_dict,
)
from .models import (
    MODEL_BY_ID,
    MODELS,
    ModelConfig,
)
from .paths import (
    PROJECT_ROOT,
    get_config_path,
    get_log_dir,
    get_log_dir_name,
    get_model_output_dir,
    get_results_dir,
    load_eval_results,
    load_model_data_for_figures,
)

__all__ = [
    # Models
    "ModelConfig",
    "MODELS",
    "MODEL_BY_ID",
    # Paths
    "PROJECT_ROOT",
    "get_results_dir",
    "get_model_output_dir",
    "get_config_path",
    "get_log_dir_name",
    "get_log_dir",
    "load_eval_results",
    "load_model_data_for_figures",
    # Checkpoints
    "CheckpointInfo",
    "parse_checkpoint_name",
    "find_all_checkpoints",
    "find_best_checkpoint",
    # Metrics
    "MetricResult",
    "calculate_metrics",
    "calculate_metrics_dict",
]
