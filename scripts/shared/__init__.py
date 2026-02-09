"""
Shared library for evaluation scripts.

This module provides centralized utilities for:
- Model configurations (Single Source of Truth)
- Path management
- Checkpoint discovery
- Metrics calculation

Usage:
    from scripts.shared import MODELS, MODEL_BY_ID, find_best_checkpoint, calculate_metrics
"""

from .checkpoints import (
    CheckpointInfo,
    find_all_checkpoints,
    find_all_seed_checkpoints,
    find_best_checkpoint,
    find_best_checkpoint_for_seed,
    find_latest_checkpoint_for_resume,
    find_seed_variants,
)
from .metrics import (
    MetricResult,
    aggregate_metrics_per_sequence,
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
    "find_all_checkpoints",
    "find_all_seed_checkpoints",
    "find_best_checkpoint",
    "find_best_checkpoint_for_seed",
    "find_latest_checkpoint_for_resume",
    "find_seed_variants",
    # Metrics
    "MetricResult",
    "aggregate_metrics_per_sequence",
    "calculate_metrics",
    "calculate_metrics_dict",
]
