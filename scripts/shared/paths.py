"""
Path utilities for the evaluation scripts.

Provides centralized path management for results, configs, and logs.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .models import MODELS, ModelConfig


# Project root (scripts/shared -> scripts -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_results_dir(variant: str) -> Path:
    """Get the results directory for a variant.

    Args:
        variant: Either "dropout" or "no_dropout"

    Returns:
        Path to results/{variant}/
    """
    return PROJECT_ROOT / "results" / variant


def get_model_output_dir(model: ModelConfig, variant: str) -> Path:
    """Get the output directory for a specific model.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Path to results/{variant}/{model_id}/
    """
    return get_results_dir(variant) / model.id


def get_config_path(model: ModelConfig, variant: str) -> Path:
    """Get the config path for a model and variant.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Absolute path to the config YAML file
    """
    if variant == "dropout":
        return PROJECT_ROOT / model.config_dropout
    else:
        return PROJECT_ROOT / model.config_no_dropout


def get_log_dir_name(model: ModelConfig, variant: str) -> str:
    """Get the lightning_logs directory name for a model.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Directory name (not path) for lightning_logs/
    """
    if variant == "dropout":
        return model.log_dir_dropout
    else:
        return model.log_dir_no_dropout


def get_log_dir(model: ModelConfig, variant: str) -> Path:
    """Get the full path to the lightning_logs directory for a model.

    Args:
        model: Model configuration
        variant: Either "dropout" or "no_dropout"

    Returns:
        Path to lightning_logs/{log_dir_name}/
    """
    return PROJECT_ROOT / "lightning_logs" / get_log_dir_name(model, variant)


def load_eval_results(variant: str) -> Dict[str, Dict[str, Any]]:
    """Load all evaluation results (eval.json) for a variant.

    Args:
        variant: Either "dropout" or "no_dropout"

    Returns:
        Dictionary mapping model_id -> eval.json contents
    """
    results = {}
    results_dir = get_results_dir(variant)

    for model in MODELS:
        eval_json = results_dir / model.id / "eval.json"
        if eval_json.exists():
            with open(eval_json, "r", encoding="utf-8") as f:
                results[model.id] = json.load(f)

    return results


def load_model_data_for_figures(variant: str = "no_dropout") -> Optional[Dict[str, Dict[str, Any]]]:
    """Load model data for paper figure generation.

    Extracts the relevant fields from eval.json files for figure generation.
    Returns None if no results are available.

    Args:
        variant: Either "dropout" or "no_dropout"

    Returns:
        Dictionary mapping model_id (uppercase, e.g., "M1") to data dict with:
        - params: int
        - accuracy: float
        - inference_ms: float (P95)
        - category: str
        Or None if no results found.
    """
    eval_results = load_eval_results(variant)

    if not eval_results:
        return None

    # Category mapping based on model type
    category_map = {
        "m1": "small",
        "m2": "small",
        "m3": "medium_baseline",
        "m4": "medium_attention",
        "m5": "medium_attention",
        "m6": "medium_attention",
    }

    model_data = {}

    for model_id, data in eval_results.items():
        model_info = data.get("model", {})
        metrics = data.get("metrics", {})
        inference = data.get("inference", {})

        # Skip if essential data is missing
        if not all([model_info.get("parameters"), metrics.get("accuracy"), inference.get("p95_ms")]):
            continue

        # Use uppercase model ID for compatibility with existing figure code
        upper_id = model_id.upper()
        model_data[upper_id] = {
            "params": model_info.get("parameters", 0),
            "accuracy": metrics.get("accuracy", 0.0),
            "inference_ms": inference.get("p95_ms", 0.0),
            "category": category_map.get(model_id, "unknown"),
        }

    return model_data if model_data else None
