"""
Metrics calculation utilities.

Provides consistent metric calculation across all evaluation scripts.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MetricResult:
    """Container for evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    accuracy: float
    accuracy_threshold: float
    num_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "accuracy": self.accuracy,
            "accuracy_threshold": self.accuracy_threshold,
            "num_samples": self.num_samples,
        }


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.05,
) -> MetricResult:
    """Calculate evaluation metrics.

    Args:
        predictions: Model predictions [N, 1] or [N,]
        targets: Ground truth targets [N, 1] or [N,]
        threshold: Threshold for accuracy calculation (default: 0.05)

    Returns:
        MetricResult with all calculated metrics
    """
    # Flatten arrays
    preds = predictions.flatten()
    targs = targets.flatten()

    # MSE
    mse = float(np.mean((preds - targs) ** 2))

    # RMSE
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(preds - targs)))

    # MAPE (avoid division by zero)
    mape = float(np.mean(np.abs((preds - targs) / (targs + 1e-8))) * 100)

    # R² Score
    ss_res = np.sum((preds - targs) ** 2)
    ss_tot = np.sum((targs - np.mean(targs)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    # Accuracy (predictions within threshold)
    correct = np.abs(preds - targs) < threshold
    accuracy = float(np.mean(correct) * 100)

    return MetricResult(
        mse=mse,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        accuracy=accuracy,
        accuracy_threshold=threshold,
        num_samples=len(preds),
    )


def calculate_metrics_dict(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """Calculate evaluation metrics and return as dictionary.

    This is a convenience wrapper for backwards compatibility with
    existing code that expects a dictionary return value.

    Args:
        predictions: Model predictions [N, 1] or [N,]
        targets: Ground truth targets [N, 1] or [N,]
        threshold: Threshold for accuracy calculation (default: 0.05)

    Returns:
        Dictionary with all calculated metrics
    """
    return calculate_metrics(predictions, targets, threshold).to_dict()
