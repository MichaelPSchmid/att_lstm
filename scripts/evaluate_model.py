"""
Evaluation script for trained LSTM-Attention models.

Loads a trained checkpoint, evaluates on test set, and measures CPU inference time.

Usage:
    python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --config config/model_configs/m1_small_baseline.yaml
    python scripts/evaluate_model.py --checkpoint path/to/checkpoint.ckpt --config config/model_configs/m1_small_baseline.yaml --output results/m1_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from config_loader import load_config, get_model_class
from data_module import TimeSeriesDataModule
from config import get_preprocessed_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained LSTM-Attention model"
    )

    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to model config file (e.g., config/model_configs/m1_small_baseline.yaml)"
    )
    parser.add_argument(
        "--base-config", "-b",
        type=str,
        default=None,
        help="Path to base config file (default: config/base_config.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for evaluation"
    )
    parser.add_argument(
        "--skip-inference-test",
        action="store_true",
        help="Skip CPU inference time measurement"
    )

    return parser.parse_args()


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    accuracy_threshold: float = 0.05
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions [N, 1]
        targets: Ground truth targets [N, 1]
        accuracy_threshold: Threshold for accuracy calculation

    Returns:
        Dictionary with metrics
    """
    # Flatten arrays
    preds = predictions.flatten()
    targs = targets.flatten()

    # MSE
    mse = np.mean((preds - targs) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(preds - targs))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((preds - targs) / (targs + 1e-8))) * 100

    # R² Score
    ss_res = np.sum((preds - targs) ** 2)
    ss_tot = np.sum((targs - np.mean(targs)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Accuracy (predictions within threshold)
    correct = np.abs(preds - targs) < accuracy_threshold
    accuracy = np.mean(correct) * 100

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "accuracy": float(accuracy),
        "accuracy_threshold": accuracy_threshold,
        "num_samples": len(preds)
    }


def measure_inference_time(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup_iterations: int = 100,
    num_samples: int = 1000,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Measure CPU inference time for a single sample.

    Args:
        model: PyTorch model
        sample_input: Single input sample [1, seq_len, features]
        warmup_iterations: Number of warm-up iterations
        num_samples: Number of samples to average
        device: Device to run on ("cpu" recommended for embedded relevance)

    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    sample_input = sample_input.to(device)

    # Warm-up
    print(f"  Warming up ({warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(sample_input)

    # Measure inference time
    print(f"  Measuring ({num_samples} samples)...")
    times = []
    with torch.no_grad():
        for _ in range(num_samples):
            start = time.perf_counter()
            _ = model(sample_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "device": device,
        "warmup_iterations": warmup_iterations,
        "num_samples": num_samples
    }


def run_test_evaluation(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: str = "cuda"
) -> tuple:
    """
    Run model on test set and collect predictions.

    Args:
        model: PyTorch model
        test_dataloader: Test DataLoader
        device: Device to run on

    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            X_batch, Y_batch = batch
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(Y_batch.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def main():
    args = parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load configuration
    config = load_config(args.config, args.base_config)
    model_name = config["model"]["name"]
    model_type = config["model"]["type"]

    print(f"Model: {model_name}")
    print(f"Type: {model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print("-" * 60)

    # Set seed for reproducibility
    seed = config["training"]["seed"]
    pl.seed_everything(seed)

    # Get data paths
    data_config = config["data"]
    paths = get_preprocessed_paths(
        vehicle=data_config["vehicle"],
        window_size=data_config["window_size"],
        predict_size=data_config["predict_size"],
        step_size=data_config["step_size"],
        suffix="sF",
        variant=data_config["variant"]
    )

    # Create data module
    batch_size = args.batch_size or config["training"]["batch_size"]
    data_module = TimeSeriesDataModule(
        feature_path=str(paths["features"]),
        target_path=str(paths["targets"]),
        batch_size=batch_size
    )
    data_module.setup()

    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model_class = get_model_class(model_type)
    model = model_class.load_from_checkpoint(args.checkpoint)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation device: {device}")

    # Run test evaluation
    print("\nRunning test evaluation...")
    test_loader = data_module.test_dataloader()
    predictions, targets = run_test_evaluation(model, test_loader, device)

    # Calculate metrics
    accuracy_threshold = config["evaluation"]["accuracy_threshold"]
    metrics = calculate_metrics(predictions, targets, accuracy_threshold)

    print("\n" + "=" * 60)
    print("Test Set Metrics")
    print("=" * 60)
    print(f"  MSE:        {metrics['mse']:.6f}")
    print(f"  RMSE:       {metrics['rmse']:.6f}")
    print(f"  MAE:        {metrics['mae']:.6f}")
    print(f"  MAPE:       {metrics['mape']:.2f}%")
    print(f"  R²:         {metrics['r2']:.6f}")
    print(f"  Accuracy:   {metrics['accuracy']:.2f}% (threshold: {accuracy_threshold})")
    print(f"  Samples:    {metrics['num_samples']:,}")

    # Measure CPU inference time
    inference_results = None
    if not args.skip_inference_test:
        print("\n" + "=" * 60)
        print("CPU Inference Time Measurement")
        print("=" * 60)

        eval_config = config["evaluation"]["inference"]
        sample_input = torch.randn(1, data_config["window_size"], config["model"]["input_size"])

        inference_results = measure_inference_time(
            model,
            sample_input,
            warmup_iterations=eval_config["warmup_iterations"],
            num_samples=eval_config["num_samples"],
            device=eval_config["device"]
        )

        target_ms = config["evaluation"]["target_inference_ms"]
        meets_target = inference_results["mean_ms"] < target_ms

        print(f"  Mean:       {inference_results['mean_ms']:.3f} ms")
        print(f"  Std:        {inference_results['std_ms']:.3f} ms")
        print(f"  Min:        {inference_results['min_ms']:.3f} ms")
        print(f"  Max:        {inference_results['max_ms']:.3f} ms")
        print(f"  P50:        {inference_results['p50_ms']:.3f} ms")
        print(f"  P95:        {inference_results['p95_ms']:.3f} ms")
        print(f"  P99:        {inference_results['p99_ms']:.3f} ms")
        print(f"  Target:     <{target_ms} ms {'[PASS]' if meets_target else '[FAIL]'}")

    # Compile results
    results = {
        "model": {
            "name": model_name,
            "type": model_type,
            "parameters": num_params,
            "checkpoint": args.checkpoint
        },
        "data": {
            "variant": data_config["variant"],
            "vehicle": data_config["vehicle"],
            "window_size": data_config["window_size"]
        },
        "metrics": metrics,
        "inference": inference_results,
        "config_path": args.config
    }

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
