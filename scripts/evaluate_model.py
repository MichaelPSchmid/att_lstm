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
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating evaluation plots"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results/figures/{model_name}/)"
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


def generate_evaluation_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Dict[str, float],
    model_name: str,
    output_dir: Path,
    num_timeline_samples: int = 1000
) -> Dict[str, str]:
    """
    Generate evaluation plots and save them to disk.

    Args:
        predictions: Model predictions [N, 1]
        targets: Ground truth targets [N, 1]
        metrics: Dictionary with calculated metrics
        model_name: Name of the model for titles
        output_dir: Directory to save plots
        num_timeline_samples: Number of samples for timeline plot

    Returns:
        Dictionary with paths to saved plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = predictions.flatten()
    targs = targets.flatten()
    residuals = preds - targs

    saved_plots = {}

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Predicted vs Actual Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample for scatter plot if too many points
    max_scatter_points = 10000
    if len(preds) > max_scatter_points:
        idx = np.random.choice(len(preds), max_scatter_points, replace=False)
        scatter_preds = preds[idx]
        scatter_targs = targs[idx]
    else:
        scatter_preds = preds
        scatter_targs = targs

    ax.scatter(scatter_targs, scatter_preds, alpha=0.3, s=1, c='blue')

    # Perfect prediction line
    min_val = min(scatter_targs.min(), scatter_preds.min())
    max_val = max(scatter_targs.max(), scatter_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'{model_name}\nPredicted vs Actual (R² = {metrics["r2"]:.4f})', fontsize=14)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    scatter_path = output_dir / f'{model_name}_scatter.png'
    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots['scatter'] = str(scatter_path)

    # 2. Residual Distribution (Histogram)
    fig, ax = plt.subplots(figsize=(10, 6))

    mu, std = residuals.mean(), residuals.std()
    ax.hist(residuals, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=mu, color='red', linestyle='-', linewidth=2, label=f'Mean = {mu:.4f}')
    ax.axvline(x=mu-std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=mu+std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Std = {std:.4f}')

    ax.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name}\nResidual Distribution (MAE = {metrics["mae"]:.4f})', fontsize=14)
    ax.legend()

    residual_path = output_dir / f'{model_name}_residuals.png'
    fig.savefig(residual_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots['residuals'] = str(residual_path)

    # 3. Prediction Timeline
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Select a contiguous segment for timeline
    start_idx = len(preds) // 2  # Start from middle
    end_idx = min(start_idx + num_timeline_samples, len(preds))
    timeline_range = range(start_idx, end_idx)

    # Top plot: Actual vs Predicted
    axes[0].plot(timeline_range, targs[start_idx:end_idx], 'b-', linewidth=1, label='Actual', alpha=0.8)
    axes[0].plot(timeline_range, preds[start_idx:end_idx], 'r-', linewidth=1, label='Predicted', alpha=0.8)
    axes[0].set_ylabel('Steering Torque', fontsize=12)
    axes[0].set_title(f'{model_name}\nPrediction Timeline (samples {start_idx}-{end_idx})', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom plot: Error
    axes[1].fill_between(timeline_range, residuals[start_idx:end_idx], 0,
                         where=(residuals[start_idx:end_idx] >= 0), color='green', alpha=0.5, label='Overpredict')
    axes[1].fill_between(timeline_range, residuals[start_idx:end_idx], 0,
                         where=(residuals[start_idx:end_idx] < 0), color='red', alpha=0.5, label='Underpredict')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=metrics['accuracy_threshold'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axhline(y=-metrics['accuracy_threshold'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Error', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    timeline_path = output_dir / f'{model_name}_timeline.png'
    fig.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots['timeline'] = str(timeline_path)

    # 4. Error Distribution by Magnitude
    fig, ax = plt.subplots(figsize=(10, 6))

    abs_errors = np.abs(residuals)
    thresholds = [0.01, 0.025, 0.05, 0.1, 0.2]
    percentages = [(abs_errors < t).mean() * 100 for t in thresholds]

    bars = ax.bar([f'<{t}' for t in thresholds], percentages, color='steelblue', edgecolor='black')
    ax.axhline(y=metrics['accuracy'], color='red', linestyle='--', linewidth=2,
               label=f'Accuracy @ {metrics["accuracy_threshold"]} = {metrics["accuracy"]:.1f}%')

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Error Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Samples (%)', fontsize=12)
    ax.set_title(f'{model_name}\nCumulative Error Distribution', fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend()

    error_dist_path = output_dir / f'{model_name}_error_distribution.png'
    fig.savefig(error_dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_plots['error_distribution'] = str(error_dist_path)

    return saved_plots


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

    # Generate plots
    plot_paths = None
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("Generating Evaluation Plots")
        print("=" * 60)

        if args.plots_dir:
            plots_dir = Path(args.plots_dir)
        else:
            plots_dir = project_root / "results" / "figures" / model_name

        plot_paths = generate_evaluation_plots(
            predictions, targets, metrics, model_name, plots_dir
        )

        print(f"  Scatter plot:      {plot_paths['scatter']}")
        print(f"  Residual dist:     {plot_paths['residuals']}")
        print(f"  Timeline:          {plot_paths['timeline']}")
        print(f"  Error distribution: {plot_paths['error_distribution']}")

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
        "plots": plot_paths,
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
