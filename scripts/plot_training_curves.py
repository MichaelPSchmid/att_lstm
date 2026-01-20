"""
Plot training curves from TensorBoard logs.

Reads TensorBoard event files and generates publication-ready plots.

Usage:
    python scripts/plot_training_curves.py --logdir lightning_logs/M1_Small_Baseline/version_0
    python scripts/plot_training_curves.py --logdir lightning_logs/M1_Small_Baseline/version_0 --output results/figures/M1_Small_Baseline/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot training curves from TensorBoard logs"
    )

    parser.add_argument(
        "--logdir", "-l",
        type=str,
        required=True,
        help="Path to TensorBoard log directory (contains events.out.tfevents.*)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as logdir)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name for plot titles (default: inferred from logdir)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)"
    )

    return parser.parse_args()


def load_tensorboard_logs(logdir: str) -> Dict[str, List[tuple]]:
    """
    Load scalar data from TensorBoard event files.

    Args:
        logdir: Path to directory containing event files

    Returns:
        Dictionary mapping tag names to list of (step, value) tuples
    """
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def extract_epoch_metrics(data: Dict[str, List[tuple]]) -> Dict[str, np.ndarray]:
    """
    Extract per-epoch metrics from TensorBoard data.

    Args:
        data: Raw TensorBoard data

    Returns:
        Dictionary with arrays for each metric
    """
    metrics = {}

    for tag, values in data.items():
        if values:
            steps = np.array([v[0] for v in values])
            vals = np.array([v[1] for v in values])
            metrics[tag] = {'steps': steps, 'values': vals}

    return metrics


def plot_loss_curves(
    metrics: Dict,
    model_name: str,
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Plot training and validation loss curves.

    Args:
        metrics: Dictionary with metric data
        model_name: Name for plot title
        output_path: Directory to save plot
        fmt: Output format
        dpi: DPI for raster formats

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find loss metrics
    train_loss_key = None
    val_loss_key = None

    for key in metrics.keys():
        if 'train_loss' in key.lower() and 'epoch' in key.lower():
            train_loss_key = key
        elif 'train_loss' in key.lower() and train_loss_key is None:
            train_loss_key = key
        if 'val_loss' in key.lower() and 'epoch' not in key.lower():
            val_loss_key = key

    if train_loss_key and train_loss_key in metrics:
        train_data = metrics[train_loss_key]
        ax.plot(train_data['steps'], train_data['values'],
                'b-', linewidth=2, label='Training Loss', alpha=0.8)

    if val_loss_key and val_loss_key in metrics:
        val_data = metrics[val_loss_key]
        ax.plot(val_data['steps'], val_data['values'],
                'r-', linewidth=2, label='Validation Loss', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'{model_name}\nTraining and Validation Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis to start from 0 or slightly below minimum
    if train_loss_key or val_loss_key:
        all_vals = []
        if train_loss_key:
            all_vals.extend(metrics[train_loss_key]['values'])
        if val_loss_key:
            all_vals.extend(metrics[val_loss_key]['values'])
        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            ax.set_ylim(bottom=max(0, min_val - 0.1 * (max_val - min_val)))

    plt.tight_layout()
    save_path = output_path / f'{model_name}_loss_curves.{fmt}'
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(save_path)


def plot_metrics_curves(
    metrics: Dict,
    model_name: str,
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Plot validation metrics (R², Accuracy, RMSE) over epochs.

    Args:
        metrics: Dictionary with metric data
        model_name: Name for plot title
        output_path: Directory to save plot
        fmt: Output format
        dpi: DPI for raster formats

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Find metric keys
    r2_key = None
    acc_key = None
    rmse_key = None

    for key in metrics.keys():
        key_lower = key.lower()
        if 'val_r2' in key_lower and 'avg' not in key_lower:
            r2_key = key
        elif 'val_accuracy' in key_lower or 'val_abs_accuracy' in key_lower:
            acc_key = key
        elif 'val_rmse' in key_lower and 'avg' not in key_lower:
            rmse_key = key

    # R² plot
    if r2_key and r2_key in metrics:
        data = metrics[r2_key]
        axes[0].plot(data['steps'], data['values'], 'g-', linewidth=2)
        axes[0].set_ylabel('R²', fontsize=12)
        axes[0].set_title('Validation R²', fontsize=12)
        if len(data['values']) > 0:
            final_r2 = data['values'][-1]
            axes[0].axhline(y=final_r2, color='gray', linestyle='--', alpha=0.5)
            axes[0].text(0.95, 0.05, f'Final: {final_r2:.4f}', transform=axes[0].transAxes,
                        ha='right', va='bottom', fontsize=10, color='gray')
    else:
        axes[0].text(0.5, 0.5, 'No R² data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Validation R²', fontsize=12)

    # Accuracy plot
    if acc_key and acc_key in metrics:
        data = metrics[acc_key]
        axes[1].plot(data['steps'], np.array(data['values']) * 100, 'b-', linewidth=2)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Validation Accuracy', fontsize=12)
        if len(data['values']) > 0:
            final_acc = data['values'][-1] * 100
            axes[1].axhline(y=final_acc, color='gray', linestyle='--', alpha=0.5)
            axes[1].text(0.95, 0.05, f'Final: {final_acc:.1f}%', transform=axes[1].transAxes,
                        ha='right', va='bottom', fontsize=10, color='gray')
    else:
        axes[1].text(0.5, 0.5, 'No Accuracy data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Validation Accuracy', fontsize=12)

    # RMSE plot
    if rmse_key and rmse_key in metrics:
        data = metrics[rmse_key]
        axes[2].plot(data['steps'], data['values'], 'r-', linewidth=2)
        axes[2].set_ylabel('RMSE', fontsize=12)
        axes[2].set_title('Validation RMSE', fontsize=12)
        if len(data['values']) > 0:
            final_rmse = data['values'][-1]
            axes[2].axhline(y=final_rmse, color='gray', linestyle='--', alpha=0.5)
            axes[2].text(0.95, 0.05, f'Final: {final_rmse:.4f}', transform=axes[2].transAxes,
                        ha='right', va='bottom', fontsize=10, color='gray')
    else:
        axes[2].text(0.5, 0.5, 'No RMSE data', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Validation RMSE', fontsize=12)

    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} - Validation Metrics', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_path / f'{model_name}_metrics_curves.{fmt}'
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(save_path)


def plot_combined(
    metrics: Dict,
    model_name: str,
    output_path: Path,
    fmt: str = "png",
    dpi: int = 150
) -> str:
    """
    Plot combined training overview (loss + key metrics).

    Args:
        metrics: Dictionary with metric data
        model_name: Name for plot title
        output_path: Directory to save plot
        fmt: Output format
        dpi: DPI for raster formats

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Find keys
    train_loss_key = val_loss_key = r2_key = acc_key = None
    for key in metrics.keys():
        key_lower = key.lower()
        if 'train_loss' in key_lower and 'epoch' in key_lower:
            train_loss_key = key
        elif 'val_loss' in key_lower:
            val_loss_key = key
        elif 'val_r2' in key_lower and 'avg' not in key_lower:
            r2_key = key
        elif 'val_accuracy' in key_lower or 'val_abs_accuracy' in key_lower:
            acc_key = key

    # Top-left: Loss curves
    ax = axes[0, 0]
    if train_loss_key:
        data = metrics[train_loss_key]
        ax.plot(data['steps'], data['values'], 'b-', linewidth=2, label='Train', alpha=0.8)
    if val_loss_key:
        data = metrics[val_loss_key]
        ax.plot(data['steps'], data['values'], 'r-', linewidth=2, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: R²
    ax = axes[0, 1]
    if r2_key:
        data = metrics[r2_key]
        ax.plot(data['steps'], data['values'], 'g-', linewidth=2)
        if len(data['values']) > 0:
            ax.axhline(y=data['values'][-1], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Validation R²')
    ax.grid(True, alpha=0.3)

    # Bottom-left: Accuracy
    ax = axes[1, 0]
    if acc_key:
        data = metrics[acc_key]
        ax.plot(data['steps'], np.array(data['values']) * 100, 'b-', linewidth=2)
        if len(data['values']) > 0:
            ax.axhline(y=data['values'][-1] * 100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Learning rate (if available)
    ax = axes[1, 1]
    lr_key = None
    for key in metrics.keys():
        if 'lr' in key.lower() or 'learning_rate' in key.lower():
            lr_key = key
            break

    if lr_key:
        data = metrics[lr_key]
        ax.plot(data['steps'], data['values'], 'm-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
    else:
        # Show epoch info instead
        ax.text(0.5, 0.5, 'No LR data\n(using ReduceLROnPlateau)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Learning Rate')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} - Training Overview', fontsize=14)
    plt.tight_layout()

    save_path = output_path / f'{model_name}_training_overview.{fmt}'
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(save_path)


def main():
    args = parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: Log directory not found: {logdir}")
        sys.exit(1)

    # Determine model name
    if args.name:
        model_name = args.name
    else:
        # Infer from path (e.g., lightning_logs/M1_Small_Baseline/version_0 -> M1_Small_Baseline)
        parts = logdir.parts
        for i, part in enumerate(parts):
            if part == 'lightning_logs' and i + 1 < len(parts):
                model_name = parts[i + 1]
                break
        else:
            model_name = logdir.name

    # Determine output directory
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = logdir

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Curves Plotter")
    print("=" * 60)
    print(f"Log directory: {logdir}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_path}")
    print("-" * 60)

    # Load TensorBoard logs
    print("\nLoading TensorBoard logs...")
    try:
        data = load_tensorboard_logs(str(logdir))
    except Exception as e:
        print(f"Error loading logs: {e}")
        sys.exit(1)

    if not data:
        print("No scalar data found in logs.")
        sys.exit(1)

    print(f"Found {len(data)} metrics:")
    for tag in sorted(data.keys()):
        print(f"  - {tag} ({len(data[tag])} points)")

    # Extract metrics
    metrics = extract_epoch_metrics(data)

    # Generate plots
    print("\nGenerating plots...")

    loss_path = plot_loss_curves(metrics, model_name, output_path, args.format, args.dpi)
    print(f"  Loss curves: {loss_path}")

    metrics_path = plot_metrics_curves(metrics, model_name, output_path, args.format, args.dpi)
    print(f"  Metrics curves: {metrics_path}")

    overview_path = plot_combined(metrics, model_name, output_path, args.format, args.dpi)
    print(f"  Training overview: {overview_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
