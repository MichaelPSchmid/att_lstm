"""
Bootstrap Confidence Interval Evaluation for LSTM-Attention Models.

Calculates 95% confidence intervals for Accuracy, RMSE, and R² using
bootstrap resampling on the test set predictions.

Usage:
    python scripts/bootstrap_evaluation.py
    python scripts/bootstrap_evaluation.py --n-bootstrap 2000
    python scripts/bootstrap_evaluation.py --dropout  # Evaluate dropout models
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from config.loader import load_config, get_model_class
from model.data_module import TimeSeriesDataModule
from config.settings import get_preprocessed_paths


# Model configurations
MODELS_NO_DROPOUT = [
    ("M1", "Small Baseline", "m1_small_baseline.yaml", "M1_Small_Baseline"),
    ("M2", "Small + Simple Attn", "m2_small_simple_attn.yaml", "M2_Small_Simple_Attention"),
    ("M3", "Medium Baseline", "m3_medium_baseline.yaml", "M3_Medium_Baseline"),
    ("M4", "Medium + Simple Attn", "m4_medium_simple_attn.yaml", "M4_Medium_Simple_Attention"),
    ("M5", "Medium + Additive Attn", "m5_medium_additive_attn.yaml", "M5_Medium_Additive_Attention"),
    ("M6", "Medium + Scaled DP", "m6_medium_scaled_dp_attn.yaml", "M6_Medium_Scaled_DP_Attention"),
]

MODELS_DROPOUT = [
    ("M1", "Small Baseline", "m1_small_baseline_dropout.yaml", "M1_Small_Baseline_Dropout"),
    ("M2", "Small + Simple Attn", "m2_small_simple_attn_dropout.yaml", "M2_Small_Simple_Attention_Dropout"),
    ("M3", "Medium Baseline", "m3_medium_baseline_dropout.yaml", "M3_Medium_Baseline_Dropout"),
    ("M4", "Medium + Simple Attn", "m4_medium_simple_attn_dropout.yaml", "M4_Medium_Simple_Attention_Dropout"),
    ("M5", "Medium + Additive Attn", "m5_medium_additive_attn_dropout.yaml", "M5_Medium_Additive_Attention_Dropout"),
    ("M6", "Medium + Scaled DP", "m6_medium_scaled_dp_attn_dropout.yaml", "M6_Medium_Scaled_DP_Attention_Dropout"),
]


def find_best_checkpoint(model_dir: str) -> Optional[Path]:
    """
    Find the best checkpoint for a model (lowest val_loss).

    Args:
        model_dir: Directory name in lightning_logs/

    Returns:
        Path to best checkpoint or None if not found
    """
    logs_dir = project_root / "lightning_logs" / model_dir

    if not logs_dir.exists():
        return None

    # Find latest version
    versions = sorted(logs_dir.glob("version_*"), key=lambda x: int(x.name.split("_")[1]))
    if not versions:
        return None

    latest_version = versions[-1]
    checkpoints_dir = latest_version / "checkpoints"

    if not checkpoints_dir.exists():
        return None

    # Find checkpoint with lowest val_loss
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Parse val_loss from filename and find minimum
    best_ckpt = None
    best_loss = float("inf")

    for ckpt in checkpoints:
        # Parse: ModelName-epoch=XX-val_loss=X.XXXX.ckpt
        try:
            loss_str = ckpt.stem.split("val_loss=")[1]
            loss = float(loss_str)
            if loss < best_loss:
                best_loss = loss
                best_ckpt = ckpt
        except (IndexError, ValueError):
            continue

    return best_ckpt


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with accuracy, rmse, r2
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Accuracy: percentage of predictions within threshold
    accuracy = np.mean(np.abs(y_true - y_pred) <= 0.05) * 100

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "accuracy": accuracy,
        "rmse": rmse,
        "r2": r2
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper for each metric
    """
    np.random.seed(seed)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n_samples = len(y_true)

    # Store bootstrap metrics
    bootstrap_accuracy = []
    bootstrap_rmse = []
    bootstrap_r2 = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate metrics for this bootstrap sample
        metrics = calculate_metrics(y_true_boot, y_pred_boot)
        bootstrap_accuracy.append(metrics["accuracy"])
        bootstrap_rmse.append(metrics["rmse"])
        bootstrap_r2.append(metrics["r2"])

    # Convert to arrays
    bootstrap_accuracy = np.array(bootstrap_accuracy)
    bootstrap_rmse = np.array(bootstrap_rmse)
    bootstrap_r2 = np.array(bootstrap_r2)

    # Calculate confidence intervals
    alpha = (1 - confidence) / 2

    def get_ci_stats(values: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": float(np.percentile(values, alpha * 100)),
            "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
            "distribution": values  # Keep for later analysis
        }

    return {
        "accuracy": get_ci_stats(bootstrap_accuracy),
        "rmse": get_ci_stats(bootstrap_rmse),
        "r2": get_ci_stats(bootstrap_r2)
    }


def run_test_evaluation(
    model: torch.nn.Module,
    test_dataloader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
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


def format_metric_with_ci(stats: Dict[str, Any], fmt: str = ".2f") -> str:
    """Format metric with confidence interval."""
    mean = stats["mean"]
    std = stats["std"]

    if fmt == ".2f":
        return f"{mean:.2f} ± {std:.2f}"
    elif fmt == ".4f":
        return f"{mean:.4f} ± {std:.4f}"
    elif fmt == ".3f":
        return f"{mean:.3f} ± {std:.3f}"
    else:
        return f"{mean:{fmt}} ± {std:{fmt}}"


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate Markdown table from results."""
    lines = [
        "| Model | Accuracy (%) | RMSE | R² |",
        "|-------|-------------|------|-----|"
    ]

    for r in results:
        name = f"{r['model_id']} {r['model_name']}"
        acc = format_metric_with_ci(r["accuracy"], ".2f")
        rmse = format_metric_with_ci(r["rmse"], ".4f")
        r2 = format_metric_with_ci(r["r2"], ".3f")
        lines.append(f"| {name} | {acc} | {rmse} | {r2} |")

    return "\n".join(lines)


def generate_latex_table(results: List[Dict], caption: str = "Bootstrap Confidence Intervals") -> str:
    """Generate LaTeX table from results."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:bootstrap_ci}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Model & Description & Accuracy (\\%) & RMSE & R² \\\\",
        "\\midrule"
    ]

    for r in results:
        model_id = r["model_id"]
        name = r["model_name"]

        # Format with proper ± symbol for LaTeX
        acc_mean = r["accuracy"]["mean"]
        acc_std = r["accuracy"]["std"]
        rmse_mean = r["rmse"]["mean"]
        rmse_std = r["rmse"]["std"]
        r2_mean = r["r2"]["mean"]
        r2_std = r["r2"]["std"]

        acc_str = f"${acc_mean:.2f} \\pm {acc_std:.2f}$"
        rmse_str = f"${rmse_mean:.4f} \\pm {rmse_std:.4f}$"
        r2_str = f"${r2_mean:.3f} \\pm {r2_std:.3f}$"

        lines.append(f"{model_id} & {name} & {acc_str} & {rmse_str} & {r2_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def permutation_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "accuracy",
    n_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Perform permutation test to compare two models.

    Args:
        y_true: Ground truth values
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        metric: Which metric to compare ("accuracy", "rmse", "r2")
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dictionary with observed difference, p-value, and significance
    """
    np.random.seed(seed)

    y_true = y_true.flatten()
    y_pred_a = y_pred_a.flatten()
    y_pred_b = y_pred_b.flatten()

    # Calculate observed difference
    metrics_a = calculate_metrics(y_true, y_pred_a)
    metrics_b = calculate_metrics(y_true, y_pred_b)
    observed_diff = metrics_b[metric] - metrics_a[metric]

    # Combine predictions
    combined = np.stack([y_pred_a, y_pred_b], axis=1)

    # Permutation test
    perm_diffs = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Randomly swap predictions for each sample
        swap_mask = np.random.randint(0, 2, size=len(y_true))
        perm_a = np.where(swap_mask == 0, combined[:, 0], combined[:, 1])
        perm_b = np.where(swap_mask == 0, combined[:, 1], combined[:, 0])

        metrics_perm_a = calculate_metrics(y_true, perm_a)
        metrics_perm_b = calculate_metrics(y_true, perm_b)
        perm_diffs.append(metrics_perm_b[metric] - metrics_perm_a[metric])

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "metric": metric
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Confidence Interval Evaluation"
    )
    parser.add_argument(
        "--n-bootstrap", "-n",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)"
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Evaluate dropout models instead of no-dropout"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/bootstrap",
        help="Output directory for results"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="Compare two models with permutation test (e.g., M3 M4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Select model set
    models = MODELS_DROPOUT if args.dropout else MODELS_NO_DROPOUT
    model_set_name = "dropout" if args.dropout else "no_dropout"

    print("=" * 70)
    print("Bootstrap Confidence Interval Evaluation")
    print("=" * 70)
    print(f"Model set: {model_set_name}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Random seed: {args.seed}")
    print("-" * 70)

    # Set seed
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)

    # Setup data module (load once, reuse for all models)
    print("\nLoading data...")
    config = load_config(f"config/model_configs/{models[0][2]}")
    data_config = config["data"]
    paths = get_preprocessed_paths(
        vehicle=data_config["vehicle"],
        window_size=data_config["window_size"],
        predict_size=data_config["predict_size"],
        step_size=data_config["step_size"],
        suffix="sF",
        variant=data_config["variant"]
    )

    data_module = TimeSeriesDataModule(
        feature_path=str(paths["features"]),
        target_path=str(paths["targets"]),
        batch_size=config["training"]["batch_size"]
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Store results and predictions for later comparison
    all_results = []
    all_predictions = {}
    y_true_common = None

    # Evaluate each model
    print("\n" + "=" * 70)
    print("Evaluating Models")
    print("=" * 70)

    for model_id, model_name, config_file, log_dir in models:
        print(f"\n{model_id}: {model_name}")
        print("-" * 50)

        # Find checkpoint
        checkpoint = find_best_checkpoint(log_dir)
        if checkpoint is None:
            print(f"  WARNING: No checkpoint found for {log_dir}, skipping...")
            continue

        print(f"  Checkpoint: {checkpoint.name}")

        # Load config and model
        config = load_config(f"config/model_configs/{config_file}")
        model_class = get_model_class(config["model"]["type"])
        model = model_class.load_from_checkpoint(str(checkpoint))

        # Get predictions
        print("  Running inference on test set...")
        predictions, targets = run_test_evaluation(model, test_loader, device)

        # Store for later comparison
        all_predictions[model_id] = predictions
        if y_true_common is None:
            y_true_common = targets

        # Calculate point estimates
        point_metrics = calculate_metrics(targets, predictions)
        print(f"  Point estimates:")
        print(f"    Accuracy: {point_metrics['accuracy']:.2f}%")
        print(f"    RMSE:     {point_metrics['rmse']:.4f}")
        print(f"    R²:       {point_metrics['r2']:.4f}")

        # Bootstrap confidence intervals
        print(f"  Computing bootstrap CIs ({args.n_bootstrap} samples)...")
        ci_results = bootstrap_confidence_intervals(
            targets, predictions,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed
        )

        print(f"  95% Confidence Intervals:")
        print(f"    Accuracy: {ci_results['accuracy']['mean']:.2f} ± {ci_results['accuracy']['std']:.2f}")
        print(f"    RMSE:     {ci_results['rmse']['mean']:.4f} ± {ci_results['rmse']['std']:.4f}")
        print(f"    R²:       {ci_results['r2']['mean']:.3f} ± {ci_results['r2']['std']:.3f}")

        # Store results (without full distribution for JSON serialization)
        result = {
            "model_id": model_id,
            "model_name": model_name,
            "config_file": config_file,
            "checkpoint": str(checkpoint),
            "accuracy": {
                "mean": ci_results["accuracy"]["mean"],
                "std": ci_results["accuracy"]["std"],
                "ci_lower": ci_results["accuracy"]["ci_lower"],
                "ci_upper": ci_results["accuracy"]["ci_upper"]
            },
            "rmse": {
                "mean": ci_results["rmse"]["mean"],
                "std": ci_results["rmse"]["std"],
                "ci_lower": ci_results["rmse"]["ci_lower"],
                "ci_upper": ci_results["rmse"]["ci_upper"]
            },
            "r2": {
                "mean": ci_results["r2"]["mean"],
                "std": ci_results["r2"]["std"],
                "ci_lower": ci_results["r2"]["ci_lower"],
                "ci_upper": ci_results["r2"]["ci_upper"]
            }
        }
        all_results.append(result)

        # Clean up
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Generate tables
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    # Markdown table
    md_table = generate_markdown_table(all_results)
    print("\nMarkdown Table:")
    print(md_table)

    # LaTeX table
    latex_caption = f"Bootstrap 95\\% Confidence Intervals ({args.n_bootstrap} samples)"
    if args.dropout:
        latex_caption += " - Models with Dropout"
    latex_table = generate_latex_table(all_results, caption=latex_caption)
    print("\nLaTeX Table:")
    print(latex_table)

    # Optional: Model comparison with permutation test
    if args.compare and len(args.compare) == 2:
        model_a, model_b = args.compare
        if model_a in all_predictions and model_b in all_predictions:
            print("\n" + "=" * 70)
            print(f"Permutation Test: {model_a} vs {model_b}")
            print("=" * 70)

            for metric in ["accuracy", "rmse", "r2"]:
                result = permutation_test(
                    y_true_common,
                    all_predictions[model_a],
                    all_predictions[model_b],
                    metric=metric,
                    n_permutations=10000,
                    seed=args.seed
                )

                sig_str = "***" if result["significant"] else ""
                print(f"  {metric.upper()}:")
                print(f"    Difference ({model_b} - {model_a}): {result['observed_diff']:.4f}")
                print(f"    p-value: {result['p_value']:.4f} {sig_str}")
        else:
            print(f"\nWARNING: Could not compare {model_a} and {model_b} - predictions not available")

    # Save results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / f"bootstrap_results_{model_set_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_set": model_set_name,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "results": all_results
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Markdown file
    md_path = output_dir / f"bootstrap_table_{model_set_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Bootstrap Confidence Intervals ({model_set_name})\n\n")
        f.write(f"Bootstrap samples: {args.n_bootstrap}\n\n")
        f.write(md_table)
    print(f"Markdown table saved to: {md_path}")

    # LaTeX file
    latex_path = output_dir / f"bootstrap_table_{model_set_name}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    # Save bootstrap distributions as .npy for later analysis
    # (Re-run bootstrap to get distributions - or modify above to keep them)
    print("\nSaving bootstrap distributions...")
    for result in all_results:
        model_id = result["model_id"]
        if model_id in all_predictions:
            # Re-run bootstrap to save distributions
            ci = bootstrap_confidence_intervals(
                y_true_common, all_predictions[model_id],
                n_bootstrap=args.n_bootstrap, seed=args.seed
            )

            dist_path = output_dir / f"bootstrap_dist_{model_id}_{model_set_name}.npy"
            np.save(dist_path, {
                "accuracy": ci["accuracy"]["distribution"],
                "rmse": ci["rmse"]["distribution"],
                "r2": ci["r2"]["distribution"]
            })

    print(f"Bootstrap distributions saved to: {output_dir}/bootstrap_dist_*.npy")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
