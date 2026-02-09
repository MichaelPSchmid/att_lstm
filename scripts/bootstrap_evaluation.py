"""
Bootstrap Confidence Interval Evaluation for LSTM-Attention Models.

Calculates 95% confidence intervals for Accuracy, RMSE, and R² using
bootstrap resampling on the test set predictions.

Supports multi-seed evaluation: when multiple seeds are available for a model,
runs bootstrap per seed and aggregates using the law of total variance.

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

# Import from shared library
from scripts.shared import (
    MODELS,
    MODEL_BY_ID,
    PROJECT_ROOT,
    find_all_seed_checkpoints,
    find_best_checkpoint,
    get_config_path,
    calculate_metrics_dict,
)


# Relevant model comparison pairs for permutation tests.
# Each tuple: (model_a, model_b, category_label)
# Convention: model_a is the simpler/baseline model.
COMPARISON_PAIRS = [
    # Baseline vs Attention (same architecture size)
    ("M3", "M4", "Baseline vs Attention"),
    ("M5", "M6", "Baseline vs Attention"),
    ("M5", "M7", "Baseline vs Attention"),
    ("M5", "M8", "Baseline vs Attention"),
    # Attention mechanism comparisons (same size)
    ("M6", "M7", "Attention vs Attention"),
    ("M6", "M8", "Attention vs Attention"),
    ("M7", "M8", "Attention vs Attention"),
    # MLP vs LSTM
    ("M1", "M3", "MLP vs LSTM"),
    ("M2", "M5", "MLP vs LSTM"),
]


def _calculate_bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for bootstrap evaluation (simplified version).

    Uses only the three metrics needed for bootstrap: accuracy, rmse, r2.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    accuracy = np.mean(np.abs(y_true - y_pred) <= 0.05) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"accuracy": accuracy, "rmse": rmse, "r2": r2}


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
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
        metrics = _calculate_bootstrap_metrics(y_true_boot, y_pred_boot)
        bootstrap_accuracy.append(metrics["accuracy"])
        bootstrap_rmse.append(metrics["rmse"])
        bootstrap_r2.append(metrics["r2"])

    # Convert to arrays
    bootstrap_accuracy = np.array(bootstrap_accuracy)
    bootstrap_rmse = np.array(bootstrap_rmse)
    bootstrap_r2 = np.array(bootstrap_r2)

    # Calculate confidence intervals
    alpha = (1 - confidence) / 2

    def get_ci_stats(values: np.ndarray) -> Dict[str, Any]:
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


def aggregate_multi_seed_bootstrap(
    seed_bootstrap_results: Dict[int, Dict[str, Dict[str, Any]]],
    seed_point_metrics: Dict[int, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate bootstrap results across multiple seeds.

    Uses the law of total variance:
        Var_total = E[Var_bootstrap(seed)] + Var(E[metric(seed)])

    The first term captures within-seed (bootstrap sampling) uncertainty,
    the second captures between-seed (model initialization) uncertainty.

    Args:
        seed_bootstrap_results: seed -> {metric_name -> {mean, std, ...}}
        seed_point_metrics: seed -> {metric_name -> point_estimate}

    Returns:
        Aggregated {metric_name -> {mean, std, std_bootstrap, std_seed,
                                     ci_lower, ci_upper}}
    """
    metrics = ["accuracy", "rmse", "r2"]
    seeds = sorted(seed_bootstrap_results.keys())
    n_seeds = len(seeds)

    result = {}
    for metric in metrics:
        # Point estimates from each seed
        point_values = np.array([seed_point_metrics[s][metric] for s in seeds])

        # Bootstrap std from each seed
        bootstrap_stds = np.array([
            seed_bootstrap_results[s][metric]["std"] for s in seeds
        ])

        # Mean point estimate across seeds
        mean_val = float(np.mean(point_values))

        # Between-seed std (ddof=1 for unbiased estimate from small sample)
        std_seed = float(np.std(point_values, ddof=1)) if n_seeds > 1 else 0.0

        # Mean bootstrap variance (within-seed uncertainty)
        mean_bootstrap_var = float(np.mean(bootstrap_stds ** 2))
        std_bootstrap = float(np.sqrt(mean_bootstrap_var))

        # Combined std (law of total variance)
        std_combined = float(np.sqrt(mean_bootstrap_var + std_seed ** 2))

        # CI from combined std (normal approximation)
        ci_lower = mean_val - 1.96 * std_combined
        ci_upper = mean_val + 1.96 * std_combined

        result[metric] = {
            "mean": mean_val,
            "std": std_combined,
            "std_bootstrap": std_bootstrap,
            "std_seed": std_seed,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "per_seed_values": [float(v) for v in point_values],
        }

    return result


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
        return f"{mean:.2f} \u00b1 {std:.2f}"
    elif fmt == ".4f":
        return f"{mean:.4f} \u00b1 {std:.4f}"
    elif fmt == ".3f":
        return f"{mean:.3f} \u00b1 {std:.3f}"
    else:
        return f"{mean:{fmt}} \u00b1 {std:{fmt}}"


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate Markdown table from results."""
    lines = [
        "| Model | Seeds | Accuracy (%) | RMSE | R\u00b2 |",
        "|-------|-------|-------------|------|-----|"
    ]

    for r in results:
        name = f"{r['model_id']} {r['model_name']}"
        n_seeds = r.get("num_seeds", 1)
        acc = format_metric_with_ci(r["accuracy"], ".2f")
        rmse = format_metric_with_ci(r["rmse"], ".4f")
        r2 = format_metric_with_ci(r["r2"], ".3f")
        lines.append(f"| {name} | {n_seeds} | {acc} | {rmse} | {r2} |")

    return "\n".join(lines)


def generate_latex_table(results: List[Dict], caption: str = "Bootstrap Confidence Intervals") -> str:
    """Generate LaTeX table from results."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:bootstrap_ci}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Model & Description & Seeds & Accuracy (\\%) & RMSE & R\u00b2 \\\\",
        "\\midrule"
    ]

    for r in results:
        model_id = r["model_id"]
        name = r["model_name"]
        n_seeds = r.get("num_seeds", 1)

        # Format with proper +/- symbol for LaTeX
        acc_mean = r["accuracy"]["mean"]
        acc_std = r["accuracy"]["std"]
        rmse_mean = r["rmse"]["mean"]
        rmse_std = r["rmse"]["std"]
        r2_mean = r["r2"]["mean"]
        r2_std = r["r2"]["std"]

        acc_str = f"${acc_mean:.2f} \\pm {acc_std:.2f}$"
        rmse_str = f"${rmse_mean:.4f} \\pm {rmse_std:.4f}$"
        r2_str = f"${r2_mean:.3f} \\pm {r2_std:.3f}$"

        lines.append(f"{model_id} & {name} & {n_seeds} & {acc_str} & {rmse_str} & {r2_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


def _significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def run_all_comparisons(
    pairs: List[Tuple[str, str, str]],
    all_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run permutation tests for all comparison pairs.

    Skips pairs where one or both models have no predictions.

    Args:
        pairs: List of (model_a, model_b, category) tuples
        all_predictions: model_id -> predictions array
        y_true: Ground truth values
        n_permutations: Number of permutations per test
        seed: Random seed

    Returns:
        List of comparison result dicts
    """
    comparison_results = []

    for model_a, model_b, category in pairs:
        if model_a not in all_predictions or model_b not in all_predictions:
            continue

        pair_result = {
            "model_a": model_a,
            "model_b": model_b,
            "category": category,
        }

        for metric in ["accuracy", "rmse", "r2"]:
            result = permutation_test(
                y_true,
                all_predictions[model_a],
                all_predictions[model_b],
                metric=metric,
                n_permutations=n_permutations,
                seed=seed,
            )
            pair_result[metric] = result

        comparison_results.append(pair_result)

    return comparison_results


def generate_comparison_markdown(comparison_results: List[Dict[str, Any]]) -> str:
    """Generate Markdown table for permutation test results."""
    lines = [
        "| Comparison | Category | \u0394 Acc (%) | p | \u0394 RMSE | p | \u0394 R\u00b2 | p |",
        "|------------|----------|-----------|---|--------|---|------|---|",
    ]

    for c in comparison_results:
        label = f"{c['model_a']} \u2192 {c['model_b']}"
        cat = c["category"]

        acc = c["accuracy"]
        rmse = c["rmse"]
        r2 = c["r2"]

        acc_diff = f"{acc['observed_diff']:+.2f}"
        rmse_diff = f"{rmse['observed_diff']:+.4f}"
        r2_diff = f"{r2['observed_diff']:+.3f}"

        acc_p = f"{acc['p_value']:.3f}{_significance_stars(acc['p_value'])}"
        rmse_p = f"{rmse['p_value']:.3f}{_significance_stars(rmse['p_value'])}"
        r2_p = f"{r2['p_value']:.3f}{_significance_stars(r2['p_value'])}"

        lines.append(
            f"| {label} | {cat} | {acc_diff} | {acc_p} | {rmse_diff} | {rmse_p} | {r2_diff} | {r2_p} |"
        )

    lines.append("")
    lines.append("Significance: \\*p<0.05, \\*\\*p<0.01, \\*\\*\\*p<0.001")

    return "\n".join(lines)


def generate_comparison_latex(
    comparison_results: List[Dict[str, Any]],
    caption: str = "Permutation Test Results",
) -> str:
    """Generate LaTeX table for permutation test results."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:permutation_tests}",
        "\\begin{tabular}{llrcrcrcc}",
        "\\toprule",
        "Comparison & Category & $\\Delta$ Acc (\\%) & $p$ & $\\Delta$ RMSE & $p$ & $\\Delta$ R$^2$ & $p$ \\\\",
        "\\midrule",
    ]

    prev_category = None
    for c in comparison_results:
        label = f"{c['model_a']} vs {c['model_b']}"
        cat = c["category"]

        # Add midrule between categories
        if prev_category is not None and cat != prev_category:
            lines.append("\\midrule")
        prev_category = cat

        acc = c["accuracy"]
        rmse = c["rmse"]
        r2 = c["r2"]

        def fmt_p_latex(p_val: float) -> str:
            stars = _significance_stars(p_val)
            if p_val < 0.001:
                return f"$<$0.001{stars}"
            return f"{p_val:.3f}{stars}"

        acc_diff = f"{acc['observed_diff']:+.2f}"
        rmse_diff = f"{rmse['observed_diff']:+.4f}"
        r2_diff = f"{r2['observed_diff']:+.3f}"

        lines.append(
            f"{label} & {cat} & {acc_diff} & {fmt_p_latex(acc['p_value'])} "
            f"& {rmse_diff} & {fmt_p_latex(rmse['p_value'])} "
            f"& {r2_diff} & {fmt_p_latex(r2['p_value'])} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{8}{l}{\\footnotesize *$p<0.05$, **$p<0.01$, ***$p<0.001$ (10\\,000 permutations)} \\\\",
        "\\end{tabular}",
        "\\end{table}",
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

    For multi-seed models, pass the averaged predictions across seeds.

    Args:
        y_true: Ground truth values
        y_pred_a: Predictions from model A (or averaged across seeds)
        y_pred_b: Predictions from model B (or averaged across seeds)
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
    metrics_a = _calculate_bootstrap_metrics(y_true, y_pred_a)
    metrics_b = _calculate_bootstrap_metrics(y_true, y_pred_b)
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

        metrics_perm_a = _calculate_bootstrap_metrics(y_true, perm_a)
        metrics_perm_b = _calculate_bootstrap_metrics(y_true, perm_b)
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
        help="Compare a specific model pair (e.g., --compare M3 M4)"
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="Skip all permutation tests (bootstrap CIs only)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for significance tests (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility"
    )

    args = parser.parse_args()

    # Select model set
    variant = "dropout" if args.dropout else "no_dropout"
    model_set_name = variant

    print("=" * 70)
    print("Bootstrap Confidence Interval Evaluation (Multi-Seed)")
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
    first_model = MODELS[0]
    config_path = get_config_path(first_model, variant)
    config = load_config(str(config_path))
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
    all_predictions = {}  # model_id -> averaged predictions (for permutation test)
    all_seed_distributions = {}  # model_id -> {seed -> {metric -> distribution array}}
    y_true_common = None

    # Evaluate each model
    print("\n" + "=" * 70)
    print("Evaluating Models")
    print("=" * 70)

    for model_cfg in MODELS:
        model_id = model_cfg.id.upper()  # M1, M2, etc.
        model_name = model_cfg.name.split(" ", 1)[1] if " " in model_cfg.name else model_cfg.name
        print(f"\n{model_id}: {model_name}")
        print("-" * 50)

        # Find all seed checkpoints
        seed_checkpoints = find_all_seed_checkpoints(model_cfg, variant)

        # Fallback: try base model (no _seedN suffix)
        if not seed_checkpoints:
            base_ckpt = find_best_checkpoint(model_cfg, variant)
            if base_ckpt is None:
                print(f"  WARNING: No checkpoints found for {model_cfg.name}, skipping...")
                continue
            # Use base model as single "seed"
            from scripts.shared import find_all_checkpoints
            all_ckpts = find_all_checkpoints(model_cfg, variant)
            if all_ckpts:
                seed_checkpoints = {42: (all_ckpts[0].path, all_ckpts[0].val_loss)}
            else:
                print(f"  WARNING: Could not load checkpoint metadata, skipping...")
                continue

        seeds = sorted(seed_checkpoints.keys())
        print(f"  Found {len(seeds)} seed(s): {seeds}")

        # Per-seed evaluation
        seed_predictions: Dict[int, np.ndarray] = {}
        seed_point_metrics: Dict[int, Dict[str, float]] = {}
        seed_bootstrap_ci: Dict[int, Dict[str, Dict[str, Any]]] = {}

        for seed in seeds:
            ckpt_path, val_loss = seed_checkpoints[seed]
            print(f"\n  Seed {seed}:")
            print(f"    Checkpoint: {ckpt_path.name}")
            print(f"    Val Loss: {val_loss:.6f}")

            # Load config and model
            cfg_path = get_config_path(model_cfg, variant)
            cfg = load_config(str(cfg_path))
            model_class = get_model_class(cfg["model"]["type"])
            model = model_class.load_from_checkpoint(str(ckpt_path))

            # Get predictions
            print("    Running inference on test set...")
            predictions, targets = run_test_evaluation(model, test_loader, device)

            if y_true_common is None:
                y_true_common = targets

            seed_predictions[seed] = predictions

            # Point estimates
            point_metrics = _calculate_bootstrap_metrics(targets, predictions)
            seed_point_metrics[seed] = point_metrics
            print(f"    Point estimates:")
            print(f"      Accuracy: {point_metrics['accuracy']:.2f}%")
            print(f"      RMSE:     {point_metrics['rmse']:.4f}")
            print(f"      R\u00b2:       {point_metrics['r2']:.4f}")

            # Bootstrap CIs for this seed
            print(f"    Computing bootstrap CIs ({args.n_bootstrap} samples)...")
            ci = bootstrap_confidence_intervals(
                targets, predictions,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed
            )
            seed_bootstrap_ci[seed] = ci

            print(f"    95% CI:")
            print(f"      Accuracy: {ci['accuracy']['mean']:.2f} \u00b1 {ci['accuracy']['std']:.2f}")
            print(f"      RMSE:     {ci['rmse']['mean']:.4f} \u00b1 {ci['rmse']['std']:.4f}")
            print(f"      R\u00b2:       {ci['r2']['mean']:.3f} \u00b1 {ci['r2']['std']:.3f}")

            # Clean up
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Aggregate across seeds
        if len(seeds) > 1:
            print(f"\n  Aggregating across {len(seeds)} seeds (law of total variance)...")
            aggregated = aggregate_multi_seed_bootstrap(seed_bootstrap_ci, seed_point_metrics)

            for metric in ["accuracy", "rmse", "r2"]:
                m = aggregated[metric]
                fmt = ".2f" if metric == "accuracy" else (".4f" if metric == "rmse" else ".3f")
                suffix = "%" if metric == "accuracy" else ""
                print(f"    {metric.upper()}: {m['mean']:{fmt}}{suffix} \u00b1 {m['std']:{fmt}}"
                      f"  (bootstrap: {m['std_bootstrap']:{fmt}}, "
                      f"seed: {m['std_seed']:{fmt}})")
        else:
            # Single seed: use bootstrap CI directly
            single_seed = seeds[0]
            ci = seed_bootstrap_ci[single_seed]
            aggregated = {}
            for metric in ["accuracy", "rmse", "r2"]:
                aggregated[metric] = {
                    "mean": ci[metric]["mean"],
                    "std": ci[metric]["std"],
                    "std_bootstrap": ci[metric]["std"],
                    "std_seed": 0.0,
                    "ci_lower": ci[metric]["ci_lower"],
                    "ci_upper": ci[metric]["ci_upper"],
                    "per_seed_values": [seed_point_metrics[single_seed][metric]],
                }

        # Store averaged predictions for permutation test
        avg_predictions = np.mean(
            [seed_predictions[s] for s in seeds], axis=0
        )
        all_predictions[model_id] = avg_predictions

        # Cache per-seed bootstrap distributions for saving later
        all_seed_distributions[model_id] = {}
        for seed in seeds:
            ci = seed_bootstrap_ci[seed]
            all_seed_distributions[model_id][seed] = {
                metric: ci[metric]["distribution"] for metric in ["accuracy", "rmse", "r2"]
            }

        # Build result entry
        result = {
            "model_id": model_id,
            "model_name": model_name,
            "config_file": str(get_config_path(model_cfg, variant).relative_to(PROJECT_ROOT)),
            "num_seeds": len(seeds),
            "seeds": seeds,
            "accuracy": {k: v for k, v in aggregated["accuracy"].items()
                         if k != "distribution"},
            "rmse": {k: v for k, v in aggregated["rmse"].items()
                     if k != "distribution"},
            "r2": {k: v for k, v in aggregated["r2"].items()
                   if k != "distribution"},
            "per_seed": {},
        }

        # Add per-seed details
        for seed in seeds:
            ckpt_path, val_loss = seed_checkpoints[seed]
            pm = seed_point_metrics[seed]
            ci = seed_bootstrap_ci[seed]
            result["per_seed"][str(seed)] = {
                "checkpoint": str(ckpt_path),
                "val_loss": val_loss,
                "point_metrics": pm,
                "bootstrap_ci": {
                    metric: {k: v for k, v in ci[metric].items() if k != "distribution"}
                    for metric in ["accuracy", "rmse", "r2"]
                },
            }

        all_results.append(result)

    if not all_results:
        print("\nNo models evaluated. Check that checkpoints exist.")
        return

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
    multi_seed_models = [r for r in all_results if r.get("num_seeds", 1) > 1]
    if multi_seed_models:
        n = multi_seed_models[0]["num_seeds"]
        latex_caption += f", {n} seeds"
    latex_table = generate_latex_table(all_results, caption=latex_caption)
    print("\nLaTeX Table:")
    print(latex_table)

    # Permutation tests
    comparison_results = []
    if not args.no_comparisons:
        # Determine which pairs to test
        if args.compare:
            # Single custom pair
            pairs = [(args.compare[0], args.compare[1], "Custom")]
        else:
            # All predefined relevant pairs
            pairs = COMPARISON_PAIRS

        # Filter to pairs where both models have predictions
        available = set(all_predictions.keys())
        valid_pairs = [(a, b, c) for a, b, c in pairs if a in available and b in available]
        skipped = [(a, b, c) for a, b, c in pairs if a not in available or b not in available]

        if valid_pairs:
            multi_seed = any(r.get("num_seeds", 1) > 1 for r in all_results)

            print("\n" + "=" * 70)
            print(f"Permutation Tests ({len(valid_pairs)} comparisons, "
                  f"{args.n_permutations} permutations each)")
            print("=" * 70)

            if multi_seed:
                print("  (Using averaged predictions across seeds)")

            comparison_results = run_all_comparisons(
                valid_pairs, all_predictions, y_true_common,
                n_permutations=args.n_permutations, seed=args.seed,
            )

            # Print results
            for c in comparison_results:
                print(f"\n  {c['model_a']} vs {c['model_b']}  [{c['category']}]")
                for metric in ["accuracy", "rmse", "r2"]:
                    m = c[metric]
                    stars = _significance_stars(m["p_value"])
                    fmt = ".2f" if metric == "accuracy" else (".4f" if metric == "rmse" else ".3f")
                    print(f"    {metric.upper():>8}: \u0394={m['observed_diff']:{fmt}}  "
                          f"p={m['p_value']:.4f} {stars}")

            # Print comparison tables
            print("\n" + "-" * 70)
            print("Comparison Markdown Table:")
            comp_md = generate_comparison_markdown(comparison_results)
            print(comp_md)

            print("\nComparison LaTeX Table:")
            comp_latex = generate_comparison_latex(
                comparison_results,
                caption=f"Permutation Test Results ({args.n_permutations} permutations)"
            )
            print(comp_latex)

        if skipped:
            print(f"\n  Skipped (missing predictions): "
                  f"{', '.join(f'{a} vs {b}' for a, b, _ in skipped)}")

    # Save results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results (bootstrap + comparisons)
    json_path = output_dir / f"bootstrap_results_{model_set_name}.json"
    # Strip non-serializable fields from comparison results
    comparison_for_json = []
    for c in comparison_results:
        c_clean = {
            "model_a": c["model_a"],
            "model_b": c["model_b"],
            "category": c["category"],
        }
        for metric in ["accuracy", "rmse", "r2"]:
            c_clean[metric] = c[metric]
        comparison_for_json.append(c_clean)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_set": model_set_name,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "results": all_results,
            "comparisons": comparison_for_json,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Markdown file
    multi_seed = any(r.get("num_seeds", 1) > 1 for r in all_results)
    md_path = output_dir / f"bootstrap_table_{model_set_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Bootstrap Confidence Intervals ({model_set_name})\n\n")
        f.write(f"Bootstrap samples: {args.n_bootstrap}\n\n")
        if multi_seed:
            f.write("Uncertainty combines bootstrap sampling variance and between-seed variance "
                    "(law of total variance).\n\n")
        f.write(md_table)
        if comparison_results:
            f.write(f"\n\n## Permutation Tests ({args.n_permutations} permutations)\n\n")
            f.write(generate_comparison_markdown(comparison_results))
    print(f"Markdown table saved to: {md_path}")

    # LaTeX file
    latex_path = output_dir / f"bootstrap_table_{model_set_name}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
        if comparison_results:
            f.write("\n\n")
            f.write(generate_comparison_latex(
                comparison_results,
                caption=f"Permutation Test Results ({args.n_permutations} permutations)"
            ))
    print(f"LaTeX table saved to: {latex_path}")

    # Save cached bootstrap distributions as .npy (no recomputation)
    print("\nSaving bootstrap distributions...")
    for r_entry in all_results:
        mid = r_entry["model_id"]
        if mid not in all_seed_distributions:
            continue

        seed_dists = all_seed_distributions[mid]

        # Save per-seed distributions
        for seed, dists in seed_dists.items():
            dist_path = output_dir / f"bootstrap_dist_{mid}_seed{seed}_{model_set_name}.npy"
            np.save(dist_path, dists)

        # Save combined distribution (average across seeds)
        if len(seed_dists) > 1:
            combined = {}
            seeds = sorted(seed_dists.keys())
            for metric in ["accuracy", "rmse", "r2"]:
                combined[metric] = np.mean(
                    [seed_dists[s][metric] for s in seeds], axis=0
                )
            dist_path = output_dir / f"bootstrap_dist_{mid}_{model_set_name}.npy"
            np.save(dist_path, combined)
        else:
            # Single seed: combined = per-seed
            single_seed = list(seed_dists.keys())[0]
            dist_path = output_dir / f"bootstrap_dist_{mid}_{model_set_name}.npy"
            np.save(dist_path, seed_dists[single_seed])

    print(f"Bootstrap distributions saved to: {output_dir}/bootstrap_dist_*.npy")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
