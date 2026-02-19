"""
Sequence-Level Statistical Evaluation for LSTM-Attention Models.

Fixes problems P4/P6 from the evaluation pipeline redesign:
- P4: Permutation test & Cohen's d on sequence-level (no autocorrelation)
- P6: Block bootstrap resamples whole sequences

Uses per-sequence metrics as the unit of analysis (~500 independent
observations) instead of per-sample metrics (~220k correlated observations).

Statistical methods:
- Block bootstrap: resample whole sequences for CIs
- Paired sign-flip permutation test: swap models per sequence
- Paired Cohen's d + Hedge's g: standardized effect size on sequences

Usage:
    python scripts/sequence_level_evaluation.py
    python scripts/sequence_level_evaluation.py --n-bootstrap 2000
    python scripts/sequence_level_evaluation.py --dropout
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from scripts.shared import (
    MODELS,
    PROJECT_ROOT,
    aggregate_metrics_per_sequence,
    find_all_checkpoints,
    find_all_seed_checkpoints,
    find_best_checkpoint,
    get_config_path,
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

# Metrics used for sequence-level analysis
# accuracy, r2: higher = better (no sign-flip for Cohen's d)
# rmse, mae: lower = better (sign-flip applied)
METRICS = ["accuracy", "rmse", "mae", "r2"]


# =============================================================================
# Core Statistical Functions
# =============================================================================

def bootstrap_ci_sequences(
    seq_metric_values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Block bootstrap confidence intervals on per-sequence metric values.

    Resamples whole sequences with replacement and computes the mean metric
    for each bootstrap sample. This respects within-sequence correlation.

    Args:
        seq_metric_values: Per-sequence metric values [n_sequences]
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    n_seq = len(seq_metric_values)

    # Vectorized: generate all bootstrap indices at once
    indices = rng.choice(n_seq, size=(n_bootstrap, n_seq), replace=True)
    boot_means = np.mean(seq_metric_values[indices], axis=1)

    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(boot_means)),
        "std": float(np.std(boot_means)),
        "ci_lower": float(np.percentile(boot_means, alpha * 100)),
        "ci_upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
    }


def cohens_d_paired_sequences(
    seq_metric_a: np.ndarray,
    seq_metric_b: np.ndarray,
) -> Dict[str, Any]:
    """Paired Cohen's d with Hedge's g correction on per-sequence metrics.

    Computes standardized effect size from paired per-sequence metrics.
    Positive d means model B has higher values than model A.
    Convention: for error metrics (MAE, RMSE), swap sign so positive = B better.

    Args:
        seq_metric_a: Per-sequence metric for model A [n_sequences]
        seq_metric_b: Per-sequence metric for model B [n_sequences]

    Returns:
        Dictionary with cohens_d, hedges_g, effect_size, n_sequences
    """
    diffs = seq_metric_b - seq_metric_a
    n = len(diffs)
    std_diffs = float(np.std(diffs, ddof=1))

    if std_diffs < 1e-12:
        d = 0.0
    else:
        d = float(np.mean(diffs) / std_diffs)

    # Hedge's g: bias-corrected Cohen's d for small samples
    # J(df) = 1 - 3/(4*df - 1), where df = n-1 for paired data
    df = n - 1
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    g = d * correction

    return {
        "cohens_d": d,
        "hedges_g": float(g),
        "effect_size": _effect_size_category(abs(d)),
        "n_sequences": n,
    }


def permutation_test_sequences(
    seq_metric_a: np.ndarray,
    seq_metric_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired sign-flip permutation test on per-sequence metrics.

    Under H0 (no difference between models), each paired difference is
    equally likely to be positive or negative. We randomly flip signs
    of the paired differences to build the null distribution.

    This properly handles within-sequence correlation because entire
    sequences are kept together (only the model assignment is permuted).

    Args:
        seq_metric_a: Per-sequence metric for model A [n_sequences]
        seq_metric_b: Per-sequence metric for model B [n_sequences]
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with observed_diff, p_value, significant
    """
    diffs = seq_metric_b - seq_metric_a
    observed = float(np.mean(diffs))
    n_seq = len(diffs)

    # Vectorized: generate all sign-flip matrices at once
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_seq))
    perm_stats = np.mean(diffs[np.newaxis, :] * signs, axis=1)

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(perm_stats) >= np.abs(observed)))

    return {
        "observed_diff": observed,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def multi_seed_sequence_analysis(
    seed_bootstrap_results: Dict[int, Dict[str, Dict[str, float]]],
    seed_point_metrics: Dict[int, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate sequence-level bootstrap results across multiple seeds.

    Uses the law of total variance:
        Var_total = E[Var_bootstrap(seed)] + Var(E[metric(seed)])

    The first term captures within-seed (bootstrap sampling) uncertainty,
    the second captures between-seed (model initialization) uncertainty.

    Args:
        seed_bootstrap_results: seed -> {metric -> {mean, std, ci_lower, ci_upper}}
        seed_point_metrics: seed -> {metric -> point_estimate}

    Returns:
        Aggregated {metric -> {mean, std, std_bootstrap, std_seed,
                               ci_lower, ci_upper, per_seed_values}}
    """
    seeds = sorted(seed_bootstrap_results.keys())
    n_seeds = len(seeds)

    result = {}
    for metric in METRICS:
        # Point estimates from each seed
        point_values = np.array([seed_point_metrics[s][metric] for s in seeds])

        # Bootstrap std from each seed
        bootstrap_stds = np.array([
            seed_bootstrap_results[s][metric]["std"] for s in seeds
        ])

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


# =============================================================================
# Helper Functions
# =============================================================================

def _effect_size_category(d: float) -> str:
    """Categorize Cohen's d effect size (Cohen, 1988)."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _compute_seq_metric_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    sequence_ids: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Compute per-sequence metric arrays from sample-level data.

    Args:
        predictions: Sample-level predictions [N]
        targets: Sample-level targets [N]
        sequence_ids: Per-sample sequence IDs [N]
        threshold: Accuracy threshold

    Returns:
        Dictionary mapping metric name to per-sequence values array.
        Note: r2 array may be shorter than others if sequences with
        near-constant targets (SS_tot < 1e-12) are excluded.
    """
    rows, _ = aggregate_metrics_per_sequence(
        predictions, targets, sequence_ids, threshold
    )

    # Per-sequence R² (excludes sequences with near-constant targets)
    preds = predictions.flatten()
    targs = targets.flatten()
    seq_ids = np.asarray(sequence_ids)
    unique_ids = np.unique(seq_ids)
    r2_values = []
    for sid in unique_ids:
        mask = seq_ids == sid
        p = preds[mask]
        t = targs[mask]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        if ss_tot < 1e-12:
            continue
        r2_values.append(1.0 - ss_res / ss_tot)

    return {
        "accuracy": np.array([r["accuracy"] for r in rows]),
        "rmse": np.array([r["rmse"] for r in rows]),
        "mae": np.array([r["mae"] for r in rows]),
        "r2": np.array(r2_values),
    }


def run_test_evaluation(
    model: torch.nn.Module,
    test_dataloader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on test set and collect predictions.

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

    return (
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def run_all_comparisons(
    pairs: List[Tuple[str, str, str]],
    all_seq_metrics: Dict[str, Dict[str, np.ndarray]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run pairwise comparisons on sequence-level metrics.

    For each pair of models, runs permutation tests and computes
    Cohen's d + Hedge's g per metric.

    Sign convention for Cohen's d:
        - Accuracy: positive d = B higher (better). No flip.
        - RMSE, MAE: positive d = B lower (better). Sign-flip applied.

    Args:
        pairs: List of (model_a, model_b, category) tuples
        all_seq_metrics: model_id -> {metric -> per-sequence values array}
        n_permutations: Number of permutations per test
        seed: Random seed

    Returns:
        List of comparison result dicts
    """
    # Error metrics where lower = better (need sign-flip)
    error_metrics = {"rmse", "mae"}

    comparison_results = []

    for model_a, model_b, category in pairs:
        if model_a not in all_seq_metrics or model_b not in all_seq_metrics:
            continue

        metrics_a = all_seq_metrics[model_a]
        metrics_b = all_seq_metrics[model_b]

        pair_result = {
            "model_a": model_a,
            "model_b": model_b,
            "category": category,
            "n_sequences": len(metrics_a[METRICS[0]]),
        }

        # Permutation tests + Cohen's d for each metric
        for metric in METRICS:
            perm = permutation_test_sequences(
                metrics_a[metric], metrics_b[metric],
                n_permutations=n_permutations,
                seed=seed,
            )

            d_result = cohens_d_paired_sequences(
                metrics_a[metric], metrics_b[metric],
            )

            # Sign-flip for error metrics: positive d = B better (lower)
            if metric in error_metrics:
                d_result["cohens_d"] = -d_result["cohens_d"]
                d_result["hedges_g"] = -d_result["hedges_g"]

            pair_result[metric] = {
                "observed_diff": perm["observed_diff"],
                "p_value": perm["p_value"],
                "significant": perm["significant"],
                "metric": metric,
                "cohens_d": d_result["cohens_d"],
                "hedges_g": d_result["hedges_g"],
                "effect_size": d_result["effect_size"],
            }

        comparison_results.append(pair_result)

    return comparison_results


# =============================================================================
# Table Generation
# =============================================================================

def format_metric_with_ci(stats: Dict[str, Any], fmt: str = ".2f") -> str:
    """Format metric with confidence interval."""
    return f"{stats['mean']:{fmt}} \u00b1 {stats['std']:{fmt}}"


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate Markdown table from sequence-level bootstrap results."""
    lines = [
        "| Model | Seeds | Seqs | Accuracy (%) | RMSE | MAE | R² |",
        "|-------|-------|------|-------------|------|-----|-----|",
    ]

    for r in results:
        name = f"{r['model_id']} {r['model_name']}"
        n_seeds = r.get("num_seeds", 1)
        n_seq = r.get("n_test_sequences", "?")
        acc = format_metric_with_ci(r["accuracy"], ".2f")
        rmse = format_metric_with_ci(r["rmse"], ".4f")
        mae = format_metric_with_ci(r["mae"], ".4f")
        r2 = format_metric_with_ci(r["r2"], ".3f")
        lines.append(
            f"| {name} | {n_seeds} | {n_seq} "
            f"| {acc} | {rmse} | {mae} | {r2} |"
        )

    return "\n".join(lines)


def generate_latex_table(
    results: List[Dict],
    caption: str = "Sequence-Level Bootstrap Confidence Intervals",
) -> str:
    """Generate LaTeX table from sequence-level results."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:sequence_level_ci}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Model & Description & Seeds & Seqs "
        "& Accuracy (\\%) & RMSE & MAE & $R^2$ \\\\",
        "\\midrule",
    ]

    for r in results:
        model_id = r["model_id"]
        name = r["model_name"]
        n_seeds = r.get("num_seeds", 1)
        n_seq = r.get("n_test_sequences", "?")

        acc_str = (f"${r['accuracy']['mean']:.2f} \\pm "
                   f"{r['accuracy']['std']:.2f}$")
        rmse_str = (f"${r['rmse']['mean']:.4f} \\pm "
                    f"{r['rmse']['std']:.4f}$")
        mae_str = (f"${r['mae']['mean']:.4f} \\pm "
                   f"{r['mae']['std']:.4f}$")
        r2_str = (f"${r['r2']['mean']:.3f} \\pm "
                  f"{r['r2']['std']:.3f}$")

        lines.append(
            f"{model_id} & {name} & {n_seeds} & {n_seq} "
            f"& {acc_str} & {rmse_str} & {mae_str} & {r2_str} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_comparison_markdown(comparison_results: List[Dict[str, Any]]) -> str:
    """Generate Markdown table for sequence-level comparison results."""
    lines = [
        "| Comparison | Category | Seqs "
        "| \u0394 Acc (%) | \u0394 RMSE | \u0394 MAE | \u0394 R\u00b2 "
        "| d(Acc) | d(RMSE) | d(MAE) | d(R\u00b2) |",
        "|------------|----------|------"
        "|-----------|--------|------|------"
        "|--------|---------|--------|--------|",
    ]

    for c in comparison_results:
        label = f"{c['model_a']} \u2192 {c['model_b']}"
        cat = c["category"]
        n_seq = c.get("n_sequences", "?")

        acc_diff = f"{c['accuracy']['observed_diff']:+.2f}"
        rmse_diff = f"{c['rmse']['observed_diff']:+.4f}"
        mae_diff = f"{c['mae']['observed_diff']:+.4f}"
        r2_diff = f"{c['r2']['observed_diff']:+.4f}"

        d_acc = f"{c['accuracy']['cohens_d']:+.3f}"
        d_rmse = f"{c['rmse']['cohens_d']:+.3f}"
        d_mae = f"{c['mae']['cohens_d']:+.3f}"
        d_r2 = f"{c['r2']['cohens_d']:+.3f}"

        # Add significance stars from permutation test
        acc_stars = _significance_stars(c["accuracy"]["p_value"])
        rmse_stars = _significance_stars(c["rmse"]["p_value"])
        mae_stars = _significance_stars(c["mae"]["p_value"])
        r2_stars = _significance_stars(c["r2"]["p_value"])

        lines.append(
            f"| {label} | {cat} | {n_seq} "
            f"| {acc_diff}{acc_stars} | {rmse_diff}{rmse_stars} "
            f"| {mae_diff}{mae_stars} | {r2_diff}{r2_stars} "
            f"| {d_acc} | {d_rmse} | {d_mae} | {d_r2} |"
        )

    lines.append("")
    lines.append(
        "Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, "
        "0.5-0.8 medium, >0.8 large"
    )
    lines.append(
        "d sign convention: positive = B better "
        "(higher accuracy/R\u00b2, lower RMSE/MAE)"
    )
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")

    return "\n".join(lines)


def generate_comparison_latex(
    comparison_results: List[Dict[str, Any]],
    caption: str = "Sequence-Level Pairwise Model Comparisons",
) -> str:
    """Generate LaTeX table for sequence-level comparison results."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:sequence_level_comparisons}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Comparison & Category & $\\Delta$ Acc (\\%) & $\\Delta$ MAE "
        "& $d$(Acc) & $d$(RMSE) & $d$(MAE) & $d$($R^2$) \\\\",
        "\\midrule",
    ]

    prev_category = None
    for c in comparison_results:
        label = f"{c['model_a']} vs {c['model_b']}"
        cat = c["category"]

        if prev_category is not None and cat != prev_category:
            lines.append("\\midrule")
        prev_category = cat

        acc_diff = f"{c['accuracy']['observed_diff']:+.2f}"
        mae_diff = f"{c['mae']['observed_diff']:+.4f}"
        d_acc = f"{c['accuracy']['cohens_d']:+.3f}"
        d_rmse = f"{c['rmse']['cohens_d']:+.3f}"
        d_mae = f"{c['mae']['cohens_d']:+.3f}"
        d_r2 = f"{c['r2']['cohens_d']:+.3f}"

        lines.append(
            f"{label} & {cat} & {acc_diff} & {mae_diff} "
            f"& {d_acc} & {d_rmse} & {d_mae} & {d_r2} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{8}{l}{\\footnotesize Cohen's $d$: $|d|<0.2$ negligible, "
        "$0.2$--$0.5$ small, $0.5$--$0.8$ medium, $>0.8$ large} \\\\",
        "\\multicolumn{8}{l}{\\footnotesize Positive $d$ = B better "
        "(higher accuracy/$R^2$, lower RMSE/MAE)} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sequence-Level Statistical Evaluation"
    )
    parser.add_argument(
        "--n-bootstrap", "-n",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Evaluate dropout models instead of no-dropout",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/bootstrap",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="Skip all permutation tests (bootstrap CIs only)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for significance tests (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Accuracy threshold (default: 0.05)",
    )

    args = parser.parse_args()

    variant = "dropout" if args.dropout else "no_dropout"

    print("=" * 70)
    print("Sequence-Level Statistical Evaluation")
    print("=" * 70)
    print(f"Model set: {variant}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Accuracy threshold: {args.threshold}")
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
        variant=data_config["variant"],
    )

    data_module = TimeSeriesDataModule(
        feature_path=str(paths["features"]),
        target_path=str(paths["targets"]),
        sequence_ids_path=str(paths["sequence_ids"]),
        batch_size=config["training"]["batch_size"],
        split_seed=data_config.get("split_seed", 0),
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # Get test sequence IDs
    test_sequence_ids = data_module.get_split_sequence_ids("test")
    n_test_sequences = len(np.unique(test_sequence_ids))
    print(f"Test sequences: {n_test_sequences}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Store results
    all_results = []
    all_seq_metrics = {}  # model_id -> {metric -> per-sequence values}
    y_true_common = None

    print("\n" + "=" * 70)
    print("Evaluating Models (Sequence-Level)")
    print("=" * 70)

    for model_cfg in MODELS:
        model_id = model_cfg.id.upper()
        model_name = (model_cfg.name.split(" ", 1)[1]
                      if " " in model_cfg.name else model_cfg.name)
        print(f"\n{model_id}: {model_name}")
        print("-" * 50)

        # Find all seed checkpoints
        seed_checkpoints = find_all_seed_checkpoints(model_cfg, variant)

        if not seed_checkpoints:
            base_ckpt = find_best_checkpoint(model_cfg, variant)
            if base_ckpt is None:
                print(f"  WARNING: No checkpoints found, skipping...")
                continue
            all_ckpts = find_all_checkpoints(model_cfg, variant)
            if all_ckpts:
                seed_checkpoints = {42: (all_ckpts[0].path, all_ckpts[0].val_loss)}
            else:
                print(f"  WARNING: Could not load checkpoint metadata, skipping...")
                continue

        seeds = sorted(seed_checkpoints.keys())
        print(f"  Found {len(seeds)} seed(s): {seeds}")

        # Per-seed evaluation
        seed_seq_metrics: Dict[int, Dict[str, np.ndarray]] = {}
        seed_point_metrics: Dict[int, Dict[str, float]] = {}
        seed_bootstrap_ci: Dict[int, Dict[str, Dict[str, float]]] = {}

        for train_seed in seeds:
            ckpt_path, val_loss = seed_checkpoints[train_seed]
            print(f"\n  Seed {train_seed}:")
            print(f"    Checkpoint: {ckpt_path.name}")
            print(f"    Val Loss: {val_loss:.6f}")

            # Load model and run inference
            cfg_path = get_config_path(model_cfg, variant)
            cfg = load_config(str(cfg_path))
            model_class = get_model_class(cfg["model"]["type"])
            model = model_class.load_from_checkpoint(str(ckpt_path))

            print("    Running inference on test set...")
            predictions, targets = run_test_evaluation(model, test_loader, device)

            if y_true_common is None:
                y_true_common = targets

            # Compute per-sequence metrics
            seq_arrays = _compute_seq_metric_arrays(
                predictions, targets, test_sequence_ids,
                threshold=args.threshold,
            )
            seed_seq_metrics[train_seed] = seq_arrays

            # Point estimates (mean of per-sequence metrics)
            point = {m: float(np.mean(seq_arrays[m])) for m in METRICS}
            seed_point_metrics[train_seed] = point

            print(f"    Per-sequence point estimates ({n_test_sequences} sequences):")
            print(f"      Accuracy: {point['accuracy']:.2f}%")
            print(f"      RMSE:     {point['rmse']:.4f}")
            print(f"      MAE:      {point['mae']:.4f}")
            print(f"      R²:       {point['r2']:.4f}")

            # Block bootstrap CIs
            print(f"    Block bootstrap CIs ({args.n_bootstrap} samples)...")
            ci = {}
            for metric in METRICS:
                ci[metric] = bootstrap_ci_sequences(
                    seq_arrays[metric],
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
            seed_bootstrap_ci[train_seed] = ci

            print(f"    95% CI (sequence-level):")
            print(f"      Accuracy: {ci['accuracy']['mean']:.2f} "
                  f"\u00b1 {ci['accuracy']['std']:.2f}")
            print(f"      RMSE:     {ci['rmse']['mean']:.4f} "
                  f"\u00b1 {ci['rmse']['std']:.4f}")
            print(f"      MAE:      {ci['mae']['mean']:.4f} "
                  f"\u00b1 {ci['mae']['std']:.4f}")
            print(f"      R²:       {ci['r2']['mean']:.4f} "
                  f"\u00b1 {ci['r2']['std']:.4f}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Multi-seed aggregation
        if len(seeds) > 1:
            print(f"\n  Aggregating across {len(seeds)} seeds "
                  f"(law of total variance)...")
            aggregated = multi_seed_sequence_analysis(
                seed_bootstrap_ci, seed_point_metrics
            )
            for metric in METRICS:
                m = aggregated[metric]
                fmt = ".2f" if metric == "accuracy" else ".4f"
                suffix = "%" if metric == "accuracy" else ""
                print(f"    {metric.upper()}: {m['mean']:{fmt}}{suffix} "
                      f"\u00b1 {m['std']:{fmt}}"
                      f"  (bootstrap: {m['std_bootstrap']:{fmt}}, "
                      f"seed: {m['std_seed']:{fmt}})")
        else:
            single_seed = seeds[0]
            ci = seed_bootstrap_ci[single_seed]
            aggregated = {}
            for metric in METRICS:
                aggregated[metric] = {
                    "mean": ci[metric]["mean"],
                    "std": ci[metric]["std"],
                    "std_bootstrap": ci[metric]["std"],
                    "std_seed": 0.0,
                    "ci_lower": ci[metric]["ci_lower"],
                    "ci_upper": ci[metric]["ci_upper"],
                    "per_seed_values": [seed_point_metrics[single_seed][metric]],
                }

        # Store averaged per-sequence metrics for comparisons
        if len(seeds) > 1:
            avg_seq = {}
            for metric in METRICS:
                avg_seq[metric] = np.mean(
                    [seed_seq_metrics[s][metric] for s in seeds], axis=0
                )
        else:
            avg_seq = seed_seq_metrics[seeds[0]]
        all_seq_metrics[model_id] = avg_seq

        # Build result entry
        result = {
            "model_id": model_id,
            "model_name": model_name,
            "config_file": str(
                get_config_path(model_cfg, variant).relative_to(PROJECT_ROOT)
            ),
            "num_seeds": len(seeds),
            "seeds": seeds,
            "n_test_sequences": n_test_sequences,
        }
        for metric in METRICS:
            result[metric] = aggregated[metric]

        # Per-seed details
        result["per_seed"] = {}
        for train_seed in seeds:
            ckpt_path, val_loss = seed_checkpoints[train_seed]
            result["per_seed"][str(train_seed)] = {
                "checkpoint": str(ckpt_path),
                "val_loss": float(val_loss),
                "point_metrics": seed_point_metrics[train_seed],
                "bootstrap_ci": seed_bootstrap_ci[train_seed],
            }

        all_results.append(result)

    if not all_results:
        print("\nNo models evaluated. Check that checkpoints exist.")
        return

    # =========================================================================
    # Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Results Summary (Sequence-Level)")
    print("=" * 70)

    md_table = generate_markdown_table(all_results)
    print("\nMarkdown Table:")
    print(md_table)

    latex_caption = (
        f"Sequence-Level Bootstrap 95\\% Confidence Intervals "
        f"({args.n_bootstrap} samples, {n_test_sequences} test sequences)"
    )
    if args.dropout:
        latex_caption += " -- Models with Dropout"
    multi_seed_models = [r for r in all_results if r.get("num_seeds", 1) > 1]
    if multi_seed_models:
        n = multi_seed_models[0]["num_seeds"]
        latex_caption += f", {n} seeds"
    latex_table = generate_latex_table(all_results, caption=latex_caption)
    print("\nLaTeX Table:")
    print(latex_table)

    # =========================================================================
    # Permutation Tests
    # =========================================================================
    comparison_results = []
    if not args.no_comparisons:
        pairs = COMPARISON_PAIRS
        available = set(all_seq_metrics.keys())
        valid_pairs = [
            (a, b, c) for a, b, c in pairs
            if a in available and b in available
        ]
        skipped = [
            (a, b, c) for a, b, c in pairs
            if a not in available or b not in available
        ]

        if valid_pairs:
            print("\n" + "=" * 70)
            print(f"Sequence-Level Permutation Tests "
                  f"({len(valid_pairs)} comparisons, "
                  f"{args.n_permutations} permutations)")
            print("=" * 70)

            comparison_results = run_all_comparisons(
                valid_pairs, all_seq_metrics,
                n_permutations=args.n_permutations, seed=args.seed,
            )

            for c in comparison_results:
                print(f"\n  {c['model_a']} vs {c['model_b']}  [{c['category']}]")
                for metric in METRICS:
                    m = c[metric]
                    stars = _significance_stars(m["p_value"])
                    fmt = ".2f" if metric == "accuracy" else ".4f"
                    print(f"    {metric.upper():>8}: "
                          f"\u0394={m['observed_diff']:{fmt}}  "
                          f"p={m['p_value']:.4f} {stars}  "
                          f"d={m['cohens_d']:+.3f} ({m['effect_size']})")

            print("\n" + "-" * 70)
            print("Comparison Table (Sequence-Level):")
            comp_md = generate_comparison_markdown(comparison_results)
            print(comp_md)

            print("\nComparison LaTeX Table:")
            comp_latex = generate_comparison_latex(
                comparison_results,
                caption=(
                    f"Sequence-Level Pairwise Comparisons "
                    f"({args.n_permutations} permutations, "
                    f"{n_test_sequences} test sequences)"
                ),
            )
            print(comp_latex)

        if skipped:
            print(f"\n  Skipped (missing predictions): "
                  f"{', '.join(f'{a} vs {b}' for a, b, _ in skipped)}")

    # =========================================================================
    # Save Results
    # =========================================================================
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"sequence_level_results_{variant}.json"
    comparison_for_json = []
    for c in comparison_results:
        c_clean = {
            "model_a": c["model_a"],
            "model_b": c["model_b"],
            "category": c["category"],
            "n_sequences": c["n_sequences"],
        }
        for metric in METRICS:
            c_clean[metric] = c[metric]
        comparison_for_json.append(c_clean)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "analysis_level": "sequence",
            "model_set": variant,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "n_test_sequences": n_test_sequences,
            "accuracy_threshold": args.threshold,
            "seed": args.seed,
            "results": all_results,
            "comparisons": comparison_for_json,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Markdown
    md_path = output_dir / f"sequence_level_table_{variant}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Sequence-Level Bootstrap CIs ({variant})\n\n")
        f.write(f"Bootstrap samples: {args.n_bootstrap}  \n")
        f.write(f"Test sequences: {n_test_sequences}  \n")
        f.write(f"Accuracy threshold: {args.threshold}\n\n")
        multi_seed = any(r.get("num_seeds", 1) > 1 for r in all_results)
        if multi_seed:
            f.write(
                "Uncertainty combines bootstrap sampling variance and "
                "between-seed variance (law of total variance).\n\n"
            )
        f.write(md_table)
        if comparison_results:
            f.write(
                f"\n\n## Sequence-Level Permutation Tests "
                f"({args.n_permutations} permutations)\n\n"
            )
            f.write(generate_comparison_markdown(comparison_results))
    print(f"Markdown table saved to: {md_path}")

    # LaTeX
    latex_path = output_dir / f"sequence_level_table_{variant}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
        if comparison_results:
            f.write("\n\n")
            f.write(generate_comparison_latex(
                comparison_results,
                caption=(
                    f"Sequence-Level Permutation Test Results "
                    f"({args.n_permutations} permutations)"
                ),
            ))
    print(f"LaTeX table saved to: {latex_path}")

    print("\n" + "=" * 70)
    print("Sequence-level evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
