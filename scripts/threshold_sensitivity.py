"""
Threshold sensitivity analysis for model accuracy rankings.

Analyzes how model accuracy rankings change across different error thresholds
to verify ranking stability.

Usage:
    python scripts/threshold_sensitivity.py
    python scripts/threshold_sensitivity.py --thresholds 0.03 0.05 0.07 0.10
    python scripts/threshold_sensitivity.py --models m1 m4 m6
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from config.loader import load_config, get_model_class
from config.settings import get_preprocessed_paths
from model.data_module import TimeSeriesDataModule
from scripts.shared import MODELS, MODEL_BY_ID, find_best_checkpoint, get_config_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis for model rankings"
    )
    parser.add_argument(
        "--thresholds", "-t",
        type=float,
        nargs="+",
        default=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
        help="List of thresholds to evaluate"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=None,
        help="Model IDs to evaluate (default: all models)"
    )
    parser.add_argument(
        "--variant", "-v",
        type=str,
        default="no_dropout",
        choices=["no_dropout", "dropout"],
        help="Model variant to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """Calculate accuracy for a given threshold."""
    return float(np.mean(np.abs(y_true - y_pred) <= threshold) * 100)


def get_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on dataloader and collect predictions."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            X, Y = batch
            X = X.to(device)
            outputs = model(X)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(Y.numpy())

    return np.concatenate(all_preds).flatten(), np.concatenate(all_targets).flatten()


def main():
    args = parse_args()

    # Determine which models to evaluate
    if args.models:
        model_ids = [m.lower() for m in args.models]
        models_to_eval = [MODEL_BY_ID[m] for m in model_ids if m in MODEL_BY_ID]
    else:
        models_to_eval = MODELS

    if not models_to_eval:
        print("Error: No valid models specified")
        sys.exit(1)

    print("=" * 70)
    print("Threshold Sensitivity Analysis")
    print("=" * 70)
    print(f"Models: {[m.id.upper() for m in models_to_eval]}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Variant: {args.variant}")
    print("-" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load predictions for all models
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    # Load data module once (all models use same data)
    first_model = models_to_eval[0]
    config_path = get_config_path(first_model, args.variant)
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
        batch_size=args.batch_size
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()

    print("\nLoading models and generating predictions...")

    for model_cfg in models_to_eval:
        model_id = model_cfg.id.upper()
        print(f"  {model_id}...", end=" ", flush=True)

        # Find checkpoint
        ckpt_path = find_best_checkpoint(model_cfg, args.variant)
        if ckpt_path is None:
            print("SKIP (no checkpoint)")
            continue

        # Load config and model
        config_path = get_config_path(model_cfg, args.variant)
        config = load_config(str(config_path))
        model_class = get_model_class(config["model"]["type"])

        try:
            model = model_class.load_from_checkpoint(str(ckpt_path))
            y_pred, y_true = get_predictions(model, test_loader, device)
            predictions[model_id] = {"y_true": y_true, "y_pred": y_pred}
            print("OK")
        except Exception as e:
            print(f"ERROR ({e})")
            continue

    if not predictions:
        print("\nError: No models could be loaded")
        sys.exit(1)

    # Calculate accuracy for each threshold
    print("\n" + "=" * 70)
    print("Accuracy by Threshold")
    print("=" * 70)

    model_names = list(predictions.keys())

    # Build header
    header = f"{'Threshold':>10}"
    for name in model_names:
        header += f" {name:>8}"
    print(header)
    print("-" * len(header))

    results: List[Dict] = []

    for thresh in args.thresholds:
        row = {"threshold": thresh}
        line = f"{thresh:>10.2f}"

        for name in model_names:
            y_true = predictions[name]["y_true"]
            y_pred = predictions[name]["y_pred"]
            acc = calculate_accuracy(y_true, y_pred, thresh)
            row[name] = acc
            line += f" {acc:>7.2f}%"

        results.append(row)
        print(line)

    # Rankings
    print("\n" + "=" * 70)
    print("Rankings by Threshold (best to worst)")
    print("=" * 70)

    for row in results:
        thresh = row["threshold"]
        model_accs = {m: row[m] for m in model_names}
        ranking = sorted(model_accs.keys(), key=lambda x: model_accs[x], reverse=True)

        # Format ranking with accuracies
        ranking_str = " > ".join([f"{m}({model_accs[m]:.1f}%)" for m in ranking[:4]])
        if len(ranking) > 4:
            ranking_str += " > ..."

        print(f"e={thresh:.2f}: {ranking_str}")

    # Check ranking stability
    print("\n" + "=" * 70)
    print("Ranking Stability Analysis")
    print("=" * 70)

    # Get rankings for each threshold
    all_rankings = []
    for row in results:
        model_accs = {m: row[m] for m in model_names}
        ranking = sorted(model_accs.keys(), key=lambda x: model_accs[x], reverse=True)
        all_rankings.append(ranking)

    # Check if top-N is consistent
    for top_n in [1, 2, 3]:
        tops = [tuple(r[:top_n]) for r in all_rankings]
        unique_tops = set(tops)

        if len(unique_tops) == 1:
            print(f"  Top-{top_n}: STABLE across all thresholds {list(unique_tops)[0]}")
        else:
            print(f"  Top-{top_n}: VARIES - {len(unique_tops)} different combinations")
            for ut in unique_tops:
                count = tops.count(ut)
                print(f"           {ut}: {count}/{len(tops)} thresholds")

    # Rank correlation between first and last threshold
    first_ranking = all_rankings[0]
    last_ranking = all_rankings[-1]

    # Simple rank comparison
    rank_changes = []
    for m in model_names:
        r1 = first_ranking.index(m)
        r2 = last_ranking.index(m)
        if r1 != r2:
            rank_changes.append((m, r1 + 1, r2 + 1))

    if rank_changes:
        print(f"\n  Rank changes (e={args.thresholds[0]} → e={args.thresholds[-1]}):")
        for m, r1, r2 in rank_changes:
            direction = "↑" if r2 < r1 else "↓"
            print(f"    {m}: #{r1} → #{r2} {direction}")
    else:
        print(f"\n  No rank changes between e={args.thresholds[0]} and e={args.thresholds[-1]}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
