#!/usr/bin/env python3
"""Compute sequence-level R² for all models and seeds.

Loads each model checkpoint, runs inference on the test set,
computes per-sequence R², and aggregates across seeds.

Output: r2_values.csv with columns model, r2_mean, r2_std

Usage:
    python scripts/compute_sequence_r2.py
    python scripts/compute_sequence_r2.py --models m3 m5 m7
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_model_class, load_config
from config.settings import get_preprocessed_paths
from model.data_module import TimeSeriesDataModule
from scripts.shared.checkpoints import find_all_seed_checkpoints
from scripts.shared.models import MODEL_BY_ID, MODELS, ModelConfig

logger = logging.getLogger(__name__)

VARIANT = "no_dropout"
SEEDS = [7, 42, 94, 123, 231]


def compute_per_sequence_r2(
    predictions: np.ndarray,
    targets: np.ndarray,
    sequence_ids: np.ndarray,
) -> Tuple[float, List[float]]:
    """Compute R² per sequence, then return mean R² across sequences.

    Args:
        predictions: Model predictions [N,] or [N, 1]
        targets: Ground truth targets [N,] or [N, 1]
        sequence_ids: Per-sample sequence IDs [N,]

    Returns:
        Tuple of (mean_r2, list_of_per_sequence_r2)
    """
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
            # Skip sequences with near-zero variance (constant target)
            continue

        r2 = 1.0 - ss_res / ss_tot
        r2_values.append(r2)

    mean_r2 = float(np.mean(r2_values))
    return mean_r2, r2_values


def run_inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference on test set.

    Args:
        model: Loaded PyTorch model
        test_loader: Test DataLoader
        device: Device string ('cuda' or 'cpu')

    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            X_batch, Y_batch = batch
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(Y_batch.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return predictions, targets


def evaluate_model_r2(
    model_config: ModelConfig,
    test_loader: torch.utils.data.DataLoader,
    test_sequence_ids: np.ndarray,
) -> Dict[str, float]:
    """Compute sequence-level R² for one model across all seeds.

    Args:
        model_config: Model configuration from registry
        test_loader: Shared test DataLoader (loaded once)
        test_sequence_ids: Shared test sequence IDs

    Returns:
        Dict with r2_mean and r2_std
    """
    config_path = PROJECT_ROOT / model_config.config_no_dropout
    config = load_config(str(config_path))
    model_type = config["model"]["type"]
    model_class = get_model_class(model_type)

    # Find checkpoints for all seeds
    seed_checkpoints = find_all_seed_checkpoints(model_config, VARIANT)

    seed_r2_values = []

    for seed in SEEDS:
        if seed not in seed_checkpoints:
            logger.warning(
                "No checkpoint for %s seed %d, skipping",
                model_config.name, seed,
            )
            continue

        checkpoint_path, val_loss = seed_checkpoints[seed]
        print(f"    Seed {seed}: {checkpoint_path.name} (val_loss={val_loss:.6f})")

        model = model_class.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu"
        )
        predictions, targets = run_inference(model, test_loader, "cpu")

        mean_r2, _ = compute_per_sequence_r2(
            predictions, targets, test_sequence_ids
        )
        seed_r2_values.append(mean_r2)
        print(f"      R² = {mean_r2:.6f}")

        # Free memory
        del model
        import gc
        gc.collect()

    if not seed_r2_values:
        return {"r2_mean": float("nan"), "r2_std": float("nan")}

    return {
        "r2_mean": float(np.mean(seed_r2_values)),
        "r2_std": float(np.std(seed_r2_values)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute sequence-level R² for all models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="r2_values.csv",
        help="Output CSV path (default: r2_values.csv)",
    )
    args = parser.parse_args()

    # Select models
    if args.models:
        models = [MODEL_BY_ID[m] for m in args.models if m in MODEL_BY_ID]
    else:
        models = MODELS

    print("=" * 60)
    print("Sequence-Level R² Computation")
    print("=" * 60)
    print(f"Models: {len(models)}")
    print(f"Seeds: {SEEDS}")
    print(f"Variant: {VARIANT}")
    print(f"Device: cpu (forced for memory safety)")
    print()

    # Load data ONCE (shared across all models)
    print("Loading data...")
    first_config_path = PROJECT_ROOT / models[0].config_no_dropout
    first_config = load_config(str(first_config_path))
    data_config = first_config["data"]
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
        batch_size=256,
        split_seed=data_config.get("split_seed", 0),
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    test_sequence_ids = data_module.get_split_sequence_ids("test")
    print(f"Test samples: {len(test_sequence_ids)}, "
          f"sequences: {len(np.unique(test_sequence_ids))}")
    print()

    results = []

    for model_config in models:
        print(f"  {model_config.name}:")
        r2_result = evaluate_model_r2(
            model_config, test_loader, test_sequence_ids
        )
        results.append({
            "model": model_config.id.upper(),
            "model_name": model_config.name,
            "r2_mean": r2_result["r2_mean"],
            "r2_std": r2_result["r2_std"],
        })
        print(
            f"    -> R² = {r2_result['r2_mean']:.4f} "
            f"+/- {r2_result['r2_std']:.4f}"
        )
        print()

    # Save CSV
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "model_name", "r2_mean", "r2_std"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    # Print summary table
    print(f"\n{'Model':<8} {'R² Mean':>10} {'R² Std':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['model']:<8} {r['r2_mean']:>10.4f} {r['r2_std']:>10.4f}")


if __name__ == "__main__":
    main()
