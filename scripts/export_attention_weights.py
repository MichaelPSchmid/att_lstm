#!/usr/bin/env python3
"""Export averaged attention weights as CSV for paper figures.

Loads attention_test.npy (test-set averaged weights) for each attention
model and seed, averages across seeds, and exports as CSV.

Output: figures/attention_weights_M{4,6,7,8}.csv

Usage:
    python scripts/export_attention_weights.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

ATTENTION_MODELS = {
    "M4": "M4_Small_Simple_Attention",
    "M6": "M6_Medium_Simple_Attention",
    "M7": "M7_Medium_Additive_Attention",
    "M8": "M8_Medium_Scaled_DP_Attention",
}
SEEDS = [7, 42, 94, 123, 231]


def main():
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Attention Weight Export")
    print("=" * 60)

    for model_id, model_name in ATTENTION_MODELS.items():
        print(f"\n  {model_id} ({model_name}):")

        seed_weights = []
        for seed in SEEDS:
            npy_path = (
                PROJECT_ROOT
                / "attention_weights"
                / f"{model_name}_seed{seed}"
                / "attention_test.npy"
            )
            if not npy_path.exists():
                print(f"    WARNING: Missing {npy_path.name} for seed {seed}")
                continue

            weights = np.load(npy_path)
            seed_weights.append(weights)
            print(f"    seed {seed}: shape={weights.shape}, sum={weights.sum():.6f}")

        if not seed_weights:
            print(f"    ERROR: No attention weights found for {model_id}")
            continue

        # Average across seeds
        avg_weights = np.mean(seed_weights, axis=0)

        # Normalize to sum=1.0
        avg_weights = avg_weights / avg_weights.sum()

        print(f"    Averaged: shape={avg_weights.shape}, sum={avg_weights.sum():.6f}")

        # Plausibility checks
        last_5_pct = avg_weights[-5:].sum() * 100
        last_10_pct = avg_weights[-10:].sum() * 100
        last_20_pct = avg_weights[-20:].sum() * 100
        peak_pos = np.argmax(avg_weights)
        print(f"    Last 5: {last_5_pct:.1f}%, Last 10: {last_10_pct:.1f}%, "
              f"Last 20: {last_20_pct:.1f}%, Peak: {peak_pos}")

        # Export CSV
        csv_path = output_dir / f"attention_weights_{model_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "weight"])
            for t, w in enumerate(avg_weights):
                writer.writerow([t, f"{w:.8f}"])

        print(f"    Saved: {csv_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
