#!/usr/bin/env python3
"""Export prediction CSVs for representative test sequences.

Selects good/median/difficult sequences based on RMSE percentiles
and exports Ground Truth + predictions from M3, M5, M6 (seed 42).

Output: figures/prediction_seq_{good,median,difficult}.csv

Usage:
    python scripts/export_prediction_csvs.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

MODELS = ["M3_Small_Baseline", "M5_Medium_Baseline", "M6_Medium_Simple_Attention"]
MODEL_SHORT = {"M3_Small_Baseline": "M3", "M5_Medium_Baseline": "M5",
               "M6_Medium_Simple_Attention": "M6"}
SEED = 42


def load_predictions(model_name: str) -> pd.DataFrame:
    """Load predictions CSV for a model."""
    csv_path = (
        PROJECT_ROOT / "results" / "paper" / model_name
        / f"seed_{SEED}" / f"{model_name}_predictions.csv"
    )
    return pd.read_csv(csv_path)


def compute_sequence_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE per sequence."""
    grouped = df.groupby("sequence_id").apply(
        lambda g: np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2)),
        include_groups=False,
    )
    return grouped.reset_index().rename(columns={0: "rmse"})


def select_sequences(seq_rmse: pd.DataFrame) -> dict:
    """Select good, median, and difficult sequences by RMSE percentile."""
    # Percentile targets
    targets = {
        "good": (10, 25),       # P10-P25
        "median": (45, 55),     # around median
        "difficult": (75, 90),  # P75-P90
    }

    selected = {}
    for label, (lo, hi) in targets.items():
        lo_val = np.percentile(seq_rmse["rmse"], lo)
        hi_val = np.percentile(seq_rmse["rmse"], hi)
        candidates = seq_rmse[
            (seq_rmse["rmse"] >= lo_val) & (seq_rmse["rmse"] <= hi_val)
        ]
        if candidates.empty:
            # Fallback: nearest to midpoint
            mid = np.percentile(seq_rmse["rmse"], (lo + hi) / 2)
            idx = (seq_rmse["rmse"] - mid).abs().idxmin()
            candidates = seq_rmse.loc[[idx]]

        # Pick the one closest to the midpoint of the range
        mid_rmse = (lo_val + hi_val) / 2
        best_idx = (candidates["rmse"] - mid_rmse).abs().idxmin()
        selected[label] = int(candidates.loc[best_idx, "sequence_id"])

    return selected


def main():
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Prediction CSV Export")
    print("=" * 60)

    # Load predictions for all three models
    predictions = {}
    for model_name in MODELS:
        df = load_predictions(model_name)
        predictions[model_name] = df
        print(f"  Loaded {model_name}: {len(df)} samples, "
              f"{df['sequence_id'].nunique()} sequences")

    # Use M5 (Medium Baseline) for sequence selection
    ref_model = "M5_Medium_Baseline"
    seq_rmse = compute_sequence_rmse(predictions[ref_model])
    print(f"\n  Sequence RMSE stats (M5):")
    print(f"    Mean:   {seq_rmse['rmse'].mean():.4f}")
    print(f"    Median: {seq_rmse['rmse'].median():.4f}")
    print(f"    P10:    {np.percentile(seq_rmse['rmse'], 10):.4f}")
    print(f"    P90:    {np.percentile(seq_rmse['rmse'], 90):.4f}")

    selected = select_sequences(seq_rmse)
    print(f"\n  Selected sequences:")
    for label, seq_id in selected.items():
        rmse_val = seq_rmse[seq_rmse["sequence_id"] == seq_id]["rmse"].values[0]
        print(f"    {label}: seq_id={seq_id}, RMSE={rmse_val:.4f}")

    # Export CSVs
    for label, seq_id in selected.items():
        rows = []

        # Get samples for this sequence from each model
        ref_df = predictions[ref_model]
        seq_mask = ref_df["sequence_id"] == seq_id
        seq_data = ref_df[seq_mask].sort_values("sample_idx")

        for i, (_, row) in enumerate(seq_data.iterrows()):
            out_row = {
                "timestep": i,
                "ground_truth": f"{row['y_true']:.8f}",
            }
            for model_name in MODELS:
                model_df = predictions[model_name]
                model_seq = model_df[model_df["sequence_id"] == seq_id].sort_values("sample_idx")
                short = MODEL_SHORT[model_name]
                pred_val = model_seq.iloc[i]["y_pred"]
                out_row[f"{short}_pred"] = f"{pred_val:.8f}"
            rows.append(out_row)

        csv_path = output_dir / f"prediction_seq_{label}.csv"
        fieldnames = ["timestep", "ground_truth", "M3_pred", "M5_pred", "M6_pred"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_samples = len(rows)
        gt_range = (float(seq_data["y_true"].min()), float(seq_data["y_true"].max()))
        print(f"    Saved {csv_path.name}: {n_samples} samples, "
              f"GT range [{gt_range[0]:.3f}, {gt_range[1]:.3f}]")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
