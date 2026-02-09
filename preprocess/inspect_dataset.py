"""
Inspect preprocessed dataset by loading and displaying random samples.

Usage:
    python preprocess/inspect_dataset.py                          # Default: sF suffix
    python preprocess/inspect_dataset.py --suffix sF_NewFeatures  # Different suffix
    python preprocess/inspect_dataset.py --index 42               # Specific sample index
    python preprocess/inspect_dataset.py --stats                  # Show statistics only
"""

import argparse
import pickle
import random
import sys
import os
from pathlib import Path

import numpy as np

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_preprocessed_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect preprocessed dataset")
    parser.add_argument(
        "--vehicle", "-v",
        default="HYUNDAI_SONATA_2020",
        help="Vehicle name (default: HYUNDAI_SONATA_2020)"
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=50,
        help="Window size (default: 50)"
    )
    parser.add_argument(
        "--predict-size", "-p",
        type=int,
        default=1,
        help="Prediction size (default: 1)"
    )
    parser.add_argument(
        "--step-size", "-s",
        type=int,
        default=1,
        help="Step size (default: 1)"
    )
    parser.add_argument(
        "--suffix",
        default="sF",
        help="Dataset suffix (default: sF)"
    )
    parser.add_argument(
        "--variant",
        choices=["paper", "full"],
        default="full",
        help="Dataset variant (default: full)"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=None,
        help="Specific sample index to display (default: random)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics only, no sample display"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=1,
        help="Number of random samples to display (default: 1)"
    )
    return parser.parse_args()


def load_data_file(file_path: Path):
    """Load data from .npy or .pkl file."""
    npy_path = file_path.with_suffix('.npy')

    if npy_path.exists():
        return np.load(npy_path, allow_pickle=True)
    elif file_path.exists():
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No file found at {file_path} or {npy_path}")


def main():
    args = parse_args()

    # Get paths
    paths = get_preprocessed_paths(
        vehicle=args.vehicle,
        window_size=args.window_size,
        predict_size=args.predict_size,
        step_size=args.step_size,
        suffix=args.suffix,
        variant=args.variant
    )

    print("=" * 60)
    print("Dataset Inspection")
    print("=" * 60)
    print(f"Vehicle:      {args.vehicle}")
    print(f"Window size:  {args.window_size}")
    print(f"Predict size: {args.predict_size}")
    print(f"Step size:    {args.step_size}")
    print(f"Suffix:       {args.suffix}")
    print(f"Variant:      {args.variant}")
    print(f"Directory:    {paths['dir']}")
    print("-" * 60)

    # Load data
    print("Loading data...")
    try:
        X = load_data_file(paths["features"])
        Y = load_data_file(paths["targets"])
        sequence_ids = load_data_file(paths["sequence_ids"])
        time_steps = load_data_file(paths["time_steps"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Convert to numpy arrays if needed
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)
    if isinstance(sequence_ids, list):
        sequence_ids = np.array(sequence_ids)
    if isinstance(time_steps, list):
        time_steps = np.array(time_steps)

    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples:     {len(X):,}")
    print(f"  Feature shape:     {X.shape}")
    print(f"  Target shape:      {Y.shape}")
    print(f"  Unique sequences:  {len(np.unique(sequence_ids)):,}")
    print(f"  Feature range:     [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Target range:      [{Y.min():.4f}, {Y.max():.4f}]")

    if args.stats:
        return

    # Display samples
    print("\n" + "=" * 60)

    if args.index is not None:
        indices = [args.index]
    else:
        indices = random.sample(range(len(X)), min(args.num_samples, len(X)))

    for idx in indices:
        print(f"\nSample Index: {idx}")
        print(f"  Sequence ID:    {sequence_ids[idx]}")
        print(f"  Time Step:      {time_steps[idx]}")
        print(f"  Features (X):   shape={X[idx].shape}")
        print(f"  Target (Y):     {Y[idx].flatten()}")

        # Show feature summary (first and last row)
        if len(X[idx].shape) == 2:
            print(f"  First row:      {X[idx][0]}")
            print(f"  Last row:       {X[idx][-1]}")


if __name__ == "__main__":
    main()
