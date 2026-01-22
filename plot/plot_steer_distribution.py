"""
Plot steering torque value distribution.

Usage:
    python plot/plot_steer_distribution.py                    # Default: full range [-1, 1]
    python plot/plot_steer_distribution.py --range fine       # Fine range [-0.25, 0.25]
    python plot/plot_steer_distribution.py --vehicle TOYOTA_HIGHLANDER_2020 --num-csvs 20399
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_raw_data_path, PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steering torque distribution")
    parser.add_argument(
        "--range", "-r",
        choices=["full", "fine"],
        default="full",
        help="Bin range: 'full' for [-1, 1] with 0.25 steps, 'fine' for [-0.25, 0.25] with 0.05 steps"
    )
    parser.add_argument(
        "--vehicle", "-v",
        default="HYUNDAI_SONATA_2020",
        help="Vehicle name (default: HYUNDAI_SONATA_2020)"
    )
    parser.add_argument(
        "--num-csvs", "-n",
        type=int,
        default=5001,
        help="Number of CSVs in dataset (default: 5001)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: plot/output/steer_distribution_{range}.png)"
    )
    return parser.parse_args()


def plot_steer_distribution(data, bins, output_path, title_suffix=""):
    """Plot steering torque distribution histogram."""
    # Calculate frequency for each bin
    hist, bin_edges = np.histogram(data['steer'], bins=bins)

    # Calculate probabilities for each bin
    probabilities = hist / len(data)

    # Print intervals and probabilities
    print(f"Steer Value Probability Distribution{title_suffix}:")
    for i in range(len(bins) - 1):
        print(f"  [{bin_edges[i]:+.2f}, {bin_edges[i+1]:+.2f}): {probabilities[i]:.4f}")

    # Calculate bar width from bin edges
    bar_width = bins[1] - bins[0]

    # Plot the probability distribution as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        bin_edges[:-1],
        probabilities,
        width=bar_width * 0.9,
        align='edge',
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )

    # Add labels and title
    plt.xlabel('Steer Value', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Probability Distribution of Steering Torque{title_suffix}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(bin_edges, rotation=45)

    # Save and show
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    plt.show()


def main():
    args = parse_args()

    # Load data
    input_path = get_raw_data_path(args.vehicle, args.num_csvs)
    print(f"Loading data from {input_path}...")
    data = pd.read_pickle(input_path)
    print(f"Loaded {len(data):,} samples")

    # Define bins based on range selection
    if args.range == "full":
        bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        title_suffix = " [-1, 1]"
    else:  # fine
        bins = [-0.25 + i * 0.05 for i in range(11)]
        title_suffix = " [-0.25, 0.25]"

    # Determine output path
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        output_path = PROJECT_ROOT / "plot" / "output" / f"steer_distribution_{args.range}.png"

    # Plot
    plot_steer_distribution(data, bins, output_path, title_suffix)


if __name__ == "__main__":
    main()
