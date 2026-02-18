#!/usr/bin/env python3
"""Generate paper revision figures (PGF format).

Creates three figures for the paper revision:
  - Figure 2: Attention Weights (M6, M7, M8) - 3 subplots
  - Figure 3: Inference-Accuracy Tradeoff - scatter plot
  - Figure 4: Prediction Timeseries - 3 subplots

Uses the same styling conventions as generate_figures.py.

Usage:
    python scripts/generate_paper_revision_figures.py
    python scripts/generate_paper_revision_figures.py --figure attention
    python scripts/generate_paper_revision_figures.py --figure tradeoff
    python scripts/generate_paper_revision_figures.py --figure timeseries
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# =============================================================================
# STYLE SETUP (consistent with generate_figures.py)
# =============================================================================

# Paper colors from task spec
MATLAB_BLUE = (0/255, 114/255, 189/255)
MATLAB_ORANGE = (217/255, 83/255, 25/255)
MATLAB_PURPLE = (126/255, 47/255, 142/255)

COLORS = {
    'mlp': '#7f7f7f',
    'lstm': '#1f77b4',
    'lstm_attn': '#ff7f0e',
    'ground_truth': '#000000',
    'M3': MATLAB_BLUE,
    'M5': MATLAB_ORANGE,
    'M6': MATLAB_PURPLE,
    'M6_attn': MATLAB_BLUE,
    'M7_attn': MATLAB_ORANGE,
    'M8_attn': MATLAB_PURPLE,
}

MARKERS = {
    'mlp': 's',
    'lstm': 'o',
    'lstm_attn': '^',
}


def setup_style():
    """Configure matplotlib with SciencePlots IEEE style."""
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        'axes.grid': False,
        'text.usetex': True,
        'pgf.texsystem': 'pdflatex',
        'pgf.rcfonts': False,
        'pgf.preamble': '\n'.join([
            r'\usepackage[utf8]{inputenc}',
            r'\usepackage[T1]{fontenc}',
            r'\usepackage{amsmath}',
            r'\usepackage{siunitx}',
            r'\providecommand{\mathdefault}[1]{#1}',
        ]),
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'figure.figsize': (3.5, 2.5),
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'none',
        'savefig.dpi': 300,
    })


setup_style()


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> List[Path]:
    """Save figure as PGF, PDF, and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in ['pgf', 'pdf', 'png']:
        path = output_dir / f'{name}.{fmt}'
        fig.savefig(path)
        saved.append(path)
    plt.close(fig)
    return saved


# =============================================================================
# FIGURE 2: ATTENTION WEIGHTS
# =============================================================================

def fig_attention_weights(output_dir: Path) -> List[Path]:
    """Create attention weights plot with 3 subplots (M6, M7, M8).

    Each subplot shows the averaged attention weight distribution over
    50 timesteps. A horizontal dashed line at 1/50 = 0.02 marks the
    uniform reference.
    """
    figures_dir = PROJECT_ROOT / "figures"

    models = [
        ("M6", "Simple Attention", COLORS['M6_attn']),
        ("M7", "Additive Attention", COLORS['M7_attn']),
        ("M8", "Scaled Dot-Product", COLORS['M8_attn']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=False)

    for ax, (model_id, label, color) in zip(axes, models):
        csv_path = figures_dir / f"attention_weights_{model_id}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        # Load CSV
        timesteps = []
        weights = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timesteps.append(int(row["timestep"]))
                weights.append(float(row["weight"]))

        timesteps = np.array(timesteps)
        weights = np.array(weights)

        # Plot (explicit solid linestyle to override scienceplots cycling)
        ax.plot(timesteps, weights, color=color, linewidth=1.2, linestyle='-')
        ax.axhline(y=1.0/50, color='gray', linestyle='--', linewidth=0.7,
                    label=r'Uniform ($\frac{1}{50}$)')
        ax.set_xlabel('Time Step')
        ax.set_title(label, fontsize=8)
        ax.set_xlim(0, 49)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel('Attention Weight')

    # Add legend only to first subplot
    axes[0].legend(fontsize=6, loc='upper left')

    fig.tight_layout()
    return save_figure(fig, output_dir, 'attention_weights_plot')


# =============================================================================
# FIGURE 3: INFERENCE-ACCURACY TRADEOFF
# =============================================================================

def fig_inference_accuracy_tradeoff(output_dir: Path) -> List[Path]:
    """Create inference time vs accuracy scatter plot.

    Data is hardcoded from the task specification (no pipeline access needed).
    M3 is highlighted as Pareto-optimal.
    """
    # Data from task document
    model_data = {
        'M1': {'accuracy': 68.98, 'inference_p95_ms': 0.07, 'type': 'mlp',
                'label': 'M1'},
        'M2': {'accuracy': 73.44, 'inference_p95_ms': 0.06, 'type': 'mlp',
                'label': 'M2'},
        'M3': {'accuracy': 79.46, 'inference_p95_ms': 0.79, 'type': 'lstm',
                'label': 'M3'},
        'M4': {'accuracy': 79.41, 'inference_p95_ms': 0.79, 'type': 'lstm_attn',
                'label': 'M4'},
        'M5': {'accuracy': 79.63, 'inference_p95_ms': 2.62, 'type': 'lstm',
                'label': 'M5'},
        'M6': {'accuracy': 79.60, 'inference_p95_ms': 2.68, 'type': 'lstm_attn',
                'label': 'M6'},
        'M7': {'accuracy': 79.73, 'inference_p95_ms': 3.83, 'type': 'lstm_attn',
                'label': 'M7'},
        'M8': {'accuracy': 79.26, 'inference_p95_ms': 3.59, 'type': 'lstm_attn',
                'label': 'M8'},
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    marker_sizes = {'mlp': 35, 'lstm': 45, 'lstm_attn': 50}

    plotted_types = set()
    for model_id, data in model_data.items():
        mtype = data['type']

        label = {
            'mlp': 'MLP Baseline',
            'lstm': 'LSTM Baseline',
            'lstm_attn': 'LSTM + Attention'
        }.get(mtype) if mtype not in plotted_types else None
        plotted_types.add(mtype)

        ax.scatter(
            data['inference_p95_ms'], data['accuracy'],
            marker=MARKERS[mtype], s=marker_sizes[mtype],
            c=COLORS[mtype], edgecolors='white', linewidths=0.5,
            label=label, zorder=4
        )

    # Highlight M3 as Pareto-optimal
    m3 = model_data['M3']
    ax.scatter(
        m3['inference_p95_ms'], m3['accuracy'],
        marker='o', s=120, facecolors='none',
        edgecolors='#d62728', linewidths=1.5, zorder=5
    )

    # Model labels (carefully placed to avoid overlap in dense LSTM cluster)
    label_offsets = {
        'M1': (5, -2),    'M2': (5, -2),
        'M3': (5, 3),     'M4': (5, -5),
        'M5': (-8, 3),    'M6': (-5, -5),
        'M7': (5, 3),     'M8': (5, -5),
    }

    for model_id, data in model_data.items():
        x_off, y_off = label_offsets[model_id]
        ha = 'left' if x_off >= 0 else 'right'
        va = 'bottom' if y_off > 0 else 'top'

        text = model_id
        if model_id == 'M3':
            text = r'\textbf{M3}'

        ax.annotate(
            text,
            (data['inference_p95_ms'], data['accuracy']),
            xytext=(x_off, y_off),
            textcoords='offset points',
            fontsize=7, ha=ha, va=va
        )

    ax.set_xscale('log')
    ax.set_xlim(0.04, 7)
    ax.set_ylim(68, 81)
    ax.set_xticks([0.05, 0.1, 0.5, 1, 2, 5])
    ax.set_xticklabels(['0.05', '0.1', '0.5', '1', '2', '5'])

    ax.set_xlabel(r'Inference Time P95 (ms)')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.legend(loc='lower right', fontsize=7)

    return save_figure(fig, output_dir, 'inference_accuracy_tradeoff')


# =============================================================================
# FIGURE 4: PREDICTION TIMESERIES
# =============================================================================

def fig_prediction_timeseries(output_dir: Path) -> List[Path]:
    """Create prediction timeseries plot with 3 subplots.

    Shows ground truth vs predictions for M3, M5, M6 on
    good/median/difficult sequences.
    """
    figures_dir = PROJECT_ROOT / "figures"

    categories = [
        ("good", "Good Prediction"),
        ("median", "Median Prediction"),
        ("difficult", "Difficult Prediction"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5), sharex=True)

    for ax, (category, title_prefix) in zip(axes, categories):
        csv_path = figures_dir / f"prediction_seq_{category}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        # Load CSV
        data = {'timestep': [], 'ground_truth': [],
                'M3_pred': [], 'M5_pred': [], 'M6_pred': []}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data['timestep'].append(int(row['timestep']))
                data['ground_truth'].append(float(row['ground_truth']))
                data['M3_pred'].append(float(row['M3_pred']))
                data['M5_pred'].append(float(row['M5_pred']))
                data['M6_pred'].append(float(row['M6_pred']))

        t = np.array(data['timestep'])
        gt = np.array(data['ground_truth'])
        m3 = np.array(data['M3_pred'])
        m5 = np.array(data['M5_pred'])
        m6 = np.array(data['M6_pred'])

        # Compute RMSE for title (using M5 as reference = Medium Baseline)
        rmse_m5 = np.sqrt(np.mean((gt - m5) ** 2))

        # Plot
        ax.plot(t, gt, color=COLORS['ground_truth'], linewidth=1.0,
                linestyle='-', label='Ground Truth')
        ax.plot(t, m3, color=COLORS['M3'], linewidth=0.8,
                linestyle='--', label='M3 (Small)')
        ax.plot(t, m5, color=COLORS['M5'], linewidth=0.8,
                linestyle='-.', label='M5 (Medium)')
        ax.plot(t, m6, color=COLORS['M6'], linewidth=0.8,
                linestyle=':', label='M6 (+ Attn)')

        ax.set_title(
            f'{title_prefix} (RMSE$_{{\\mathrm{{M5}}}}={rmse_m5:.3f}$)',
            fontsize=8,
        )
        ax.set_ylabel('Torque (norm.)')

    axes[-1].set_xlabel('Time Step')
    axes[0].legend(fontsize=6, loc='best', ncol=2)
    fig.tight_layout()

    return save_figure(fig, output_dir, 'prediction_timeseries')


# =============================================================================
# MAIN
# =============================================================================

FIGURES = {
    'attention': ('Attention Weights', fig_attention_weights),
    'tradeoff': ('Inference-Accuracy Tradeoff', fig_inference_accuracy_tradeoff),
    'timeseries': ('Prediction Timeseries', fig_prediction_timeseries),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper revision figures"
    )
    parser.add_argument(
        "--figure",
        choices=list(FIGURES.keys()),
        default=None,
        help="Generate specific figure (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory (default: figures/)",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("Paper Revision Figure Generation")
    print("=" * 60)

    if args.figure:
        figures_to_generate = {args.figure: FIGURES[args.figure]}
    else:
        figures_to_generate = FIGURES

    for fig_key, (fig_name, fig_func) in figures_to_generate.items():
        print(f"\n  Generating: {fig_name}...")
        paths = fig_func(output_dir)
        for p in paths:
            print(f"    -> {p.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
