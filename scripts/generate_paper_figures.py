#!/usr/bin/env python3
"""
Generate publication-ready figures for the paper.

Uses PGF export for perfect LaTeX font integration. The figures inherit
all font settings from the LaTeX document they are included in.

Outputs:
- figures/fig_inference_tradeoff.pgf  (LaTeX-native, inherits document fonts)
- figures/fig_attention_weights.pgf
- figures/fig_dropout_effect.pgf
- figures/*.png                        (Preview images)

Usage:
    python scripts/generate_paper_figures.py

In LaTeX:
    \\begin{figure}
        \\centering
        \\input{figures/fig_inference_tradeoff.pgf}
        \\caption{...}
    \\end{figure}
"""

import matplotlib
matplotlib.use('pgf')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configure matplotlib for PGF export with document font inheritance
# Font sizes are set explicitly to match typical IEEE paper body text (~9pt)
# when the figure is scaled to \columnwidth
plt.rcParams.update({
    # LaTeX integration
    'text.usetex': True,
    'pgf.rcfonts': False,  # Don't embed fonts, inherit from document
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': '\n'.join([
        r'\usepackage{siunitx}',
        r'\usepackage{amsmath}',
    ]),
    # Figure sizing (in inches)
    # IEEE single column: 3.5in, double column: 7.16in
    'figure.figsize': (3.5, 2.5),
    # Font sizes (will inherit document font family, but need explicit sizes)
    # These sizes work well for IEEE single-column figures
    'font.size': 8,           # Base font size
    'axes.labelsize': 8,      # Axis labels
    'axes.titlesize': 8,      # Axis title
    'legend.fontsize': 7,     # Legend
    'xtick.labelsize': 7,     # Tick labels
    'ytick.labelsize': 7,
    # Minimal padding (LaTeX handles margins)
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    # Line and marker defaults
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    # Axis styling
    'axes.linewidth': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    # Tick styling
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Define colors (matching paper's matlabXXX colors)
COLORS = {
    'blue': '#0072BD',
    'orange': '#D95319',
    'yellow': '#EDB120',
    'purple': '#7E2F8E',
    'green': '#77AC30',
}

# Model data from experiments
MODEL_DATA = {
    'M1': {'params': 84801, 'accuracy': 82.57, 'inference_ms': 1.11, 'category': 'small'},
    'M2': {'params': 84866, 'accuracy': 81.50, 'inference_ms': 1.16, 'category': 'small'},
    'M3': {'params': 597633, 'accuracy': 87.84, 'inference_ms': 2.40, 'category': 'medium_baseline'},
    'M4': {'params': 597762, 'accuracy': 90.25, 'inference_ms': 2.44, 'category': 'medium_attention'},
    'M5': {'params': 630529, 'accuracy': 88.34, 'inference_ms': 2.88, 'category': 'medium_attention'},
    'M6': {'params': 597633, 'accuracy': 88.17, 'inference_ms': 2.46, 'category': 'medium_attention'},
}


def load_attention_weights(base_path: Path) -> dict:
    """Load attention weights from CSV files."""
    weights = {}

    # M4 Simple Attention
    m4_path = base_path / 'M4_Medium_Simple_Attention' / 'attention_all_epochs.csv'
    if m4_path.exists():
        df = pd.read_csv(m4_path)
        # Get last epoch column
        last_epoch_col = [c for c in df.columns if c.startswith('epoch_')][-1]
        weights['M4'] = df[last_epoch_col].values

    # M5 Additive Attention
    m5_path = base_path / 'M5_Medium_Additive_Attention' / 'attention_all_epochs.csv'
    if m5_path.exists():
        df = pd.read_csv(m5_path)
        last_epoch_col = [c for c in df.columns if c.startswith('epoch_')][-1]
        weights['M5'] = df[last_epoch_col].values

    # M6 Scaled DP Attention
    m6_path = base_path / 'M6_Medium_Scaled_DP_Attention' / 'attention_all_epochs.csv'
    if m6_path.exists():
        df = pd.read_csv(m6_path)
        last_epoch_col = [c for c in df.columns if c.startswith('epoch_')][-1]
        weights['M6'] = df[last_epoch_col].values

    return weights


def save_figure(fig, output_dir: Path, name: str):
    """Save figure as PGF (for LaTeX) and PNG (for preview)."""
    # PGF for LaTeX inclusion (inherits document fonts)
    fig.savefig(output_dir / f'{name}.pgf')
    # PNG for quick preview
    fig.savefig(output_dir / f'{name}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}.pgf + {name}.png")


def create_inference_tradeoff_figure(output_dir: Path):
    """Create Figure: Accuracy vs Inference Time scatter plot."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Small models (blue circles)
    small_x = [MODEL_DATA['M1']['inference_ms'], MODEL_DATA['M2']['inference_ms']]
    small_y = [MODEL_DATA['M1']['accuracy'], MODEL_DATA['M2']['accuracy']]
    ax.scatter(small_x, small_y, c=COLORS['blue'], marker='o', s=40,
               label='Small (85K)', zorder=3, edgecolors='white', linewidths=0.5)

    # Medium baseline (green square)
    ax.scatter([MODEL_DATA['M3']['inference_ms']], [MODEL_DATA['M3']['accuracy']],
               c=COLORS['green'], marker='s', s=45,
               label='Medium Baseline (598K)', zorder=3, edgecolors='white', linewidths=0.5)

    # Medium + Attention (orange triangles)
    attn_x = [MODEL_DATA['M4']['inference_ms'], MODEL_DATA['M5']['inference_ms'], MODEL_DATA['M6']['inference_ms']]
    attn_y = [MODEL_DATA['M4']['accuracy'], MODEL_DATA['M5']['accuracy'], MODEL_DATA['M6']['accuracy']]
    ax.scatter(attn_x, attn_y, c=COLORS['orange'], marker='^', s=50,
               label='Medium + Attention', zorder=3, edgecolors='white', linewidths=0.5)

    # Add model labels with offsets in POINTS (not data coordinates)
    # This gives consistent visual spacing regardless of axis scales
    label_config = {
        'M1': {'offset': (5, 3), 'ha': 'left'},      # Right-above
        'M2': {'offset': (5, -8), 'ha': 'left'},     # Right-below
        'M3': {'offset': (-5, 3), 'ha': 'right'},    # Left-above
        'M4': {'offset': (5, 5), 'ha': 'left'},      # Right-above
        'M5': {'offset': (5, 3), 'ha': 'left'},      # Right-above
        'M6': {'offset': (5, -8), 'ha': 'left'},     # Right-below
    }
    for name, data in MODEL_DATA.items():
        cfg = label_config[name]
        weight = 'bold' if name == 'M4' else 'normal'
        ax.annotate(name, (data['inference_ms'], data['accuracy']),
                   xytext=cfg['offset'],
                   textcoords='offset points',  # Offset in points, not data coords
                   fontweight=weight, ha=cfg['ha'])

    # Formatting
    ax.set_xlabel('Inference Time P95 (ms)')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(80, 92)
    ax.legend(loc='lower right', framealpha=0.9)

    save_figure(fig, output_dir, 'fig_inference_tradeoff')


def create_attention_weights_figure(output_dir: Path, weights: dict):
    """Create Figure: Attention Weight Distributions."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Subsample for cleaner visualization
    indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    time_subset = [-50 + i for i in indices]

    # M4 Simple Attention
    if 'M4' in weights:
        m4_subset = [weights['M4'][i] for i in indices]
        ax.plot(time_subset, m4_subset, color=COLORS['orange'],
                marker='^', markersize=3, label='M4 Simple')

    # M5 Additive Attention
    if 'M5' in weights:
        m5_subset = [weights['M5'][i] for i in indices]
        ax.plot(time_subset, m5_subset, color=COLORS['green'],
                marker='s', markersize=3, label='M5 Additive')

    # M6 Scaled DP Attention
    if 'M6' in weights:
        m6_subset = [weights['M6'][i] for i in indices]
        ax.plot(time_subset, m6_subset, color=COLORS['blue'],
                linestyle='--', marker='o', markersize=2, label='M6 Scaled DP')

    # Formatting (LaTeX math mode for tick labels)
    ax.set_xlabel('Time Step (relative to prediction)')
    ax.set_ylabel('Attention Weight')
    ax.set_xlim(-52, 2)
    ax.set_ylim(0, 0.30)
    ax.set_xticks([-50, -40, -30, -20, -10, -1])
    ax.set_xticklabels([r'$t_{-50}$', r'$t_{-40}$', r'$t_{-30}$', r'$t_{-20}$', r'$t_{-10}$', r'$t_{-1}$'])
    ax.legend(loc='upper left', framealpha=0.9)

    save_figure(fig, output_dir, 'fig_attention_weights')


def create_dropout_comparison_figure(output_dir: Path):
    """Create Figure: Dropout Effect Bar Chart."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    no_dropout = [82.57, 81.50, 87.84, 90.25, 88.34, 88.17]
    with_dropout = [80.49, 80.07, 86.29, 84.31, 85.39, 85.23]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, no_dropout, width, label='No Dropout', color=COLORS['blue'])
    ax.bar(x + width/2, with_dropout, width, label=r'Dropout${}=0.2$', color=COLORS['orange'], alpha=0.7)

    # Add delta labels
    for i, (nd, wd) in enumerate(zip(no_dropout, with_dropout)):
        delta = wd - nd
        ax.annotate(rf'{delta:.1f}\%', xy=(x[i] + width/2, wd + 0.3),
                   ha='center', va='bottom', fontsize=6, color='red')

    ax.set_xlabel('Model')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(78, 92)
    ax.legend(loc='upper right')
    ax.grid(axis='y', zorder=0)

    save_figure(fig, output_dir, 'fig_dropout_effect')


def main():
    """Generate all paper figures."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'docs' / 'paper' / 'figures'
    attention_weights_dir = project_root / 'attention_weights'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Paper Figures (PGF for LaTeX font inheritance)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Load attention weights
    print("Loading attention weights...")
    weights = load_attention_weights(attention_weights_dir)
    if weights:
        print(f"  Loaded: {list(weights.keys())}")
    print()

    # Generate figures
    print("Creating figures...")

    # Figure: Accuracy vs Inference Time
    create_inference_tradeoff_figure(output_dir)

    # Figure: Attention Weights
    if weights:
        create_attention_weights_figure(output_dir, weights)
    else:
        print("  WARNING: No attention weights found, skipping attention figure")

    # Figure: Dropout Effect
    create_dropout_comparison_figure(output_dir)

    print()
    print("=" * 60)
    print("Done! Include in LaTeX with:")
    print()
    print(r"  \begin{figure}")
    print(r"      \centering")
    print(r"      \input{figures/fig_inference_tradeoff.pgf}")
    print(r"      \caption{Accuracy vs. Inference Time}")
    print(r"  \end{figure}")
    print()
    print("Or scaled to column width:")
    print()
    print(r"  \resizebox{\columnwidth}{!}{\input{figures/fig_inference_tradeoff.pgf}}")
    print("=" * 60)


if __name__ == '__main__':
    main()
