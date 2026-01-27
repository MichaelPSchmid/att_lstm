#!/usr/bin/env python3
"""
Generate publication-ready figures for the paper (no LaTeX required).

Outputs:
- figures/fig_inference_tradeoff.pdf
- figures/fig_attention_weights.pdf
- figures/fig_dropout_effect.pdf

Usage:
    python scripts/generate_paper_figures_simple.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': (4.5, 3.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
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
    'M1': {'params': 84801, 'accuracy': 82.57, 'rmse': 0.0408, 'inference_ms': 1.11, 'category': 'small'},
    'M2': {'params': 84866, 'accuracy': 81.50, 'rmse': 0.0423, 'inference_ms': 1.16, 'category': 'small'},
    'M3': {'params': 597633, 'accuracy': 87.84, 'rmse': 0.0338, 'inference_ms': 2.40, 'category': 'medium_baseline'},
    'M4': {'params': 597762, 'accuracy': 90.25, 'rmse': 0.0311, 'inference_ms': 2.44, 'category': 'medium_attention'},
    'M5': {'params': 630529, 'accuracy': 88.34, 'rmse': 0.0332, 'inference_ms': 2.88, 'category': 'medium_attention'},
    'M6': {'params': 597633, 'accuracy': 88.17, 'rmse': 0.0334, 'inference_ms': 2.46, 'category': 'medium_attention'},
}

DROPOUT_DATA = {
    'M1': {'no_dropout': 82.57, 'dropout': 80.49},
    'M2': {'no_dropout': 81.50, 'dropout': 80.07},
    'M3': {'no_dropout': 87.84, 'dropout': 86.29},
    'M4': {'no_dropout': 90.25, 'dropout': 84.31},
    'M5': {'no_dropout': 88.34, 'dropout': 85.39},
    'M6': {'no_dropout': 88.17, 'dropout': 85.23},
}


def load_attention_weights(base_path: Path) -> dict:
    """Load attention weights from CSV files."""
    weights = {}

    # M4 Simple Attention
    m4_path = base_path / 'M4_Medium_Simple_Attention' / 'attention_all_epochs.csv'
    if m4_path.exists():
        df = pd.read_csv(m4_path)
        epoch_cols = sorted([c for c in df.columns if c.startswith('epoch_')],
                           key=lambda x: int(x.split('_')[1]))
        weights['M4'] = df[epoch_cols[-1]].values
        print(f"  M4: Loaded {len(weights['M4'])} weights from {epoch_cols[-1]}")

    # M5 Additive Attention
    m5_path = base_path / 'M5_Medium_Additive_Attention' / 'attention_all_epochs.csv'
    if m5_path.exists():
        df = pd.read_csv(m5_path)
        epoch_cols = sorted([c for c in df.columns if c.startswith('epoch_')],
                           key=lambda x: int(x.split('_')[1]))
        weights['M5'] = df[epoch_cols[-1]].values
        print(f"  M5: Loaded {len(weights['M5'])} weights from {epoch_cols[-1]}")

    # M6 Scaled DP Attention
    m6_path = base_path / 'M6_Medium_Scaled_DP_Attention' / 'attention_all_epochs.csv'
    if m6_path.exists():
        df = pd.read_csv(m6_path)
        epoch_cols = sorted([c for c in df.columns if c.startswith('epoch_')],
                           key=lambda x: int(x.split('_')[1]))
        weights['M6'] = df[epoch_cols[-1]].values
        print(f"  M6: Loaded {len(weights['M6'])} weights from {epoch_cols[-1]}")

    return weights


def create_inference_tradeoff_figure(output_dir: Path):
    """Create Figure: Accuracy vs Inference Time scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 4))

    # Small models (blue circles)
    small_models = ['M1', 'M2']
    small_x = [MODEL_DATA[m]['inference_ms'] for m in small_models]
    small_y = [MODEL_DATA[m]['accuracy'] for m in small_models]
    ax.scatter(small_x, small_y, c=COLORS['blue'], marker='o', s=100,
               label='Small (85K params)', zorder=3, edgecolors='white', linewidths=1)

    # Medium baseline (green square)
    ax.scatter([MODEL_DATA['M3']['inference_ms']], [MODEL_DATA['M3']['accuracy']],
               c=COLORS['green'], marker='s', s=120,
               label='Medium Baseline (598K)', zorder=3, edgecolors='white', linewidths=1)

    # Medium + Attention (orange triangles)
    attn_models = ['M4', 'M5', 'M6']
    attn_x = [MODEL_DATA[m]['inference_ms'] for m in attn_models]
    attn_y = [MODEL_DATA[m]['accuracy'] for m in attn_models]
    ax.scatter(attn_x, attn_y, c=COLORS['orange'], marker='^', s=120,
               label='Medium + Attention', zorder=3, edgecolors='white', linewidths=1)

    # Highlight M4 (best model) with dashed circle
    circle = plt.Circle((MODEL_DATA['M4']['inference_ms'], MODEL_DATA['M4']['accuracy']),
                         0.15, fill=False, color=COLORS['orange'], linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # Add model labels
    label_config = {
        'M1': {'offset': (0.12, 0.6), 'ha': 'left'},
        'M2': {'offset': (0.12, -0.8), 'ha': 'left'},
        'M3': {'offset': (-0.15, 0.0), 'ha': 'right'},
        'M4': {'offset': (0.0, 1.2), 'ha': 'center'},
        'M5': {'offset': (0.15, 0.0), 'ha': 'left'},
        'M6': {'offset': (0.12, -0.8), 'ha': 'left'},
    }
    for name, data in MODEL_DATA.items():
        cfg = label_config[name]
        weight = 'bold' if name == 'M4' else 'normal'
        ax.annotate(name, (data['inference_ms'], data['accuracy']),
                   xytext=(data['inference_ms'] + cfg['offset'][0],
                          data['accuracy'] + cfg['offset'][1]),
                   fontsize=9, fontweight=weight, ha=cfg['ha'])

    # Formatting
    ax.set_xlabel('Inference Time P95 (ms)', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(80, 92)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

    # Save
    output_path = output_dir / 'fig_inference_tradeoff.pdf'
    fig.savefig(output_path)
    fig.savefig(output_dir / 'fig_inference_tradeoff.png')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_attention_weights_figure(output_dir: Path, weights: dict):
    """Create Figure: Attention Weight Distributions."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Time steps (relative to prediction)
    n_steps = 50

    # Subsample indices for cleaner visualization
    indices = list(range(0, 50, 5)) + [49]  # Every 5th step + last
    time_labels = [f't-{50-i}' for i in indices]
    time_positions = [-50 + i for i in indices]

    # M4 Simple Attention
    if 'M4' in weights:
        m4_subset = [weights['M4'][i] for i in indices]
        ax.plot(time_positions, m4_subset, color=COLORS['orange'], linewidth=2,
                marker='^', markersize=6, label='M4 Simple', zorder=3)

    # M5 Additive Attention
    if 'M5' in weights:
        m5_subset = [weights['M5'][i] for i in indices]
        ax.plot(time_positions, m5_subset, color=COLORS['green'], linewidth=2,
                marker='s', markersize=5, label='M5 Additive', zorder=3)

    # M6 Scaled DP Attention
    if 'M6' in weights:
        m6_subset = [weights['M6'][i] for i in indices]
        ax.plot(time_positions, m6_subset, color=COLORS['blue'], linewidth=2,
                linestyle='--', marker='o', markersize=4, label='M6 Scaled DP', zorder=3)

    # Formatting
    ax.set_xlabel('Time Step (relative to prediction)', fontsize=10)
    ax.set_ylabel('Attention Weight', fontsize=10)
    ax.set_xlim(-52, 2)
    ax.set_ylim(0, 0.30)
    ax.set_xticks([-50, -40, -30, -20, -10, -1])
    ax.set_xticklabels(['t-50', 't-40', 't-30', 't-20', 't-10', 't-1'])
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

    # Add annotation for key insight
    ax.annotate('M5 concentrates\n25% on t-1',
               xy=(-1, 0.255), xytext=(-15, 0.22),
               fontsize=7, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Save
    output_path = output_dir / 'fig_attention_weights.pdf'
    fig.savefig(output_path)
    fig.savefig(output_dir / 'fig_attention_weights.png')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_dropout_comparison_figure(output_dir: Path):
    """Create Figure: Dropout Effect Bar Chart."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    models = list(DROPOUT_DATA.keys())
    no_dropout = [DROPOUT_DATA[m]['no_dropout'] for m in models]
    with_dropout = [DROPOUT_DATA[m]['dropout'] for m in models]
    deltas = [wd - nd for nd, wd in zip(no_dropout, with_dropout)]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_dropout, width, label='No Dropout',
                   color=COLORS['blue'], zorder=3)
    bars2 = ax.bar(x + width/2, with_dropout, width, label='Dropout=0.2',
                   color=COLORS['orange'], alpha=0.8, zorder=3)

    # Add delta labels above bars
    for i, (nd, wd, delta) in enumerate(zip(no_dropout, with_dropout, deltas)):
        color = 'red' if abs(delta) > 3 else 'darkred'
        ax.annotate(f'{delta:.1f}%', xy=(x[i] + width/2, wd + 0.5),
                   ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

    # Highlight M4's large drop
    ax.annotate('', xy=(3 + width/2, 84.31), xytext=(3 + width/2, 90.25),
               arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(78, 93)
    ax.legend(loc='upper right', fontsize=8)

    # Save
    output_path = output_dir / 'fig_dropout_effect.pdf'
    fig.savefig(output_path)
    fig.savefig(output_dir / 'fig_dropout_effect.png')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_model_comparison_table(output_dir: Path):
    """Create a summary table as LaTeX."""
    latex = r"""
\begin{table}[t]
    \centering
    \caption{Model Performance Summary}
    \label{tab:model_summary}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Model} & \textbf{Params} & \textbf{Acc. (\%)} & \textbf{RMSE} & \textbf{Inf. (ms)} \\
        \midrule
"""
    for name, data in MODEL_DATA.items():
        params_str = f"{data['params']/1000:.0f}K"
        bold = r'\textbf{' if name == 'M4' else ''
        unbold = '}' if name == 'M4' else ''
        latex += f"        {bold}{name}{unbold} & {params_str} & {bold}{data['accuracy']:.2f}{unbold} & {bold}{data['rmse']:.4f}{unbold} & {data['inference_ms']:.2f} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    output_path = output_dir / 'table_model_summary.tex'
    output_path.write_text(latex)
    print(f"  Saved: {output_path}")


def main():
    """Generate all paper figures."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'docs' / 'paper' / 'figures'
    attention_weights_dir = project_root / 'attention_weights'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Load attention weights
    print("Loading attention weights...")
    weights = load_attention_weights(attention_weights_dir)
    print()

    # Generate figures
    print("Creating figures...")

    # Figure 1: Accuracy vs Inference Time
    create_inference_tradeoff_figure(output_dir)

    # Figure 2: Attention Weights
    if weights:
        create_attention_weights_figure(output_dir, weights)
    else:
        print("  WARNING: No attention weights found, skipping attention figure")

    # Figure 3: Dropout Effect
    create_dropout_comparison_figure(output_dir)

    # Bonus: LaTeX table
    create_model_comparison_table(output_dir)

    print()
    print("=" * 60)
    print("Done! Files saved to:")
    for f in sorted(output_dir.glob('*')):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
