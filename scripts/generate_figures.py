#!/usr/bin/env python3
"""
Unified figure generation script for paper and analysis plots.

Generates all figures with consistent styling for LaTeX (PGF) and preview (PDF).

Available figures:
  Paper Figures:
    - fig_inference_tradeoff    Accuracy vs. Inference Time (all 8 models)
    - fig_attention_simple      Attention weights XY-plot (M6 Simple Attention)
    - fig_attention_additive    Attention heatmap (M7 Additive Attention)
    - fig_attention_scaled      Attention heatmap (M8 Scaled DP Attention)
    - fig_att_combined          All three attention mechanisms in one line plot
    - fig_dropout_effect        Dropout comparison bar chart
    - fig_prediction_timeseries Prediction vs. Ground Truth timeseries (M6)

  Training Figures (per model):
    - {model}_loss_curves       Training and validation loss
    - {model}_metrics_curves    R², Accuracy, RMSE over epochs
    - {model}_training_overview Combined training overview

Usage:
    # Generate all paper and training figures
    python scripts/generate_figure.py --all
    
    # Generate all paper figures
    python scripts/generate_figures.py --paper

    # Generate specific figure
    python scripts/generate_figures.py --figure inference_tradeoff

    # Generate training curves for a model
    python scripts/generate_figures.py --training M4_Medium_Simple_Attention

    # Generate all training curves
    python scripts/generate_figures.py --training-all

    # List available figures
    python scripts/generate_figures.py --list
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # Flexible backend for all output formats

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 - activates styles

# Lazy imports for attention heatmaps (only loaded when needed)
_torch_imported = False
_torch = None
_F = None

# =============================================================================
# UNIFIED STYLE CONFIGURATION
# =============================================================================

# Color palette (colorblind-friendly, print-safe)
COLORS = {
    # Model type colors
    'mlp': '#7f7f7f',           # Gray - MLP baselines
    'lstm': '#1f77b4',          # Blue - LSTM baselines
    'lstm_attn': '#ff7f0e',     # Orange - LSTM + Attention

    # Semantic colors
    'primary': '#1f77b4',       # Blue
    'secondary': '#ff7f0e',     # Orange
    'tertiary': '#2ca02c',      # Green
    'highlight': '#d62728',     # Red
    'neutral': '#7f7f7f',       # Gray

    # Plot-specific
    'train': '#1f77b4',         # Blue - training
    'val': '#ff7f0e',           # Orange - validation
    'ground_truth': '#1f77b4',  # Blue
    'prediction': '#ff7f0e',    # Orange
    'residual_pos': '#2ca02c',  # Green - positive residuals
    'residual_neg': '#d62728',  # Red - negative residuals
    'threshold': '#7f7f7f',     # Gray - threshold lines
}

# Markers for different model types
MARKERS = {
    'mlp': 's',        # Square
    'lstm': 'o',       # Circle
    'lstm_attn': '^',  # Triangle up
}


def setup_style():
    """Configure matplotlib with SciencePlots IEEE style and project customizations."""
    # Use SciencePlots IEEE style as base
    plt.style.use(['science', 'ieee'])

    # Project-specific overrides
    plt.rcParams.update({
        # Disable grid (IEEE convention: minimal, no grids)
        'axes.grid': False,

        # LaTeX configuration - use pdflatex with Computer Modern fonts
        'text.usetex': True,
        'pgf.texsystem': 'pdflatex',
        'pgf.rcfonts': False,
        'pgf.preamble': '\n'.join([
            r'\usepackage[utf8]{inputenc}',
            r'\usepackage[T1]{fontenc}',
            r'\usepackage{amsmath}',
            r'\usepackage{siunitx}',
            r'\providecommand{\mathdefault}[1]{#1}',  # Fix for PGF compatibility
        ]),
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],

        # Figure sizing (IEEE single column: 3.5in)
        'figure.figsize': (3.5, 2.5),

        # Spacing
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Lines and markers
        'lines.linewidth': 1.0,
        'lines.markersize': 4,

        # Legend
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'none',

        # PNG output quality
        'savefig.dpi': 300,
    })


setup_style()


# =============================================================================
# MODEL DATA
# =============================================================================

def load_model_data(comparison_path: Optional[Path] = None,
                    bootstrap_path: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Load model data from evaluation results JSON.

    Args:
        comparison_path: Path to comparison.json (default: results/no_dropout/comparison.json)
        bootstrap_path: Path to bootstrap results JSON (default: results/bootstrap/bootstrap_results_no_dropout.json)

    Returns:
        Dictionary with model data in the expected format
    """
    import json

    if comparison_path is None:
        comparison_path = PROJECT_ROOT / 'results' / 'no_dropout' / 'comparison.json'

    if bootstrap_path is None:
        bootstrap_path = PROJECT_ROOT / 'results' / 'bootstrap' / 'bootstrap_results_no_dropout.json'

    # Load comparison results
    with open(comparison_path, 'r', encoding='utf-8') as f:
        comparison = json.load(f)

    # Load bootstrap results for confidence intervals (optional)
    bootstrap_data = {}
    if bootstrap_path.exists():
        with open(bootstrap_path, 'r', encoding='utf-8') as f:
            bootstrap = json.load(f)
            for model in bootstrap.get('models', []):
                model_id = model['model_id']
                bootstrap_data[model_id] = {
                    'accuracy_ci': (model['ci_lower'], model['ci_upper'])
                }

    # Map model type from name to category
    def get_model_type(name: str) -> str:
        name_lower = name.lower()
        if 'mlp' in name_lower:
            return 'mlp'
        elif 'attention' in name_lower or 'attn' in name_lower or 'simple' in name_lower or 'additive' in name_lower or 'scaled' in name_lower:
            return 'lstm_attn'
        else:
            return 'lstm'

    model_data = {}
    for model_key, data in comparison['models'].items():
        model_id = model_key.upper()  # m1 -> M1

        # Get bootstrap CI if available
        accuracy_ci = bootstrap_data.get(model_id, {}).get('accuracy_ci', (None, None))

        model_data[model_id] = {
            'name': data['name'],
            'type': get_model_type(data['name']),
            'accuracy': data['metrics']['accuracy'],
            'accuracy_ci': accuracy_ci,
            'r2': data['metrics']['r2'],
            'rmse': data['metrics']['rmse'],
            'inference_p95_ms': data['inference']['p95_ms'],
            'inference_p95_std_ms': data['inference']['p95_std_ms'],
            'params': data['parameters'],
        }

    return model_data


# Load model data from results (lazy initialization)
_MODEL_DATA_CACHE = None


def get_model_data() -> Dict[str, Dict]:
    """Get model data, loading from JSON on first access."""
    global _MODEL_DATA_CACHE
    if _MODEL_DATA_CACHE is None:
        try:
            _MODEL_DATA_CACHE = load_model_data()
            print(f"Loaded model data for {len(_MODEL_DATA_CACHE)} models from comparison.json")
        except FileNotFoundError as e:
            print(f"WARNING: Could not load model data: {e}")
            print("Using empty model data - some figures may not generate")
            _MODEL_DATA_CACHE = {}
    return _MODEL_DATA_CACHE


# For backwards compatibility, MODEL_DATA is now a property-like access
# Use get_model_data() in functions instead
MODEL_DATA = None  # Deprecated, use get_model_data() instead

# Dropout comparison data (only LSTM models M3-M8)
# NOTE: Hardcoded values - no dropout comparison.json available
# TODO: Load from results/dropout/comparison.json when available
DROPOUT_DATA = {
    'M3': {'no_dropout': 82.60, 'dropout': 80.49},
    'M4': {'no_dropout': 81.91, 'dropout': 80.07},
    'M5': {'no_dropout': 87.85, 'dropout': 86.29},
    'M6': {'no_dropout': 90.25, 'dropout': 84.31},
    'M7': {'no_dropout': 88.34, 'dropout': 85.39},
    'M8': {'no_dropout': 89.80, 'dropout': 85.23},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig: plt.Figure, output_dir: Path, name: str,
                formats: List[str] = ['pgf', 'pdf', 'png']) -> List[Path]:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_dir: Output directory
        name: Base filename (without extension)
        formats: List of formats to save (default: pgf, pdf, png)

    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for fmt in formats:
        path = output_dir / f'{name}.{fmt}'
        fig.savefig(path)
        saved.append(path)

    plt.close(fig)
    return saved


# =============================================================================
# PAPER FIGURES
# =============================================================================

def fig_inference_tradeoff(output_dir: Path) -> List[Path]:
    """
    Create Accuracy vs. Inference Time scatter plot.

    Shows all 8 models with different markers for MLP, LSTM, and LSTM+Attention.
    Uses logarithmic x-axis to accommodate the wide inference time range.
    M6 (best model) is highlighted with a red ring.
    """
    model_data = get_model_data()
    if not model_data:
        print("  WARNING: No model data available, skipping")
        return []

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Marker configuration
    marker_sizes = {'mlp': 35, 'lstm': 45, 'lstm_attn': 50}

    # Plot each model type
    plotted_types = set()
    for model_id, data in model_data.items():
        mtype = data['type']

        # Label only first of each type for legend
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

    # Add model labels (offset in points for consistent spacing)
    label_offsets = {
        # (x_points, y_points): positive x = right, positive y = up
        'M1': (5, 1),    'M2': (5, 1),
        'M3': (-5, 1),     'M4': (5, 0),
        'M5': (-5, 0),   'M6': (-5, 1),
        'M7': (1, -5),    'M8': (5, 0),
    }

    for model_id, data in model_data.items():
        x_off, y_off = label_offsets[model_id]
        ha = 'left' if x_off > 0 else 'right'
        va = 'bottom' if y_off > 0 else 'top'

        ax.annotate(
            model_id,
            (data['inference_p95_ms'], data['accuracy']),
            xytext=(x_off, y_off),
            textcoords='offset points',
            fontsize=7, ha=ha, va=va
        )

    # Logarithmic x-axis
    ax.set_xscale('log')
    ax.set_xlim(0.04, 7)
    ax.set_ylim(68, 93)
    ax.set_xticks([0.05, 0.1, 0.5, 1, 2, 5])
    ax.set_xticklabels(['0.05', '0.1', '0.5', '1', '2', '5'])

    ax.set_xlabel(r'Inference Time P95 (ms)')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.legend(loc='lower right', fontsize=7)


    return save_figure(fig, output_dir, 'fig_inference_tradeoff')


def fig_attention_simple(output_dir: Path) -> List[Path]:
    """
    Create attention weight distribution plot for M6 (Simple Attention).

    Shows how Simple Attention distributes weights across timesteps.
    X-axis: Time step, Y-axis: Attention weight.
    """
    # Load M6 attention weights
    weight_path = PROJECT_ROOT / 'attention_weights' / 'M6_Medium_Simple_Attention' / 'attention_test.npy'

    if not weight_path.exists():
        print("  WARNING: M6 attention weights not found, skipping")
        return []

    weights = np.load(weight_path)
    print(f"  Loaded M6 attention weights: shape={weights.shape}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    seq_len = len(weights)
    time_positions = list(range(-seq_len, 0))

    ax.plot(
        time_positions, weights,
        color=COLORS['primary'], linewidth=1.2,
        marker='o', markersize=3, markevery=5
    )

    ax.set_xlabel('Time Step (relative to prediction)')
    ax.set_ylabel('Attention Weight')
    ax.set_xlim(-seq_len - 2, 2)
    ax.set_ylim(0, None)
    ax.set_xticks([-50, -40, -30, -20, -10, -1])
    ax.set_xticklabels([r'$t_{-50}$', r'$t_{-40}$', r'$t_{-30}$', r'$t_{-20}$', r'$t_{-10}$', r'$t_{-1}$'])


    return save_figure(fig, output_dir, 'fig_attention_simple')


def _ensure_torch():
    """Lazy import torch only when needed for attention heatmaps."""
    global _torch_imported, _torch, _F
    if not _torch_imported:
        import torch
        import torch.nn.functional as F
        _torch = torch
        _F = F
        _torch_imported = True
    return _torch, _F


def _compute_additive_attention_matrix(lstm_output, attention_module):
    """Compute full additive attention matrix (batch, seq_len, seq_len)."""
    torch, F = _ensure_torch()
    batch_size, seq_len, hidden_size = lstm_output.shape

    w_x = attention_module.w(lstm_output).unsqueeze(2).expand(-1, -1, seq_len, -1)
    u_x = attention_module.u(lstm_output).unsqueeze(1).expand(-1, seq_len, -1, -1)

    e = torch.tanh(w_x + u_x)
    e = torch.matmul(e, attention_module.v).squeeze(-1)

    return F.softmax(e, dim=-1)


def _compute_scaled_dp_attention_matrix(lstm_output, hidden_size):
    """Compute full scaled dot-product attention matrix (batch, seq_len, seq_len)."""
    torch, F = _ensure_torch()

    e = torch.bmm(lstm_output, lstm_output.permute(0, 2, 1))
    e = e / math.sqrt(hidden_size + 1e-8)

    return F.softmax(e, dim=-1)


def _extract_attention_matrix(model, dataloader, model_type: str) -> np.ndarray:
    """
    Extract averaged full attention matrix from model on test data.

    Args:
        model: Loaded PyTorch Lightning model
        dataloader: Test dataloader
        model_type: 'additive' or 'scaled_dp'

    Returns:
        avg_attention: (seq_len, seq_len) averaged attention matrix
    """
    torch, _ = _ensure_torch()
    model.eval()
    model.cpu()

    attention_sum = None
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.cpu()

            lstm_output, _ = model.lstm(x)

            if model_type == 'additive':
                attention = _compute_additive_attention_matrix(lstm_output, model.attention)
            else:
                attention = _compute_scaled_dp_attention_matrix(lstm_output, model.hidden_size)

            batch_attention = attention.sum(dim=0)
            if attention_sum is None:
                attention_sum = batch_attention
            else:
                attention_sum += batch_attention

            total_samples += x.size(0)

    return (attention_sum / total_samples).numpy()


def _create_attention_heatmap(attention_matrix: np.ndarray, title: str,
                               output_dir: Path, filename: str) -> List[Path]:
    """Create and save attention heatmap figure."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    im = ax.imshow(attention_matrix, aspect='auto', cmap='viridis', origin='lower')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel('Key Position (input timestep)')
    ax.set_ylabel('Query Position (output timestep)')
    ax.set_title(title, fontsize=9)

    seq_len = attention_matrix.shape[0]
    tick_positions = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
    tick_labels = [rf'$t_{{-{seq_len-i}}}$' for i in tick_positions]
    tick_labels[-1] = r'$t_{-1}$'

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    fig.tight_layout()

    return save_figure(fig, output_dir, filename)


def _load_model_and_data(config_path: str, log_dir: str):
    """Load model from checkpoint and setup test dataloader."""
    import pytorch_lightning as pl
    from config.loader import load_config, get_model_class
    from config.settings import get_preprocessed_paths
    from model.data_module import TimeSeriesDataModule

    config = load_config(str(PROJECT_ROOT / config_path))
    pl.seed_everything(config["training"]["seed"])

    # Find best checkpoint
    log_path = PROJECT_ROOT / log_dir
    checkpoints = list((log_path / 'checkpoints').glob('*.ckpt'))
    if not checkpoints:
        return None, None, None
    checkpoint = min(checkpoints, key=lambda p: float(p.stem.split('val_loss=')[1]))

    # Setup data
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
        sequence_ids_path=str(paths["sequence_ids"]),
        batch_size=256,
        split_seed=data_config.get("split_seed", 0),
    )
    data_module.setup()

    # Load model
    model_class = get_model_class(config["model"]["type"])
    model = model_class.load_from_checkpoint(str(checkpoint))

    return model, data_module.test_dataloader(), checkpoint.name


def fig_attention_additive(output_dir: Path) -> List[Path]:
    """
    Create attention weight heatmap for M7 (Additive Attention).

    Loads the trained M7 model, runs inference on test data, and creates
    a heatmap showing the full 2D attention matrix.
    """
    print("  Loading M7 model and extracting attention matrix...")

    model, dataloader, ckpt_name = _load_model_and_data(
        'config/model_configs/m7_medium_additive_attn.yaml',
        'lightning_logs/M7_Medium_Additive_Attention/version_0'
    )

    if model is None:
        print("  WARNING: M7 checkpoint not found, skipping")
        return []

    print(f"  Checkpoint: {ckpt_name}")

    attention_matrix = _extract_attention_matrix(model, dataloader, 'additive')
    print(f"  Attention matrix shape: {attention_matrix.shape}")

    # Also save the matrix
    matrix_path = output_dir / 'attention_heatmap_m7_matrix.npy'
    np.save(matrix_path, attention_matrix)

    return _create_attention_heatmap(
        attention_matrix,
        'M7 Additive Attention',
        output_dir,
        'fig_attention_additive'
    )


def fig_attention_scaled(output_dir: Path) -> List[Path]:
    """
    Create attention weight heatmap for M8 (Scaled Dot-Product Attention).

    Loads the trained M8 model, runs inference on test data, and creates
    a heatmap showing the full 2D attention matrix.
    """
    print("  Loading M8 model and extracting attention matrix...")

    model, dataloader, ckpt_name = _load_model_and_data(
        'config/model_configs/m8_medium_scaled_dp_attn.yaml',
        'lightning_logs/M8_Medium_Scaled_DP_Attention/version_1'
    )

    if model is None:
        print("  WARNING: M8 checkpoint not found, skipping")
        return []

    print(f"  Checkpoint: {ckpt_name}")

    attention_matrix = _extract_attention_matrix(model, dataloader, 'scaled_dp')
    print(f"  Attention matrix shape: {attention_matrix.shape}")

    # Also save the matrix
    matrix_path = output_dir / 'attention_heatmap_m8_matrix.npy'
    np.save(matrix_path, attention_matrix)

    return _create_attention_heatmap(
        attention_matrix,
        'M8 Scaled Dot-Product Attention',
        output_dir,
        'fig_attention_scaled'
    )


def fig_attention_combined(output_dir: Path) -> List[Path]:
    """
    Create combined attention weight distribution plot for all three mechanisms.

    Shows M6 (Simple), M7 (Additive), and M8 (Scaled Dot-Product) attention
    weights as line plots in a single figure. For M7, the 2D attention matrix
    is averaged over query positions to produce a 1D distribution.
    """
    # Load attention weights
    m6_path = PROJECT_ROOT / 'results' / 'no_dropout' / 'm6' / 'M6_Medium_Simple_Attention_attention_weights.npy'
    m7_path = PROJECT_ROOT / 'results' / 'no_dropout' / 'm7' / 'M7_Medium_Additive_Attention_attention_weights.npy'
    m8_path = PROJECT_ROOT / 'results' / 'no_dropout' / 'm8' / 'M8_Medium_Scaled_DP_Attention_attention_weights.npy'

    missing = []
    for name, path in [('M6', m6_path), ('M7', m7_path), ('M8', m8_path)]:
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"  WARNING: Attention weights not found for {', '.join(missing)}, skipping")
        return []

    m6_weights = np.load(m6_path)  # Shape: (50,)
    m7_weights = np.load(m7_path)  # Shape: (50, 50)
    m8_weights = np.load(m8_path)  # Shape: (50,)

    # M7 is 2D (Query × Key) - average over query positions to get key importance
    m7_weights_1d = m7_weights.mean(axis=0)

    print(f"  M6 Simple: shape={m6_weights.shape}")
    print(f"  M7 Additive: shape={m7_weights.shape} -> averaged to {m7_weights_1d.shape}")
    print(f"  M8 Scaled DP: shape={m8_weights.shape}")

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    seq_len = 50
    time_positions = list(range(-seq_len, 0))  # t-50 to t-1

    # Plot with distinct colors and line styles for B/W compatibility
    ax.plot(time_positions, m6_weights,
            color=COLORS['primary'], linewidth=1.2, linestyle='-',
            label='Simple (M6)')

    ax.plot(time_positions, m7_weights_1d,
            color=COLORS['secondary'], linewidth=1.2, linestyle='--',
            label='Additive (M7)')

    ax.plot(time_positions, m8_weights,
            color=COLORS['tertiary'], linewidth=1.2, linestyle=':',
            label='Scaled Dot-Product (M8)')

    # Configure axes
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Weight')

    # X-axis labels
    ax.set_xlim(-seq_len - 1, 0)
    ax.set_xticks([-50, -40, -30, -20, -10, -1])
    ax.set_xticklabels([r'$t_{-50}$', r'$t_{-40}$', r'$t_{-30}$', r'$t_{-20}$', r'$t_{-10}$', r'$t_{-1}$'])

    # Y-axis: start at 0
    ax.set_ylim(0, None)

    # Legend
    ax.legend(loc='upper left', fontsize=7)

    return save_figure(fig, output_dir, 'fig_attention_comparison')


def fig_dropout_effect(output_dir: Path) -> List[Path]:
    """
    Create dropout effect comparison bar chart.

    Shows accuracy drop when using dropout for each model.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    models = list(DROPOUT_DATA.keys())
    no_dropout = [DROPOUT_DATA[m]['no_dropout'] for m in models]
    with_dropout = [DROPOUT_DATA[m]['dropout'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_dropout, width,
                   label='No Dropout', color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, with_dropout, width,
                   label=r'Dropout${}=0.2$', color=COLORS['secondary'], alpha=0.8)

    # Add delta labels
    for i, (nd, wd) in enumerate(zip(no_dropout, with_dropout)):
        delta = wd - nd
        color = COLORS['highlight'] if abs(delta) > 3 else 'darkred'
        ax.annotate(
            rf'{delta:.1f}\%',
            xy=(x[i] + width/2, wd + 0.3),
            ha='center', va='bottom', fontsize=6, color=color
        )

    ax.set_xlabel('Model')
    ax.set_ylabel(r'Accuracy (\%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(78, 93)
    ax.legend(loc='upper right', fontsize=7)


    return save_figure(fig, output_dir, 'fig_dropout_effect')


def fig_prediction_timeseries(output_dir: Path) -> List[Path]:
    """
    Create prediction vs. ground truth timeseries plot.

    Shows M6 predictions overlaid on ground truth with residuals below.
    Selects a segment with interesting dynamics (high variance, steering both ways).
    """
    import torch
    import pytorch_lightning as pl
    from config.loader import load_config, get_model_class
    from config.settings import get_preprocessed_paths
    from model.data_module import TimeSeriesDataModule

    # Configuration - M6 is the best model (Medium + Simple Attention)
    checkpoint_dir = PROJECT_ROOT / 'lightning_logs' / 'M6_Medium_Simple_Attention' / 'version_2' / 'checkpoints'
    config_path = PROJECT_ROOT / 'config' / 'model_configs' / 'm6_medium_simple_attn.yaml'
    num_samples = 800  # ~80 seconds @ 10Hz

    # Find best checkpoint
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        print("  WARNING: No M6 checkpoints found, using synthetic data")
        # Synthetic fallback
        time = np.linspace(0, 80, num_samples)
        targets = 0.3 * np.sin(0.1 * time) + 0.1 * np.sin(0.5 * time)
        predictions = targets + 0.02 * np.random.randn(num_samples)
    else:
        checkpoint = min(checkpoints, key=lambda p: float(p.stem.split('val_loss=')[1]))
        print(f"  Using checkpoint: {checkpoint.name}")

        # Load config and set seed
        config = load_config(str(config_path))
        pl.seed_everything(config["training"]["seed"])

        # Load data
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
            sequence_ids_path=str(paths["sequence_ids"]),
            batch_size=512,
            split_seed=data_config.get("split_seed", 0),
        )
        data_module.setup()

        # Load model
        model_class = get_model_class(config["model"]["type"])
        model = model_class.load_from_checkpoint(str(checkpoint))
        model.eval()
        model.cpu()

        # Get test data
        all_features = []
        all_targets = []
        for batch in data_module.test_dataloader():
            X_batch, Y_batch = batch
            all_features.append(X_batch)
            all_targets.append(Y_batch)

        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()

        # Find segment with high variance and bidirectional steering
        best_start = 0
        best_score = 0
        for start in range(0, len(all_targets) - num_samples, num_samples // 4):
            segment = all_targets[start:start + num_samples]
            variance = np.var(segment)
            has_dynamics = np.max(segment) > 0.1 and np.min(segment) < -0.1
            score = variance * (1.5 if has_dynamics else 0.5)
            if score > best_score:
                best_score = score
                best_start = start

        print(f"  Selected segment {best_start}:{best_start + num_samples}")

        # Generate predictions
        segment_features = all_features[best_start:best_start + num_samples]
        targets = all_targets[best_start:best_start + num_samples]
        with torch.no_grad():
            predictions = model(segment_features).numpy().flatten()

        time = np.arange(num_samples) / 10.0

    # Calculate metrics
    residuals = predictions - targets
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((targets - np.mean(targets))**2)
    accuracy = np.mean(np.abs(residuals) < 0.05) * 100

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})

    # Top: Predictions vs Ground Truth
    ax1 = axes[0]
    ax1.plot(time, targets, color=COLORS['ground_truth'], linewidth=0.8,
             label='Ground Truth', alpha=0.9)
    ax1.plot(time, predictions, color=COLORS['prediction'], linewidth=0.8,
             linestyle='--', label='Prediction', alpha=0.9)
    ax1.set_ylabel('Steering Torque\n(normalized)', fontsize=8)
    ax1.legend(loc='upper right', fontsize=6)
    ax1.set_ylim(-0.6, 0.6)


    # Metrics annotation
    metrics_text = f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\nAcc: {accuracy:.1f}\\%'
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=6,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Bottom: Residuals
    ax2 = axes[1]
    ax2.fill_between(time, 0, residuals, where=(residuals >= 0),
                     color=COLORS['residual_pos'], alpha=0.5, linewidth=0)
    ax2.fill_between(time, 0, residuals, where=(residuals < 0),
                     color=COLORS['residual_neg'], alpha=0.5, linewidth=0)
    ax2.plot(time, residuals, color='black', linewidth=0.3, alpha=0.5)

    # Threshold lines
    ax2.axhline(y=0.05, color=COLORS['threshold'], linestyle='--', linewidth=0.5)
    ax2.axhline(y=-0.05, color=COLORS['threshold'], linestyle='--', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.3)
    ax2.fill_between(time, -0.05, 0.05, color=COLORS['threshold'], alpha=0.1)

    ax2.set_ylabel('Residual', fontsize=8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax2.text(time[-1] * 0.99, 0.055, r'$\pm 0.05$', fontsize=5, ha='right',
             va='bottom', color=COLORS['threshold'])


    fig.align_ylabels(axes)

    return save_figure(fig, output_dir, 'fig_prediction_timeseries')


# =============================================================================
# TRAINING FIGURES
# =============================================================================

def load_tensorboard_logs(logdir: Path) -> Dict[str, Dict]:
    """Load TensorBoard event files and extract metrics."""
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        epochs = np.arange(len(events))
        values = np.array([e.value for e in events])
        metrics[tag] = {'epochs': epochs, 'values': values}

    return metrics


def fig_training_loss(metrics: Dict, model_name: str, output_dir: Path) -> List[Path]:
    """Create training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Find loss keys
    train_key = next((k for k in metrics if 'train_loss' in k.lower() and 'epoch' in k.lower()), None)
    val_key = next((k for k in metrics if 'val_loss' in k.lower()), None)

    if train_key:
        data = metrics[train_key]
        ax.plot(data['epochs'], data['values'], color=COLORS['train'],
                linewidth=1.5, label='Training', alpha=0.8)

    if val_key:
        data = metrics[val_key]
        ax.plot(data['epochs'], data['values'], color=COLORS['val'],
                linewidth=1.5, label='Validation', alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend(loc='upper right', fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


    # Set y limits
    if train_key or val_key:
        all_vals = []
        if train_key:
            all_vals.extend(metrics[train_key]['values'])
        if val_key:
            all_vals.extend(metrics[val_key]['values'])
        if all_vals:
            min_val, max_val = min(all_vals), max(all_vals)
            ax.set_ylim(bottom=max(0, min_val - 0.1 * (max_val - min_val)))

    return save_figure(fig, output_dir, f'{model_name}_loss_curves')


def fig_training_metrics(metrics: Dict, model_name: str, output_dir: Path) -> List[Path]:
    """Create R², Accuracy, RMSE curves over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Find metric keys
    r2_key = next((k for k in metrics if 'val_r2' in k.lower() and 'avg' not in k.lower()), None)
    acc_key = next((k for k in metrics if 'val_accuracy' in k.lower() or 'val_abs_accuracy' in k.lower()), None)
    rmse_key = next((k for k in metrics if 'val_rmse' in k.lower() and 'avg' not in k.lower()), None)

    plot_data = [
        (r2_key, r'$R^2$', COLORS['tertiary'], axes[0], lambda x: x),
        (acc_key, r'Accuracy (\%)', COLORS['primary'], axes[1], lambda x: x * 100),
        (rmse_key, 'RMSE', COLORS['highlight'], axes[2], lambda x: x),
    ]

    for key, ylabel, color, ax, transform in plot_data:
        if key and key in metrics:
            data = metrics[key]
            values = transform(data['values'])
            ax.plot(data['epochs'], values, color=color, linewidth=1.5)

            if len(values) > 0:
                final = values[-1]
                ax.axhline(y=final, color=COLORS['neutral'], linestyle='--',
                          linewidth=0.5, alpha=0.5)
                fmt = '.4f' if 'R' in ylabel or 'RMSE' in ylabel else '.1f'
                ax.text(0.95, 0.05, f'Final: {final:{fmt}}', transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=6, color=COLORS['neutral'])

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    

    fig.tight_layout()
    return save_figure(fig, output_dir, f'{model_name}_metrics_curves')


def fig_training_overview(metrics: Dict, model_name: str, output_dir: Path) -> List[Path]:
    """Create combined training overview."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # Find keys
    train_key = next((k for k in metrics if 'train_loss' in k.lower() and 'epoch' in k.lower()), None)
    val_key = next((k for k in metrics if 'val_loss' in k.lower()), None)
    r2_key = next((k for k in metrics if 'val_r2' in k.lower() and 'avg' not in k.lower()), None)
    acc_key = next((k for k in metrics if 'val_accuracy' in k.lower() or 'val_abs_accuracy' in k.lower()), None)

    # Loss plot
    ax = axes[0, 0]
    if train_key:
        data = metrics[train_key]
        ax.plot(data['epochs'], data['values'], color=COLORS['train'], label='Train')
    if val_key:
        data = metrics[val_key]
        ax.plot(data['epochs'], data['values'], color=COLORS['val'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=6)


    # R² plot
    ax = axes[0, 1]
    if r2_key:
        data = metrics[r2_key]
        ax.plot(data['epochs'], data['values'], color=COLORS['tertiary'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$R^2$')


    # Accuracy plot
    ax = axes[1, 0]
    if acc_key:
        data = metrics[acc_key]
        ax.plot(data['epochs'], data['values'] * 100, color=COLORS['primary'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Accuracy (\%)')


    # Summary
    ax = axes[1, 1]
    ax.axis('off')

    lines = [f"\\textbf{{{model_name}}}", ""]
    if val_key and len(metrics[val_key]['values']) > 0:
        lines.append(f"Val Loss: {metrics[val_key]['values'][-1]:.6f}")
    if r2_key and len(metrics[r2_key]['values']) > 0:
        lines.append(f"$R^2$: {metrics[r2_key]['values'][-1]:.4f}")
    if acc_key and len(metrics[acc_key]['values']) > 0:
        lines.append(f"Accuracy: {metrics[acc_key]['values'][-1] * 100:.2f}\\%")

    ax.text(0.5, 0.5, '\n'.join(lines), ha='center', va='center',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return save_figure(fig, output_dir, f'{model_name}_training_overview')


# =============================================================================
# MAIN
# =============================================================================

PAPER_FIGURES = {
    'inference_tradeoff': fig_inference_tradeoff,
    'attention_simple': fig_attention_simple,
    'attention_additive': fig_attention_additive,
    'attention_scaled': fig_attention_scaled,
    'att_combined': fig_attention_combined,
    'dropout_effect': fig_dropout_effect,
    'prediction_timeseries': fig_prediction_timeseries,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper and training figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate ALL figures (paper + training)')
    parser.add_argument('--paper', action='store_true',
                       help='Generate all paper figures')
    parser.add_argument('--figure', '-f', type=str, choices=list(PAPER_FIGURES.keys()),
                       help='Generate specific paper figure')
    parser.add_argument('--training', '-t', type=str, metavar='MODEL',
                       help='Generate training curves for model (e.g., M4_Medium_Simple_Attention)')
    parser.add_argument('--training-all', action='store_true',
                       help='Generate training curves for all models')
    parser.add_argument('--output', '-o', type=str, default='figures',
                       help='Output directory (default: figures)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available figures')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available paper figures:")
        for name in PAPER_FIGURES:
            print(f"  - {name}")
        print("\nTraining figures (generated per model):")
        print("  - {model}_loss_curves")
        print("  - {model}_metrics_curves")
        print("  - {model}_training_overview")
        return

    output_dir = PROJECT_ROOT / args.output

    # --all implies both --paper and --training-all
    if args.all:
        args.paper = True
        args.training_all = True

    print("=" * 60)
    print("Figure Generation")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print()

    # Paper figures
    if args.paper or args.figure:
        figures_to_generate = [args.figure] if args.figure else list(PAPER_FIGURES.keys())

        for name in figures_to_generate:
            print(f"Creating {name}...")
            try:
                paths = PAPER_FIGURES[name](output_dir)
                for p in paths:
                    print(f"  -> {p.name}")
            except Exception as e:
                print(f"  ERROR: {e}")
        print()

    # Training figures
    if args.training or args.training_all:
        logs_dir = PROJECT_ROOT / 'lightning_logs'

        if args.training_all:
            model_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        else:
            model_dirs = [logs_dir / args.training]

        for model_dir in model_dirs:
            if not model_dir.exists():
                print(f"WARNING: {model_dir} not found")
                continue

            # Find latest version
            versions = sorted(model_dir.glob('version_*'), key=lambda p: int(p.name.split('_')[1]))
            if not versions:
                continue

            logdir = versions[-1]
            model_name = model_dir.name
            print(f"Creating training figures for {model_name}...")

            try:
                metrics = load_tensorboard_logs(logdir)
                train_output = output_dir / 'training' / model_name

                for fig_func in [fig_training_loss, fig_training_metrics, fig_training_overview]:
                    paths = fig_func(metrics, model_name, train_output)
                    for p in paths:
                        print(f"  -> {p.name}")
            except Exception as e:
                print(f"  ERROR: {e}")
        print()

    if not any([args.all, args.paper, args.figure, args.training, args.training_all]):
        print("No action specified. Use --help for usage.")
        return

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
