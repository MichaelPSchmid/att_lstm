"""
Unified batch runner for training and evaluation jobs.

Replaces the separate .bat and .sh scripts with a single, platform-independent
Python script that handles all batch operations.

Output Structure:
    results/
    ├── no_dropout/
    │   ├── summary.md          # Auto-generated overview
    │   ├── comparison.json     # All metrics for comparison
    │   ├── m1/
    │   │   ├── eval.json
    │   │   ├── scatter.png
    │   │   └── ...
    │   └── m2/
    └── dropout/
        └── ...

Usage:
    # Training
    python scripts/batch_runner.py train --variant no_dropout
    python scripts/batch_runner.py train --variant dropout
    python scripts/batch_runner.py train --variant dropout --models m1 m2 m3

    # Evaluation
    python scripts/batch_runner.py evaluate --variant no_dropout
    python scripts/batch_runner.py evaluate --variant dropout

    # Both (train + evaluate)
    python scripts/batch_runner.py all --variant dropout

    # List available models/checkpoints
    python scripts/batch_runner.py list --variant dropout
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    type: str
    config_no_dropout: str
    config_dropout: str


# All available models
MODELS = [
    ModelConfig(
        id="m1",
        name="M1 Small Baseline",
        type="LSTM (64, 3)",
        config_no_dropout="config/model_configs/m1_small_baseline.yaml",
        config_dropout="config/model_configs/m1_small_baseline_dropout.yaml",
    ),
    ModelConfig(
        id="m2",
        name="M2 Small + Simple Attn",
        type="LSTM + Attention (64, 3)",
        config_no_dropout="config/model_configs/m2_small_simple_attn.yaml",
        config_dropout="config/model_configs/m2_small_simple_attn_dropout.yaml",
    ),
    ModelConfig(
        id="m3",
        name="M3 Medium Baseline",
        type="LSTM (128, 5)",
        config_no_dropout="config/model_configs/m3_medium_baseline.yaml",
        config_dropout="config/model_configs/m3_medium_baseline_dropout.yaml",
    ),
    ModelConfig(
        id="m4",
        name="M4 Medium + Simple Attn",
        type="LSTM + Attention (128, 5)",
        config_no_dropout="config/model_configs/m4_medium_simple_attn.yaml",
        config_dropout="config/model_configs/m4_medium_simple_attn_dropout.yaml",
    ),
    ModelConfig(
        id="m5",
        name="M5 Medium + Additive Attn",
        type="LSTM + Additive (128, 5)",
        config_no_dropout="config/model_configs/m5_medium_additive_attn.yaml",
        config_dropout="config/model_configs/m5_medium_additive_attn_dropout.yaml",
    ),
    ModelConfig(
        id="m6",
        name="M6 Medium + Scaled DP",
        type="LSTM + Scaled DP (128, 5)",
        config_no_dropout="config/model_configs/m6_medium_scaled_dp_attn.yaml",
        config_dropout="config/model_configs/m6_medium_scaled_dp_attn_dropout.yaml",
    ),
]

MODEL_BY_ID = {m.id: m for m in MODELS}


# =============================================================================
# Path helpers
# =============================================================================

def get_results_dir(variant: str) -> Path:
    """Get the results directory for a variant."""
    return PROJECT_ROOT / "results" / variant


def get_model_output_dir(model: ModelConfig, variant: str) -> Path:
    """Get the output directory for a specific model."""
    return get_results_dir(variant) / model.id


def get_config_path(model: ModelConfig, variant: str) -> Path:
    """Get the config path for a model and variant."""
    if variant == "dropout":
        return PROJECT_ROOT / model.config_dropout
    else:
        return PROJECT_ROOT / model.config_no_dropout


def get_log_dir_name(model: ModelConfig, variant: str) -> str:
    """Get the expected lightning_logs directory name for a model."""
    if variant == "dropout":
        config_name = Path(model.config_dropout).stem
    else:
        config_name = Path(model.config_no_dropout).stem

    # Map config names to log directory names
    name_mapping = {
        "m1_small_baseline": "M1_Small_Baseline",
        "m1_small_baseline_dropout": "M1_Small_Baseline_Dropout",
        "m2_small_simple_attn": "M2_Small_Simple_Attention",
        "m2_small_simple_attn_dropout": "M2_Small_Simple_Attention_Dropout",
        "m3_medium_baseline": "M3_Medium_Baseline",
        "m3_medium_baseline_dropout": "M3_Medium_Baseline_Dropout",
        "m4_medium_simple_attn": "M4_Medium_Simple_Attention",
        "m4_medium_simple_attn_dropout": "M4_Medium_Simple_Attention_Dropout",
        "m5_medium_additive_attn": "M5_Medium_Additive_Attention",
        "m5_medium_additive_attn_dropout": "M5_Medium_Additive_Attention_Dropout",
        "m6_medium_scaled_dp_attn": "M6_Medium_Scaled_DP_Attention",
        "m6_medium_scaled_dp_attn_dropout": "M6_Medium_Scaled_DP_Attention_Dropout",
    }

    return name_mapping.get(config_name, config_name)


def find_best_checkpoint(model: ModelConfig, variant: str) -> Optional[Path]:
    """
    Find the best checkpoint for a model (lowest val_loss).

    Searches through all versions and returns the checkpoint with lowest val_loss.
    """
    log_dir_name = get_log_dir_name(model, variant)
    log_dir = PROJECT_ROOT / "lightning_logs" / log_dir_name

    if not log_dir.exists():
        return None

    checkpoints = list(log_dir.glob("version_*/checkpoints/*.ckpt"))

    if not checkpoints:
        return None

    def extract_val_loss(ckpt_path: Path) -> float:
        name = ckpt_path.stem
        try:
            val_loss_part = name.split("val_loss=")[1]
            return float(val_loss_part)
        except (IndexError, ValueError):
            return float("inf")

    best_checkpoint = min(checkpoints, key=extract_val_loss)
    return best_checkpoint


# =============================================================================
# Summary generation
# =============================================================================

def load_eval_results(variant: str) -> Dict[str, Dict[str, Any]]:
    """Load all evaluation results for a variant."""
    results = {}
    results_dir = get_results_dir(variant)

    for model in MODELS:
        eval_json = results_dir / model.id / "eval.json"
        if eval_json.exists():
            with open(eval_json, "r", encoding="utf-8") as f:
                results[model.id] = json.load(f)

    return results


def generate_comparison_json(variant: str) -> Path:
    """Generate comparison.json with all metrics."""
    results = load_eval_results(variant)
    results_dir = get_results_dir(variant)

    comparison = {
        "variant": variant,
        "generated": datetime.now().isoformat(),
        "models": {}
    }

    for model_id, data in results.items():
        model = MODEL_BY_ID[model_id]
        comparison["models"][model_id] = {
            "name": model.name,
            "type": model.type,
            "parameters": data.get("model", {}).get("parameters"),
            "metrics": data.get("metrics", {}),
            "inference": data.get("inference", {}),
        }

    output_path = results_dir / "comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return output_path


def generate_summary_md(variant: str) -> Path:
    """Generate summary.md with overview table."""
    results = load_eval_results(variant)
    results_dir = get_results_dir(variant)

    # Find best model
    best_model_id = None
    best_r2 = -1
    for model_id, data in results.items():
        r2 = data.get("metrics", {}).get("r2", 0)
        if r2 > best_r2:
            best_r2 = r2
            best_model_id = model_id

    lines = [
        f"# Evaluation Results - {variant.replace('_', ' ').title()}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "| Model | Type | Parameters | R2 | Accuracy | RMSE | Inference (ms) |",
        "|-------|------|------------|-----|----------|------|----------------|",
    ]

    for model in MODELS:
        if model.id not in results:
            continue

        data = results[model.id]
        metrics = data.get("metrics", {})
        inference = data.get("inference", {})
        params = data.get("model", {}).get("parameters", "?")

        r2 = metrics.get("r2", 0)
        accuracy = metrics.get("accuracy", 0)
        rmse = metrics.get("rmse", 0)
        inf_p95 = inference.get("p95_ms", 0)

        # Highlight best model
        if model.id == best_model_id:
            name = f"**{model.name}**"
            r2_str = f"**{r2:.3f}**"
            acc_str = f"**{accuracy:.2f}%**"
        else:
            name = model.name
            r2_str = f"{r2:.3f}"
            acc_str = f"{accuracy:.2f}%"

        lines.append(
            f"| {name} | {model.type} | {params:,} | {r2_str} | {acc_str} | {rmse:.4f} | {inf_p95:.2f} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        f"- **Best Model:** {MODEL_BY_ID[best_model_id].name} (R2={best_r2:.3f})" if best_model_id else "- No results available",
        f"- **Variant:** {'With Dropout (0.2)' if variant == 'dropout' else 'No Dropout'}",
        f"- **Models Evaluated:** {len(results)}/{len(MODELS)}",
        "",
        "## Model Details",
        "",
    ])

    for model in MODELS:
        if model.id not in results:
            continue

        data = results[model.id]
        metrics = data.get("metrics", {})
        inference = data.get("inference", {})

        lines.extend([
            f"### {model.name}",
            "",
            f"- **R2:** {metrics.get('r2', 0):.4f}",
            f"- **Accuracy:** {metrics.get('accuracy', 0):.2f}%",
            f"- **RMSE:** {metrics.get('rmse', 0):.4f}",
            f"- **MAE:** {metrics.get('mae', 0):.4f}",
            f"- **Inference (P95):** {inference.get('p95_ms', 0):.2f} ms",
            f"- **Checkpoint:** `{data.get('model', {}).get('checkpoint', 'N/A')}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        f"*Results directory: `results/{variant}/`*",
    ])

    output_path = results_dir / "summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def generate_summaries(variant: str) -> None:
    """Generate all summary files for a variant."""
    print_subheader("Generating Summary Reports")

    comparison_path = generate_comparison_json(variant)
    print(f"  comparison.json: {comparison_path.relative_to(PROJECT_ROOT)}")

    summary_path = generate_summary_md(variant)
    print(f"  summary.md:      {summary_path.relative_to(PROJECT_ROOT)}")


# =============================================================================
# Output helpers
# =============================================================================

def print_header(text: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_subheader(text: str) -> None:
    """Print a formatted subheader."""
    print()
    print("-" * 40)
    print(text)
    print("-" * 40)


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# =============================================================================
# Commands
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Run training for specified models."""
    variant = args.variant
    model_ids = args.models or [m.id for m in MODELS]

    print_header(f"Training - {variant.replace('_', ' ').title()}")
    print(f"Models: {', '.join(model_ids)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    failed = []

    for i, model_id in enumerate(model_ids, 1):
        if model_id not in MODEL_BY_ID:
            print(f"\nWARNING: Unknown model '{model_id}', skipping...")
            continue

        model = MODEL_BY_ID[model_id]
        config_path = get_config_path(model, variant)

        if not config_path.exists():
            print(f"\nWARNING: Config not found: {config_path}, skipping...")
            failed.append(model_id)
            continue

        print_subheader(f"[{i}/{len(model_ids)}] {model.name}")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_model.py"),
            "--config", str(config_path),
        ]

        success = run_command(cmd, f"Training {model.name}")

        if not success:
            failed.append(model_id)
            if not args.continue_on_error:
                print("\nERROR: Training failed. Use --continue-on-error to continue anyway.")
                return 1

    print_header("Training Complete")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1

    print("All models trained successfully!")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation for specified models."""
    variant = args.variant
    model_ids = args.models or [m.id for m in MODELS]

    print_header(f"Evaluation - {variant.replace('_', ' ').title()}")
    print(f"Models: {', '.join(model_ids)}")
    print(f"Output: results/{variant}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure results directory exists
    results_dir = get_results_dir(variant)
    results_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    evaluated = []

    for i, model_id in enumerate(model_ids, 1):
        if model_id not in MODEL_BY_ID:
            print(f"\nWARNING: Unknown model '{model_id}', skipping...")
            continue

        model = MODEL_BY_ID[model_id]
        config_path = get_config_path(model, variant)
        checkpoint_path = find_best_checkpoint(model, variant)
        output_dir = get_model_output_dir(model, variant)

        print_subheader(f"[{i}/{len(model_ids)}] {model.name}")

        if not config_path.exists():
            print(f"  WARNING: Config not found: {config_path}, skipping...")
            failed.append(model_id)
            continue

        if checkpoint_path is None:
            print(f"  WARNING: No checkpoint found for {model.name}, skipping...")
            failed.append(model_id)
            continue

        print(f"  Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
        print(f"  Output dir: {output_dir.relative_to(PROJECT_ROOT)}/")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
            "--checkpoint", str(checkpoint_path),
            "--config", str(config_path),
            "--output", str(output_dir / "eval.json"),
            "--plots-dir", str(output_dir),
        ]

        if args.save_predictions:
            cmd.append("--save-predictions")

        if args.no_plots:
            cmd.append("--no-plots")

        success = run_command(cmd, f"Evaluating {model.name}")

        if success:
            evaluated.append(model_id)
        else:
            failed.append(model_id)
            if not args.continue_on_error:
                print("\nERROR: Evaluation failed. Use --continue-on-error to continue anyway.")
                return 1

    # Generate summary reports if we have results
    if evaluated:
        generate_summaries(variant)

    print_header("Evaluation Complete")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: results/{variant}/")

    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1

    print("All models evaluated successfully!")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run both training and evaluation."""
    print_header(f"Full Pipeline - {args.variant.replace('_', ' ').title()}")

    result = cmd_train(args)
    if result != 0 and not args.continue_on_error:
        return result

    result = cmd_evaluate(args)
    return result


def cmd_list(args: argparse.Namespace) -> int:
    """List available models and their checkpoints."""
    variant = args.variant

    print_header(f"Available Models - {variant.replace('_', ' ').title()}")

    for model in MODELS:
        config_path = get_config_path(model, variant)
        checkpoint_path = find_best_checkpoint(model, variant)
        output_dir = get_model_output_dir(model, variant)
        eval_exists = (output_dir / "eval.json").exists()

        config_status = "OK" if config_path.exists() else "MISSING"
        eval_status = "OK" if eval_exists else "MISSING"

        print(f"\n{model.id}: {model.name}")
        print(f"  Config:     [{config_status}] {config_path.relative_to(PROJECT_ROOT)}")

        if checkpoint_path:
            val_loss = checkpoint_path.stem.split("val_loss=")[1]
            print(f"  Checkpoint: [OK] val_loss={val_loss}")
            print(f"              {checkpoint_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  Checkpoint: [MISSING] Not found")

        print(f"  Results:    [{eval_status}] results/{variant}/{model.id}/")

    # Summary info
    results_dir = get_results_dir(variant)
    summary_exists = (results_dir / "summary.md").exists()
    comparison_exists = (results_dir / "comparison.json").exists()

    print(f"\nSummary Files:")
    print(f"  summary.md:      [{'OK' if summary_exists else 'MISSING'}]")
    print(f"  comparison.json: [{'OK' if comparison_exists else 'MISSING'}]")

    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified batch runner for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
  results/
  ├── no_dropout/
  │   ├── summary.md          # Auto-generated overview
  │   ├── comparison.json     # All metrics for comparison
  │   ├── m1/
  │   │   ├── eval.json
  │   │   ├── scatter.png
  │   │   └── ...
  │   └── m2/
  └── dropout/
      └── ...

Examples:
  %(prog)s train --variant no_dropout
  %(prog)s train --variant dropout --models m1 m2 m3
  %(prog)s evaluate --variant dropout
  %(prog)s all --variant dropout
  %(prog)s list --variant no_dropout
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--variant", "-v",
            choices=["dropout", "no_dropout"],
            default="no_dropout",
            help="Model variant (default: no_dropout)"
        )
        p.add_argument(
            "--models", "-m",
            nargs="+",
            choices=[m.id for m in MODELS],
            help="Specific models to process (default: all)"
        )
        p.add_argument(
            "--continue-on-error", "-c",
            action="store_true",
            help="Continue processing even if a model fails"
        )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    add_common_args(train_parser)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    add_common_args(eval_parser)
    eval_parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions as CSV"
    )
    eval_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )

    # All command (train + evaluate)
    all_parser = subparsers.add_parser("all", help="Train and evaluate models")
    add_common_args(all_parser)
    all_parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions as CSV"
    )
    all_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available models and checkpoints")
    list_parser.add_argument(
        "--variant", "-v",
        choices=["dropout", "no_dropout"],
        default="no_dropout",
        help="Model variant (default: no_dropout)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "all":
        return cmd_all(args)
    elif args.command == "list":
        return cmd_list(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
