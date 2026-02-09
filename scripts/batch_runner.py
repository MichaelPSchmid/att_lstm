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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from shared library
from scripts.shared import (
    MODEL_BY_ID,
    MODELS,
    PROJECT_ROOT,
    find_all_seed_checkpoints,
    find_best_checkpoint,
    find_latest_checkpoint_for_resume,
    get_config_path,
    get_model_output_dir,
    get_results_dir,
    load_eval_results,
)


def aggregate_seed_results(seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation results from multiple seeds.

    Args:
        seed_results: Dictionary mapping seed -> eval.json contents

    Returns:
        Aggregated results with mean and std for each metric
    """
    import numpy as np

    if not seed_results:
        return {}

    seeds = sorted(seed_results.keys())
    first_result = seed_results[seeds[0]]

    # Start with model info from first seed (same for all)
    aggregated = {
        "model": first_result.get("model", {}),
        "data": first_result.get("data", {}),
        "seeds": seeds,
        "num_seeds": len(seeds),
    }

    # Aggregate metrics
    metric_keys = first_result.get("metrics", {}).keys()
    metrics_aggregated = {}

    for key in metric_keys:
        values = [seed_results[s].get("metrics", {}).get(key) for s in seeds]
        values = [v for v in values if v is not None]
        if values:
            metrics_aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    aggregated["metrics"] = metrics_aggregated

    # Aggregate inference times
    inference_keys = first_result.get("inference", {}).keys()
    inference_aggregated = {}

    for key in inference_keys:
        values = [seed_results[s].get("inference", {}).get(key) for s in seeds]
        values = [v for v in values if v is not None]
        if values:
            if isinstance(values[0], (int, float)):
                inference_aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            else:
                inference_aggregated[key] = values[0]  # Non-numeric, take first

    aggregated["inference"] = inference_aggregated

    # Keep per-seed results for reference
    aggregated["per_seed"] = {str(s): seed_results[s] for s in seeds}

    return aggregated


def generate_comparison_json(variant: str) -> Path:
    """Generate comparison.json with all metrics (supports multi-seed)."""
    results = load_eval_results(variant)
    results_dir = get_results_dir(variant)

    comparison = {
        "variant": variant,
        "generated": datetime.now().isoformat(),
        "models": {}
    }

    for model_id, data in results.items():
        model = MODEL_BY_ID[model_id]
        model_data = data.get("model", {})
        comparison["models"][model_id] = {
            "name": model.name,
            "type": model.type,
            "parameters": model_data.get("parameters"),
            "flops": model_data.get("flops"),
            "flops_formatted": model_data.get("flops_formatted"),
            "macs": model_data.get("macs"),
            "macs_formatted": model_data.get("macs_formatted"),
            "metrics": data.get("metrics", {}),
            "inference": data.get("inference", {}),
            "num_seeds": data.get("num_seeds", 1),
            "seeds": data.get("seeds", []),
        }

    output_path = results_dir / "comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return output_path


def _get_metric_value(metrics: Dict[str, Any], key: str, default: float = 0) -> float:
    """Extract metric value, handling both single values and aggregated {mean, std} format."""
    val = metrics.get(key, default)
    if isinstance(val, dict):
        return val.get("mean", default)
    return val if val is not None else default


def _format_metric(metrics: Dict[str, Any], key: str, fmt: str = ".4f", suffix: str = "") -> str:
    """Format metric for display, showing mean +/- std if aggregated."""
    val = metrics.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, dict):
        mean = val.get("mean", 0)
        std = val.get("std", 0)
        return f"{mean:{fmt}}{suffix} +/- {std:{fmt}}"
    return f"{val:{fmt}}{suffix}"


def generate_summary_md(variant: str) -> Path:
    """Generate summary.md with overview table (supports multi-seed aggregation)."""
    results = load_eval_results(variant)
    results_dir = get_results_dir(variant)

    # Find best model (using mean R2 for aggregated results)
    best_model_id = None
    best_r2 = -1
    for model_id, data in results.items():
        r2 = _get_metric_value(data.get("metrics", {}), "r2", 0)
        if r2 > best_r2:
            best_r2 = r2
            best_model_id = model_id

    # Check if any results have multiple seeds
    has_multi_seed = any(data.get("num_seeds", 1) > 1 for data in results.values())

    lines = [
        f"# Evaluation Results - {variant.replace('_', ' ').title()}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if has_multi_seed:
        lines.append("> Results show mean +/- std across multiple seeds")

    lines.extend([
        "",
        "## Overview",
        "",
        "| Model | Type | Params | R2 | Accuracy | RMSE | Seeds |",
        "|-------|------|--------|-----|----------|------|-------|",
    ])

    for model in MODELS:
        if model.id not in results:
            continue

        data = results[model.id]
        metrics = data.get("metrics", {})
        model_data = data.get("model", {})
        params = model_data.get("parameters", "?")
        num_seeds = data.get("num_seeds", 1)

        r2 = _get_metric_value(metrics, "r2", 0)
        accuracy = _get_metric_value(metrics, "accuracy", 0)
        rmse = _get_metric_value(metrics, "rmse", 0)

        # Format with std if multi-seed
        if num_seeds > 1:
            r2_str = _format_metric(metrics, "r2", ".3f")
            acc_str = _format_metric(metrics, "accuracy", ".1f", "%")
            rmse_str = _format_metric(metrics, "rmse", ".4f")
        else:
            r2_str = f"{r2:.3f}"
            acc_str = f"{accuracy:.1f}%"
            rmse_str = f"{rmse:.4f}"

        # Highlight best model
        if model.id == best_model_id:
            name = f"**{model.name}**"
            r2_str = f"**{r2_str}**"
            acc_str = f"**{acc_str}**"

        else:
            name = model.name

        lines.append(
            f"| {name} | {model.type} | {params:,} | {r2_str} | {acc_str} | {rmse_str} | {num_seeds} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        f"- **Best Model:** {MODEL_BY_ID[best_model_id].name} (R2={best_r2:.3f})" if best_model_id else "- No results available",
        f"- **Variant:** {'With Dropout (0.2)' if variant == 'dropout' else 'No Dropout'}",
        f"- **Models Evaluated:** {len(results)}/{len(MODELS)}",
    ])

    if has_multi_seed:
        lines.append("- **Note:** Results aggregated across multiple random seeds")

    lines.extend([
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
        model_data = data.get("model", {})
        num_seeds = data.get("num_seeds", 1)
        seeds = data.get("seeds", [])

        lines.extend([
            f"### {model.name}",
            "",
            f"- **Parameters:** {model_data.get('parameters', 'N/A'):,}",
            f"- **FLOPs:** {model_data.get('flops_formatted', 'N/A')}",
            f"- **MACs:** {model_data.get('macs_formatted', 'N/A')}",
            f"- **R2:** {_format_metric(metrics, 'r2', '.4f')}",
            f"- **Accuracy:** {_format_metric(metrics, 'accuracy', '.2f', '%')}",
            f"- **RMSE:** {_format_metric(metrics, 'rmse', '.4f')}",
            f"- **MAE:** {_format_metric(metrics, 'mae', '.4f')}",
        ])

        # Inference time handling
        inf_p95 = inference.get("p95_ms")
        if isinstance(inf_p95, dict):
            lines.append(f"- **Inference (P95):** {inf_p95.get('mean', 0):.2f} +/- {inf_p95.get('std', 0):.2f} ms")
        elif inf_p95 is not None:
            lines.append(f"- **Inference (P95):** {inf_p95:.2f} ms")

        if num_seeds > 1:
            lines.append(f"- **Seeds:** {seeds}")
        else:
            lines.append(f"- **Checkpoint:** `{model_data.get('checkpoint', 'N/A')}`")

        lines.append("")

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
    auto_resume = getattr(args, 'auto_resume', False)

    print_header(f"Training - {variant.replace('_', ' ').title()}")
    print(f"Models: {', '.join(model_ids)}")
    if auto_resume:
        print(f"Auto-resume: ENABLED (will continue from existing checkpoints)")
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

        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])

        # Check for existing checkpoint to resume from
        if auto_resume:
            resume_info = find_latest_checkpoint_for_resume(model, variant, seed=args.seed)
            if resume_info is not None:
                ckpt_path, epoch, val_loss = resume_info
                print(f"  Found checkpoint to resume: epoch {epoch}, val_loss={val_loss:.6f}")
                print(f"  Path: {ckpt_path.relative_to(PROJECT_ROOT)}")
                cmd.extend(["--resume", str(ckpt_path)])

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
    """Run evaluation for specified models (with multi-seed support)."""
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
        output_dir = get_model_output_dir(model, variant)

        # Find all seed variants for this model
        seed_checkpoints = find_all_seed_checkpoints(model, variant)

        print_subheader(f"[{i}/{len(model_ids)}] {model.name}")

        if not config_path.exists():
            print(f"  WARNING: Config not found: {config_path}, skipping...")
            failed.append(model_id)
            continue

        if not seed_checkpoints:
            print(f"  WARNING: No checkpoints found for {model.name}, skipping...")
            failed.append(model_id)
            continue

        seeds = sorted(seed_checkpoints.keys())
        print(f"  Found {len(seeds)} seed(s): {seeds}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_results: Dict[int, Dict[str, Any]] = {}
        seed_failed = False

        for seed in seeds:
            checkpoint_path, val_loss = seed_checkpoints[seed]
            seed_suffix = f"_seed{seed}" if seed != 42 else ""

            print(f"\n  Seed {seed}:")
            print(f"    Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
            print(f"    Val Loss: {val_loss:.6f}")

            # Output path for this seed
            if len(seeds) > 1:
                seed_output = output_dir / f"eval_seed{seed}.json"
                seed_plots_dir = output_dir / f"seed_{seed}"
                seed_plots_dir.mkdir(parents=True, exist_ok=True)
            else:
                seed_output = output_dir / "eval.json"
                seed_plots_dir = output_dir

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
                "--checkpoint", str(checkpoint_path),
                "--config", str(config_path),
                "--output", str(seed_output),
                "--plots-dir", str(seed_plots_dir),
            ]

            if args.save_predictions:
                cmd.append("--save-predictions")

            if args.no_plots:
                cmd.append("--no-plots")

            success = run_command(cmd, f"Evaluating {model.name} (seed {seed})")

            if success and seed_output.exists():
                with open(seed_output, "r", encoding="utf-8") as f:
                    seed_results[seed] = json.load(f)
            elif not success:
                seed_failed = True
                if not args.continue_on_error:
                    print("\nERROR: Evaluation failed. Use --continue-on-error to continue anyway.")
                    return 1

        # Aggregate results if multiple seeds
        if len(seed_results) > 1:
            print(f"\n  Aggregating {len(seed_results)} seed results...")
            aggregated = aggregate_seed_results(seed_results)
            aggregated_path = output_dir / "eval.json"
            with open(aggregated_path, "w", encoding="utf-8") as f:
                json.dump(aggregated, f, indent=2)
            print(f"  Aggregated results: {aggregated_path.relative_to(PROJECT_ROOT)}")

            # Print summary
            metrics = aggregated.get("metrics", {})
            if "r2" in metrics:
                r2 = metrics["r2"]
                print(f"  R2: {r2['mean']:.4f} +/- {r2['std']:.4f}")
            if "accuracy" in metrics:
                acc = metrics["accuracy"]
                print(f"  Accuracy: {acc['mean']:.2f}% +/- {acc['std']:.2f}%")

        if seed_failed:
            failed.append(model_id)
        else:
            evaluated.append(model_id)

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
    """List available models and their checkpoints (with seed variants)."""
    variant = args.variant

    print_header(f"Available Models - {variant.replace('_', ' ').title()}")

    for model in MODELS:
        config_path = get_config_path(model, variant)
        output_dir = get_model_output_dir(model, variant)
        eval_exists = (output_dir / "eval.json").exists()

        # Find all seed variants
        seed_checkpoints = find_all_seed_checkpoints(model, variant)

        config_status = "OK" if config_path.exists() else "MISSING"
        eval_status = "OK" if eval_exists else "MISSING"

        print(f"\n{model.id}: {model.name}")
        print(f"  Config:     [{config_status}] {config_path.relative_to(PROJECT_ROOT)}")

        if seed_checkpoints:
            seeds = sorted(seed_checkpoints.keys())
            print(f"  Seeds:      {len(seeds)} trained: {seeds}")
            for seed in seeds:
                ckpt_path, val_loss = seed_checkpoints[seed]
                seed_label = f"seed {seed}" if seed != 42 else "seed 42 (default)"
                print(f"    - {seed_label}: val_loss={val_loss:.6f}")
        else:
            print(f"  Checkpoint: [MISSING] Not found")

        print(f"  Results:    [{eval_status}] results/{variant}/{model.id}/")

        # Show per-seed eval files if they exist
        seed_evals = list(output_dir.glob("eval_seed*.json"))
        if seed_evals:
            print(f"              Per-seed evals: {len(seed_evals)}")

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
        p.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed (passed to train_model.py). Appends '_seedX' to model name."
        )

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--auto-resume", "-r",
            action="store_true",
            help="Automatically resume from existing checkpoints if available"
        )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    add_common_args(train_parser)
    add_train_args(train_parser)

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
    add_train_args(all_parser)
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
