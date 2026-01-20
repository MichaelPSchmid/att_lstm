"""
Compare evaluation results from multiple models.

Reads JSON result files and generates comparison tables in Markdown and LaTeX format.

Usage:
    python scripts/compare_results.py results/m1_results.json results/m2_results.json
    python scripts/compare_results.py results/*.json --output results/comparison.md
    python scripts/compare_results.py results/*.json --latex --output results/comparison.tex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from multiple models"
    )

    parser.add_argument(
        "result_files",
        type=str,
        nargs="+",
        help="JSON result files to compare"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output as LaTeX table"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="rmse",
        choices=["rmse", "mse", "mae", "r2", "accuracy", "inference", "params"],
        help="Sort results by this metric (default: rmse)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: best first)"
    )

    return parser.parse_args()


def load_results(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load results from JSON files.

    Args:
        file_paths: List of paths to JSON result files

    Returns:
        List of result dictionaries
    """
    results = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = path
                results.append(data)
        except FileNotFoundError:
            print(f"Warning: File not found: {path}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {path}: {e}", file=sys.stderr)

    return results


def sort_results(
    results: List[Dict[str, Any]],
    sort_by: str,
    ascending: bool = False
) -> List[Dict[str, Any]]:
    """
    Sort results by specified metric.

    Args:
        results: List of result dictionaries
        sort_by: Metric to sort by
        ascending: If True, sort ascending; otherwise best first

    Returns:
        Sorted list of results
    """
    def get_sort_key(r: Dict) -> float:
        if sort_by == "inference":
            if r.get("inference") is None:
                return float("inf")
            return r["inference"]["mean_ms"]
        elif sort_by == "params":
            return r["model"]["parameters"]
        elif sort_by in ["r2", "accuracy"]:
            # Higher is better for these metrics
            return -r["metrics"][sort_by] if not ascending else r["metrics"][sort_by]
        else:
            # Lower is better for mse, rmse, mae
            return r["metrics"][sort_by]

    reverse = not ascending and sort_by not in ["r2", "accuracy"]
    return sorted(results, key=get_sort_key, reverse=False if ascending else False)


def format_number(value: float, precision: int = 4) -> str:
    """Format number with specified precision."""
    if abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate comparison table in Markdown format.

    Args:
        results: List of result dictionaries

    Returns:
        Markdown table string
    """
    lines = []

    # Header
    lines.append("# Model Comparison Results\n")
    lines.append("## Metrics Comparison\n")

    # Table header
    lines.append("| Model | Type | Params | MSE | RMSE | MAE | R² | Acc (%) |")
    lines.append("|-------|------|--------|-----|------|-----|-------|---------|")

    # Find best values for highlighting
    best_mse = min(r["metrics"]["mse"] for r in results)
    best_rmse = min(r["metrics"]["rmse"] for r in results)
    best_mae = min(r["metrics"]["mae"] for r in results)
    best_r2 = max(r["metrics"]["r2"] for r in results)
    best_acc = max(r["metrics"]["accuracy"] for r in results)

    # Data rows
    for r in results:
        m = r["metrics"]
        model = r["model"]

        # Add bold for best values
        mse_str = f"**{format_number(m['mse'])}**" if m["mse"] == best_mse else format_number(m["mse"])
        rmse_str = f"**{format_number(m['rmse'])}**" if m["rmse"] == best_rmse else format_number(m["rmse"])
        mae_str = f"**{format_number(m['mae'])}**" if m["mae"] == best_mae else format_number(m["mae"])
        r2_str = f"**{format_number(m['r2'])}**" if m["r2"] == best_r2 else format_number(m["r2"])
        acc_str = f"**{m['accuracy']:.2f}**" if m["accuracy"] == best_acc else f"{m['accuracy']:.2f}"

        params_str = f"{model['parameters']:,}"

        lines.append(
            f"| {model['name']} | {model['type']} | {params_str} | "
            f"{mse_str} | {rmse_str} | {mae_str} | {r2_str} | {acc_str} |"
        )

    # Inference time table (if available)
    results_with_inference = [r for r in results if r.get("inference") is not None]
    if results_with_inference:
        lines.append("\n## Inference Time (CPU)\n")
        lines.append("| Model | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) | Target (<10ms) |")
        lines.append("|-------|-----------|----------|----------|----------|----------------|")

        best_mean = min(r["inference"]["mean_ms"] for r in results_with_inference)

        for r in results_with_inference:
            inf = r["inference"]
            model = r["model"]

            mean_str = f"**{inf['mean_ms']:.3f}**" if inf["mean_ms"] == best_mean else f"{inf['mean_ms']:.3f}"
            meets_target = "✓" if inf["mean_ms"] < 10.0 else "✗"

            lines.append(
                f"| {model['name']} | {mean_str} | {inf['std_ms']:.3f} | "
                f"{inf['p95_ms']:.3f} | {inf['p99_ms']:.3f} | {meets_target} |"
            )

    # Summary statistics
    lines.append("\n## Summary\n")
    lines.append(f"- **Number of models:** {len(results)}")
    lines.append(f"- **Best RMSE:** {best_rmse:.6f}")
    lines.append(f"- **Best R²:** {best_r2:.6f}")
    lines.append(f"- **Best Accuracy:** {best_acc:.2f}%")

    if results_with_inference:
        best_inference = min(r["inference"]["mean_ms"] for r in results_with_inference)
        lines.append(f"- **Fastest Inference:** {best_inference:.3f} ms")

    return "\n".join(lines)


def generate_latex_table(results: List[Dict[str, Any]]) -> str:
    """
    Generate comparison table in LaTeX format.

    Args:
        results: List of result dictionaries

    Returns:
        LaTeX table string
    """
    lines = []

    # Metrics table
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model Performance Comparison}")
    lines.append("\\label{tab:model-comparison}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Type & Params & MSE & RMSE & MAE & $R^2$ & Acc (\\%) \\\\")
    lines.append("\\midrule")

    # Find best values for highlighting
    best_mse = min(r["metrics"]["mse"] for r in results)
    best_rmse = min(r["metrics"]["rmse"] for r in results)
    best_mae = min(r["metrics"]["mae"] for r in results)
    best_r2 = max(r["metrics"]["r2"] for r in results)
    best_acc = max(r["metrics"]["accuracy"] for r in results)

    for r in results:
        m = r["metrics"]
        model = r["model"]

        # Use textbf for best values
        mse_str = f"\\textbf{{{format_number(m['mse'])}}}" if m["mse"] == best_mse else format_number(m["mse"])
        rmse_str = f"\\textbf{{{format_number(m['rmse'])}}}" if m["rmse"] == best_rmse else format_number(m["rmse"])
        mae_str = f"\\textbf{{{format_number(m['mae'])}}}" if m["mae"] == best_mae else format_number(m["mae"])
        r2_str = f"\\textbf{{{format_number(m['r2'])}}}" if m["r2"] == best_r2 else format_number(m["r2"])
        acc_str = f"\\textbf{{{m['accuracy']:.2f}}}" if m["accuracy"] == best_acc else f"{m['accuracy']:.2f}"

        # Escape underscores in model name
        model_name = model["name"].replace("_", "\\_")
        model_type = model["type"].replace("_", "\\_")

        lines.append(
            f"{model_name} & {model_type} & {model['parameters']:,} & "
            f"{mse_str} & {rmse_str} & {mae_str} & {r2_str} & {acc_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Inference time table
    results_with_inference = [r for r in results if r.get("inference") is not None]
    if results_with_inference:
        lines.append("")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{CPU Inference Time Comparison}")
        lines.append("\\label{tab:inference-time}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Model & Mean (ms) & Std (ms) & P95 (ms) & P99 (ms) & Target \\\\")
        lines.append("\\midrule")

        for r in results_with_inference:
            inf = r["inference"]
            model_name = r["model"]["name"].replace("_", "\\_")
            meets_target = "\\checkmark" if inf["mean_ms"] < 10.0 else "$\\times$"

            lines.append(
                f"{model_name} & {inf['mean_ms']:.3f} & {inf['std_ms']:.3f} & "
                f"{inf['p95_ms']:.3f} & {inf['p99_ms']:.3f} & {meets_target} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Load results
    results = load_results(args.result_files)

    if not results:
        print("Error: No valid result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} result files", file=sys.stderr)

    # Sort results
    results = sort_results(results, args.sort_by, args.ascending)

    # Generate output
    if args.latex:
        output = generate_latex_table(results)
    else:
        output = generate_markdown_table(results)

    # Write or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"Output written to: {output_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
