#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    DEFAULT_MAX_CYCLE_LENGTH,
    DEFAULT_MIN_CYCLE_LENGTH,
    build_cycle_examples,
    count_by_participant,
    load_mcphases_data,
    summarize_loaded_data,
)
from src.features import build_feature_table
from src.openai_report import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    load_aggregate_summary,
    summarize_with_openai,
    write_openai_outputs,
)
from src.report import build_benchmark_summary, write_json, write_markdown_report


def _print_summary(summary: dict) -> None:
    print(f"Participants: {summary['participants']}")
    print(f"Inferred complete cycles: {summary['cycles']}")
    print(f"Eligible source/target examples: {summary['eligible_examples']}")
    print("Tables inspected:")
    for table, rows in sorted(summary["table_rows"].items()):
        detail = f"{rows} rows" if rows is not None else "schema only; row count skipped"
        print(f"  - {table}: {detail}")
    print("Variables used:")
    if summary["tables_used"]:
        for table, variables in sorted(summary["tables_used"].items()):
            print(f"  - {table}: {', '.join(variables)}")
    else:
        print("  - none")


def _load_and_build(args: argparse.Namespace):
    loaded = load_mcphases_data(args.data_dir)
    examples = build_cycle_examples(
        loaded.flow_rows,
        min_cycle_length=args.min_cycle_length,
        max_cycle_length=args.max_cycle_length,
    )
    feature_table = build_feature_table(examples, loaded.measurements)
    summary = summarize_loaded_data(loaded, examples)
    return loaded, examples, feature_table, summary


def inspect_command(args: argparse.Namespace) -> None:
    _, examples, _, summary = _load_and_build(args)
    _print_summary(summary)
    counts = count_by_participant(examples)
    if counts:
        print("Eligible examples per participant:")
        print(f"  min={min(counts.values())}, median={sorted(counts.values())[len(counts)//2]}, max={max(counts.values())}")


def evaluate_command(args: argparse.Namespace) -> None:
    from src.evaluate import evaluate_feature_table, write_csv
    from src.plots import (
        save_delta_mae_vs_history,
        save_mae_by_track,
        save_model_delta_vs_ridge,
        save_model_track_heatmap,
        save_predicted_vs_observed,
        save_target_distribution,
    )

    _, examples, feature_table, summary = _load_and_build(args)
    _print_summary(summary)
    print(f"Feature variables used: {feature_table.variables_used}")
    scores, fold_scores, predictions = evaluate_feature_table(feature_table)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "scores.csv", scores)
    write_csv(output_dir / "fold_scores.csv", fold_scores)
    write_csv(output_dir / "predictions.csv", predictions)
    save_mae_by_track(output_dir / "mae_by_track.png", scores)
    save_delta_mae_vs_history(output_dir / "mae_delta_vs_history.png", scores)
    save_model_track_heatmap(output_dir / "model_track_heatmap.png", scores)
    save_model_delta_vs_ridge(output_dir / "model_delta_vs_ridge.png", scores)
    save_predicted_vs_observed(output_dir / "predicted_vs_observed.png", predictions)
    save_target_distribution(
        output_dir / "target_distribution.png",
        [float(row["target_cycle_length"]) for row in feature_table.rows],
    )
    aggregate_summary = build_benchmark_summary(summary, feature_table, scores, fold_scores)
    write_json(output_dir / "benchmark_summary.json", aggregate_summary)
    write_markdown_report(output_dir / "benchmark_report.md", aggregate_summary)
    print(f"Saved results to {output_dir.resolve()}")
    print("Scores:")
    for row in scores:
        print(
            f"  - {row['model_label']} / {row['track']}: MAE={float(row['mae']):.3f}, "
            f"95% CI=({float(row['mae_ci_low']):.3f}, {float(row['mae_ci_high']):.3f}), "
            f"delta_vs_history={float(row['delta_mae_vs_history']):+.3f}, "
            f"within_7_days={float(row['within_7_days_pct']):.1f}%"
        )


def export_public_command(args: argparse.Namespace) -> None:
    source = Path(args.results_dir) / "benchmark_summary.json"
    payload = load_aggregate_summary(source)
    write_json(args.output_file, payload)
    print(f"Saved privacy-checked aggregate artifact to {Path(args.output_file).resolve()}")


def summarize_command(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    payload = load_aggregate_summary(results_dir / "benchmark_summary.json")
    try:
        result = summarize_with_openai(
            payload,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI summary failed: {exc}") from exc
    write_openai_outputs(
        results_dir / "openai_report.json",
        results_dir / "openai_report.md",
        result,
    )
    print(f"Saved aggregate-only OpenAI report to {(results_dir / 'openai_report.md').resolve()}")
    print(f"Model: {result['model']}")
    if result.get("request_id"):
        print(f"Request ID: {result['request_id']}")
    if result.get("usage"):
        print(f"Usage: {result['usage']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mcPHASES CycleBench")
    parser.add_argument("--min-cycle-length", type=int, default=DEFAULT_MIN_CYCLE_LENGTH)
    parser.add_argument("--max-cycle-length", type=int, default=DEFAULT_MAX_CYCLE_LENGTH)
    parser.add_argument("--output-dir", default="results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect local mcPHASES data and inferred examples.")
    inspect.add_argument("--data-dir", required=True)
    inspect.set_defaults(func=inspect_command)

    evaluate = subparsers.add_parser("evaluate", help="Run participant-disjoint benchmark on local mcPHASES data.")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.set_defaults(func=evaluate_command)

    summarize = subparsers.add_parser(
        "summarize",
        help="Use OpenAI to interpret aggregate benchmark_summary.json only.",
    )
    summarize.add_argument("--results-dir", default="results")
    summarize.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    summarize.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    summarize.set_defaults(func=summarize_command)

    export_public = subparsers.add_parser(
        "export-public",
        help="Export a validated aggregate summary for static publication.",
    )
    export_public.add_argument("--results-dir", default="results")
    export_public.add_argument("--output-file", default="docs/data/benchmark_summary.json")
    export_public.set_defaults(func=export_public_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ModuleNotFoundError as exc:
        parser.error(
            f"Missing dependency '{exc.name}'. Activate the project environment with "
            "'source .venv/bin/activate', or install dependencies with "
            "'python3 -m pip install -r requirements.txt'."
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
