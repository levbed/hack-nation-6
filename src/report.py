from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .features import FeatureTable, feature_tracks


FORBIDDEN_AGGREGATE_KEYS = {
    "participant_id",
    "example_id",
    "predictions",
    "source_start_day",
    "target_start_day",
    "target_end_day",
}


def _finite(values: list[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return clean


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _evidence_label(score: dict[str, Any]) -> str:
    if score["track"] == score.get("reference_track") and score["model"] == score.get("reference_model"):
        return "reference"
    lower = float(score.get("delta_mae_ci_low", math.nan))
    upper = float(score.get("delta_mae_ci_high", math.nan))
    if math.isfinite(upper) and upper < 0:
        return "lower MAE than history in participant bootstrap"
    if math.isfinite(lower) and lower > 0:
        return "higher MAE than history in participant bootstrap"
    return "inconclusive versus history"


def assert_aggregate_payload_safe(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_AGGREGATE_KEYS:
                raise ValueError(f"Participant-level field is not allowed in aggregate payload: {path}.{key}")
            assert_aggregate_payload_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_aggregate_payload_safe(child, f"{path}[{index}]")


def build_benchmark_summary(
    summary: dict[str, Any],
    feature_table: FeatureTable,
    scores: list[dict[str, Any]],
    fold_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    target_values = _finite([row.get("target_cycle_length") for row in feature_table.rows])
    coverage: dict[str, dict[str, float]] = {}
    for feature_name in sorted(
        name
        for name in {key for row in feature_table.rows for key in row}
        if name.endswith("_coverage")
    ):
        values = _finite([row.get(feature_name) for row in feature_table.rows])
        if values:
            coverage[feature_name.removesuffix("_coverage")] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "examples_with_data": len(values),
            }

    track_details: list[dict[str, Any]] = []
    for score in scores:
        model = str(score["model"])
        track = str(score["track"])
        folds = [
            row for row in fold_scores
            if row["model"] == model and row["track"] == track
        ]
        selected_hyperparameters = sorted(
            {
                str(row["selected_hyperparameters"])
                for row in folds
                if row.get("selected_hyperparameters") not in {None, ""}
            }
        )
        feature_counts = [int(row.get("feature_count", 0)) for row in folds]
        track_details.append(
            {
                "model": model,
                "model_label": str(score["model_label"]),
                "track": track,
                "reference_model": str(score["reference_model"]),
                "reference_track": str(score["reference_track"]),
                "n": int(score["n"]),
                "mae": float(score["mae"]),
                "mae_ci_low": float(score.get("mae_ci_low", math.nan)),
                "mae_ci_high": float(score.get("mae_ci_high", math.nan)),
                "rmse": float(score["rmse"]),
                "median_absolute_error": float(score["median_absolute_error"]),
                "within_3_days_pct": float(score["within_3_days_pct"]),
                "within_7_days_pct": float(score["within_7_days_pct"]),
                "mean_signed_error": float(score["mean_signed_error"]),
                "delta_mae_vs_history": float(score["delta_mae_vs_history"]),
                "delta_mae_ci_low": float(score.get("delta_mae_ci_low", math.nan)),
                "delta_mae_ci_high": float(score.get("delta_mae_ci_high", math.nan)),
                "delta_mae_vs_ridge_same_track": float(
                    score.get("delta_mae_vs_ridge_same_track", math.nan)
                ),
                "delta_mae_vs_ridge_ci_low": float(
                    score.get("delta_mae_vs_ridge_ci_low", math.nan)
                ),
                "delta_mae_vs_ridge_ci_high": float(
                    score.get("delta_mae_vs_ridge_ci_high", math.nan)
                ),
                "evidence": _evidence_label(score),
                "selected_hyperparameters": selected_hyperparameters,
                "feature_count_min": min(feature_counts, default=0),
                "feature_count_max": max(feature_counts, default=0),
            }
        )

    payload = {
        "benchmark": str(summary.get("benchmark", "mcPHASES CycleBench")),
        "protocol_version": "2.1",
        "dataset": {
            "id": str(summary.get("dataset_id", "mcphases")),
            "version": str(summary.get("dataset_version", "unknown")),
            "doi": str(summary.get("dataset_doi", "")),
        },
        "dataset_version": str(summary.get("dataset_version", "unknown")),
        "task": "Predict the length in days of cycle t+1 using only information available before it begins.",
        "intended_use": "Exploratory research benchmark; not for diagnosis, fertility planning, treatment, or perimenopause prediction.",
        "cohort_flow": {
            "participants": int(summary["participants"]),
            "inferred_complete_cycles": int(summary["cycles"]),
            "eligible_examples": int(summary["eligible_examples"]),
            "eligibility_exclusions": summary.get("exclusions", {}),
        },
        "target_distribution_days": {
            "minimum": min(target_values, default=math.nan),
            "q1": _percentile(target_values, 0.25),
            "median": statistics.median(target_values) if target_values else math.nan,
            "q3": _percentile(target_values, 0.75),
            "maximum": max(target_values, default=math.nan),
        },
        "evaluation": {
            "outer_split": "participant-disjoint GroupKFold",
            "outer_folds": len({int(row["fold"]) for row in fold_scores}),
            "selection_metric": "inner participant-disjoint GroupKFold MAE",
            "models": [
                {"id": "ridge", "label": "Ridge", "role": "primary linear analysis"},
                {"id": "rbf_svr", "label": "RBF-SVR", "role": "nonlinear sensitivity analysis"},
                {
                    "id": "hist_gradient_boosting",
                    "label": "HistGradientBoosting",
                    "role": "nonlinear sensitivity analysis",
                },
            ],
            "uncertainty": "95% participant-clustered bootstrap intervals with 2,000 replicates",
        },
        "variables_used": feature_table.variables_used,
        "feature_counts_by_track": {
            name: len(features) for name, features in feature_tracks(feature_table).items()
        },
        "source_cycle_coverage": coverage,
        "scores": track_details,
    }
    assert_aggregate_payload_safe(payload)
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    assert_aggregate_payload_safe(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_markdown_report(path: str | Path, payload: dict[str, Any]) -> None:
    assert_aggregate_payload_safe(payload)
    cohort = payload["cohort_flow"]
    target = payload["target_distribution_days"]
    lines = [
        f"# {payload['benchmark']} Report",
        "",
        payload["task"],
        "",
        "## Cohort Flow",
        "",
        f"- Participants: {cohort['participants']}",
        f"- Complete cycles: {cohort['inferred_complete_cycles']}",
        f"- Eligible source/target examples: {cohort['eligible_examples']}",
        f"- Target cycle length: {target['minimum']:.0f}-{target['maximum']:.0f} days "
        f"(median {target['median']:.1f}, IQR {target['q1']:.1f}-{target['q3']:.1f})",
        "",
        "## Results",
        "",
        "| Model | Track | MAE (95% CI) | Delta vs model history (95% CI) | Within 7 days | Evidence |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for score in payload["scores"]:
        lines.append(
            f"| {score['model_label']} | {score['track']} | {score['mae']:.2f} "
            f"({score['mae_ci_low']:.2f}, {score['mae_ci_high']:.2f}) | "
            f"{score['delta_mae_vs_history']:+.3f} "
            f"({score['delta_mae_ci_low']:+.3f}, {score['delta_mae_ci_high']:+.3f}) | "
            f"{score['within_7_days_pct']:.1f}% | {score['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- Outer split: {payload['evaluation']['outer_split']} "
            f"({payload['evaluation']['outer_folds']} folds)",
            f"- Model selection: {payload['evaluation']['selection_metric']}",
            f"- Uncertainty: {payload['evaluation']['uncertainty']}",
            "- Imputation, scaling, feature availability, and model fitting occur inside training folds.",
            "",
            "## Responsible Use",
            "",
            payload["intended_use"],
            "",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
