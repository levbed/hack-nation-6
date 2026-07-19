from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcphases-cyclebench-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2364AA", "#2A9D8F", "#E9C46A", "#E76F51", "#6D597A", "#457B9D", "#8A9A5B", "#B56576"]


def _finish(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def save_mae_by_track(path: str | Path, scores: list[dict[str, Any]]) -> None:
    selected = [row for row in scores if row["model"] in {"ridge", "baseline"}]
    selected.sort(key=lambda row: float(row["mae"]))
    labels = [str(row["track"]).replace("history_plus_", "hist + ").replace("_", " ") for row in selected]
    values = [float(row["mae"]) for row in selected]
    lower = [max(0.0, value - float(row.get("mae_ci_low", value))) for value, row in zip(values, selected)]
    upper = [max(0.0, float(row.get("mae_ci_high", value)) - value) for value, row in zip(values, selected)]
    positions = list(range(len(selected)))

    plt.figure(figsize=(9.5, max(4.8, 0.62 * len(selected))))
    plt.barh(positions, values, color=[COLORS[index % len(COLORS)] for index in positions], alpha=0.9)
    plt.errorbar(values, positions, xerr=[lower, upper], fmt="none", ecolor="#20242A", capsize=3, linewidth=1.2)
    plt.yticks(positions, labels)
    plt.xlabel("Mean absolute error (days), 95% participant-bootstrap CI")
    plt.title("Primary Ridge analysis and baselines")
    plt.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    plt.gca().set_axisbelow(True)
    plt.gca().invert_yaxis()
    _finish(path)


def save_delta_mae_vs_history(path: str | Path, scores: list[dict[str, Any]]) -> None:
    selected = [
        row for row in scores
        if row["model"] != "baseline" and row["track"] != "history_only"
    ]
    selected.sort(key=lambda row: (str(row["model"]), float(row["delta_mae_vs_history"])))
    labels = [
        f"{row['model_label']} | "
        f"{str(row['track']).replace('history_plus_', 'hist + ').replace('_', ' ')}"
        for row in selected
    ]
    values = [float(row["delta_mae_vs_history"]) for row in selected]
    lower = [value - float(row["delta_mae_ci_low"]) for value, row in zip(values, selected)]
    upper = [float(row["delta_mae_ci_high"]) - value for value, row in zip(values, selected)]
    positions = list(range(len(selected)))

    plt.figure(figsize=(9.5, max(4.8, 0.62 * len(selected))))
    if not selected:
        plt.text(0.5, 0.5, "No optional feature tracks available", ha="center", va="center")
        plt.axis("off")
        plt.title("Incremental value over history-only")
        _finish(path)
        return
    colors = ["#2A9D8F" if value < 0 else "#E76F51" for value in values]
    plt.barh(positions, values, color=colors, alpha=0.9)
    plt.errorbar(values, positions, xerr=[lower, upper], fmt="none", ecolor="#20242A", capsize=3, linewidth=1.2)
    plt.axvline(0.0, color="#20242A", linewidth=1)
    plt.yticks(positions, labels)
    plt.xlabel("MAE difference versus history only (days); lower is better")
    plt.title("Incremental value over each model's history-only track")
    plt.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    plt.gca().set_axisbelow(True)
    plt.gca().invert_yaxis()
    _finish(path)


def save_predicted_vs_observed(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    selected = [
        row for row in predictions
        if row["model"] == "ridge" and row["track"] == "full_multimodal"
    ]
    if not selected:
        selected = [
            row for row in predictions
            if row["model"] == "ridge" and row["track"] == "history_only"
        ]
    title = (
        "Ridge full multimodal: predicted vs observed"
        if selected and selected[0]["track"] == "full_multimodal"
        else "Ridge history-only: predicted vs observed"
    )
    observed = [float(row["observed_cycle_length"]) for row in selected]
    predicted = [float(row["predicted_cycle_length"]) for row in selected]
    minimum = min(observed + predicted, default=10.0) - 2
    maximum = max(observed + predicted, default=40.0) + 2

    plt.figure(figsize=(6.6, 6.2))
    plt.scatter(observed, predicted, color="#2364AA", alpha=0.65, edgecolor="white", linewidth=0.4)
    plt.plot([minimum, maximum], [minimum, maximum], color="#E76F51", linewidth=1.4, label="Perfect prediction")
    plt.xlim(minimum, maximum)
    plt.ylim(minimum, maximum)
    plt.xlabel("Observed cycle length (days)")
    plt.ylabel("Predicted cycle length (days)")
    plt.title(title)
    plt.grid(color="#D9DEE3", linewidth=0.7)
    plt.legend(frameon=False)
    _finish(path)


def save_target_distribution(path: str | Path, target_values: list[float]) -> None:
    plt.figure(figsize=(7.8, 4.8))
    plt.hist(target_values, bins=min(16, max(5, len(set(target_values)))), color="#457B9D", edgecolor="white")
    plt.xlabel("Target cycle length (days)")
    plt.ylabel("Eligible examples")
    plt.title("Target distribution")
    plt.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    plt.gca().set_axisbelow(True)
    _finish(path)


def save_model_track_heatmap(path: str | Path, scores: list[dict[str, Any]]) -> None:
    selected = [row for row in scores if row["model"] != "baseline"]
    model_order = ["ridge", "rbf_svr", "hist_gradient_boosting"]
    models = [model for model in model_order if any(row["model"] == model for row in selected)]
    tracks = sorted({str(row["track"]) for row in selected})
    lookup = {(str(row["model"]), str(row["track"])): float(row["mae"]) for row in selected}
    matrix = np.asarray([[lookup.get((model, track), np.nan) for track in tracks] for model in models])

    plt.figure(figsize=(max(9.0, 1.35 * len(tracks)), 4.8))
    image = plt.imshow(matrix, cmap="YlGnBu_r", aspect="auto")
    plt.colorbar(image, label="MAE (days; lower is better)")
    plt.xticks(
        range(len(tracks)),
        [track.replace("history_plus_", "hist + ").replace("_", " ") for track in tracks],
        rotation=28,
        ha="right",
    )
    labels = {str(row["model"]): str(row["model_label"]) for row in selected}
    plt.yticks(range(len(models)), [labels[model] for model in models])
    for row_index in range(len(models)):
        for column_index in range(len(tracks)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9)
    plt.title("Model-by-track next-cycle length error")
    _finish(path)


def save_model_delta_vs_ridge(path: str | Path, scores: list[dict[str, Any]]) -> None:
    selected = [
        row for row in scores
        if row["model"] not in {"ridge", "baseline"}
    ]
    selected.sort(key=lambda row: (str(row["model"]), float(row["delta_mae_vs_ridge_same_track"])))
    labels = [
        f"{row['model_label']} | "
        f"{str(row['track']).replace('history_plus_', 'hist + ').replace('_', ' ')}"
        for row in selected
    ]
    values = [float(row["delta_mae_vs_ridge_same_track"]) for row in selected]
    lower = [value - float(row["delta_mae_vs_ridge_ci_low"]) for value, row in zip(values, selected)]
    upper = [float(row["delta_mae_vs_ridge_ci_high"]) - value for value, row in zip(values, selected)]
    positions = list(range(len(selected)))

    plt.figure(figsize=(10.5, max(5.2, 0.52 * len(selected))))
    colors = ["#2A9D8F" if value < 0 else "#E76F51" for value in values]
    plt.barh(positions, values, color=colors, alpha=0.9)
    plt.errorbar(values, positions, xerr=[lower, upper], fmt="none", ecolor="#20242A", capsize=3)
    plt.axvline(0.0, color="#20242A", linewidth=1)
    plt.yticks(positions, labels)
    plt.xlabel("MAE difference versus Ridge on the same track (days); lower is better")
    plt.title("Nonlinear model comparison")
    plt.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    plt.gca().set_axisbelow(True)
    plt.gca().invert_yaxis()
    _finish(path)
