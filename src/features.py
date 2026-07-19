from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .data import CycleExample


HISTORY_FEATURES = [
    "history_previous_cycle_length",
    "history_mean_cycle_length",
    "history_median_cycle_length",
    "history_std_cycle_length",
    "history_prior_cycle_count",
]

WEARABLE_METRICS = [
    "sleep_minutes_asleep",
    "sleep_minutes_awake",
    "sleep_time_in_bed",
    "sleep_efficiency",
    "resting_heart_rate",
    "steps",
    "active_minutes",
]

HORMONE_METRICS = [
    "hormone_e3g",
    "hormone_lh",
    "hormone_pdg",
    "hormone_fsh",
]

SYMPTOM_METRICS = [
    "selfreport_appetite",
    "selfreport_exercise_level",
    "selfreport_headaches",
    "selfreport_cramps",
    "selfreport_sore_breasts",
    "selfreport_fatigue",
    "selfreport_sleep_issue",
    "selfreport_mood_swing",
    "selfreport_stress",
    "selfreport_food_cravings",
    "selfreport_indigestion",
    "selfreport_bloating",
]

METABOLIC_STRESS_METRICS = [
    "glucose",
    "wearable_stress_score",
]


@dataclass
class FeatureTable:
    rows: list[dict[str, Any]]
    history_features: list[str]
    wearable_features: list[str]
    hormone_features: list[str]
    symptom_features: list[str]
    metabolic_stress_features: list[str]
    variables_used: dict[str, list[str]]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else math.nan
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def _add_stats(row: dict[str, Any], prefix: str, values: list[float], covered_days: set[int], cycle_days: int) -> list[str]:
    if not values:
        return []
    names = [
        f"{prefix}_mean",
        f"{prefix}_max",
        f"{prefix}_std",
        f"{prefix}_count",
        f"{prefix}_coverage",
    ]
    row[names[0]] = _mean(values)
    row[names[1]] = max(values)
    row[names[2]] = _std(values)
    row[names[3]] = float(len(values))
    row[names[4]] = len(covered_days) / cycle_days if cycle_days > 0 else math.nan
    return names


def build_feature_table(
    examples: list[CycleExample],
    measurements: dict[tuple[str, str, int], dict[str, list[float]]],
) -> FeatureTable:
    rows: list[dict[str, Any]] = []
    wearable_features: set[str] = set()
    hormone_features: set[str] = set()
    symptom_features: set[str] = set()
    metabolic_stress_features: set[str] = set()
    variables_used: dict[str, set[str]] = defaultdict(set)

    for example in examples:
        history = list(example.history_lengths)
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "participant_id": example.participant_id,
            "study_interval": example.study_interval,
            "source_cycle_index": example.source_cycle_index,
            "source_start_day": example.source_start_day,
            "target_start_day": example.target_start_day,
            "target_end_day": example.target_end_day,
            "target_cycle_length": example.target_cycle_length,
            "history_previous_cycle_length": example.previous_cycle_length,
            "history_mean_cycle_length": _mean(history),
            "history_median_cycle_length": _median(history),
            "history_std_cycle_length": _std(history),
            "history_prior_cycle_count": float(len(history)),
            "feature_min_day": example.source_start_day,
            "feature_max_day": example.target_start_day - 1,
        }
        cycle_days = max(1, example.target_start_day - example.source_start_day)
        by_metric: dict[str, list[float]] = defaultdict(list)
        days_by_metric: dict[str, set[int]] = defaultdict(set)
        observed_feature_days: list[int] = []
        for day in range(example.source_start_day, example.target_start_day):
            day_measurements = measurements.get((example.participant_id, example.study_interval, day), {})
            if day_measurements:
                observed_feature_days.append(day)
            for metric, values in day_measurements.items():
                usable_values = [float(value) for value in values if not math.isnan(float(value))]
                if usable_values:
                    by_metric[metric].extend(usable_values)
                    days_by_metric[metric].add(day)
        row["observed_feature_max_day"] = max(observed_feature_days) if observed_feature_days else math.nan

        for metric in WEARABLE_METRICS:
            names = _add_stats(row, metric, by_metric.get(metric, []), days_by_metric.get(metric, set()), cycle_days)
            if names:
                wearable_features.update(names)
                variables_used["wearables"].add(metric)

        for metric in HORMONE_METRICS:
            names = _add_stats(row, metric, by_metric.get(metric, []), days_by_metric.get(metric, set()), cycle_days)
            if names:
                hormone_features.update(names)
                variables_used["hormones"].add(metric)

        for metric in SYMPTOM_METRICS:
            names = _add_stats(row, metric, by_metric.get(metric, []), days_by_metric.get(metric, set()), cycle_days)
            if names:
                symptom_features.update(names)
                variables_used["symptoms"].add(metric)

        for metric in METABOLIC_STRESS_METRICS:
            names = _add_stats(row, metric, by_metric.get(metric, []), days_by_metric.get(metric, set()), cycle_days)
            if names:
                metabolic_stress_features.update(names)
                variables_used["glucose_stress"].add(metric)

        rows.append(row)

    return FeatureTable(
        rows=rows,
        history_features=HISTORY_FEATURES.copy(),
        wearable_features=sorted(wearable_features),
        hormone_features=sorted(hormone_features),
        symptom_features=sorted(symptom_features),
        metabolic_stress_features=sorted(metabolic_stress_features),
        variables_used={key: sorted(value) for key, value in variables_used.items()},
    )


def feature_tracks(feature_table: FeatureTable) -> dict[str, list[str]]:
    history = feature_table.history_features
    wearables = feature_table.wearable_features
    hormones = feature_table.hormone_features
    symptoms = feature_table.symptom_features
    glucose_stress = feature_table.metabolic_stress_features
    tracks = {"history_only": history}
    if wearables:
        tracks["history_plus_wearables"] = history + wearables
    if hormones:
        tracks["history_plus_hormones"] = history + hormones
    if symptoms:
        tracks["history_plus_symptoms"] = history + symptoms
    if glucose_stress:
        tracks["history_plus_glucose_stress"] = history + glucose_stress
    if wearables or hormones or symptoms or glucose_stress:
        tracks["full_multimodal"] = history + wearables + hormones + symptoms + glucose_stress
    return tracks
