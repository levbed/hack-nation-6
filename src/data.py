from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MIN_CYCLE_LENGTH = 10
DEFAULT_MAX_CYCLE_LENGTH = 90
MAX_MENSTRUAL_GAP_DAYS = 2

ORDINAL_SELF_REPORT_VALUES = {
    "not at all": 0.0,
    "very low": 1.0,
    "very low/little": 1.0,
    "low": 2.0,
    "moderate": 3.0,
    "high": 4.0,
    "very high": 5.0,
}


@dataclass(frozen=True)
class CycleExample:
    example_id: str
    participant_id: str
    study_interval: str
    source_cycle_index: int
    source_start_day: int
    target_start_day: int
    target_end_day: int
    previous_cycle_length: float
    target_cycle_length: float
    history_lengths: tuple[float, ...]


@dataclass
class LoadedData:
    flow_rows: list[dict[str, Any]]
    measurements: dict[tuple[str, str, int], dict[str, list[float]]]
    tables_used: dict[str, list[str]]
    table_rows: dict[str, int | None]
    table_columns: dict[str, list[str]]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def normalize_id(value: Any) -> str:
    return str(value).strip()


def normalize_interval(value: Any) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def is_positive_flow(flow_volume: Any, phase: Any = None) -> bool:
    flow = str(flow_volume or "").strip().lower()
    if flow and flow not in {"not at all", "none", "no", "0", "nan"}:
        return True
    return str(phase or "").strip().lower() == "menstrual"


def safe_ordinal(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in ORDINAL_SELF_REPORT_VALUES:
        return ORDINAL_SELF_REPORT_VALUES[text]
    number = safe_float(text)
    return number if number is not None and 0.0 <= number <= 5.0 else None


def find_data_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root}. "
            "Replace /path/to/mcphases with the location of your downloaded mcPHASES data."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root}")
    if (root / "hormones_and_selfreport.csv").exists():
        return root
    matches = [p for p in root.iterdir() if p.is_dir() and (p / "hormones_and_selfreport.csv").exists()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find hormones_and_selfreport.csv in {root} or exactly one child directory."
    )


def _read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def _add_measurement(
    measurements: dict[tuple[str, str, int], dict[str, list[float]]],
    pid: str,
    interval: str,
    day: int | None,
    name: str,
    value: Any,
) -> bool:
    number = safe_float(value)
    if day is None or number is None:
        return False
    measurements[(pid, interval, day)][name].append(number)
    return True


def load_mcphases_data(data_dir: str | Path) -> LoadedData:
    base = find_data_dir(data_dir)
    flow_rows: list[dict[str, Any]] = []
    measurements: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tables_used: dict[str, list[str]] = defaultdict(list)
    table_rows: dict[str, int | None] = {}
    table_columns: dict[str, list[str]] = {}

    for csv_path in sorted(base.glob("*.csv")):
        table_rows[csv_path.name] = None
        table_columns[csv_path.name] = _read_header(csv_path)

    hormone_path = base / "hormones_and_selfreport.csv"
    if not hormone_path.exists():
        raise FileNotFoundError(f"Missing required table: {hormone_path}")

    table_rows[hormone_path.name] = 0
    with hormone_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            table_rows[hormone_path.name] += 1
            pid = normalize_id(row.get("id"))
            interval = normalize_interval(row.get("study_interval"))
            day = safe_int(row.get("day_in_study"))
            if not pid or day is None:
                continue
            flow_rows.append(
                {
                    "participant_id": pid,
                    "study_interval": interval,
                    "day": day,
                    "flow_volume": row.get("flow_volume"),
                    "phase": row.get("phase"),
                }
            )
            hormone_aliases = {
                "lh": "hormone_lh",
                "estrogen": "hormone_e3g",
                "e3g": "hormone_e3g",
                "pdg": "hormone_pdg",
                "fsh": "hormone_fsh",
            }
            for column, metric in hormone_aliases.items():
                if column in row and _add_measurement(measurements, pid, interval, day, metric, row[column]):
                    if metric not in tables_used["hormones_and_selfreport.csv"]:
                        tables_used["hormones_and_selfreport.csv"].append(metric)

            self_report_aliases = {
                "appetite": "selfreport_appetite",
                "exerciselevel": "selfreport_exercise_level",
                "headaches": "selfreport_headaches",
                "cramps": "selfreport_cramps",
                "sorebreasts": "selfreport_sore_breasts",
                "fatigue": "selfreport_fatigue",
                "sleepissue": "selfreport_sleep_issue",
                "moodswing": "selfreport_mood_swing",
                "stress": "selfreport_stress",
                "foodcravings": "selfreport_food_cravings",
                "indigestion": "selfreport_indigestion",
                "bloating": "selfreport_bloating",
            }
            for column, metric in self_report_aliases.items():
                value = safe_ordinal(row.get(column))
                if value is not None and _add_measurement(measurements, pid, interval, day, metric, value):
                    if metric not in tables_used["hormones_and_selfreport.csv"]:
                        tables_used["hormones_and_selfreport.csv"].append(metric)

    sleep_path = base / "sleep.csv"
    if sleep_path.exists():
        table_rows[sleep_path.name] = 0
        sleep_metrics = {
            "minutesasleep": "sleep_minutes_asleep",
            "minutesawake": "sleep_minutes_awake",
            "timeinbed": "sleep_time_in_bed",
            "efficiency": "sleep_efficiency",
        }
        with sleep_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[sleep_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("sleep_end_day_in_study") or row.get("sleep_start_day_in_study"))
                for column, metric in sleep_metrics.items():
                    if column in row and _add_measurement(measurements, pid, interval, day, metric, row[column]):
                        if metric not in tables_used["sleep.csv"]:
                            tables_used["sleep.csv"].append(metric)

    rhr_path = base / "resting_heart_rate.csv"
    if rhr_path.exists():
        table_rows[rhr_path.name] = 0
        with rhr_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[rhr_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("day_in_study"))
                if _add_measurement(measurements, pid, interval, day, "resting_heart_rate", row.get("value")):
                    if "resting_heart_rate" not in tables_used["resting_heart_rate.csv"]:
                        tables_used["resting_heart_rate.csv"].append("resting_heart_rate")

    active_path = base / "active_minutes.csv"
    if active_path.exists():
        table_rows[active_path.name] = 0
        with active_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[active_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("day_in_study"))
                light = safe_float(row.get("lightly")) or 0.0
                moderate = safe_float(row.get("moderately")) or 0.0
                very = safe_float(row.get("very")) or 0.0
                if _add_measurement(measurements, pid, interval, day, "active_minutes", light + moderate + very):
                    if "active_minutes" not in tables_used["active_minutes.csv"]:
                        tables_used["active_minutes.csv"].append("active_minutes")

    steps_path = base / "steps.csv"
    if steps_path.exists():
        table_rows[steps_path.name] = 0
        daily_steps: dict[tuple[str, str, int], float] = defaultdict(float)
        with steps_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[steps_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("day_in_study"))
                steps = safe_float(row.get("steps"))
                if pid and day is not None and steps is not None:
                    daily_steps[(pid, interval, day)] += steps
        for (pid, interval, day), total_steps in daily_steps.items():
            measurements[(pid, interval, day)]["steps"].append(total_steps)
        if daily_steps:
            tables_used["steps.csv"].append("steps")

    glucose_path = base / "glucose.csv"
    if glucose_path.exists():
        table_rows[glucose_path.name] = 0
        used_glucose = False
        with glucose_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[glucose_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("day_in_study"))
                used_glucose |= _add_measurement(
                    measurements, pid, interval, day, "glucose", row.get("glucose_value")
                )
        if used_glucose:
            tables_used["glucose.csv"].append("glucose")

    stress_path = base / "stress_score.csv"
    if stress_path.exists():
        table_rows[stress_path.name] = 0
        used_stress = False
        with stress_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                table_rows[stress_path.name] += 1
                pid = normalize_id(row.get("id"))
                interval = normalize_interval(row.get("study_interval"))
                day = safe_int(row.get("day_in_study"))
                used_stress |= _add_measurement(
                    measurements, pid, interval, day, "wearable_stress_score", row.get("stress_score")
                )
        if used_stress:
            tables_used["stress_score.csv"].append("wearable_stress_score")

    return LoadedData(
        flow_rows=flow_rows,
        measurements=measurements,
        tables_used={name: sorted(values) for name, values in tables_used.items()},
        table_rows=table_rows,
        table_columns=table_columns,
    )


def infer_cycle_starts(flow_rows: Iterable[dict[str, Any]], min_gap_days: int = DEFAULT_MIN_CYCLE_LENGTH) -> dict[tuple[str, str], list[int]]:
    positive_days: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in flow_rows:
        if is_positive_flow(row.get("flow_volume"), row.get("phase")):
            positive_days[(row["participant_id"], row["study_interval"])].add(int(row["day"]))

    starts_by_group: dict[tuple[str, str], list[int]] = {}
    for key, days_set in positive_days.items():
        days = sorted(days_set)
        episode_starts: list[int] = []
        last_day: int | None = None
        for day in days:
            if last_day is None or day - last_day > MAX_MENSTRUAL_GAP_DAYS:
                episode_starts.append(day)
            last_day = day
        starts: list[int] = []
        for day in episode_starts:
            if not starts or day - starts[-1] >= min_gap_days:
                starts.append(day)
        starts_by_group[key] = starts
    return starts_by_group


def build_cycle_examples(
    flow_rows: Iterable[dict[str, Any]],
    min_cycle_length: int = DEFAULT_MIN_CYCLE_LENGTH,
    max_cycle_length: int = DEFAULT_MAX_CYCLE_LENGTH,
) -> list[CycleExample]:
    starts_by_group = infer_cycle_starts(flow_rows, min_cycle_length)
    examples: list[CycleExample] = []
    for (pid, interval), starts in sorted(starts_by_group.items()):
        if len(starts) < 3:
            continue
        lengths = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        plausible = [min_cycle_length <= length <= max_cycle_length for length in lengths]
        source_cycle_number = 0
        for i in range(len(starts) - 2):
            if not (plausible[i] and plausible[i + 1]):
                continue
            history_lengths = tuple(float(lengths[j]) for j in range(i + 1) if plausible[j])
            if not history_lengths:
                continue
            source_cycle_number += 1
            example_id = f"{pid}_{interval}_{starts[i]}_{starts[i + 1]}"
            examples.append(
                CycleExample(
                    example_id=example_id,
                    participant_id=pid,
                    study_interval=interval,
                    source_cycle_index=source_cycle_number,
                    source_start_day=starts[i],
                    target_start_day=starts[i + 1],
                    target_end_day=starts[i + 2],
                    previous_cycle_length=float(lengths[i]),
                    target_cycle_length=float(lengths[i + 1]),
                    history_lengths=history_lengths,
                )
            )
    return examples


def summarize_loaded_data(loaded: LoadedData, examples: list[CycleExample]) -> dict[str, Any]:
    participants = sorted({row["participant_id"] for row in loaded.flow_rows})
    starts = infer_cycle_starts(loaded.flow_rows)
    cycle_count = sum(max(0, len(group_starts) - 1) for group_starts in starts.values())
    used_variables = sorted({metric for values in loaded.tables_used.values() for metric in values})
    return {
        "benchmark": "mcPHASES CycleBench",
        "dataset_id": "mcphases",
        "dataset_version": "1.0.0",
        "dataset_doi": "10.13026/zx6a-2c81",
        "participants": len(participants),
        "cycles": cycle_count,
        "eligible_examples": len(examples),
        "tables_used": loaded.tables_used,
        "table_rows": loaded.table_rows,
        "used_variables": used_variables,
    }


def count_by_participant(examples: Iterable[CycleExample]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in examples:
        counts[example.participant_id] += 1
    return counts
