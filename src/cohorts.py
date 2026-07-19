from __future__ import annotations

import csv
import hashlib
import statistics
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .data import (
    DEFAULT_MAX_CYCLE_LENGTH,
    DEFAULT_MIN_CYCLE_LENGTH,
    CycleExample,
    normalize_id,
    safe_int,
)


UTAH_DATASET_DOI = "10.7278/S50d-4gxs-s4hj"
UTAH_RECORD_URL = "https://hive.utah.edu/records/3zhb1-48n69"
UTAH_DATA_URL = (
    "https://hive.utah.edu/api/records/3zhb1-48n69/files/"
    "CrMcyclelength_share.csv/content"
)
UTAH_README_URL = (
    "https://hive.utah.edu/api/records/3zhb1-48n69/files/"
    "Najmabadi_README20230824.txt/content"
)
UTAH_DATA_SHA256 = "ad67a3eed731a038d6502a1eb42a9dcddf8f6c69a607a61b85f3798198043dac"
UTAH_README_SHA256 = "4bbb2c93efada3c84fe841bc5ef82820c30f8c1dce7e7ebc07d98558138a4678"
UTAH_FILENAME = "CrMcyclelength_share.csv"


@dataclass(frozen=True)
class UtahCycle:
    participant_id: str
    cycle_number: int
    start_date: date
    end_date: date
    cycle_length: int | None
    conception_cycle: str


@dataclass(frozen=True)
class UtahData:
    cycles: tuple[UtahCycle, ...]
    source_file: Path
    columns: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_utah_data(output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        (UTAH_DATA_URL, destination / UTAH_FILENAME, UTAH_DATA_SHA256),
        (UTAH_README_URL, destination / "README.txt", UTAH_README_SHA256),
    ]
    downloaded: list[Path] = []
    for url, path, expected_hash in files:
        if not path.exists() or overwrite:
            urllib.request.urlretrieve(url, path)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {expected_hash}, got {actual_hash}. "
                "The repository file may have changed; review it before use."
            )
        downloaded.append(path)
    return downloaded


def find_utah_file(data_file: str | Path) -> Path:
    path = Path(data_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Utah data path does not exist: {path}. Download it with "
            "'python3 run_benchmark.py download-utah'."
        )
    if path.is_dir():
        path = path / UTAH_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {UTAH_FILENAME} at {path}.")
    return path


def _parse_date(value: Any) -> date:
    text = str(value or "").strip()
    for date_format in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid Utah cycle date: {text!r}")


def load_utah_data(data_file: str | Path) -> UtahData:
    path = find_utah_file(data_file)
    required = {
        "new_id",
        "cycle_number",
        "cycle_start_date",
        "cycle_end_date",
        "cycle_length",
        "conception_cycle",
    }
    cycles: list[UtahCycle] = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"Utah CSV is missing required columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            participant_id = normalize_id(row.get("new_id"))
            cycle_number = safe_int(row.get("cycle_number"))
            if not participant_id or cycle_number is None:
                raise ValueError(f"Missing participant or cycle number on line {line_number}.")
            key = (participant_id, cycle_number)
            if key in seen:
                raise ValueError(f"Duplicate participant/cycle key on line {line_number}: {key}")
            seen.add(key)
            start_date = _parse_date(row.get("cycle_start_date"))
            end_date = _parse_date(row.get("cycle_end_date"))
            cycle_length = safe_int(row.get("cycle_length"))
            if cycle_length is not None and (end_date - start_date).days + 1 != cycle_length:
                raise ValueError(
                    f"Cycle length does not match inclusive dates on line {line_number}."
                )
            cycles.append(
                UtahCycle(
                    participant_id=participant_id,
                    cycle_number=cycle_number,
                    start_date=start_date,
                    end_date=end_date,
                    cycle_length=cycle_length,
                    conception_cycle=str(row.get("conception_cycle") or "").strip(),
                )
            )
    if not cycles:
        raise ValueError(f"Utah CSV has no data rows: {path}")
    return UtahData(cycles=tuple(cycles), source_file=path, columns=columns)


def _connected(left: UtahCycle, right: UtahCycle) -> bool:
    return (
        right.cycle_number == left.cycle_number + 1
        and right.start_date == left.end_date + timedelta(days=1)
    )


def build_utah_cycle_examples(
    loaded: UtahData,
    min_cycle_length: int,
    max_cycle_length: int,
) -> list[CycleExample]:
    by_participant: dict[str, list[UtahCycle]] = defaultdict(list)
    for cycle in loaded.cycles:
        by_participant[cycle.participant_id].append(cycle)

    examples: list[CycleExample] = []
    for raw_participant_id, participant_cycles in sorted(by_participant.items()):
        ordered = sorted(participant_cycles, key=lambda cycle: cycle.cycle_number)
        run: list[UtahCycle] = []
        previous: UtahCycle | None = None
        for cycle in ordered:
            if previous is not None and not _connected(previous, cycle):
                run = []
            plausible = (
                cycle.cycle_length is not None
                and min_cycle_length <= cycle.cycle_length <= max_cycle_length
            )
            if plausible:
                run.append(cycle)
            else:
                run = []

            if len(run) >= 2:
                source, target = run[-2], run[-1]
                history = tuple(float(item.cycle_length) for item in run[:-1] if item.cycle_length is not None)
                participant_id = f"utah:{raw_participant_id}"
                examples.append(
                    CycleExample(
                        example_id=f"utah_{raw_participant_id}_{source.cycle_number}_{target.cycle_number}",
                        participant_id=participant_id,
                        study_interval="utah",
                        source_cycle_index=source.cycle_number,
                        source_start_day=source.start_date.toordinal(),
                        target_start_day=target.start_date.toordinal(),
                        target_end_day=target.end_date.toordinal() + 1,
                        previous_cycle_length=float(source.cycle_length),
                        target_cycle_length=float(target.cycle_length),
                        history_lengths=history,
                    )
                )
            previous = cycle
    return examples


def summarize_utah_data(
    loaded: UtahData,
    examples: Iterable[CycleExample],
    min_cycle_length: int = DEFAULT_MIN_CYCLE_LENGTH,
    max_cycle_length: int = DEFAULT_MAX_CYCLE_LENGTH,
) -> dict[str, Any]:
    example_list = list(examples)
    lengths = [cycle.cycle_length for cycle in loaded.cycles if cycle.cycle_length is not None]
    by_participant: dict[str, list[UtahCycle]] = defaultdict(list)
    for cycle in loaded.cycles:
        by_participant[cycle.participant_id].append(cycle)
    transitions = [
        (left, right)
        for participant_cycles in by_participant.values()
        for left, right in zip(
            sorted(participant_cycles, key=lambda cycle: cycle.cycle_number),
            sorted(participant_cycles, key=lambda cycle: cycle.cycle_number)[1:],
        )
    ]
    connected = [(left, right) for left, right in transitions if _connected(left, right)]
    connected_with_lengths = [
        (left, right)
        for left, right in connected
        if left.cycle_length is not None and right.cycle_length is not None
    ]
    return {
        "benchmark": "mcPHASES CycleBench: Utah history replication",
        "dataset_id": "utah_cycle_length",
        "dataset_version": "repository record accessed by pinned checksum",
        "dataset_doi": UTAH_DATASET_DOI,
        "participants": len({cycle.participant_id for cycle in loaded.cycles}),
        "cycles": len(lengths),
        "eligible_examples": len(example_list),
        "tables_used": {
            loaded.source_file.name: [
                "cycle_number",
                "cycle_start_date",
                "cycle_end_date",
                "cycle_length",
            ]
        },
        "table_rows": {loaded.source_file.name: len(loaded.cycles)},
        "used_variables": ["cycle_number", "cycle_start_date", "cycle_end_date", "cycle_length"],
        "exclusions": {
            "cycles_missing_length": sum(
                cycle.cycle_length is None for cycle in loaded.cycles
            ),
            "cycles_outside_length_range": sum(
                not min_cycle_length <= length <= max_cycle_length for length in lengths
            ),
            "adjacent_row_transitions": len(transitions),
            "nonconsecutive_transitions": len(transitions) - len(connected),
            "connected_pairs_with_missing_length": len(connected) - len(connected_with_lengths),
            "connected_pairs_outside_length_range": sum(
                not (
                    min_cycle_length <= int(left.cycle_length) <= max_cycle_length
                    and min_cycle_length <= int(right.cycle_length) <= max_cycle_length
                )
                for left, right in connected_with_lengths
            ),
        },
    }


def add_cohort(rows: Iterable[dict[str, Any]], cohort: str) -> list[dict[str, Any]]:
    return [{"cohort": cohort, **row} for row in rows]


def build_macro_scores(cohort_scores: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(cohort_scores)
    cohorts = sorted({str(row["cohort"]) for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["track"]))].append(row)

    metrics = (
        "mae",
        "rmse",
        "median_absolute_error",
        "within_3_days_pct",
        "within_7_days_pct",
        "mean_signed_error",
    )
    output: list[dict[str, Any]] = []
    for (model, track), group in sorted(grouped.items()):
        represented = {str(row["cohort"]) for row in group}
        if represented != set(cohorts):
            continue
        output.append(
            {
                "model": model,
                "model_label": str(group[0]["model_label"]),
                "track": track,
                "cohort_count": len(cohorts),
                "cohorts": ";".join(cohorts),
                "total_examples": sum(int(row["n"]) for row in group),
                **{
                    f"macro_{metric}": statistics.mean(float(row[metric]) for row in group)
                    for metric in metrics
                },
            }
        )
    return output
