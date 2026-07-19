from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cohorts import (
    add_cohort,
    build_macro_scores,
    build_utah_cycle_examples,
    load_utah_data,
    summarize_utah_data,
)
from src.features import build_feature_table, feature_tracks


UTAH_FIXTURE = """new_id,age,cycle_number,cycle_start_date,cycle_end_date,cycle_length,conception_cycle
1,30,1,1/1/03,1/20/03,20,No
1,30,2,1/21/03,2/19/03,30,No
1,30,3,2/20/03,3/15/03,24,No
1,30,4,3/16/03,3/31/03,,Yes
1,30,5,4/1/03,4/25/03,25,No
1,30,6,4/26/03,5/21/03,26,No
2,28,1,6/1/03,6/28/03,28,No
"""


class UtahCohortTests(unittest.TestCase):
    def _load(self):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "cycles.csv"
        path.write_text(UTAH_FIXTURE)
        return temporary, load_utah_data(path)

    def test_builds_only_complete_consecutive_pairs_and_resets_history(self) -> None:
        temporary, loaded = self._load()
        self.addCleanup(temporary.cleanup)
        examples = build_utah_cycle_examples(loaded, min_cycle_length=10, max_cycle_length=90)

        self.assertEqual([example.target_cycle_length for example in examples], [30.0, 24.0, 26.0])
        self.assertEqual(examples[0].history_lengths, (20.0,))
        self.assertEqual(examples[1].history_lengths, (20.0, 30.0))
        self.assertEqual(examples[2].history_lengths, (25.0,))
        self.assertTrue(all(example.participant_id == "utah:1" for example in examples))

    def test_date_boundaries_and_empty_modalities_are_leakage_safe(self) -> None:
        temporary, loaded = self._load()
        self.addCleanup(temporary.cleanup)
        examples = build_utah_cycle_examples(loaded, min_cycle_length=10, max_cycle_length=90)
        table = build_feature_table(examples, {})

        self.assertEqual(set(feature_tracks(table)), {"history_only"})
        for row in table.rows:
            self.assertLess(row["feature_max_day"], row["target_start_day"])
            self.assertEqual(
                row["target_end_day"] - row["target_start_day"],
                row["target_cycle_length"],
            )

    def test_summary_reports_transition_exclusions(self) -> None:
        temporary, loaded = self._load()
        self.addCleanup(temporary.cleanup)
        examples = build_utah_cycle_examples(loaded, min_cycle_length=10, max_cycle_length=90)
        summary = summarize_utah_data(loaded, examples)

        self.assertEqual(summary["exclusions"]["cycles_missing_length"], 1)
        self.assertEqual(summary["exclusions"]["nonconsecutive_transitions"], 0)
        self.assertEqual(summary["exclusions"]["connected_pairs_with_missing_length"], 2)

    def test_macro_scores_include_only_tracks_shared_by_every_cohort(self) -> None:
        base = {
            "model": "ridge",
            "model_label": "Ridge",
            "track": "history_only",
            "n": 10,
            "mae": 4.0,
            "rmse": 5.0,
            "median_absolute_error": 3.0,
            "within_3_days_pct": 50.0,
            "within_7_days_pct": 80.0,
            "mean_signed_error": 0.5,
        }
        rows = add_cohort([base], "one") + add_cohort([{**base, "mae": 2.0}], "two")
        rows += add_cohort([{**base, "track": "history_plus_hormones"}], "one")

        macro = build_macro_scores(rows)

        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["macro_mae"], 3.0)
        self.assertEqual(macro[0]["total_examples"], 20)


if __name__ == "__main__":
    unittest.main()
