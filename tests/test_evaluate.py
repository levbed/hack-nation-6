from __future__ import annotations

import unittest

from src.data import safe_ordinal
from src.evaluate import (
    DEFAULT_MODEL_IDS,
    DEFAULT_RIDGE_ALPHAS,
    default_model_grids,
    evaluate_feature_table,
    participant_bootstrap_mae,
)
from src.features import HISTORY_FEATURES, FeatureTable, feature_tracks


def _evaluation_table() -> FeatureTable:
    rows = []
    for participant_index in range(6):
        for cycle_index in range(2):
            previous = 25.0 + participant_index + cycle_index
            rows.append(
                {
                    "example_id": f"row-{participant_index}-{cycle_index}",
                    "participant_id": f"P{participant_index}",
                    "study_interval": "test",
                    "source_start_day": cycle_index * 30,
                    "target_start_day": (cycle_index + 1) * 30,
                    "target_end_day": (cycle_index + 2) * 30,
                    "target_cycle_length": previous + (participant_index % 2),
                    "history_previous_cycle_length": previous,
                    "history_mean_cycle_length": previous - 0.5,
                    "history_median_cycle_length": previous - 0.5,
                    "history_std_cycle_length": 1.0 + cycle_index,
                    "history_prior_cycle_count": 2.0 + cycle_index,
                }
            )
    return FeatureTable(
        rows=rows,
        history_features=HISTORY_FEATURES.copy(),
        wearable_features=[],
        hormone_features=[],
        symptom_features=[],
        metabolic_stress_features=[],
        variables_used={},
    )


class EvaluationTests(unittest.TestCase):
    def test_self_report_ordinal_mapping(self) -> None:
        self.assertEqual(safe_ordinal("Not at all"), 0.0)
        self.assertEqual(safe_ordinal("Very Low/Little"), 1.0)
        self.assertEqual(safe_ordinal("Moderate"), 3.0)
        self.assertEqual(safe_ordinal("5"), 5.0)
        self.assertIsNone(safe_ordinal("unknown"))

    def test_nested_ridge_records_only_prespecified_alphas(self) -> None:
        _, fold_scores, _ = evaluate_feature_table(
            _evaluation_table(), model_ids=("ridge",), bootstrap_replicates=100
        )
        selected = {
            float(row["selected_alpha"])
            for row in fold_scores
            if row["model"] == "ridge"
        }
        self.assertTrue(selected)
        self.assertTrue(selected.issubset(set(DEFAULT_RIDGE_ALPHAS)))

    def test_all_prespecified_model_families_run(self) -> None:
        grids = {model: candidates[:1] for model, candidates in default_model_grids().items()}
        scores, _, _ = evaluate_feature_table(
            _evaluation_table(), model_grids=grids, bootstrap_replicates=20
        )
        models = {row["model"] for row in scores if row["track"] == "history_only"}
        self.assertEqual(models, set(DEFAULT_MODEL_IDS))

    def test_optional_multimodal_tracks_are_included_when_available(self) -> None:
        table = _evaluation_table()
        table.symptom_features = ["selfreport_stress_mean"]
        table.metabolic_stress_features = ["glucose_mean"]
        tracks = feature_tracks(table)
        self.assertIn("history_plus_symptoms", tracks)
        self.assertIn("history_plus_glucose_stress", tracks)

    def test_unavailable_multimodal_tracks_are_skipped(self) -> None:
        self.assertEqual(set(feature_tracks(_evaluation_table())), {"history_only"})

    def test_participant_bootstrap_is_paired_by_participant(self) -> None:
        predictions = []
        for participant in ["A", "B", "C"]:
            for track, errors in {"history_only": [2.0, 2.0], "better": [1.0, 1.0]}.items():
                for error in errors:
                    predictions.append(
                        {
                            "model": "ridge",
                            "track": track,
                            "participant_id": participant,
                            "error_days": error,
                        }
                    )
        intervals = participant_bootstrap_mae(predictions, replicates=200, seed=7)
        self.assertAlmostEqual(intervals[("ridge", "better")]["delta_mae_ci_low"], -1.0)
        self.assertAlmostEqual(intervals[("ridge", "better")]["delta_mae_ci_high"], -1.0)


if __name__ == "__main__":
    unittest.main()
