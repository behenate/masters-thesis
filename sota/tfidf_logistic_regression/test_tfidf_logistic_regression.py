from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd


METHOD_DIR = Path(__file__).resolve().parent
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

import run_tfidf_logistic_regression as baseline


class TfidfLogisticRegressionTests(unittest.TestCase):
    def test_build_email_text_matches_thesis_prompt_input(self) -> None:
        self.assertEqual(
            baseline.build_email_text("Meeting", "Agenda attached"),
            "Subject: Meeting\n\nAgenda attached",
        )
        self.assertEqual(baseline.build_email_text("", "Body only"), "Body only")
        self.assertEqual(baseline.build_email_text(None, None), "")

    def test_binary_metrics_include_requested_false_rates(self) -> None:
        metrics = baseline.binary_metrics(
            predictions=[0, 1, 0, 1],
            labels=[0, 0, 1, 1],
        )

        expected = {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "specificity": 0.5,
            "balanced_accuracy": 0.5,
            "false_positive_count": 1.0,
            "false_negative_count": 1.0,
            "true_positive_count": 1.0,
            "true_negative_count": 1.0,
            "false_positive_rate": 0.5,
            "false_negative_rate": 0.5,
            "spam_prediction_rate": 0.5,
            "classification_failure_count": 0.0,
            "classification_failure_rate": 0.0,
            "parse_failure_count": 0.0,
            "parse_failure_rate": 0.0,
        }
        for key, value in expected.items():
            self.assertTrue(math.isclose(metrics[key], value), key)

    def test_plain_sample_is_deterministic_and_limited(self) -> None:
        frame = pd.DataFrame(
            {
                "subject": [f"s{i}" for i in range(10)],
                "body": [f"b{i}" for i in range(10)],
                "label": [0, 1] * 5,
                "source": ["unit"] * 10,
            }
        )

        first = baseline.plain_sample(frame, limit=4, seed=67)
        second = baseline.plain_sample(frame, limit=4, seed=67)

        self.assertEqual(len(first), 4)
        self.assertEqual(first["subject"].tolist(), second["subject"].tolist())

    def test_summary_row_uses_thesis_compatible_columns(self) -> None:
        metrics = baseline.binary_metrics(
            predictions=[1, 0, 1, 0],
            labels=[1, 0, 0, 0],
        )
        row = baseline.summary_row(
            dataset_name="unit",
            metadata={"rows": 4, "ham_count": 3, "spam_count": 1},
            metrics=metrics,
            runtime_seconds=1.23456,
        )

        for column in baseline.SUMMARY_COLUMNS:
            self.assertIn(column, row)
        self.assertEqual(row["dataset"], "unit")
        self.assertEqual(row["method"], baseline.EVALUATION_METHOD)
        self.assertEqual(row["rows"], 4)
        self.assertEqual(row["ham_count"], 3)
        self.assertEqual(row["spam_count"], 1)
        self.assertEqual(row["runtime_seconds"], 1.2346)
        self.assertEqual(row["status"], "completed")

    def test_default_datasets_cache_dir_avoids_home_cache(self) -> None:
        args = baseline.build_parser().parse_args([])
        path = baseline.resolve_datasets_cache_dir(args.datasets_cache_dir)

        self.assertIn("/tmp", str(path))
        self.assertNotIn(".cache/huggingface", str(path))


if __name__ == "__main__":
    unittest.main()
