from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import ClassLabel, Dataset


MODULE_PATH = Path(__file__).resolve().parent / "run_minilm_logistic_regression.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_minilm_logistic_regression", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniLMLogisticRegressionTests(unittest.TestCase):
    def test_compute_metrics_matches_thesis_binary_counts_and_rates(self) -> None:
        runner = load_module()

        row = runner.compute_metrics_row(
            dataset_name="fixture",
            labels=[0, 0, 1, 1],
            predictions=[0, 1, 1, 0],
            runtime_seconds=1.25,
        )

        self.assertEqual(row["dataset"], "fixture")
        self.assertEqual(row["rows"], 4)
        self.assertEqual(row["ham_count"], 2)
        self.assertEqual(row["spam_count"], 2)
        self.assertEqual(row["false_positive_count"], 1)
        self.assertEqual(row["false_negative_count"], 1)
        self.assertEqual(row["true_positive_count"], 1)
        self.assertEqual(row["true_negative_count"], 1)
        self.assertAlmostEqual(row["accuracy"], 0.5)
        self.assertAlmostEqual(row["precision"], 0.5)
        self.assertAlmostEqual(row["recall"], 0.5)
        self.assertAlmostEqual(row["f1"], 0.5)
        self.assertAlmostEqual(row["specificity"], 0.5)
        self.assertAlmostEqual(row["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(row["false_positive_rate"], 0.5)
        self.assertAlmostEqual(row["false_negative_rate"], 0.5)
        self.assertAlmostEqual(row["runtime_seconds"], 1.25)
        self.assertEqual(row["status"], "completed")

    def test_training_split_uses_92_2_6_stratified_seeded_splits(self) -> None:
        runner = load_module()
        labels = [0, 1] * 500
        dataset = Dataset.from_dict(
            {
                "subject": [f"subject {index}" for index in range(1000)],
                "body": [f"body {index}" for index in range(1000)],
                "label": labels,
            }
        ).cast_column("label", ClassLabel(names=["ham", "spam"]))

        splits = runner.split_training_dataset(dataset, seed=67)

        self.assertEqual(len(splits["train"]), 920)
        self.assertEqual(len(splits["validation"]), 20)
        self.assertEqual(len(splits["test"]), 60)
        self.assertEqual(sum(int(value) == 1 for value in splits["train"]["label"]), 460)
        self.assertEqual(sum(int(value) == 1 for value in splits["validation"]["label"]), 10)
        self.assertEqual(sum(int(value) == 1 for value in splits["test"]["label"]), 30)

    def test_summary_columns_include_required_thesis_metrics(self) -> None:
        runner = load_module()

        required = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "false_positive_count",
            "false_negative_count",
            "false_positive_rate",
            "false_negative_rate",
            "rows",
            "ham_count",
            "spam_count",
            "runtime_seconds",
            "status",
        }

        self.assertTrue(required.issubset(set(runner.SUMMARY_COLUMNS)))

    def test_configure_huggingface_cache_defaults_to_method_local_cache(self) -> None:
        runner = load_module()

        with patch.dict(os.environ, {}, clear=True):
            cache_dir = runner.configure_huggingface_cache()

            self.assertEqual(cache_dir, MODULE_PATH.parent / ".hf_cache")
            self.assertEqual(os.environ["HF_HOME"], str(cache_dir))
            self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(cache_dir / "datasets"))


if __name__ == "__main__":
    unittest.main()
