from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from datasets import Dataset


SCRIPT_PATH = Path(__file__).with_name("run_distilbert_sequence_classification.py")


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("distilbert_baseline", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistilBertBaselineTests(unittest.TestCase):
    def test_split_constants_match_thesis_evaluation(self) -> None:
        baseline = load_baseline_module()

        self.assertEqual(baseline.SEED, 67)
        self.assertAlmostEqual(baseline.TRAIN_SPLIT, 0.92)
        self.assertAlmostEqual(baseline.VALIDATION_SPLIT, 0.02)
        self.assertAlmostEqual(baseline.TEST_SPLIT, 0.06)
        self.assertAlmostEqual(baseline.HOLDOUT_SPLIT, 0.08)

    def test_huggingface_cache_defaults_to_owned_method_directory(self) -> None:
        baseline = load_baseline_module()

        self.assertTrue(baseline.LOCAL_HF_HOME.is_relative_to(SCRIPT_PATH.parent))
        self.assertTrue(baseline.LOCAL_HF_DATASETS_CACHE.is_relative_to(SCRIPT_PATH.parent))
        self.assertEqual(baseline.os.environ["HF_HOME"], str(baseline.LOCAL_HF_HOME))
        self.assertEqual(
            baseline.os.environ["HF_DATASETS_CACHE"],
            str(baseline.LOCAL_HF_DATASETS_CACHE),
        )

    def test_compute_binary_metrics_reports_thesis_fields(self) -> None:
        baseline = load_baseline_module()

        metrics = baseline.compute_binary_metrics(labels=[0, 0, 1, 1], predictions=[0, 1, 0, 1])

        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["ham_count"], 2)
        self.assertEqual(metrics["spam_count"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["false_positive_count"], 1.0)
        self.assertEqual(metrics["false_negative_count"], 1.0)
        self.assertEqual(metrics["true_positive_count"], 1.0)
        self.assertEqual(metrics["true_negative_count"], 1.0)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)
        self.assertAlmostEqual(metrics["spam_prediction_rate"], 0.5)

    def test_stratified_sample_preserves_class_balance(self) -> None:
        baseline = load_baseline_module()
        dataset = Dataset.from_dict(
            {
                "subject": [f"subject {index}" for index in range(10)],
                "body": [f"body {index}" for index in range(10)],
                "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )

        sample = baseline.stratified_sample(dataset, limit=4, seed=baseline.SEED)

        self.assertEqual(len(sample), 4)
        self.assertEqual(sum(int(value) == 0 for value in sample["label"]), 2)
        self.assertEqual(sum(int(value) == 1 for value in sample["label"]), 2)

    def test_build_email_text_matches_thesis_format(self) -> None:
        baseline = load_baseline_module()

        self.assertEqual(
            baseline.build_email_text("Hello", "Body text"),
            "Subject: Hello\n\nBody text",
        )
        self.assertEqual(baseline.build_email_text("", "Only body"), "Only body")
        self.assertEqual(baseline.build_email_text("Only subject", ""), "Subject: Only subject")

    def test_select_threshold_uses_validation_scores(self) -> None:
        baseline = load_baseline_module()

        threshold, metrics = baseline.select_threshold(
            labels=[0, 0, 1, 1],
            scores=[0.10, 0.40, 0.45, 0.90],
        )

        self.assertGreater(threshold, 0.40)
        self.assertLessEqual(threshold, 0.45)
        self.assertAlmostEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
