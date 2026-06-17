from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

from datasets import ClassLabel, Dataset, disable_progress_bars


disable_progress_bars()


SCRIPT_PATH = Path(__file__).with_name("run_tfidf_naive_bayes.py")
SPEC = importlib.util.spec_from_file_location("tfidf_naive_bayes_runner", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TfidfNaiveBayesRunnerTests(unittest.TestCase):
    def test_build_email_text_matches_thesis_prompt_source_text(self) -> None:
        self.assertEqual(
            MODULE.build_email_text("Quarterly update", "Numbers attached"),
            "Subject: Quarterly update\n\nNumbers attached",
        )
        self.assertEqual(MODULE.build_email_text("  ", "Only body"), "Only body")
        self.assertEqual(MODULE.build_email_text(None, None), "")

    def test_binary_metrics_include_error_counts_and_rates(self) -> None:
        metrics = MODULE.compute_binary_metrics(
            labels=[0, 0, 1, 1],
            predictions=[0, 1, 0, 1],
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
        }
        for key, value in expected.items():
            self.assertTrue(math.isclose(metrics[key], value), key)

    def test_training_split_uses_92_2_6_stratified_shape(self) -> None:
        raw_dataset = Dataset.from_dict(
            {
                "subject": [f"subject {index}" for index in range(100)],
                "body": [f"body {index}" for index in range(100)],
                "label": [index % 2 for index in range(100)],
            }
        ).cast_column("label", ClassLabel(names=["ham", "spam"]))

        splits = MODULE.split_training_dataset(raw_dataset, seed=67)

        self.assertEqual(len(splits["train"]), 92)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 6)
        self.assertEqual(splits["train"]["label"].count(0), 46)
        self.assertEqual(splits["train"]["label"].count(1), 46)

    def test_load_dataset_kwargs_use_repository_cache(self) -> None:
        kwargs = MODULE.load_dataset_kwargs()

        self.assertEqual(Path(kwargs["cache_dir"]).name, ".hf_datasets_cache")
        self.assertTrue(Path(kwargs["cache_dir"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
