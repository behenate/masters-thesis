from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset


MODULE_PATH = Path(__file__).with_name("run_tfidf_linear_svm.py")
SPEC = importlib.util.spec_from_file_location("run_tfidf_linear_svm", MODULE_PATH)
svm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(svm)


def make_dataset(labels: list[int]) -> Dataset:
    return Dataset.from_dict(
        {
            "subject": [f"Subject {index}" for index in range(len(labels))],
            "body": [f"Body {index}" for index in range(len(labels))],
            "label": labels,
            "source": ["unit"] * len(labels),
        }
    )


class TfidfLinearSvmContractTests(unittest.TestCase):
    def test_build_email_text_matches_thesis_format(self) -> None:
        self.assertEqual(
            svm.build_email_text("Quarterly update", "Please review."),
            "Subject: Quarterly update\n\nPlease review.",
        )
        self.assertEqual(svm.build_email_text("", "Only body"), "Only body")
        self.assertEqual(svm.build_email_text("Only subject", ""), "Subject: Only subject")

    def test_split_training_dataset_uses_92_2_6_stratified_contract(self) -> None:
        dataset = make_dataset([0] * 50 + [1] * 50)

        split = svm.split_training_dataset(dataset, seed=67)

        self.assertEqual(len(split.train), 92)
        self.assertEqual(len(split.validation), 2)
        self.assertEqual(len(split.test), 6)
        self.assertEqual(svm.label_counts(split.train), (46, 46))
        self.assertEqual(svm.label_counts(split.validation), (1, 1))
        self.assertEqual(svm.label_counts(split.test), (3, 3))

    def test_stratified_sample_keeps_class_balance_when_limited(self) -> None:
        dataset = make_dataset([0] * 70 + [1] * 30)

        sampled = svm.stratified_sample(dataset, limit=20, seed=67)

        self.assertEqual(len(sampled), 20)
        self.assertEqual(svm.label_counts(sampled), (14, 6))

    def test_compute_metrics_returns_thesis_counts_and_rates(self) -> None:
        metrics = svm.compute_metrics([0, 0, 1, 1], [0, 1, 0, 1])

        self.assertEqual(metrics["false_positive_count"], 1.0)
        self.assertEqual(metrics["false_negative_count"], 1.0)
        self.assertEqual(metrics["true_positive_count"], 1.0)
        self.assertEqual(metrics["true_negative_count"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.5)
        self.assertEqual(metrics["spam_prediction_rate"], 0.5)

    def test_configure_datasets_cache_defaults_to_project_local_cache(self) -> None:
        old_values = {name: os.environ.pop(name, None) for name in ["HF_HOME", "HF_DATASETS_CACHE"]}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = svm.configure_datasets_cache(Path(tmpdir))

                self.assertEqual(cache_dir, Path(tmpdir) / ".hf_datasets_cache")
                self.assertEqual(os.environ["HF_HOME"], str(cache_dir))
                self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(cache_dir / "datasets"))
        finally:
            for name, value in old_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
