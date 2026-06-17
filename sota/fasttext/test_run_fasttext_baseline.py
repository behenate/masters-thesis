from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "run_fasttext_baseline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_fasttext_baseline", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastTextBaselineTests(unittest.TestCase):
    def test_binary_metrics_match_thesis_counts_and_rates(self) -> None:
        baseline = load_module()

        metrics = baseline.binary_metrics(
            predictions=[1, 0, 1, 0, 1, 0],
            labels=[1, 1, 0, 0, 1, 0],
            failure_count=0,
        )

        self.assertEqual(metrics["false_positive_count"], 1.0)
        self.assertEqual(metrics["false_negative_count"], 1.0)
        self.assertEqual(metrics["true_positive_count"], 2.0)
        self.assertEqual(metrics["true_negative_count"], 2.0)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["spam_prediction_rate"], 3 / 6)

    def test_fasttext_line_uses_label_prefix_and_single_line_text(self) -> None:
        baseline = load_module()

        line = baseline.fasttext_line(
            {
                "subject": "Prize\ninside",
                "body": "Click\tthis  offer\r\nnow",
                "label": 1,
            }
        )

        self.assertEqual(line, "__label__spam Subject: Prize inside Click this offer now")

    def test_fasttext_line_escapes_label_like_tokens_inside_email_text(self) -> None:
        baseline = load_module()

        line = baseline.fasttext_line(
            {
                "subject": "Build artifact",
                "body": "This contains __label__unexpected inside the message.",
                "label": 0,
            }
        )

        text_tokens = line.split()[1:]
        self.assertFalse(any(token.startswith("__label__") for token in text_tokens))

    def test_configure_hf_cache_defaults_to_method_local_paths(self) -> None:
        baseline = load_module()
        original = {key: os.environ.get(key) for key in ["HF_HOME", "HF_DATASETS_CACHE"]}
        for key in original:
            os.environ.pop(key, None)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cache_root = (Path(tmp_dir) / "hf-cache").resolve()
                baseline.configure_hf_cache(cache_root)

                self.assertEqual(os.environ["HF_HOME"], str(cache_root / "home"))
                self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(cache_root / "datasets"))
                self.assertTrue((cache_root / "home").is_dir())
                self.assertTrue((cache_root / "datasets").is_dir())
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_predict_label_bypasses_numpy_incompatible_public_predict(self) -> None:
        baseline = load_module()

        class FakeNativeModel:
            def predict(self, text, k, threshold, on_unicode_error):
                self.call = (text, k, threshold, on_unicode_error)
                return [(0.99, "__label__spam")]

        class FakeModel:
            def __init__(self):
                self.f = FakeNativeModel()

            def predict(self, text, k=1):
                raise AssertionError("public predict should not be called")

        model = FakeModel()
        prediction, failed = baseline.predict_label(model, {"subject": "Offer", "body": "Free", "label": 1})

        self.assertEqual(prediction, 1)
        self.assertFalse(failed)
        self.assertEqual(model.f.call, ("Subject: Offer Free\n", 1, 0.0, "strict"))

    def test_write_summary_ignores_non_schema_metadata(self) -> None:
        baseline = load_module()
        row = {column: "" for column in baseline.SUMMARY_COLUMNS}
        row.update(
            {
                "dataset": "train_subset",
                "method": baseline.METHOD,
                "status": "completed",
                "source_path": "/tmp/not-a-summary-column.parquet",
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "summary.csv"
            baseline.write_summary([row], output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("dataset,method,", text)
        self.assertNotIn("source_path", text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
