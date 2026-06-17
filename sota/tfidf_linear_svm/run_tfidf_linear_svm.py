#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
DEFAULT_SAMPLE_LIMIT = 2000
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
METHOD = "tfidf_linear_svm"
EVALUATION_METHOD = "tfidf_linear_svm"
CONFIG_ID = "tfidf_word_1_2_linear_svm_c1"

SUMMARY_COLUMNS = [
    "method",
    "evaluation_method",
    "dataset",
    "config_index",
    "config_id",
    "checkpoint_type",
    "checkpoint_step",
    "rows",
    "ham_count",
    "spam_count",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "false_positive_count",
    "false_negative_count",
    "true_positive_count",
    "true_negative_count",
    "false_positive_rate",
    "false_negative_rate",
    "spam_prediction_rate",
    "classification_failure_count",
    "classification_failure_rate",
    "eval_batch_size",
    "max_seq_length",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "checkpoint_path",
    "runtime_seconds",
    "status",
    "error_message",
    "training_runtime_seconds",
    "total_runtime_seconds",
    "training_rows",
    "validation_rows",
    "test_rows",
    "seed",
    "sample_limit",
    "training_data_path",
    "dataset_source_path",
    "model",
]


class TrainingSplit(NamedTuple):
    train: Any
    validation: Any
    test: Any


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "sota").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and sota/.")


def configure_datasets_cache(root: Path) -> Path:
    cache_dir = root / ".hf_datasets_cache"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    return cache_dir


def build_email_text(subject: str | None, body: str | None) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def ensure_class_label(dataset: Any) -> Any:
    from datasets import ClassLabel

    if isinstance(dataset.features.get("label"), ClassLabel):
        return dataset
    return dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))


def split_training_dataset(dataset: Any, seed: int = SEED) -> TrainingSplit:
    dataset = ensure_class_label(dataset)
    holdout = dataset.train_test_split(
        test_size=HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    valid_test = holdout["test"].train_test_split(
        test_size=TEST_SPLIT / HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    return TrainingSplit(
        train=holdout["train"],
        validation=valid_test["train"],
        test=valid_test["test"],
    )


def label_counts(dataset: Any) -> tuple[int, int]:
    labels = [int(value) for value in dataset["label"]]
    spam = sum(1 for value in labels if value == 1)
    ham = sum(1 for value in labels if value == 0)
    return ham, spam


def stratified_sample(dataset: Any, limit: int, seed: int) -> Any:
    from datasets import concatenate_datasets

    if limit <= 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)

    labels = [int(value) for value in dataset["label"]]
    total = len(labels)
    selected_parts = []
    remaining = limit
    label_values = sorted(set(labels))
    for offset, label_value in enumerate(label_values):
        label_indices = [index for index, value in enumerate(labels) if value == label_value]
        if offset == len(label_values) - 1:
            take = min(remaining, len(label_indices))
        else:
            take = int(round(limit * len(label_indices) / total))
            take = max(0, min(take, len(label_indices), remaining))
        if take > 0:
            part = dataset.select(label_indices).shuffle(seed=seed + int(label_value)).select(range(take))
            selected_parts.append(part)
            remaining -= take

    if not selected_parts:
        return dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    return concatenate_datasets(selected_parts).shuffle(seed=seed)


def plain_sample(dataset: Any, limit: int, seed: int) -> Any:
    if limit <= 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(limit))


def filter_non_empty(dataset: Any) -> Any:
    return dataset.filter(lambda sample: bool(build_email_text(sample["subject"], sample["body"])))


def dataset_to_xy(dataset: Any) -> tuple[list[str], list[int]]:
    subjects = dataset["subject"]
    bodies = dataset["body"]
    labels = [int(value) for value in dataset["label"]]
    texts = [build_email_text(subject, body) for subject, body in zip(subjects, bodies, strict=False)]
    return texts, labels


def compute_metrics(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)

    if len(labels_array) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "false_positive_count": 0.0,
            "false_negative_count": 0.0,
            "true_positive_count": 0.0,
            "true_negative_count": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "spam_prediction_rate": 0.0,
            "classification_failure_count": 0.0,
            "classification_failure_rate": 0.0,
        }

    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=[0, 1],
        zero_division=0,
    )
    tp = int(((predictions_array == 1) & (labels_array == 1)).sum())
    fp = int(((predictions_array == 1) & (labels_array == 0)).sum())
    fn = int(((predictions_array == 0) & (labels_array == 1)).sum())
    tn = int(((predictions_array == 0) & (labels_array == 0)).sum())
    accuracy = float((predictions_array == labels_array).mean())
    specificity = float(tn / max(tn + fp, 1))
    recall = float(per_class_recall[1])

    return {
        "accuracy": accuracy,
        "precision": float(per_class_precision[1]),
        "recall": recall,
        "f1": float(per_class_f1[1]),
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "spam_prediction_rate": float((predictions_array == 1).mean()),
        "classification_failure_count": 0.0,
        "classification_failure_rate": 0.0,
    }


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    ngram_range = (int(args.ngram_min), int(args.ngram_max))
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                    min_df=int(args.min_df),
                    max_features=int(args.max_features) if args.max_features else None,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=float(args.svm_c),
                    random_state=int(args.seed),
                    dual="auto",
                    max_iter=int(args.max_iter),
                ),
            ),
        ]
    )


def empty_summary_row(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "method": METHOD,
            "evaluation_method": EVALUATION_METHOD,
            "dataset": dataset_name,
            "config_index": 0,
            "config_id": CONFIG_ID,
            "checkpoint_type": "single_model",
            "checkpoint_step": "final",
            "seed": args.seed,
            "sample_limit": args.sample_limit,
            "model": f"tfidf_{args.ngram_min}_{args.ngram_max}_linear_svm",
        }
    )
    return row


def summary_row(
    *,
    dataset_name: str,
    metrics: dict[str, float],
    rows: int,
    ham_count: int,
    spam_count: int,
    runtime_seconds: float,
    status: str,
    error_message: str,
    args: argparse.Namespace,
    training_runtime_seconds: float,
    total_runtime_seconds: float,
    training_rows: int,
    validation_rows: int,
    test_rows: int,
    training_data_path: str,
    dataset_source_path: str,
) -> dict[str, Any]:
    row = empty_summary_row(dataset_name, args)
    row.update(
        {
            "rows": rows,
            "ham_count": ham_count,
            "spam_count": spam_count,
            "runtime_seconds": round(runtime_seconds, 4),
            "status": status,
            "error_message": error_message,
            "training_runtime_seconds": round(training_runtime_seconds, 4),
            "total_runtime_seconds": round(total_runtime_seconds, 4),
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "training_data_path": training_data_path,
            "dataset_source_path": dataset_source_path,
        }
    )
    for key, value in metrics.items():
        row[key] = value
    return row


def load_parquet_dataset(path: str) -> Any:
    from datasets import load_dataset

    return load_dataset("parquet", data_files=str(path), split="train")


def build_evaluation_dataset(
    *,
    dataset_name: str,
    training_split: TrainingSplit,
    sample_limit: int,
    seed: int,
) -> tuple[Any, str]:
    from dataset.combine import combine_datasets

    if dataset_name == "train_subset":
        return stratified_sample(training_split.train, sample_limit, seed), "training_split"

    data_path = combine_datasets(dataset_name, duplicate_detection="high")
    dataset = load_parquet_dataset(data_path)
    dataset = filter_non_empty(dataset)
    if dataset_name == "spam_ham":
        dataset = stratified_sample(dataset, sample_limit, seed)
    else:
        dataset = plain_sample(dataset, sample_limit, seed)
    return dataset, str(data_path)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    root = project_root()
    configure_datasets_cache(root)
    from datasets import disable_progress_bars
    from dataset.combine import combine_datasets

    disable_progress_bars()
    total_start = time.perf_counter()
    training_data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_parquet_dataset(training_data_path)
    split = split_training_dataset(raw_dataset, seed=int(args.seed))
    train_dataset = filter_non_empty(split.train)

    if args.train_limit is not None:
        train_dataset = train_dataset.select(range(min(int(args.train_limit), len(train_dataset))))

    train_texts, train_labels = dataset_to_xy(train_dataset)
    pipeline = build_pipeline(args)

    train_start = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    training_runtime_seconds = time.perf_counter() - train_start

    rows = []
    exit_code = 0
    for dataset_name in args.datasets:
        eval_start = time.perf_counter()
        try:
            eval_dataset, dataset_source_path = build_evaluation_dataset(
                dataset_name=dataset_name,
                training_split=split,
                sample_limit=int(args.sample_limit),
                seed=int(args.seed),
            )
            eval_texts, eval_labels = dataset_to_xy(eval_dataset)
            predictions = pipeline.predict(eval_texts)
            metrics = compute_metrics(eval_labels, predictions)
            ham_count, spam_count = label_counts(eval_dataset)
            runtime_seconds = time.perf_counter() - eval_start
            rows.append(
                summary_row(
                    dataset_name=dataset_name,
                    metrics=metrics,
                    rows=len(eval_dataset),
                    ham_count=ham_count,
                    spam_count=spam_count,
                    runtime_seconds=runtime_seconds,
                    status="completed",
                    error_message="",
                    args=args,
                    training_runtime_seconds=training_runtime_seconds,
                    total_runtime_seconds=time.perf_counter() - total_start,
                    training_rows=len(train_dataset),
                    validation_rows=len(split.validation),
                    test_rows=len(split.test),
                    training_data_path=str(training_data_path),
                    dataset_source_path=dataset_source_path,
                )
            )
        except Exception as exc:
            exit_code = 1
            runtime_seconds = time.perf_counter() - eval_start
            error_message = f"{type(exc).__name__}: {exc}"
            rows.append(
                summary_row(
                    dataset_name=dataset_name,
                    metrics={},
                    rows=0,
                    ham_count=0,
                    spam_count=0,
                    runtime_seconds=runtime_seconds,
                    status="failed",
                    error_message=error_message,
                    args=args,
                    training_runtime_seconds=training_runtime_seconds,
                    total_runtime_seconds=time.perf_counter() - total_start,
                    training_rows=len(train_dataset),
                    validation_rows=len(split.validation),
                    test_rows=len(split.test),
                    training_data_path=str(training_data_path),
                    dataset_source_path="",
                )
            )
            traceback.print_exc()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    write_summary(output_path, rows)
    print(f"summary_path={output_path}")
    return exit_code


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run TF-IDF + Linear SVM spam baseline.")
    parser.add_argument("--output", default=str(script_dir / "summary.csv"))
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
