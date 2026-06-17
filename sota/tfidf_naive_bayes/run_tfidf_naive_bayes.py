#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any


SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
METHOD = "tfidf_multinomial_naive_bayes"
MODEL_ID = "TfidfVectorizer + MultinomialNB"
SUMMARY_COLUMNS = [
    "dataset",
    "method",
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
    "parse_failure_count",
    "parse_failure_rate",
    "runtime_seconds",
    "status",
    "error_message",
    "model_id",
    "training_rows",
    "training_ham_count",
    "training_spam_count",
    "training_runtime_seconds",
    "sample_limit",
    "seed",
    "vectorizer_max_features",
    "vectorizer_ngram_range",
    "vectorizer_min_df",
    "vectorizer_max_df",
    "alpha",
    "source_path",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "lora-fine-tuning").is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and lora-fine-tuning/.")


def load_dataset_kwargs() -> dict[str, str]:
    cache_dir = project_root() / ".hf_datasets_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": str(cache_dir)}


def default_summary_path() -> Path:
    return method_dir() / "summary.csv"


def resolve_output_path(value: str | None) -> Path:
    if value is None:
        return default_summary_path()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def build_email_text(subject: Any, body: Any) -> str:
    subject_text = "" if subject is None else str(subject).strip()
    body_text = "" if body is None else str(body).strip()
    parts = []
    if subject_text:
        parts.append(f"Subject: {subject_text}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts).strip()


def label_counts_from_labels(labels: list[int]) -> tuple[int, int]:
    ham = sum(1 for label in labels if int(label) == 0)
    spam = sum(1 for label in labels if int(label) == 1)
    return ham, spam


def label_counts(dataset: Any) -> tuple[int, int]:
    return label_counts_from_labels([int(value) for value in dataset["label"]])


def split_training_dataset(raw_dataset: Any, seed: int) -> Any:
    from datasets import DatasetDict

    holdout = raw_dataset.train_test_split(
        test_size=HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    validation_test = holdout["test"].train_test_split(
        test_size=TEST_SPLIT / HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    return DatasetDict(
        {
            "train": holdout["train"],
            "validation": validation_test["train"],
            "test": validation_test["test"],
        }
    )


def plain_sample(dataset: Any, limit: int, seed: int) -> Any:
    if limit <= 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(limit))


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


def nonempty_text_rows(dataset: Any) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for sample in dataset:
        text = build_email_text(sample.get("subject"), sample.get("body"))
        if not text:
            continue
        texts.append(text)
        labels.append(int(sample["label"]))
    return texts, labels


def load_training_splits(seed: int) -> tuple[Any, dict[str, Any]]:
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train", **load_dataset_kwargs())
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
    splits = split_training_dataset(raw_dataset, seed)
    train_texts, train_labels = nonempty_text_rows(splits["train"])
    train_ham, train_spam = label_counts_from_labels(train_labels)
    metadata = {
        "source_path": str(data_path),
        "raw_rows": len(raw_dataset),
        "train_rows": len(train_texts),
        "train_ham_count": train_ham,
        "train_spam_count": train_spam,
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
    }
    return splits, metadata


def load_evaluation_dataset(
    dataset_name: str,
    *,
    training_splits: Any,
    sample_limit: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    from datasets import disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    if dataset_name == "train_subset":
        dataset = stratified_sample(training_splits["train"], sample_limit, seed)
        source_path = "training_all/train_split"
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train", **load_dataset_kwargs())
        texts, labels = nonempty_text_rows(dataset)
        if len(texts) != len(dataset):
            nonempty_indices = [
                index
                for index, sample in enumerate(dataset)
                if build_email_text(sample.get("subject"), sample.get("body"))
            ]
            dataset = dataset.select(nonempty_indices)
        if dataset_name == "spam_ham":
            dataset = stratified_sample(dataset, sample_limit, seed)
        else:
            dataset = plain_sample(dataset, sample_limit, seed)
        source_path = str(data_path)

    texts, labels = nonempty_text_rows(dataset)
    ham, spam = label_counts_from_labels(labels)
    metadata = {
        "dataset": dataset_name,
        "source_path": source_path,
        "rows": len(texts),
        "ham_count": ham,
        "spam_count": spam,
    }
    return dataset, metadata


def build_classifier(args: argparse.Namespace) -> Any:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        dtype=np.float32,
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", MultinomialNB(alpha=args.alpha)),
        ]
    )


def compute_binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    rows = len(labels)
    tp = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 0 and label == 0)
    accuracy = (tp + tn) / max(rows, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (recall + specificity) / 2
    spam_predictions = sum(1 for value in predictions if value == 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "spam_prediction_rate": float(spam_predictions / max(rows, 1)),
        "classification_failure_count": 0.0,
        "classification_failure_rate": 0.0,
        "parse_failure_count": 0.0,
        "parse_failure_rate": 0.0,
    }


def empty_metrics() -> dict[str, float]:
    return compute_binary_metrics([], [])


def summary_row(
    *,
    dataset_name: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    runtime_seconds: float,
    training_metadata: dict[str, Any],
    training_runtime_seconds: float,
    status: str = "completed",
    error_message: str = "",
) -> dict[str, Any]:
    row = {
        "dataset": dataset_name,
        "method": METHOD,
        "rows": metadata.get("rows", 0),
        "ham_count": metadata.get("ham_count", 0),
        "spam_count": metadata.get("spam_count", 0),
        "runtime_seconds": round(runtime_seconds, 4),
        "status": status,
        "error_message": error_message,
        "model_id": MODEL_ID,
        "training_rows": training_metadata.get("train_rows", ""),
        "training_ham_count": training_metadata.get("train_ham_count", ""),
        "training_spam_count": training_metadata.get("train_spam_count", ""),
        "training_runtime_seconds": round(training_runtime_seconds, 4),
        "sample_limit": args.sample_limit,
        "seed": args.seed,
        "vectorizer_max_features": args.max_features,
        "vectorizer_ngram_range": f"1,{args.ngram_max}",
        "vectorizer_min_df": args.min_df,
        "vectorizer_max_df": args.max_df,
        "alpha": args.alpha,
        "source_path": metadata.get("source_path", ""),
    }
    for key in SUMMARY_COLUMNS:
        row.setdefault(key, metrics.get(key, ""))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], float]:
    splits, training_metadata = load_training_splits(args.seed)
    train_texts, train_labels = nonempty_text_rows(splits["train"])
    if not train_texts:
        raise ValueError("Training split has no non-empty emails.")
    model = build_classifier(args)
    started = time.perf_counter()
    model.fit(train_texts, train_labels)
    training_runtime_seconds = time.perf_counter() - started
    return model, splits, training_metadata, training_runtime_seconds


def evaluate_dataset(
    *,
    model: Any,
    dataset_name: str,
    training_splits: Any,
    args: argparse.Namespace,
    training_metadata: dict[str, Any],
    training_runtime_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata: dict[str, Any] = {"dataset": dataset_name}
    try:
        dataset, metadata = load_evaluation_dataset(
            dataset_name,
            training_splits=training_splits,
            sample_limit=args.sample_limit,
            seed=args.seed,
        )
        texts, labels = nonempty_text_rows(dataset)
        metadata["rows"] = len(texts)
        metadata["ham_count"], metadata["spam_count"] = label_counts_from_labels(labels)
        if texts:
            predictions = [int(value) for value in model.predict(texts)]
            metrics = compute_binary_metrics(labels, predictions)
        else:
            metrics = empty_metrics()
        runtime_seconds = time.perf_counter() - started
        return summary_row(
            dataset_name=dataset_name,
            metadata=metadata,
            metrics=metrics,
            args=args,
            runtime_seconds=runtime_seconds,
            training_metadata=training_metadata,
            training_runtime_seconds=training_runtime_seconds,
        )
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        return summary_row(
            dataset_name=dataset_name,
            metadata=metadata,
            metrics={},
            args=args,
            runtime_seconds=runtime_seconds,
            training_metadata=training_metadata,
            training_runtime_seconds=training_runtime_seconds,
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )


def run(args: argparse.Namespace) -> int:
    project_root()
    output_path = resolve_output_path(args.output)
    log("Loading training data and fitting TF-IDF + MultinomialNB.")
    model, training_splits, training_metadata, training_runtime_seconds = train_model(args)
    log(
        "Training complete: "
        f"{training_metadata['train_rows']} rows "
        f"({training_metadata['train_ham_count']} ham, {training_metadata['train_spam_count']} spam)."
    )

    rows = []
    failures = 0
    for dataset_name in args.datasets:
        log(f"Evaluating {dataset_name}.")
        row = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            training_splits=training_splits,
            args=args,
            training_metadata=training_metadata,
            training_runtime_seconds=training_runtime_seconds,
        )
        rows.append(row)
        if row["status"] != "completed":
            failures += 1
            log(f"{dataset_name} failed: {row['error_message']}")

    write_csv(output_path, rows, SUMMARY_COLUMNS)
    metadata_path = output_path.with_suffix(".metadata.json")
    write_run_metadata(
        metadata_path,
        {
            "method": METHOD,
            "model_id": MODEL_ID,
            "summary_path": str(output_path),
            "datasets": list(args.datasets),
            "sample_limit": args.sample_limit,
            "seed": args.seed,
            "train_split": TRAIN_SPLIT,
            "validation_split": VALIDATION_SPLIT,
            "test_split": TEST_SPLIT,
            "training_metadata": training_metadata,
            "training_runtime_seconds": training_runtime_seconds,
            "vectorizer": {
                "max_features": args.max_features,
                "ngram_range": [1, args.ngram_max],
                "min_df": args.min_df,
                "max_df": args.max_df,
            },
            "alpha": args.alpha,
        },
    )
    log(f"Wrote: {output_path}")
    log(f"Wrote: {metadata_path}")
    return 1 if failures and not args.keep_going else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a TF-IDF + Multinomial Naive Bayes spam baseline."
    )
    parser.add_argument("--output", default=None, help="Summary CSV path. Defaults to sota/tfidf_naive_bayes/summary.csv.")
    parser.add_argument("--datasets", nargs="+", choices=list(DEFAULT_DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-limit", "--limit", dest="sample_limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be >= 1")
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0")
    if args.max_features < 1:
        raise SystemExit("--max-features must be >= 1")
    if args.min_df < 1:
        raise SystemExit("--min-df must be >= 1")
    if not 0.0 < args.max_df <= 1.0:
        raise SystemExit("--max-df must be in (0, 1]")
    if args.alpha <= 0:
        raise SystemExit("--alpha must be > 0")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
