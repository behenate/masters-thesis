#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


EVALUATION_METHOD = "tfidf_logistic_regression"
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
DEFAULT_SAMPLE_LIMIT = 2000
DEFAULT_DATASETS_CACHE_DIR = Path("/tmp") / "tfidf_logistic_regression_hf_datasets_cache"

SUMMARY_COLUMNS = [
    "dataset",
    "method",
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
    "parse_failure_count",
    "parse_failure_rate",
    "eval_batch_size",
    "max_seq_length",
    "model_id",
    "checkpoint_path",
    "runtime_seconds",
    "status",
    "error_message",
    "train_rows",
    "train_ham_count",
    "train_spam_count",
    "validation_rows",
    "test_rows",
    "train_runtime_seconds",
    "sample_limit",
    "seed",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "sota").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and sota/.")


def resolve_output_path(value: str | None) -> Path:
    if value is None:
        return method_dir() / "summary.csv"
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def resolve_datasets_cache_dir(value: str | None) -> Path:
    path = Path(value).expanduser() if value else DEFAULT_DATASETS_CACHE_DIR
    if not path.is_absolute():
        path = project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_email_text(subject: str | None, body: str | None) -> str:
    subject_text = clean_text(subject)
    body_text = clean_text(body)
    parts = []
    if subject_text:
        parts.append(f"Subject: {subject_text}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts).strip()


def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"spam", "1", "true"}:
            return 1
        if normalized in {"ham", "valid", "0", "false"}:
            return 0
    return int(value)


def label_counts_from_values(labels: list[int]) -> tuple[int, int]:
    spam = sum(1 for value in labels if int(value) == 1)
    ham = sum(1 for value in labels if int(value) == 0)
    return ham, spam


def dataframe_from_dataset(dataset: Any) -> pd.DataFrame:
    frame = dataset.to_pandas() if hasattr(dataset, "to_pandas") else dataset.copy()
    frame["subject"] = frame["subject"].map(clean_text)
    frame["body"] = frame["body"].map(clean_text)
    frame["label"] = frame["label"].map(normalize_label).astype(int)
    frame["text"] = [
        build_email_text(subject, body)
        for subject, body in zip(frame["subject"], frame["body"], strict=False)
    ]
    frame = frame[frame["text"].astype(bool)].reset_index(drop=True)
    return frame


def plain_sample(dataset: Any, limit: int, seed: int) -> Any:
    if limit <= 0:
        return dataset
    if isinstance(dataset, pd.DataFrame):
        if len(dataset) <= limit:
            return dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        return dataset.sample(n=limit, random_state=seed).reset_index(drop=True)
    if len(dataset) <= limit:
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(limit))


def stratified_sample(dataset: Any, limit: int, seed: int) -> Any:
    if limit <= 0:
        return dataset
    if len(dataset) <= limit:
        return plain_sample(dataset, limit, seed)

    if isinstance(dataset, pd.DataFrame):
        labels = [int(value) for value in dataset["label"].tolist()]
        selected_parts = []
        remaining = limit
        label_values = sorted(set(labels))
        total = len(labels)
        for offset, label_value in enumerate(label_values):
            label_frame = dataset[dataset["label"] == label_value]
            if offset == len(label_values) - 1:
                take = min(remaining, len(label_frame))
            else:
                take = int(round(limit * len(label_frame) / total))
                take = max(0, min(take, len(label_frame), remaining))
            if take > 0:
                selected_parts.append(label_frame.sample(n=take, random_state=seed + int(label_value)))
                remaining -= take
        if not selected_parts:
            return plain_sample(dataset, limit, seed)
        return pd.concat(selected_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    from datasets import concatenate_datasets

    labels = [int(value) for value in dataset["label"]]
    selected_parts = []
    remaining = limit
    label_values = sorted(set(labels))
    total = len(labels)
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
        return plain_sample(dataset, limit, seed)
    return concatenate_datasets(selected_parts).shuffle(seed=seed)


def split_training_dataset(seed: int, datasets_cache_dir: str | None = None) -> tuple[Any, Any, Any, str]:
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    cache_dir = resolve_datasets_cache_dir(datasets_cache_dir)
    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train", cache_dir=str(cache_dir))
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))

    holdout = raw_dataset.train_test_split(
        test_size=HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    valid_test = holdout["test"].train_test_split(
        test_size=TEST_SPLIT / HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=seed,
    )
    return holdout["train"], valid_test["train"], valid_test["test"], str(data_path)


def load_training_frame(seed: int, train_limit: int | None, datasets_cache_dir: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_dataset, validation_dataset, test_dataset, data_path = split_training_dataset(seed, datasets_cache_dir)
    if train_limit is not None:
        train_dataset = stratified_sample(train_dataset, train_limit, seed)
    train_frame = dataframe_from_dataset(train_dataset)
    ham, spam = label_counts_from_values(train_frame["label"].tolist())
    metadata = {
        "source_path": data_path,
        "train_rows": len(train_frame),
        "train_ham_count": ham,
        "train_spam_count": spam,
        "validation_rows": len(validation_dataset),
        "test_rows": len(test_dataset),
    }
    return train_frame, metadata


def load_evaluation_frame(
    dataset_name: str,
    sample_limit: int,
    seed: int,
    datasets_cache_dir: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    cache_dir = resolve_datasets_cache_dir(datasets_cache_dir)
    if dataset_name == "train_subset":
        train_dataset, _, _, data_path = split_training_dataset(seed, datasets_cache_dir)
        dataset = stratified_sample(train_dataset, sample_limit, seed)
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train", cache_dir=str(cache_dir))
        dataset = dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
        dataset = dataset.filter(lambda sample: bool(build_email_text(sample["subject"], sample["body"])))
        if dataset_name == "spam_ham":
            dataset = stratified_sample(dataset, sample_limit, seed)
        else:
            dataset = plain_sample(dataset, sample_limit, seed)

    frame = dataframe_from_dataset(dataset)
    ham, spam = label_counts_from_values(frame["label"].tolist())
    metadata = {
        "dataset": dataset_name,
        "source_path": str(data_path),
        "rows": len(frame),
        "ham_count": ham,
        "spam_count": spam,
    }
    return frame, metadata


def binary_metrics(predictions: list[int], labels: list[int]) -> dict[str, float]:
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


def config_id(args: argparse.Namespace | None = None) -> str:
    if args is None:
        return "tfidf_word_1_2_logreg_liblinear_c1"
    max_df = str(args.max_df).replace(".", "p")
    c_value = str(args.C).replace(".", "p")
    return (
        f"tfidf_word_1_{args.ngram_max}"
        f"_max{args.max_features}_mindf{args.min_df}_maxdf{max_df}"
        f"_logreg_{args.solver}_c{c_value}"
    )


def model_id(args: argparse.Namespace | None = None) -> str:
    if args is None:
        return "sklearn:TfidfVectorizer+LogisticRegression"
    return (
        "sklearn:"
        f"TfidfVectorizer(max_features={args.max_features},ngram_range=(1,{args.ngram_max}),"
        f"min_df={args.min_df},max_df={args.max_df},sublinear_tf=True)"
        f"+LogisticRegression(solver={args.solver},C={args.C})"
    )


def summary_row(
    *,
    dataset_name: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    runtime_seconds: float,
    args: argparse.Namespace | None = None,
    train_metadata: dict[str, Any] | None = None,
    train_runtime_seconds: float = 0.0,
    status: str = "completed",
    error_message: str = "",
) -> dict[str, Any]:
    train_metadata = train_metadata or {}
    row = {
        "dataset": dataset_name,
        "method": EVALUATION_METHOD,
        "config_index": 0,
        "config_id": config_id(args),
        "checkpoint_type": "sota_baseline",
        "checkpoint_step": "none",
        "rows": metadata.get("rows", 0),
        "ham_count": metadata.get("ham_count", 0),
        "spam_count": metadata.get("spam_count", 0),
        "eval_batch_size": "",
        "max_seq_length": "",
        "model_id": model_id(args),
        "checkpoint_path": "",
        "runtime_seconds": round(runtime_seconds, 4),
        "status": status,
        "error_message": error_message,
        "train_rows": train_metadata.get("train_rows", ""),
        "train_ham_count": train_metadata.get("train_ham_count", ""),
        "train_spam_count": train_metadata.get("train_spam_count", ""),
        "validation_rows": train_metadata.get("validation_rows", ""),
        "test_rows": train_metadata.get("test_rows", ""),
        "train_runtime_seconds": round(train_runtime_seconds, 4),
        "sample_limit": args.sample_limit if args is not None else "",
        "seed": args.seed if args is not None else SEED,
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


def fit_model(train_frame: pd.DataFrame, args: argparse.Namespace) -> tuple[Any, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, args.ngram_max),
                    max_features=args.max_features,
                    min_df=args.min_df,
                    max_df=args.max_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.C,
                    solver=args.solver,
                    max_iter=args.max_iter,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(train_frame["text"].tolist(), train_frame["label"].tolist())
    return pipeline, time.perf_counter() - started


def evaluate_dataset(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    predictions = [int(value) for value in model.predict(frame["text"].tolist()).tolist()]
    labels = [int(value) for value in frame["label"].tolist()]
    return binary_metrics(predictions, labels)


def run(args: argparse.Namespace) -> int:
    project_root()
    output_path = resolve_output_path(args.output)
    log("Loading training split from dataset.combine training_all.")
    train_frame, train_metadata = load_training_frame(args.seed, args.train_limit, args.datasets_cache_dir)
    log(
        "Training rows: "
        f"{train_metadata['train_rows']} "
        f"(ham={train_metadata['train_ham_count']}, spam={train_metadata['train_spam_count']})."
    )
    model, train_runtime_seconds = fit_model(train_frame, args)
    log(f"Training finished in {train_runtime_seconds:.4f}s.")

    rows = []
    for dataset_name in args.datasets:
        started = time.perf_counter()
        try:
            log(f"Preparing evaluation dataset: {dataset_name}.")
            frame, metadata = load_evaluation_frame(dataset_name, args.sample_limit, args.seed, args.datasets_cache_dir)
            metrics = evaluate_dataset(model, frame)
            runtime_seconds = time.perf_counter() - started
            rows.append(
                summary_row(
                    dataset_name=dataset_name,
                    metadata=metadata,
                    metrics=metrics,
                    runtime_seconds=runtime_seconds,
                    args=args,
                    train_metadata=train_metadata,
                    train_runtime_seconds=train_runtime_seconds,
                )
            )
            log(
                f"{dataset_name}: accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}."
            )
        except Exception as exc:
            runtime_seconds = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            rows.append(
                summary_row(
                    dataset_name=dataset_name,
                    metadata={},
                    metrics={},
                    runtime_seconds=runtime_seconds,
                    args=args,
                    train_metadata=train_metadata,
                    train_runtime_seconds=train_runtime_seconds,
                    status="failed",
                    error_message=message,
                )
            )
            log(f"{dataset_name}: failed with {message}")
            if not args.keep_going:
                raise

    write_csv(output_path, rows, SUMMARY_COLUMNS)
    log(f"Wrote: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TF-IDF + Logistic Regression spam baseline.")
    parser.add_argument("--output", default=None, help="CSV output path. Defaults to this method's summary.csv.")
    parser.add_argument("--datasets", nargs="+", choices=list(DEFAULT_DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-limit", type=int, default=None, help="Optional stratified training row cap for smoke tests.")
    parser.add_argument("--datasets-cache-dir", default=None, help=f"Hugging Face datasets cache directory. Defaults to {DEFAULT_DATASETS_CACHE_DIR}.")
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", choices=["liblinear", "lbfgs", "saga"], default="liblinear")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be >= 1")
    if args.train_limit is not None and args.train_limit < 2:
        raise SystemExit("--train-limit must be >= 2")
    if args.max_features < 1:
        raise SystemExit("--max-features must be >= 1")
    if args.ngram_max < 1:
        raise SystemExit("--ngram-max must be >= 1")
    if args.min_df < 1:
        raise SystemExit("--min-df must be >= 1")
    if not 0.0 < args.max_df <= 1.0:
        raise SystemExit("--max-df must be in (0.0, 1.0]")
    if args.C <= 0:
        raise SystemExit("--C must be > 0")
    if args.max_iter < 1:
        raise SystemExit("--max-iter must be >= 1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
