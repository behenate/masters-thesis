#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
DEFAULT_SAMPLE_LIMIT = 2000
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
METHOD = "fasttext_supervised"
FASTTEXT_INSTALL_COMMAND = "./.venv/bin/python -m pip install fasttext"
FASTTEXT_RUN_COMMAND = "./.venv/bin/python sota/fasttext/run_fasttext_baseline.py"
WHITESPACE_RE = re.compile(r"\s+")
EXPECTED_FASTTEXT_LABELS = {"__label__ham", "__label__spam"}

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
    "sample_limit",
    "seed",
    "train_rows",
    "validation_rows",
    "test_rows",
    "train_runtime_seconds",
    "runtime_seconds",
    "fasttext_epoch",
    "fasttext_lr",
    "fasttext_word_ngrams",
    "fasttext_dim",
    "fasttext_min_count",
    "fasttext_loss",
    "status",
    "error_message",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "lora-fine-tuning").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and lora-fine-tuning/.")


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def configure_hf_cache(cache_root: Path | None = None) -> Path:
    cache_root = (cache_root or method_dir() / ".cache" / "huggingface").resolve()
    hf_home = cache_root / "home"
    datasets_cache = cache_root / "datasets"
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    return cache_root


def build_email_text(subject: str | None, body: str | None) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def clean_fasttext_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("__label__", "label_")
    return WHITESPACE_RE.sub(" ", text).strip()


def label_text(label: int) -> str:
    return "spam" if int(label) == 1 else "ham"


def fasttext_line(sample: dict[str, Any]) -> str:
    text = clean_fasttext_text(build_email_text(sample.get("subject"), sample.get("body")))
    return f"__label__{label_text(int(sample['label']))} {text}".rstrip()


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


def nonempty_email(sample: dict[str, Any]) -> bool:
    return bool(build_email_text(sample.get("subject"), sample.get("body")))


def load_training_splits(seed: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    from datasets import ClassLabel, DatasetDict, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
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
    splits = DatasetDict(
        {
            "train": holdout["train"],
            "validation": valid_test["train"],
            "test": valid_test["test"],
        }
    ).filter(nonempty_email)

    train_ham, train_spam = label_counts(splits["train"])
    metadata = {
        "source_path": str(data_path),
        "raw_rows": len(raw_dataset),
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "train_ham_count": train_ham,
        "train_spam_count": train_spam,
    }
    return splits["train"], splits["validation"], splits["test"], metadata


def build_dataset_sample(dataset_name: str, sample_limit: int, seed: int) -> tuple[Any, dict[str, Any]]:
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()

    if dataset_name == "train_subset":
        data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
        raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
        holdout = raw_dataset.train_test_split(
            test_size=HOLDOUT_SPLIT,
            stratify_by_column="label",
            seed=seed,
        )
        dataset = stratified_sample(holdout["train"], sample_limit, seed).filter(nonempty_email)
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        dataset = dataset.filter(nonempty_email)
        if dataset_name == "spam_ham":
            dataset = stratified_sample(dataset, sample_limit, seed)
        else:
            dataset = plain_sample(dataset, sample_limit, seed)

    ham, spam = label_counts(dataset)
    metadata = {
        "dataset": dataset_name,
        "source_path": str(data_path),
        "rows": len(dataset),
        "ham_count": ham,
        "spam_count": spam,
    }
    return dataset, metadata


def binary_metrics(
    predictions: list[int],
    labels: list[int],
    failure_count: int,
) -> dict[str, float]:
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
        "classification_failure_count": float(failure_count),
        "classification_failure_rate": float(failure_count / max(rows, 1)),
    }


def import_fasttext() -> Any:
    try:
        return importlib.import_module("fasttext")
    except ModuleNotFoundError as exc:
        message = (
            "fasttext is not installed in ./.venv. "
            f"Install it with: {FASTTEXT_INSTALL_COMMAND}; "
            f"then run: {FASTTEXT_RUN_COMMAND}"
        )
        raise RuntimeError(message) from exc


def write_training_file(dataset: Any, path: Path) -> int:
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in dataset:
            line = fasttext_line(sample)
            if not line.split(" ", 1)[-1].strip():
                continue
            handle.write(line)
            handle.write("\n")
            rows += 1
    return rows


def predict_label(model: Any, sample: dict[str, Any]) -> tuple[int, bool]:
    text = clean_fasttext_text(build_email_text(sample.get("subject"), sample.get("body")))
    predictions = model.f.predict(f"{text}\n", 1, 0.0, "strict")
    predicted = predictions[0][1].replace("__label__", "") if predictions else ""
    if predicted == "spam":
        return 1, False
    if predicted == "ham":
        return 0, False
    return 0, True


def evaluate_dataset(model: Any, dataset: Any) -> tuple[dict[str, float], float]:
    predictions: list[int] = []
    labels: list[int] = []
    failure_count = 0
    started = time.perf_counter()

    for sample in dataset:
        prediction, failed = predict_label(model, sample)
        predictions.append(prediction)
        labels.append(int(sample["label"]))
        failure_count += int(failed)

    return binary_metrics(predictions, labels, failure_count), time.perf_counter() - started


def empty_summary_row(
    *,
    dataset_name: str,
    args: argparse.Namespace,
    train_metadata: dict[str, Any] | None,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset": dataset_name,
            "method": METHOD,
            "sample_limit": args.sample_limit,
            "seed": args.seed,
            "fasttext_epoch": args.epoch,
            "fasttext_lr": args.lr,
            "fasttext_word_ngrams": args.word_ngrams,
            "fasttext_dim": args.dim,
            "fasttext_min_count": args.min_count,
            "fasttext_loss": args.loss,
            "status": status,
            "error_message": error_message,
        }
    )
    if train_metadata:
        row.update(
            {
                "train_rows": train_metadata.get("train_rows", ""),
                "validation_rows": train_metadata.get("validation_rows", ""),
                "test_rows": train_metadata.get("test_rows", ""),
            }
        )
    return row


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(args: argparse.Namespace) -> Path:
    project_root()
    configure_hf_cache()
    fasttext = import_fasttext()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = method_dir() / output_path
    output_path = output_path.resolve()

    log("Preparing 92/2/6 training split from dataset.combine training_all.")
    train_dataset, _, _, train_metadata = load_training_splits(args.seed)
    log(
        "Training rows: "
        f"{train_metadata['train_rows']} "
        f"(ham={train_metadata['train_ham_count']}, spam={train_metadata['train_spam_count']})."
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fasttext_baseline_") as tmp_dir:
        train_path = Path(tmp_dir) / "train.txt"
        written_rows = write_training_file(train_dataset, train_path)
        train_metadata["train_rows"] = written_rows
        log(f"Wrote fastText training file with {written_rows} rows.")

        started = time.perf_counter()
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=args.lr,
            epoch=args.epoch,
            wordNgrams=args.word_ngrams,
            dim=args.dim,
            minCount=args.min_count,
            loss=args.loss,
            thread=args.thread,
            verbose=args.verbose,
        )
        train_runtime = time.perf_counter() - started
        model_labels = set(getattr(model, "labels", []))
        if model_labels != EXPECTED_FASTTEXT_LABELS:
            raise RuntimeError(f"Unexpected fastText labels: {sorted(model_labels)}")
        log(f"Training finished in {train_runtime:.2f}s.")
        log(f"Model labels: {', '.join(sorted(model_labels))}.")

        if args.model_output:
            model_output = Path(args.model_output).expanduser()
            if not model_output.is_absolute():
                model_output = method_dir() / model_output
            model_output.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_output.resolve()))
            log(f"Saved model to {model_output.resolve()}.")

        for dataset_name in args.datasets:
            row = empty_summary_row(
                dataset_name=dataset_name,
                args=args,
                train_metadata=train_metadata,
                status="failed",
            )
            row["train_runtime_seconds"] = round(train_runtime, 4)
            try:
                log(f"Evaluating {dataset_name}.")
                dataset, metadata = build_dataset_sample(dataset_name, args.sample_limit, args.seed)
                metrics, runtime = evaluate_dataset(model, dataset)
                row.update(metadata)
                row.update(metrics)
                row["runtime_seconds"] = round(runtime, 4)
                row["status"] = "completed"
                row["error_message"] = ""
            except Exception as exc:
                row["error_message"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_summary(rows, output_path)
            log(f"Updated summary: {output_path}")

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fastText supervised spam classifier and evaluate thesis baseline datasets."
    )
    parser.add_argument("--output", default="summary.csv", help="CSV path relative to sota/fasttext unless absolute.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DEFAULT_DATASETS)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--loss", default="softmax", choices=["softmax", "hs", "ova"])
    parser.add_argument("--thread", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--model-output", default="", help="Optional fastText .bin model output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary_path = run_baseline(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    log(f"Summary written to {summary_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
