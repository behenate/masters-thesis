#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


METHOD_NAME = "minilm_logistic_regression"
EVALUATION_METHOD = "sentence_bert_embeddings_logistic_regression"
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
DEFAULT_SAMPLE_LIMIT = 2000
SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
INSTALL_COMMAND = "./.venv/bin/python -m pip install sentence-transformers"
RUN_COMMAND = "./.venv/bin/python sota/minilm_logistic_regression/run_minilm_logistic_regression.py"

SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "evaluation_method",
    "model_id",
    "classifier",
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
    "training_rows",
    "training_ham_count",
    "training_spam_count",
    "training_runtime_seconds",
    "runtime_seconds",
    "status",
    "error_message",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "lora-fine-tuning").is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and lora-fine-tuning/.")


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def configure_huggingface_cache(cache_root: Path | None = None) -> Path:
    cache_dir = (cache_root or method_dir() / ".hf_cache").resolve()
    datasets_cache = cache_dir / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    return Path(os.environ["HF_HOME"])


def build_email_text(subject: str | None, body: str | None) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def label_counts(dataset: Any) -> tuple[int, int]:
    labels = [int(value) for value in dataset["label"]]
    spam = sum(1 for value in labels if value == 1)
    ham = sum(1 for value in labels if value == 0)
    return ham, spam


def split_training_dataset(dataset: Any, seed: int = SEED) -> Any:
    from datasets import DatasetDict

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
    return DatasetDict(
        {
            "train": holdout["train"],
            "validation": valid_test["train"],
            "test": valid_test["test"],
        }
    )


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


def dataset_to_texts_and_labels(dataset: Any) -> tuple[list[str], list[int]]:
    texts = [
        build_email_text(subject, body)
        for subject, body in zip(dataset["subject"], dataset["body"], strict=False)
    ]
    labels = [int(value) for value in dataset["label"]]
    return texts, labels


def filter_nonempty_emails(dataset: Any) -> Any:
    return dataset.filter(lambda sample: bool(build_email_text(sample["subject"], sample["body"])))


def load_training_splits(seed: int) -> tuple[Any, dict[str, Any]]:
    configure_huggingface_cache()

    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
    splits = split_training_dataset(raw_dataset, seed=seed)
    splits = splits.filter(lambda sample: bool(build_email_text(sample["subject"], sample["body"])))
    train_ham, train_spam = label_counts(splits["train"])
    metadata = {
        "source_path": str(data_path),
        "train_rows": len(splits["train"]),
        "train_ham_count": train_ham,
        "train_spam_count": train_spam,
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
    }
    return splits, metadata


def build_dataset_sample(
    dataset_name: str,
    sample_limit: int,
    seed: int,
    training_splits: Any,
) -> tuple[Any, dict[str, Any]]:
    configure_huggingface_cache()

    from datasets import disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()

    if dataset_name == "train_subset":
        data_path = "training_split"
        dataset = stratified_sample(training_splits["train"], sample_limit, seed)
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        dataset = filter_nonempty_emails(dataset)
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


def compute_metrics_row(
    *,
    dataset_name: str,
    labels: Sequence[int],
    predictions: Sequence[int],
    runtime_seconds: float,
    method: str = METHOD_NAME,
    evaluation_method: str = EVALUATION_METHOD,
    model_id: str = DEFAULT_MODEL_ID,
    classifier: str = "LogisticRegression",
    training_metadata: dict[str, Any] | None = None,
    status: str = "completed",
    error_message: str = "",
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    rows = int(labels_array.size)
    training_metadata = training_metadata or {}

    if rows == 0:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
        }
        tp = fp = fn = tn = 0
    else:
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
        specificity = tn / max(tn + fp, 1)
        metrics = {
            "accuracy": float((predictions_array == labels_array).mean()),
            "precision": float(per_class_precision[1]),
            "recall": float(per_class_recall[1]),
            "f1": float(per_class_f1[1]),
            "specificity": float(specificity),
            "balanced_accuracy": float((per_class_recall[1] + specificity) / 2),
        }

    ham_count = int((labels_array == 0).sum()) if rows else 0
    spam_count = int((labels_array == 1).sum()) if rows else 0
    return {
        "dataset": dataset_name,
        "method": method,
        "evaluation_method": evaluation_method,
        "model_id": model_id,
        "classifier": classifier,
        "rows": rows,
        "ham_count": ham_count,
        "spam_count": spam_count,
        **metrics,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "spam_prediction_rate": float((predictions_array == 1).mean()) if rows else 0.0,
        "classification_failure_count": 0,
        "classification_failure_rate": 0.0,
        "training_rows": int(training_metadata.get("train_rows", 0)),
        "training_ham_count": int(training_metadata.get("train_ham_count", 0)),
        "training_spam_count": int(training_metadata.get("train_spam_count", 0)),
        "training_runtime_seconds": float(training_metadata.get("training_runtime_seconds", 0.0)),
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "error_message": error_message,
    }


def error_row(
    *,
    dataset_name: str,
    runtime_seconds: float,
    model_id: str,
    training_metadata: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    return compute_metrics_row(
        dataset_name=dataset_name,
        labels=[],
        predictions=[],
        runtime_seconds=runtime_seconds,
        model_id=model_id,
        training_metadata=training_metadata,
        status="error",
        error_message=error_message,
    )


def write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def load_sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: sentence_transformers. "
            f"Install it with `{INSTALL_COMMAND}`, then run `{RUN_COMMAND}`."
        ) from exc
    return SentenceTransformer


def make_sentence_transformer(args: argparse.Namespace) -> Any:
    SentenceTransformer = load_sentence_transformer_class()
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
    }
    if args.cache_dir:
        kwargs["cache_folder"] = str(Path(args.cache_dir).expanduser())
    if args.device != "auto":
        kwargs["device"] = args.device
    return SentenceTransformer(args.model_id, **kwargs)


def encode_texts(model: Any, texts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=args.batch_size,
        show_progress_bar=not args.no_progress,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize_embeddings,
    )


def train_classifier(
    *,
    model: Any,
    train_dataset: Any,
    args: argparse.Namespace,
) -> tuple[LogisticRegression, dict[str, Any]]:
    train_started = time.perf_counter()
    train_texts, train_labels = dataset_to_texts_and_labels(train_dataset)
    log(f"Encoding {len(train_texts)} training emails with {args.model_id}")
    train_embeddings = encode_texts(model, train_texts, args)
    class_weight = None if args.class_weight == "none" else args.class_weight
    classifier = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.seed,
        class_weight=class_weight,
    )
    log("Training LogisticRegression classifier")
    classifier.fit(train_embeddings, np.asarray(train_labels, dtype=int))
    training_runtime_seconds = time.perf_counter() - train_started
    ham, spam = label_counts(train_dataset)
    metadata = {
        "train_rows": len(train_dataset),
        "train_ham_count": ham,
        "train_spam_count": spam,
        "training_runtime_seconds": training_runtime_seconds,
    }
    return classifier, metadata


def evaluate_dataset(
    *,
    dataset_name: str,
    dataset: Any,
    model: Any,
    classifier: LogisticRegression,
    args: argparse.Namespace,
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    texts, labels = dataset_to_texts_and_labels(dataset)
    log(f"Encoding and evaluating {dataset_name} ({len(texts)} rows)")
    embeddings = encode_texts(model, texts, args)
    predictions = classifier.predict(embeddings).astype(int).tolist()
    runtime_seconds = time.perf_counter() - started
    return compute_metrics_row(
        dataset_name=dataset_name,
        labels=labels,
        predictions=predictions,
        runtime_seconds=runtime_seconds,
        model_id=args.model_id,
        training_metadata=training_metadata,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniLM/Sentence-BERT embeddings plus Logistic Regression SOTA baseline.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--output", type=Path, default=method_dir() / "summary.csv")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root()
    configure_huggingface_cache()
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be >= 1")

    log("Preparing training split from dataset.combine training_all")
    training_splits, split_metadata = load_training_splits(args.seed)
    log(
        "Training split rows: "
        f"{split_metadata['train_rows']} "
        f"(ham={split_metadata['train_ham_count']}, spam={split_metadata['train_spam_count']})"
    )

    log(f"Loading SentenceTransformer model: {args.model_id}")
    model = make_sentence_transformer(args)
    classifier, training_metadata = train_classifier(
        model=model,
        train_dataset=training_splits["train"],
        args=args,
    )

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        started = time.perf_counter()
        try:
            dataset, metadata = build_dataset_sample(
                dataset_name,
                args.sample_limit,
                args.seed,
                training_splits,
            )
            log(
                f"Prepared {dataset_name}: {metadata['rows']} rows "
                f"(ham={metadata['ham_count']}, spam={metadata['spam_count']})"
            )
            rows.append(
                evaluate_dataset(
                    dataset_name=dataset_name,
                    dataset=dataset,
                    model=model,
                    classifier=classifier,
                    args=args,
                    training_metadata=training_metadata,
                )
            )
        except Exception as exc:  # Keep the summary machine-readable for partial runs.
            traceback.print_exc()
            rows.append(
                error_row(
                    dataset_name=dataset_name,
                    runtime_seconds=time.perf_counter() - started,
                    model_id=args.model_id,
                    training_metadata=training_metadata,
                    error_message=str(exc),
                )
            )

    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root() / output_path
    write_summary(output_path, rows)
    log(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
