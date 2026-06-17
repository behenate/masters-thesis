#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


LOCAL_HF_HOME = Path(__file__).resolve().parent / ".cache" / "huggingface"
LOCAL_HF_DATASETS_CACHE = LOCAL_HF_HOME / "datasets"
LOCAL_HF_HUB_CACHE = LOCAL_HF_HOME / "hub"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(LOCAL_HF_DATASETS_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_HF_HUB_CACHE))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


METHOD = "distilbert_sequence_classification"
EVALUATION_METHOD = "sequence_classification"
DEFAULT_MODEL_ID = "distilbert-base-uncased"
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
SEED = 67
TRAIN_SPLIT = 0.92
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.06
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
DEFAULT_SAMPLE_LIMIT = 2000
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_NUM_TRAIN_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06

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
    "train_batch_size",
    "num_train_epochs",
    "model_id",
    "checkpoint_path",
    "runtime_seconds",
    "training_runtime_seconds",
    "training_rows",
    "validation_rows",
    "test_rows",
    "validation_f1",
    "decision_threshold",
    "status",
    "error_message",
]


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "sota").is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and sota/.")


def build_email_text(subject: str | None, body: str | None) -> str:
    subject_text = (subject or "").strip()
    body_text = (body or "").strip()
    parts = []
    if subject_text:
        parts.append(f"Subject: {subject_text}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts).strip()


def label_counts(dataset: Any) -> tuple[int, int]:
    labels = [int(value) for value in dataset["label"]]
    spam_count = sum(1 for value in labels if value == 1)
    ham_count = sum(1 for value in labels if value == 0)
    return ham_count, spam_count


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


def filter_empty_emails(dataset: Any) -> Any:
    return dataset.filter(lambda sample: bool(build_email_text(sample["subject"], sample["body"])))


def compute_binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")

    rows = len(labels)
    ham_count = sum(1 for value in labels if int(value) == 0)
    spam_count = sum(1 for value in labels if int(value) == 1)

    tp = sum(1 for label, prediction in zip(labels, predictions) if int(label) == 1 and int(prediction) == 1)
    tn = sum(1 for label, prediction in zip(labels, predictions) if int(label) == 0 and int(prediction) == 0)
    fp = sum(1 for label, prediction in zip(labels, predictions) if int(label) == 0 and int(prediction) == 1)
    fn = sum(1 for label, prediction in zip(labels, predictions) if int(label) == 1 and int(prediction) == 0)

    accuracy = (tp + tn) / rows if rows else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    spam_prediction_rate = (tp + fp) / rows if rows else 0.0

    return {
        "rows": float(rows),
        "ham_count": float(ham_count),
        "spam_count": float(spam_count),
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
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "spam_prediction_rate": float(spam_prediction_rate),
        "classification_failure_count": 0.0,
        "classification_failure_rate": 0.0,
    }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_device(torch_module: Any) -> Any:
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def build_training_split(seed: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    from datasets import ClassLabel, load_dataset
    from dataset.combine import combine_datasets

    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
    raw_dataset = filter_empty_emails(raw_dataset)

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

    metadata = {
        "source_path": str(data_path),
        "raw_rows": len(raw_dataset),
        "train_rows": len(holdout["train"]),
        "validation_rows": len(valid_test["train"]),
        "test_rows": len(valid_test["test"]),
    }
    return holdout["train"], valid_test["train"], valid_test["test"], metadata


def build_evaluation_samples(train_split: Any, sample_limit: int, seed: int) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset
    from dataset.combine import combine_datasets

    samples: dict[str, dict[str, Any]] = {}
    train_subset = stratified_sample(train_split, sample_limit, seed)
    ham_count, spam_count = label_counts(train_subset)
    samples["train_subset"] = {
        "dataset": train_subset,
        "source_path": "training_all/train_split",
        "rows": len(train_subset),
        "ham_count": ham_count,
        "spam_count": spam_count,
    }

    for dataset_name in ("enron", "fraudulent_email_corpus", "spam_ham"):
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        dataset = filter_empty_emails(dataset)
        labels = {int(value) for value in dataset["label"]}
        if len(labels) > 1:
            sample = stratified_sample(dataset, sample_limit, seed)
        else:
            sample = plain_sample(dataset, sample_limit, seed)
        ham_count, spam_count = label_counts(sample)
        samples[dataset_name] = {
            "dataset": sample,
            "source_path": str(data_path),
            "rows": len(sample),
            "ham_count": ham_count,
            "spam_count": spam_count,
        }

    return samples


def add_text_column(dataset: Any) -> Any:
    return dataset.map(
        lambda batch: {
            "text": [
                build_email_text(subject, body)
                for subject, body in zip(batch["subject"], batch["body"])
            ]
        },
        batched=True,
    )


def tokenize_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    dataset_with_text = add_text_column(dataset)
    remove_columns = [
        column
        for column in dataset_with_text.column_names
        if column not in {"label"}
    ]
    tokenized = dataset_with_text.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
        ),
        batched=True,
        remove_columns=remove_columns,
    )
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def make_dataloader(dataset: Any, tokenizer: Any, batch_size: int, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def train_model(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    device: Any,
    args: argparse.Namespace,
) -> int:
    import math
    import torch
    from transformers import get_linear_schedule_with_warmup

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, args.max_seq_length)
    train_loader = make_dataloader(tokenized_train, tokenizer, args.train_batch_size, shuffle=True)
    total_steps = max(1, len(train_loader) * args.num_train_epochs)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.to(device)
    model.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        epoch_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += float(loss.detach().cpu())
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                mean_loss = epoch_loss / batch_index
                log(f"epoch {epoch + 1}/{args.num_train_epochs} step {global_step}/{total_steps} loss={mean_loss:.4f}")

    return global_step


def predict_scores(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    device: Any,
    batch_size: int,
    max_seq_length: int,
) -> list[float]:
    import torch

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_seq_length)
    loader = make_dataloader(tokenized_dataset, tokenizer, batch_size, shuffle=False)
    scores: list[float] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            batch_scores = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(float(value) for value in batch_scores)

    return scores


def select_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, float]]:
    import numpy as np

    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    thresholds = np.unique(
        np.concatenate([np.quantile(scores_array, np.linspace(0.0, 1.0, 401)), np.asarray([0.5])])
    )
    best: tuple[tuple[float, float, float], float, dict[str, float]] | None = None
    for threshold in thresholds:
        predictions = (scores_array >= threshold).astype(int).tolist()
        metrics = compute_binary_metrics(labels_array.tolist(), predictions)
        key = (
            float(metrics["f1"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


def base_summary_row(
    *,
    dataset_name: str,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    global_step: int | str,
    runtime_seconds: float,
    status: str,
    error_message: str = "",
    training_metadata: dict[str, Any] | None = None,
    validation_f1: float | str = "",
    decision_threshold: float | str = "",
) -> dict[str, Any]:
    training_metadata = training_metadata or {}
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "method": METHOD,
            "evaluation_method": EVALUATION_METHOD,
            "dataset": dataset_name,
            "config_index": 1,
            "config_id": f"distilbert_seq{args.max_seq_length}_lr{args.learning_rate:g}",
            "checkpoint_type": "fine_tuned_model",
            "checkpoint_step": global_step,
            "rows": metadata.get("rows", ""),
            "ham_count": metadata.get("ham_count", ""),
            "spam_count": metadata.get("spam_count", ""),
            "classification_failure_count": 0.0,
            "classification_failure_rate": 0.0,
            "eval_batch_size": args.eval_batch_size,
            "max_seq_length": args.max_seq_length,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "num_train_epochs": args.num_train_epochs,
            "model_id": args.model_id,
            "checkpoint_path": "",
            "runtime_seconds": round(runtime_seconds, 4),
            "training_runtime_seconds": training_metadata.get("training_runtime_seconds", ""),
            "training_rows": training_metadata.get("training_rows", ""),
            "validation_rows": training_metadata.get("validation_rows", ""),
            "test_rows": training_metadata.get("test_rows", ""),
            "validation_f1": validation_f1,
            "decision_threshold": decision_threshold,
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def completed_summary_row(
    *,
    dataset_name: str,
    metadata: dict[str, Any],
    metrics: dict[str, float],
    args: argparse.Namespace,
    global_step: int,
    runtime_seconds: float,
    training_metadata: dict[str, Any],
    validation_f1: float,
    decision_threshold: float,
) -> dict[str, Any]:
    row = base_summary_row(
        dataset_name=dataset_name,
        metadata=metadata,
        args=args,
        global_step=global_step,
        runtime_seconds=runtime_seconds,
        status="completed",
        training_metadata=training_metadata,
        validation_f1=validation_f1,
        decision_threshold=decision_threshold,
    )
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    return row


def failure_summary_rows(
    *,
    samples: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    status: str,
    error_message: str,
    runtime_seconds: float,
) -> list[dict[str, Any]]:
    rows = []
    for dataset_name in args.datasets:
        metadata = samples.get(dataset_name, {})
        rows.append(
            base_summary_row(
                dataset_name=dataset_name,
                metadata=metadata,
                args=args,
                global_step="",
                runtime_seconds=runtime_seconds,
                status=status,
                error_message=error_message,
            )
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a DistilBERT sequence-classification spam baseline.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-train-epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--output", default=str(method_dir() / "summary.csv"))
    parser.add_argument("--model-output", default=str(method_dir() / "trained_model"))
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = project_root() / output_path

    try:
        root = project_root()
        log(f"Project root: {root}")
        set_reproducible_seed(args.seed)

        from datasets import disable_progress_bars

        disable_progress_bars()

        log("Preparing training split from dataset.combine training_all.")
        train_split, validation_split, test_split, split_metadata = build_training_split(args.seed)
        log(
            "Split rows: "
            f"train={split_metadata['train_rows']} "
            f"validation={split_metadata['validation_rows']} "
            f"test={split_metadata['test_rows']}"
        )
        samples = build_evaluation_samples(train_split, args.sample_limit, args.seed)

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = find_device(torch)
        log(f"Using device: {device}")
        log(f"Loading model and tokenizer: {args.model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_id,
                num_labels=2,
                id2label={0: "ham", 1: "spam"},
                label2id={"ham": 0, "spam": 1},
            )
        except Exception as exc:
            runtime = time.perf_counter() - start_time
            error_message = f"{type(exc).__name__}: {exc}"
            log(f"Model load failed: {error_message}")
            rows = failure_summary_rows(
                samples=samples,
                args=args,
                status="model_load_failed",
                error_message=error_message,
                runtime_seconds=runtime,
            )
            write_summary_csv(output_path, rows)
            log(f"Summary written to: {output_path}")
            return 2

        log("Training DistilBERT on the 92% training split.")
        training_started = time.perf_counter()
        global_step = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_split,
            validation_dataset=validation_split,
            device=device,
            args=args,
        )
        training_runtime = time.perf_counter() - training_started

        log("Selecting the classification threshold on the validation split.")
        validation_labels = [int(value) for value in validation_split["label"]]
        validation_scores = predict_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=validation_split,
            device=device,
            batch_size=args.eval_batch_size,
            max_seq_length=args.max_seq_length,
        )
        decision_threshold, validation_metrics = select_threshold(validation_labels, validation_scores)
        log(
            f"Validation F1={validation_metrics['f1']:.4f}; "
            f"decision threshold={decision_threshold:.4f}."
        )

        model_output = Path(args.model_output).expanduser()
        if not model_output.is_absolute():
            model_output = project_root() / model_output
        model_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_output)
        tokenizer.save_pretrained(model_output)
        training_metadata = {
            "training_runtime_seconds": round(training_runtime, 4),
            "training_rows": len(train_split),
            "validation_rows": len(validation_split),
            "test_rows": len(test_split),
        }
        (model_output / "evaluation_config.json").write_text(
            json.dumps(
                {
                    "decision_threshold": decision_threshold,
                    "validation_metrics": validation_metrics,
                    "global_step": global_step,
                    **training_metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"Saved trained model to: {model_output}")

        summary_rows = []
        for dataset_name in args.datasets:
            eval_start = time.perf_counter()
            metadata = samples[dataset_name]
            dataset = metadata["dataset"]
            log(f"Evaluating {dataset_name}: rows={len(dataset)}")
            labels = [int(value) for value in dataset["label"]]
            scores = predict_scores(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                batch_size=args.eval_batch_size,
                max_seq_length=args.max_seq_length,
            )
            predictions = [int(score >= decision_threshold) for score in scores]
            metrics = compute_binary_metrics(labels, predictions)
            summary_rows.append(
                completed_summary_row(
                    dataset_name=dataset_name,
                    metadata=metadata,
                    metrics=metrics,
                    args=args,
                    global_step=global_step,
                    runtime_seconds=time.perf_counter() - eval_start,
                    training_metadata=training_metadata,
                    validation_f1=float(validation_metrics["f1"]),
                    decision_threshold=decision_threshold,
                )
            )

        write_summary_csv(output_path, summary_rows)
        log(f"Summary written to: {output_path}")
        return 0
    except Exception as exc:
        runtime = time.perf_counter() - start_time
        error_message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        rows = failure_summary_rows(
            samples={},
            args=args,
            status="failed",
            error_message=error_message,
            runtime_seconds=runtime,
        )
        write_summary_csv(output_path, rows)
        log(f"Summary written to: {output_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
