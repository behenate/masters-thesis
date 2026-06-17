#!/usr/bin/env python3
"""Tune the classical SOTA baselines on the training validation split.

The external datasets do not participate in configuration or threshold
selection. They are used only to evaluate the selected model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SOTA_DIR = ROOT / "sota"
HF_CACHE_DIR = SOTA_DIR / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SEED = 67
SAMPLE_LIMIT = 2000
DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
METHODS = ("tfidf_logistic_regression", "tfidf_linear_svm", "fasttext", "minilm_logistic_regression")
SUMMARY_COLUMNS = [
    "dataset", "method", "config_id", "rows", "ham_count", "spam_count",
    "accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy",
    "false_positive_count", "false_negative_count", "true_positive_count", "true_negative_count",
    "false_positive_rate", "false_negative_rate", "spam_prediction_rate",
    "classification_failure_count", "classification_failure_rate", "runtime_seconds",
    "training_runtime_seconds", "training_rows", "validation_rows", "test_rows",
    "validation_f1", "validation_source_macro_f1", "decision_threshold", "parameters",
    "sample_limit", "seed", "status", "error_message",
]
TUNING_COLUMNS = [
    "method", "config_id", "validation_rows", "validation_accuracy", "validation_precision",
    "validation_recall", "validation_f1", "validation_specificity", "validation_balanced_accuracy",
    "validation_source_macro_f1", "decision_threshold", "training_runtime_seconds", "parameters",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def add_project_root() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_email_text(subject: Any, body: Any) -> str:
    subject_text = clean_text(subject)
    body_text = clean_text(body)
    parts = []
    if subject_text:
        parts.append(f"Subject: {subject_text}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts)


def frame_from_dataset(dataset: Any) -> pd.DataFrame:
    frame = dataset.to_pandas()
    frame["subject"] = frame["subject"].map(clean_text)
    frame["body"] = frame["body"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    frame["text"] = [
        build_email_text(subject, body)
        for subject, body in zip(frame["subject"], frame["body"], strict=False)
    ]
    if "source" not in frame:
        frame["source"] = "unknown"
    frame["source"] = frame["source"].map(clean_text).replace("", "unknown")
    return frame[frame["text"].astype(bool)].reset_index(drop=True)


@dataclass
class DataBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    raw_train: Any
    test_rows: int
    training_path: str


def load_training_data(seed: int, train_limit: int | None = None) -> DataBundle:
    add_project_root()
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw = load_dataset("parquet", data_files=str(data_path), split="train")
    raw = raw.cast_column("label", ClassLabel(names=["ham", "spam"]))
    holdout = raw.train_test_split(test_size=0.08, stratify_by_column="label", seed=seed)
    valid_test = holdout["test"].train_test_split(test_size=0.75, stratify_by_column="label", seed=seed)
    train = frame_from_dataset(holdout["train"])
    validation = frame_from_dataset(valid_test["train"])
    if train_limit is not None and train_limit < len(train):
        train = (
            train.groupby("label", group_keys=False)
            .sample(n=max(1, train_limit // 2), random_state=seed)
            .sample(frac=1, random_state=seed)
            .head(train_limit)
            .reset_index(drop=True)
        )
    return DataBundle(
        train=train,
        validation=validation,
        raw_train=holdout["train"],
        test_rows=len(valid_test["test"]),
        training_path=str(data_path),
    )


def stratified_sample(dataset: Any, limit: int, seed: int) -> Any:
    from datasets import concatenate_datasets

    if limit <= 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)
    labels = [int(value) for value in dataset["label"]]
    remaining = limit
    selected_parts = []
    label_values = sorted(set(labels))
    for offset, label_value in enumerate(label_values):
        indices = [index for index, value in enumerate(labels) if value == label_value]
        take = min(remaining, len(indices)) if offset == len(label_values) - 1 else round(limit * len(indices) / len(labels))
        take = max(0, min(take, len(indices), remaining))
        if take:
            selected_parts.append(
                dataset.select(indices).shuffle(seed=seed + label_value).select(range(take))
            )
            remaining -= take
    if not selected_parts:
        return dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    return concatenate_datasets(selected_parts).shuffle(seed=seed)


def plain_sample(dataset: Any, limit: int, seed: int) -> Any:
    if limit <= 0 or len(dataset) <= limit:
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(limit))


def load_evaluation_frames(data: DataBundle, sample_limit: int, seed: int) -> dict[str, pd.DataFrame]:
    add_project_root()
    from datasets import disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()
    frames = {
        "train_subset": frame_from_dataset(
            stratified_sample(data.raw_train, sample_limit, seed)
        )
    }
    for dataset_name in DATASETS[1:]:
        path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        dataset = dataset.filter(
            lambda sample: bool(build_email_text(sample["subject"], sample["body"]))
        )
        sample = (
            stratified_sample(dataset, sample_limit, seed)
            if dataset_name == "spam_ham"
            else plain_sample(dataset, sample_limit, seed)
        )
        frames[dataset_name] = frame_from_dataset(sample)
    return frames


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "specificity": float(specificity), "balanced_accuracy": float((recall + specificity) / 2),
        "false_positive_count": fp, "false_negative_count": fn,
        "true_positive_count": tp, "true_negative_count": tn,
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "spam_prediction_rate": float(np.mean(pred == 1)),
        "classification_failure_count": 0, "classification_failure_rate": 0.0,
    }


def source_macro_f1(labels: np.ndarray, predictions: np.ndarray, sources: Sequence[str] | None) -> float:
    if sources is None:
        return float(binary_metrics(labels, predictions)["f1"])
    source_array = np.asarray(sources)
    scores = [
        float(binary_metrics(labels[source_array == source], predictions[source_array == source])["f1"])
        for source in np.unique(source_array)
    ]
    return float(np.mean(scores))


def threshold_candidates(scores: np.ndarray, default: float) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 401))
    return np.unique(np.concatenate([quantiles, np.asarray([default])]))


def select_threshold(
    labels: Sequence[int], scores: Sequence[float], default: float, sources: Sequence[str] | None = None
) -> tuple[float, dict[str, float | int], float]:
    y = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    best: tuple[tuple[float, float, float, float], float, dict[str, float | int], float] | None = None
    for threshold in threshold_candidates(score_array, default):
        pred = (score_array >= threshold).astype(np.int8)
        metrics = binary_metrics(y, pred)
        macro_f1 = source_macro_f1(y, pred, sources)
        objective = (
            macro_f1 if sources is not None else float(metrics["f1"]),
            float(metrics["f1"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - default),
        )
        if best is None or objective > best[0]:
            best = (objective, float(threshold), metrics, macro_f1)
    assert best is not None
    return best[1], best[2], best[3]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tuning_row(
    method: str,
    config_id: str,
    metrics: dict[str, Any],
    source_f1: float,
    threshold: float,
    runtime: float,
    params: dict[str, Any],
    validation_rows: int,
) -> dict[str, Any]:
    return {
        "method": method, "config_id": config_id, "validation_rows": validation_rows,
        **{f"validation_{key}": metrics[key] for key in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")},
        "validation_source_macro_f1": source_f1, "decision_threshold": threshold,
        "training_runtime_seconds": runtime, "parameters": json.dumps(params, sort_keys=True),
    }


def evaluate_selected(
    method: str,
    config_id: str,
    frames: dict[str, pd.DataFrame],
    score_function: Callable[[pd.DataFrame], np.ndarray],
    threshold: float,
    training_runtime: float,
    data: DataBundle,
    validation_metrics: dict[str, Any],
    validation_source_f1: float,
    params: dict[str, Any],
    sample_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, frame in frames.items():
        started = time.perf_counter()
        scores = score_function(frame)
        predictions = (scores >= threshold).astype(np.int8)
        metrics = binary_metrics(frame["label"].to_numpy(), predictions)
        rows.append({
            "dataset": dataset_name, "method": method, "config_id": config_id,
            "rows": len(frame), "ham_count": int(np.sum(frame["label"] == 0)),
            "spam_count": int(np.sum(frame["label"] == 1)), **metrics,
            "runtime_seconds": time.perf_counter() - started,
            "training_runtime_seconds": training_runtime, "training_rows": len(data.train),
            "validation_rows": len(data.validation), "test_rows": data.test_rows,
            "validation_f1": validation_metrics["f1"],
            "validation_source_macro_f1": validation_source_f1,
            "decision_threshold": threshold, "parameters": json.dumps(params, sort_keys=True),
            "sample_limit": sample_limit, "seed": seed, "status": "completed", "error_message": "",
        })
        log(f"{method} / {dataset_name}: F1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
    return rows


def tune_tfidf_logistic(data: DataBundle, frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    method = "tfidf_logistic_regression"
    output_dir = SOTA_DIR / method
    log("TF-IDF + logistic regression: fitting vectorizer")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=200_000,
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(data.train["text"])
    x_validation = vectorizer.transform(data.validation["text"])
    y_train = data.train["label"].to_numpy()
    y_validation = data.validation["label"].to_numpy()
    candidates = []
    best = None
    total_started = time.perf_counter()
    for c_value in (0.01, 0.1, 1.0, 10.0):
        started = time.perf_counter()
        classifier = LogisticRegression(C=c_value, solver="liblinear", max_iter=1000, random_state=args.seed)
        classifier.fit(x_train, y_train)
        threshold, metrics, macro_f1 = select_threshold(y_validation, classifier.predict_proba(x_validation)[:, 1], 0.5)
        params = {"C": c_value, "threshold": threshold, "ngram_range": [1, 2], "max_df": 0.95}
        config_id = f"tfidf_word_1_2_logreg_c{c_value:g}"
        row = tuning_row(method, config_id, metrics, macro_f1, threshold, time.perf_counter() - started, params, len(data.validation))
        candidates.append(row)
        key = (float(metrics["f1"]), float(metrics["balanced_accuracy"]), -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, classifier, threshold, metrics, macro_f1, params, config_id)
        log(f"{config_id}: validation F1={metrics['f1']:.4f}, threshold={threshold:.4f}")
    assert best is not None
    _, classifier, threshold, metrics, macro_f1, params, config_id = best
    artifact = {"vectorizer": vectorizer, "classifier": classifier, "threshold": threshold, "config_id": config_id}
    joblib.dump(artifact, output_dir / "tuned_model.joblib")
    write_csv(output_dir / "tuning_results.csv", candidates, TUNING_COLUMNS)
    rows = evaluate_selected(
        method, config_id, frames,
        lambda frame: classifier.predict_proba(vectorizer.transform(frame["text"]))[:, 1],
        threshold, time.perf_counter() - total_started, data, metrics, macro_f1, params, args.sample_limit, args.seed,
    )
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_json(output_dir / "tuned_config.json", {"config_id": config_id, **params, "validation_metrics": metrics})


def tune_tfidf_svm(data: DataBundle, frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    method = "tfidf_linear_svm"
    output_dir = SOTA_DIR / method
    y_train = data.train["label"].to_numpy()
    y_validation = data.validation["label"].to_numpy()
    vectorizer_configs = (
        ("word_1_2", {"analyzer": "word", "ngram_range": (1, 2)}),
        ("char_wb_3_5", {"analyzer": "char_wb", "ngram_range": (3, 5)}),
    )
    candidates = []
    best = None
    total_started = time.perf_counter()
    artifact_path = output_dir / "tuned_model.joblib"
    for vectorizer_name, vectorizer_args in vectorizer_configs:
        log(f"TF-IDF + SVM: fitting {vectorizer_name} vectorizer")
        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", max_features=200_000,
            min_df=2, max_df=0.95, sublinear_tf=True, norm="l2", **vectorizer_args,
        )
        x_train = vectorizer.fit_transform(data.train["text"])
        x_validation = vectorizer.transform(data.validation["text"])
        for c_value in (0.01, 0.1, 1.0, 10.0):
            started = time.perf_counter()
            classifier = LinearSVC(C=c_value, random_state=args.seed, dual="auto", max_iter=5000)
            classifier.fit(x_train, y_train)
            threshold, metrics, macro_f1 = select_threshold(y_validation, classifier.decision_function(x_validation), 0.0)
            params = {"C": c_value, "threshold": threshold, "vectorizer": vectorizer_name, "max_df": 0.95}
            config_id = f"tfidf_{vectorizer_name}_svm_c{c_value:g}"
            candidates.append(tuning_row(method, config_id, metrics, macro_f1, threshold, time.perf_counter() - started, params, len(data.validation)))
            key = (float(metrics["f1"]), float(metrics["balanced_accuracy"]), -abs(threshold))
            if best is None or key > best[0]:
                best = (key, threshold, metrics, macro_f1, params, config_id)
                joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "threshold": threshold, "config_id": config_id}, artifact_path)
            log(f"{config_id}: validation F1={metrics['f1']:.4f}, threshold={threshold:.4f}")
        del x_train, x_validation, vectorizer
        gc.collect()
    assert best is not None
    _, threshold, metrics, macro_f1, params, config_id = best
    artifact = joblib.load(artifact_path)
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    write_csv(output_dir / "tuning_results.csv", candidates, TUNING_COLUMNS)
    rows = evaluate_selected(
        method, config_id, frames,
        lambda frame: classifier.decision_function(vectorizer.transform(frame["text"])),
        threshold, time.perf_counter() - total_started, data, metrics, macro_f1, params, args.sample_limit, args.seed,
    )
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_json(output_dir / "tuned_config.json", {"config_id": config_id, **params, "validation_metrics": metrics})


FASTTEXT_WHITESPACE = re.compile(r"\s+")


def clean_fasttext_text(text: str) -> str:
    return FASTTEXT_WHITESPACE.sub(" ", text.replace("\x00", " ").replace("__label__", "label_")).strip()


def fasttext_scores(model: Any, texts: Sequence[str]) -> np.ndarray:
    clean = [clean_fasttext_text(text) or "empty" for text in texts]
    labels, probabilities = model.predict(clean, k=2)
    scores = []
    for row_labels, row_probabilities in zip(labels, probabilities, strict=False):
        mapping = dict(zip(row_labels, row_probabilities, strict=False))
        scores.append(float(mapping.get("__label__spam", 0.0)))
    return np.asarray(scores)


def tune_fasttext(data: DataBundle, frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    import fasttext

    method = "fasttext"
    output_dir = SOTA_DIR / method
    cache_dir = output_dir / ".tuning_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"train_seed{args.seed}.txt"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in data.train.itertuples(index=False):
            label = "spam" if int(row.label) == 1 else "ham"
            handle.write(f"__label__{label} {clean_fasttext_text(row.text)}\n")
    y_validation = data.validation["label"].to_numpy()
    validation_texts = data.validation["text"].tolist()
    candidates = []
    best = None
    completed_configs: set[tuple[Any, ...]] = set()
    model_path = output_dir / "tuned_model.bin"
    total_started = time.perf_counter()

    def run_candidate(params: dict[str, Any]) -> None:
        nonlocal best
        signature = tuple(params[key] for key in ("lr", "epoch", "wordNgrams", "dim", "minn", "maxn"))
        if signature in completed_configs:
            return
        completed_configs.add(signature)
        started = time.perf_counter()
        try:
            model = fasttext.train_supervised(
                input=str(train_path), lr=params["lr"], epoch=params["epoch"],
                wordNgrams=params["wordNgrams"], dim=params["dim"], minn=params["minn"], maxn=params["maxn"],
                minCount=2, loss="softmax", thread=args.fasttext_threads, seed=args.seed, verbose=0,
            )
        except RuntimeError as exc:
            log(f"Skipping unstable fastText configuration {signature}: {exc}")
            return
        scores = fasttext_scores(model, validation_texts)
        threshold, metrics, macro_f1 = select_threshold(y_validation, scores, 0.5)
        config_id = (
            f"fasttext_lr{params['lr']:g}_ep{params['epoch']}_w{params['wordNgrams']}"
            f"_d{params['dim']}_char{params['minn']}_{params['maxn']}"
        )
        full_params = {**params, "threshold": threshold, "thread": args.fasttext_threads, "seed": args.seed}
        candidates.append(tuning_row(method, config_id, metrics, macro_f1, threshold, time.perf_counter() - started, full_params, len(data.validation)))
        key = (float(metrics["f1"]), float(metrics["balanced_accuracy"]), -abs(threshold - 0.5))
        if best is None or key > best[0]:
            model.save_model(str(model_path))
            best = (key, threshold, metrics, macro_f1, full_params, config_id)
        log(f"{config_id}: validation F1={metrics['f1']:.4f}, threshold={threshold:.4f}")
        del model
        gc.collect()

    # Stage 1 selects learning rate and epoch with the previous representation.
    for lr in (0.05, 0.1, 0.25):
        for epoch in (5, 10, 20):
            run_candidate({"lr": lr, "epoch": epoch, "wordNgrams": 2, "dim": 100, "minn": 0, "maxn": 0})
    assert best is not None
    stage_one_params = best[4]
    # Stage 2 compares representation parameters using the best optimization settings.
    for minn, maxn in ((0, 0), (3, 6)):
        for word_ngrams in (1, 2, 3):
            for dim in (50, 100, 200):
                params = {"lr": stage_one_params["lr"], "epoch": stage_one_params["epoch"], "wordNgrams": word_ngrams, "dim": dim, "minn": minn, "maxn": maxn}
                run_candidate(params)

    assert best is not None
    _, threshold, metrics, macro_f1, params, config_id = best
    model = fasttext.load_model(str(model_path))
    write_csv(output_dir / "tuning_results.csv", candidates, TUNING_COLUMNS)
    rows = evaluate_selected(
        method, config_id, frames, lambda frame: fasttext_scores(model, frame["text"].tolist()),
        threshold, time.perf_counter() - total_started, data, metrics, macro_f1, params, args.sample_limit, args.seed,
    )
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_json(output_dir / "tuned_config.json", {"config_id": config_id, **params, "validation_metrics": metrics})


def l2_normalize(array: np.ndarray) -> np.ndarray:
    return normalize(array, norm="l2", axis=1, copy=True).astype(np.float32)


def encode_full(model: Any, texts: Sequence[str], max_length: int, batch_size: int) -> np.ndarray:
    model.max_seq_length = max_length
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)


def encode_subject_body(model: Any, frame: pd.DataFrame, max_length: int, batch_size: int) -> np.ndarray:
    subjects = [f"Subject: {value}" if value else "Subject:" for value in frame["subject"]]
    subject_embeddings = encode_full(model, subjects, max_length, batch_size)
    body_embeddings = encode_full(model, frame["body"].tolist(), max_length, batch_size)
    return np.concatenate([subject_embeddings, body_embeddings], axis=1)


def encode_chunks(model: Any, texts: Sequence[str], max_length: int, batch_size: int) -> np.ndarray:
    tokenizer = model.tokenizer
    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunks: list[str] = []
    owners: list[int] = []
    for owner, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode("empty", add_special_tokens=False)
        for offset in range(0, len(token_ids), content_length):
            chunks.append(tokenizer.decode(token_ids[offset : offset + content_length], skip_special_tokens=True))
            owners.append(owner)
    chunk_embeddings = encode_full(model, chunks, max_length, batch_size)
    result = np.zeros((len(texts), chunk_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.int32)
    for owner, embedding in zip(owners, chunk_embeddings, strict=False):
        result[owner] += embedding
        counts[owner] += 1
    result /= counts[:, None]
    return result


def minilm_embeddings(
    model: Any, representation: str, frame: pd.DataFrame, batch_size: int, cache_path: Path | None = None
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)["embeddings"]
        if cached.shape[0] == len(frame):
            log(f"Loading cached embeddings: {cache_path.name}")
            return cached
        log(f"Ignoring cache with {cached.shape[0]} rows; expected {len(frame)}")
    if representation == "full256":
        embeddings = encode_full(model, frame["text"].tolist(), 256, batch_size)
    elif representation == "full512":
        embeddings = encode_full(model, frame["text"].tolist(), 512, batch_size)
    elif representation == "chunks256":
        embeddings = encode_chunks(model, frame["text"].tolist(), 256, batch_size)
    elif representation == "subject_body256":
        embeddings = encode_subject_body(model, frame, 256, batch_size)
    else:
        raise ValueError(f"Unknown MiniLM representation: {representation}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def tune_minilm(data: DataBundle, frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    method = "minilm_logistic_regression"
    output_dir = SOTA_DIR / method
    cache_dir = output_dir / ".embedding_cache"
    snapshots = sorted((output_dir / ".hf_cache" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").glob("*"))
    model_id = str(snapshots[-1]) if args.local_files_only and snapshots else "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(
        model_id, local_files_only=args.local_files_only,
        device=None if args.minilm_device == "auto" else args.minilm_device,
    )
    y_train = data.train["label"].to_numpy()
    y_validation = data.validation["label"].to_numpy()
    sources = data.validation["source"].tolist()
    candidates = []
    best = None
    artifact_path = output_dir / "tuned_classifier.joblib"
    total_started = time.perf_counter()
    for representation in args.minilm_representations:
        log(f"MiniLM: preparing {representation} embeddings")
        x_train_raw = minilm_embeddings(model, representation, data.train, args.minilm_batch_size, cache_dir / f"train_{representation}_seed{args.seed}.npz")
        x_validation_raw = minilm_embeddings(model, representation, data.validation, args.minilm_batch_size, cache_dir / f"validation_{representation}_seed{args.seed}.npz")
        for normalized in (False, True):
            x_train = l2_normalize(x_train_raw) if normalized else x_train_raw
            x_validation = l2_normalize(x_validation_raw) if normalized else x_validation_raw
            for class_weight in (None, "balanced"):
                for c_value in (0.01, 0.1, 1.0, 10.0):
                    started = time.perf_counter()
                    classifier = LogisticRegression(C=c_value, max_iter=1000, random_state=args.seed, class_weight=class_weight)
                    classifier.fit(x_train, y_train)
                    threshold, metrics, macro_f1 = select_threshold(
                        y_validation, classifier.predict_proba(x_validation)[:, 1], 0.5, sources=sources,
                    )
                    params = {"representation": representation, "normalized": normalized, "class_weight": class_weight or "none", "C": c_value, "threshold": threshold}
                    config_id = f"minilm_{representation}_{'norm' if normalized else 'raw'}_cw{params['class_weight']}_c{c_value:g}"
                    candidates.append(tuning_row(method, config_id, metrics, macro_f1, threshold, time.perf_counter() - started, params, len(data.validation)))
                    key = (macro_f1, float(metrics["f1"]), float(metrics["balanced_accuracy"]), -abs(threshold - 0.5))
                    if best is None or key > best[0]:
                        best = (key, threshold, metrics, macro_f1, params, config_id)
                        joblib.dump({"classifier": classifier, "threshold": threshold, "config_id": config_id, **params}, artifact_path)
                    log(f"{config_id}: source-macro F1={macro_f1:.4f}, F1={metrics['f1']:.4f}")
        del x_train_raw, x_validation_raw
        gc.collect()
    assert best is not None
    _, threshold, metrics, macro_f1, params, config_id = best
    artifact = joblib.load(artifact_path)
    classifier = artifact["classifier"]
    representation = params["representation"]
    normalized = params["normalized"]

    def score(frame: pd.DataFrame) -> np.ndarray:
        embeddings = minilm_embeddings(model, representation, frame, args.minilm_batch_size)
        if normalized:
            embeddings = l2_normalize(embeddings)
        return classifier.predict_proba(embeddings)[:, 1]

    write_csv(output_dir / "tuning_results.csv", candidates, TUNING_COLUMNS)
    rows = evaluate_selected(
        method, config_id, frames, score, threshold, time.perf_counter() - total_started,
        data, metrics, macro_f1, params, args.sample_limit, args.seed,
    )
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_json(output_dir / "tuned_config.json", {"config_id": config_id, **params, "validation_metrics": metrics, "validation_source_macro_f1": macro_f1})


def previous_training_runtime(output_dir: Path) -> float:
    summary_path = output_dir / "summary.csv"
    if not summary_path.is_file():
        return 0.0
    with summary_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), {})
    value = row.get("training_runtime_seconds", "")
    return float(value) if value else 0.0


def evaluate_saved_method(
    method: str,
    data: DataBundle,
    frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    output_dir = SOTA_DIR / method
    config = json.loads((output_dir / "tuned_config.json").read_text(encoding="utf-8"))
    config_id = config["config_id"]
    threshold = float(config["threshold"])
    validation_metrics = config["validation_metrics"]
    validation_source_f1 = float(config.get("validation_source_macro_f1", validation_metrics["f1"]))
    params = {
        key: value
        for key, value in config.items()
        if key not in {"config_id", "validation_metrics", "validation_source_macro_f1"}
    }
    training_runtime = previous_training_runtime(output_dir)

    if method in {"tfidf_logistic_regression", "tfidf_linear_svm"}:
        artifact = joblib.load(output_dir / "tuned_model.joblib")
        vectorizer = artifact["vectorizer"]
        classifier = artifact["classifier"]

        def score(frame: pd.DataFrame) -> np.ndarray:
            features = vectorizer.transform(frame["text"])
            if hasattr(classifier, "predict_proba"):
                return classifier.predict_proba(features)[:, 1]
            return classifier.decision_function(features)

    elif method == "fasttext":
        import fasttext

        model = fasttext.load_model(str(output_dir / "tuned_model.bin"))

        def score(frame: pd.DataFrame) -> np.ndarray:
            return fasttext_scores(model, frame["text"].tolist())

    elif method == "minilm_logistic_regression":
        from sentence_transformers import SentenceTransformer

        snapshots = sorted(
            (
                output_dir
                / ".hf_cache"
                / "hub"
                / "models--sentence-transformers--all-MiniLM-L6-v2"
                / "snapshots"
            ).glob("*")
        )
        model_id = str(snapshots[-1]) if snapshots else "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(
            model_id,
            local_files_only=bool(snapshots),
            device=None if args.minilm_device == "auto" else args.minilm_device,
        )
        artifact = joblib.load(output_dir / "tuned_classifier.joblib")
        classifier = artifact["classifier"]
        representation = config["representation"]
        normalized = bool(config["normalized"])

        def score(frame: pd.DataFrame) -> np.ndarray:
            embeddings = minilm_embeddings(
                model, representation, frame, args.minilm_batch_size
            )
            if normalized:
                embeddings = l2_normalize(embeddings)
            return classifier.predict_proba(embeddings)[:, 1]

    else:
        raise ValueError(f"Unsupported saved method: {method}")

    rows = evaluate_selected(
        method,
        config_id,
        frames,
        score,
        threshold,
        training_runtime,
        data,
        validation_metrics,
        validation_source_f1,
        params,
        args.sample_limit,
        args.seed,
    )
    write_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sample-limit", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-limit", type=int, default=None, help="Only for smoke tests.")
    parser.add_argument("--fasttext-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--minilm-batch-size", type=int, default=128)
    parser.add_argument("--minilm-device", default="auto")
    parser.add_argument("--minilm-representations", nargs="+", choices=("full256", "full512", "chunks256", "subject_body256"), default=["full256", "full512", "chunks256", "subject_body256"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate persisted selected models without repeating parameter search.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be positive")
    log("Loading the deterministic 92/2/6 training split")
    data = load_training_data(args.seed, args.train_limit)
    log(f"Rows: train={len(data.train)}, validation={len(data.validation)}, test={data.test_rows}")
    if args.evaluate_only:
        log("Preparing canonical evaluation samples")
    else:
        log("Preparing external datasets; they will not be used for configuration selection")
    frames = load_evaluation_frames(data, args.sample_limit, args.seed)
    functions = {
        "tfidf_logistic_regression": tune_tfidf_logistic,
        "tfidf_linear_svm": tune_tfidf_svm,
        "fasttext": tune_fasttext,
        "minilm_logistic_regression": tune_minilm,
    }
    for method in args.methods:
        if args.evaluate_only:
            log(f"Re-evaluating selected model: {method}")
            evaluate_saved_method(method, data, frames, args)
            log(f"Completed evaluation: {method}")
        else:
            log(f"Starting tuning: {method}")
            functions[method](data, frames, args)
            log(f"Completed tuning: {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
