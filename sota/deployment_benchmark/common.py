from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import psutil


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BENCHMARK_DIR / "artifacts"
SAMPLE_DIR = BENCHMARK_DIR / "sample"
WORKER_RESULTS_DIR = BENCHMARK_DIR / "worker_results"
SAMPLE_PATH = SAMPLE_DIR / "spam_ham_1000.parquet"
SEED = 67
SAMPLE_LIMIT = 1000

METHODS = (
    "tfidf_naive_bayes",
    "tfidf_logistic_regression",
    "tfidf_linear_svm",
    "fasttext",
    "minilm_logistic_regression",
    "distilbert_sequence_classification",
    "qwen3_lora_next_token",
)

ARTIFACT_PATHS = {
    "tfidf_naive_bayes": ARTIFACT_DIR / "tfidf_naive_bayes.joblib",
    "tfidf_logistic_regression": ARTIFACT_DIR / "tfidf_logistic_regression.joblib",
    "tfidf_linear_svm": ARTIFACT_DIR / "tfidf_linear_svm.joblib",
    "fasttext": ARTIFACT_DIR / "fasttext.bin",
    "minilm_logistic_regression": ARTIFACT_DIR / "minilm_logistic_regression.joblib",
    "distilbert_sequence_classification": ROOT / "sota" / "distilbert_sequence_classification" / "trained_model",
}

QWEN_ADAPTER_PATH = (
    ROOT
    / "lora-fine-tuning"
    / "methods"
    / "03_causal_lm_next_token"
    / "results"
    / "03.06.2026"
    / "results"
    / "qwen3_clm_sweep_20260603_084911"
    / "08_lora_capacity_seq512_lr1e-4_r32_a64_do0p05"
    / "trainer_output"
    / "checkpoint-3000"
)


def ensure_directories() -> None:
    for path in (ARTIFACT_DIR, SAMPLE_DIR, WORKER_RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def build_email_text(subject: Any, body: Any) -> str:
    subject_text = "" if subject is None else str(subject).strip()
    body_text = "" if body is None else str(body).strip()
    parts = []
    if subject_text:
        parts.append(f"Subject: {subject_text}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts).strip()


def prepare_sample(path: Path = SAMPLE_PATH, limit: int = SAMPLE_LIMIT, seed: int = SEED) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from dataset.combine import combine_datasets

    data_path = combine_datasets("spam_ham", duplicate_detection="high")
    table = pq.read_table(data_path)
    subjects = table.column("subject").to_pylist()
    bodies = table.column("body").to_pylist()
    labels = [int(value) for value in table.column("label").to_pylist()]
    valid_indices = [
        index
        for index, (subject, body) in enumerate(zip(subjects, bodies, strict=True))
        if build_email_text(subject, body)
    ]
    valid_labels = [labels[index] for index in valid_indices]
    selected_indices: list[int] = []
    remaining = limit
    label_values = sorted(set(valid_labels))
    for offset, label_value in enumerate(label_values):
        indices = [index for index in valid_indices if labels[index] == label_value]
        if offset == len(label_values) - 1:
            take = min(remaining, len(indices))
        else:
            take = int(round(limit * len(indices) / len(valid_indices)))
            take = max(0, min(take, len(indices), remaining))
        if take:
            random.Random(seed + label_value).shuffle(indices)
            selected_indices.extend(indices[:take])
            remaining -= take

    random.Random(seed).shuffle(selected_indices)
    sample = table.take(pa.array(selected_indices, type=pa.int64()))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, path)
    sample_labels = [int(value) for value in sample.column("label").to_pylist()]
    return {
        "path": str(path),
        "source_path": str(data_path),
        "rows": sample.num_rows,
        "ham_count": sum(value == 0 for value in sample_labels),
        "spam_count": sum(value == 1 for value in sample_labels),
        "seed": seed,
        "sha256": sha256_file(path),
    }


def load_sample(path: Path = SAMPLE_PATH) -> tuple[list[str], list[int], dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["subject", "body", "label"])
    subjects = table.column("subject").to_pylist()
    bodies = table.column("body").to_pylist()
    labels = [int(value) for value in table.column("label").to_pylist()]
    texts = [build_email_text(subject, body) for subject, body in zip(subjects, bodies, strict=True)]
    lengths = [len(text) for text in texts]
    metadata = {
        "sample_sha256": sha256_file(path),
        "input_characters_mean": statistics.fmean(lengths),
        "input_characters_median": statistics.median(lengths),
        "input_characters_p95": percentile(lengths, 95),
    }
    return texts, labels, metadata


def cached_snapshot_path(model_id: str, cache_roots: Iterable[Path]) -> Path | None:
    cache_name = "models--" + model_id.replace("/", "--")
    for cache_root in cache_roots:
        snapshots = cache_root / cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if candidates:
            return candidates[-1]
    return None


def percentile(values: list[int | float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def binary_metrics(labels: list[int], predictions: Iterable[int]) -> dict[str, float]:
    predicted = [int(value) for value in predictions]
    if len(labels) != len(predicted):
        raise ValueError("Prediction count does not match label count")
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predicted, strict=True))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predicted, strict=True))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predicted, strict=True))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_positive_count": tp,
        "true_negative_count": tn,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_file_size(paths: Iterable[Path]) -> int:
    resolved: set[Path] = set()
    total = 0
    for original in paths:
        if not original.exists():
            continue
        candidates = original.rglob("*") if original.is_dir() else [original]
        for candidate in candidates:
            if candidate.is_dir():
                continue
            try:
                real = candidate.resolve(strict=True)
            except FileNotFoundError:
                continue
            if real in resolved:
                continue
            resolved.add(real)
            total += real.stat().st_size
    return total


def synchronize_accelerator() -> None:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    elif torch_module.backends.mps.is_available():
        torch_module.mps.synchronize()


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.reset()

    def _read(self) -> tuple[int, int, int]:
        rss = self.process.memory_info().rss
        accelerator_allocated = 0
        accelerator_driver = 0
        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    accelerator_allocated = int(torch_module.cuda.memory_allocated())
                    accelerator_driver = int(torch_module.cuda.memory_reserved())
                elif torch_module.backends.mps.is_available():
                    accelerator_allocated = int(torch_module.mps.current_allocated_memory())
                    accelerator_driver = int(torch_module.mps.driver_allocated_memory())
            except (AttributeError, RuntimeError):
                pass
        return rss, accelerator_allocated, accelerator_driver

    def _sample(self) -> None:
        rss, allocated, driver = self._read()
        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_accelerator_allocated = max(self.peak_accelerator_allocated, allocated)
            self.peak_accelerator_driver = max(self.peak_accelerator_driver, driver)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def reset(self) -> None:
        rss, allocated, driver = self._read()
        with getattr(self, "_lock", threading.Lock()):
            self.peak_rss = rss
            self.peak_accelerator_allocated = allocated
            self.peak_accelerator_driver = driver

    def snapshot(self) -> dict[str, int]:
        self._sample()
        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            try:
                if torch_module.cuda.is_available():
                    self.peak_accelerator_allocated = max(
                        self.peak_accelerator_allocated,
                        int(torch_module.cuda.max_memory_allocated()),
                    )
                    self.peak_accelerator_driver = max(
                        self.peak_accelerator_driver,
                        int(torch_module.cuda.max_memory_reserved()),
                    )
            except (AttributeError, RuntimeError):
                pass
        with self._lock:
            return {
                "peak_rss_bytes": self.peak_rss,
                "peak_accelerator_allocated_bytes": self.peak_accelerator_allocated,
                "peak_accelerator_driver_bytes": self.peak_accelerator_driver,
            }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> float:
    return time.perf_counter()
