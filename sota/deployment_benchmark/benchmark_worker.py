from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import psutil

from common import (
    ARTIFACT_PATHS,
    QWEN_ADAPTER_PATH,
    ROOT,
    SAMPLE_PATH,
    ResourceMonitor,
    binary_metrics,
    bytes_to_mib,
    cached_snapshot_path,
    ensure_directories,
    load_module,
    load_sample,
    percentile,
    synchronize_accelerator,
    unique_file_size,
    write_json,
)


def selected_device() -> str:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return "cpu"
    if torch_module.cuda.is_available():
        return f"cuda:{torch_module.cuda.get_device_name(0)}"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def benchmark_batch_size(default: int) -> int:
    raw_value = os.environ.get("BENCHMARK_BATCH_SIZE")
    if raw_value is None:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError("BENCHMARK_BATCH_SIZE must be positive")
    return value


def snapshot_path(model_id: str, additional_cache_roots: list[Path] | None = None) -> Path | None:
    cache_roots = list(additional_cache_roots or [])
    cache_roots.extend(
        [
            Path(os.environ.get("HF_HUB_CACHE", "")),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    return cached_snapshot_path(model_id, [path for path in cache_roots if str(path)])


def benchmark_sklearn(method: str, texts: list[str]) -> tuple[list[int], dict[str, Any], Callable[[], None]]:
    artifact = ARTIFACT_PATHS[method]
    started = time.perf_counter()
    loaded = joblib.load(artifact)
    load_seconds = time.perf_counter() - started
    if isinstance(loaded, dict):
        vectorizer = loaded["vectorizer"]
        classifier = loaded["classifier"]
        threshold = float(loaded["threshold"])

        def predict(values: list[str]) -> list[int]:
            features = vectorizer.transform(values)
            if hasattr(classifier, "predict_proba"):
                scores = classifier.predict_proba(features)[:, 1]
            else:
                scores = classifier.decision_function(features)
            return [int(value >= threshold) for value in scores]
    else:
        model = loaded
        vectorizer = model.named_steps.get("tfidf")
        classifier = model.steps[-1][1]

        def predict(values: list[str]) -> list[int]:
            return [int(value) for value in model.predict(values)]

    predict(texts[:32])
    started = time.perf_counter()
    predictions = predict(texts)
    inference_seconds = time.perf_counter() - started
    classifier_parameters = getattr(classifier, "coef_", None)
    if classifier_parameters is None:
        classifier_parameters = getattr(classifier, "feature_log_prob_", None)
    details = {
        "load_seconds": load_seconds,
        "preprocessing_seconds": 0.0,
        "forward_seconds": inference_seconds,
        "inference_seconds": inference_seconds,
        "batch_size": len(texts),
        "device": "cpu",
        "artifact_size_bytes": unique_file_size([artifact]),
        "feature_count": len(getattr(vectorizer, "vocabulary_", {})),
        "classifier_coefficient_count": int(classifier_parameters.size) if classifier_parameters is not None else 0,
    }
    return predictions, details, lambda: None


def benchmark_fasttext(texts: list[str]) -> tuple[list[int], dict[str, Any], Callable[[], None]]:
    module = load_module(
        ROOT / "sota" / "fasttext" / "run_fasttext_baseline.py",
        "deployment_benchmark_fasttext",
    )
    fasttext = module.import_fasttext()
    artifact = ARTIFACT_PATHS["fasttext"]
    started = time.perf_counter()
    model = fasttext.load_model(str(artifact))
    metadata_path = artifact.with_suffix(artifact.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("threshold", 0.5))
    load_seconds = time.perf_counter() - started

    def predict(values: list[str]) -> list[int]:
        cleaned = [module.clean_fasttext_text(text) or "empty" for text in values]
        labels, probabilities = model.predict(cleaned, k=2)
        return [
            int(dict(zip(row_labels, row_probabilities, strict=False)).get("__label__spam", 0.0) >= threshold)
            for row_labels, row_probabilities in zip(labels, probabilities, strict=False)
        ]

    predict(texts[:32])
    started = time.perf_counter()
    predictions = predict(texts)
    inference_seconds = time.perf_counter() - started
    details = {
        "load_seconds": load_seconds,
        "preprocessing_seconds": 0.0,
        "forward_seconds": inference_seconds,
        "inference_seconds": inference_seconds,
        "batch_size": len(texts),
        "device": "cpu",
        "artifact_size_bytes": unique_file_size([artifact]),
        "feature_count": len(model.words),
        "embedding_dimension": int(model.get_dimension()),
    }
    return predictions, details, lambda: None


def benchmark_minilm(texts: list[str]) -> tuple[list[int], dict[str, Any], Callable[[], None]]:
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    artifact = ARTIFACT_PATHS["minilm_logistic_regression"]
    model_snapshot = snapshot_path(
        "sentence-transformers/all-MiniLM-L6-v2",
        [ROOT / "sota" / "minilm_logistic_regression" / ".hf_cache" / "hub"],
    )
    if model_snapshot is None:
        raise FileNotFoundError("MiniLM snapshot not found")
    started = time.perf_counter()
    model = SentenceTransformer(str(model_snapshot), local_files_only=True)
    model.max_seq_length = 256
    saved = joblib.load(artifact)
    classifier = saved["classifier"]
    threshold = float(saved["threshold"])
    load_seconds = time.perf_counter() - started
    batch_size = benchmark_batch_size(64)

    def split_text(values: list[str]) -> tuple[list[str], list[str]]:
        subjects, bodies = [], []
        for text in values:
            subject, separator, body = text.partition("\n\n")
            if subject.startswith("Subject: "):
                subjects.append(subject)
                bodies.append(body if separator else "")
            else:
                subjects.append("Subject:")
                bodies.append(text)
        return subjects, bodies

    def encode(values: list[str]) -> np.ndarray:
        subjects, bodies = split_text(values)
        subject_embeddings = model.encode(
            subjects,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        body_embeddings = model.encode(
            bodies,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings = np.concatenate([subject_embeddings, body_embeddings], axis=1)
        return normalize(embeddings, norm="l2", axis=1)

    encode(texts[:32])
    synchronize_accelerator()
    started = time.perf_counter()
    embeddings = encode(texts)
    predictions = [int(value >= threshold) for value in classifier.predict_proba(embeddings)[:, 1]]
    synchronize_accelerator()
    inference_seconds = time.perf_counter() - started
    artifact_paths = [artifact] + ([model_snapshot] if model_snapshot else [])
    details = {
        "load_seconds": load_seconds,
        "preprocessing_seconds": 0.0,
        "forward_seconds": inference_seconds,
        "inference_seconds": inference_seconds,
        "batch_size": batch_size,
        "device": str(getattr(model, "device", selected_device())),
        "artifact_size_bytes": unique_file_size(artifact_paths),
        "feature_count": int(classifier.coef_.shape[1]),
        "classifier_coefficient_count": int(classifier.coef_.size),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "max_seq_length": 256,
    }
    return predictions, details, lambda: None


def benchmark_distilbert(texts: list[str]) -> tuple[list[int], dict[str, Any], Callable[[], None]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    artifact = ARTIFACT_PATHS["distilbert_sequence_classification"]
    evaluation_config = json.loads((artifact / "evaluation_config.json").read_text(encoding="utf-8"))
    threshold = float(evaluation_config["decision_threshold"])
    device, _ = qwen_device(torch)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(artifact, local_files_only=True)
    model.to(device)
    model.eval()
    synchronize_accelerator()
    load_seconds = time.perf_counter() - started
    batch_size = benchmark_batch_size(32)

    def predict(values: list[str]) -> list[int]:
        predictions: list[int] = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = tokenizer(
                    values[start : start + batch_size], padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                scores = torch.softmax(model(**batch).logits, dim=-1)[:, 1].detach().cpu().tolist()
                predictions.extend(int(score >= threshold) for score in scores)
        return predictions

    predict(texts[:batch_size])
    synchronize_accelerator()
    started = time.perf_counter()
    predictions = predict(texts)
    synchronize_accelerator()
    inference_seconds = time.perf_counter() - started
    details = {
        "load_seconds": load_seconds,
        "preprocessing_seconds": 0.0,
        "forward_seconds": inference_seconds,
        "inference_seconds": inference_seconds,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "artifact_size_bytes": unique_file_size([artifact]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "max_seq_length": 512,
    }

    def cleanup() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return predictions, details, cleanup


def qwen_device(torch_module: Any) -> tuple[Any, Any]:
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), torch_module.bfloat16
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps"), torch_module.float16
    return torch_module.device("cpu"), torch_module.float32


def benchmark_qwen(texts: list[str], labels: list[int]) -> tuple[list[int], dict[str, Any], Callable[[], None]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = load_module(
        ROOT
        / "lora-fine-tuning"
        / "methods"
        / "03_causal_lm_next_token"
        / "notebooks"
        / "qwen3_0.6b_casual_lm_sweep.py",
        "deployment_benchmark_qwen_base",
    )
    evaluator = load_module(
        ROOT
        / "lora-fine-tuning"
        / "methods"
        / "03_causal_lm_next_token"
        / "notebooks"
        / "evaluate_checkpoints.py",
        "deployment_benchmark_qwen_evaluator",
    )
    device, dtype = qwen_device(torch)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base.MODEL_ID,
            dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base.MODEL_ID,
            torch_dtype=dtype,
            local_files_only=True,
        )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, QWEN_ADAPTER_PATH)
    model.to(device)
    model.eval()
    synchronize_accelerator()
    load_seconds = time.perf_counter() - started

    label_token_ids = {
        base.NEGATIVE_LABEL_TEXT: base.encode_text(tokenizer, base.NEGATIVE_LABEL_TEXT),
        base.POSITIVE_LABEL_TEXT: base.encode_text(tokenizer, base.POSITIVE_LABEL_TEXT),
    }
    if any(len(value) != 1 for value in label_token_ids.values()):
        raise RuntimeError(f"Expected one token per label: {label_token_ids}")
    label_ids = torch.tensor(
        [label_token_ids[base.NEGATIVE_LABEL_TEXT][0], label_token_ids[base.POSITIVE_LABEL_TEXT][0]],
        dtype=torch.long,
        device=device,
    )

    started = time.perf_counter()
    prompt_rows = []
    raw_token_lengths = []
    for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
        subject, separator, body = text.partition("\n\n")
        if subject.startswith("Subject: "):
            subject_value = subject.removeprefix("Subject: ")
            body_value = body if separator else ""
        else:
            subject_value = ""
            body_value = text
        tokenized = base.build_tokenized_sample(
            tokenizer,
            {"subject": subject_value, "body": body_value, "label": label},
            512,
        )
        prompt_rows.append((len(tokenized["prompt_input_ids"]), index, tokenized["prompt_input_ids"]))
        raw_token_lengths.append(int(tokenized["raw_email_tokens"]))
    prompt_rows.sort(key=lambda item: item[0])
    preprocessing_seconds = time.perf_counter() - started

    batch_size = benchmark_batch_size(16)

    def forward(rows: list[tuple[int, int, list[int]]]) -> list[tuple[int, int]]:
        output = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                batch = evaluator.pad_prompt_batch(
                    torch,
                    [item[2] for item in chunk],
                    tokenizer.pad_token_id,
                    device,
                )
                logits, _ = evaluator.forward_last_token_logits(model, batch, None)
                predictions = logits.index_select(dim=-1, index=label_ids).argmax(dim=-1).detach().cpu().tolist()
                output.extend((item[1], int(prediction)) for item, prediction in zip(chunk, predictions, strict=True))
        return output

    middle = max(0, len(prompt_rows) // 2 - batch_size // 2)
    forward(prompt_rows[middle : middle + batch_size])
    synchronize_accelerator()
    started = time.perf_counter()
    indexed_predictions = forward(prompt_rows)
    synchronize_accelerator()
    forward_seconds = time.perf_counter() - started
    predictions = [0] * len(texts)
    for index, prediction in indexed_predictions:
        predictions[index] = prediction

    base_snapshot = snapshot_path(base.MODEL_ID)
    adapter_files = [QWEN_ADAPTER_PATH / "adapter_config.json", QWEN_ADAPTER_PATH / "adapter_model.safetensors"]
    artifact_paths = adapter_files + ([base_snapshot] if base_snapshot else [])
    details = {
        "load_seconds": load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "forward_seconds": forward_seconds,
        "inference_seconds": preprocessing_seconds + forward_seconds,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "artifact_size_bytes": unique_file_size(artifact_paths),
        "adapter_size_bytes": unique_file_size(adapter_files),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "max_seq_length": 512,
        "input_tokens_mean": statistics.fmean(raw_token_lengths),
        "input_tokens_median": statistics.median(raw_token_lengths),
        "input_tokens_p95": percentile(raw_token_lengths, 95),
        "batching": "length_sorted",
    }

    def cleanup() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return predictions, details, cleanup


RUNNERS = {
    "tfidf_naive_bayes": lambda texts, labels: benchmark_sklearn("tfidf_naive_bayes", texts),
    "tfidf_logistic_regression": lambda texts, labels: benchmark_sklearn("tfidf_logistic_regression", texts),
    "tfidf_linear_svm": lambda texts, labels: benchmark_sklearn("tfidf_linear_svm", texts),
    "fasttext": lambda texts, labels: benchmark_fasttext(texts),
    "minilm_logistic_regression": lambda texts, labels: benchmark_minilm(texts),
    "distilbert_sequence_classification": lambda texts, labels: benchmark_distilbert(texts),
    "qwen3_lora_next_token": benchmark_qwen,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure isolated inference resources for one method.")
    parser.add_argument("--method", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_directories()

    process = psutil.Process(os.getpid())
    monitor = ResourceMonitor()
    monitor.start()
    process_started = time.perf_counter()
    texts, labels, sample_metadata = load_sample(args.sample)
    rss_after_data = process.memory_info().rss
    monitor.reset()
    load_phase_started = time.perf_counter()
    predictions, details, cleanup = RUNNERS[args.method](texts, labels)
    total_runner_seconds = time.perf_counter() - load_phase_started
    resources = monitor.snapshot()
    rss_after_inference = process.memory_info().rss
    monitor.stop()

    metrics = binary_metrics(labels, predictions)
    inference_seconds = float(details["inference_seconds"])
    result = {
        "method": args.method,
        "status": "completed",
        "rows": len(texts),
        "ham_count": sum(value == 0 for value in labels),
        "spam_count": sum(value == 1 for value in labels),
        **sample_metadata,
        **details,
        **metrics,
        "throughput_samples_per_second": len(texts) / inference_seconds,
        "latency_mean_ms_per_sample": inference_seconds * 1000.0 / len(texts),
        "rss_after_data_mib": bytes_to_mib(rss_after_data),
        "rss_after_inference_mib": bytes_to_mib(rss_after_inference),
        "peak_rss_mib": bytes_to_mib(resources["peak_rss_bytes"]),
        "peak_rss_delta_mib": bytes_to_mib(max(0, resources["peak_rss_bytes"] - rss_after_data)),
        "peak_accelerator_allocated_mib": bytes_to_mib(resources["peak_accelerator_allocated_bytes"]),
        "peak_accelerator_driver_mib": bytes_to_mib(resources["peak_accelerator_driver_bytes"]),
        "artifact_size_mib": bytes_to_mib(details["artifact_size_bytes"]),
        "total_worker_seconds": time.perf_counter() - process_started,
        "runner_seconds": total_runner_seconds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
    }
    if "adapter_size_bytes" in details:
        result["adapter_size_mib"] = bytes_to_mib(details["adapter_size_bytes"])
    write_json(args.output, result)
    cleanup()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
