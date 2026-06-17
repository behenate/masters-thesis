from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from common import (
    ARTIFACT_PATHS,
    BENCHMARK_DIR,
    METHODS,
    QWEN_ADAPTER_PATH,
    ROOT,
    SAMPLE_PATH,
    WORKER_RESULTS_DIR,
    ensure_directories,
    prepare_sample,
    write_json,
)


CSV_COLUMNS = [
    "method",
    "status",
    "rows",
    "ham_count",
    "spam_count",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_positive_count",
    "false_negative_count",
    "load_seconds",
    "preprocessing_seconds",
    "forward_seconds",
    "inference_seconds",
    "throughput_samples_per_second",
    "latency_mean_ms_per_sample",
    "batch_size",
    "device",
    "dtype",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "rss_after_inference_mib",
    "peak_accelerator_allocated_mib",
    "peak_accelerator_driver_mib",
    "artifact_size_mib",
    "adapter_size_mib",
    "parameter_count",
    "feature_count",
    "embedding_dimension",
    "max_seq_length",
    "input_tokens_mean",
    "input_tokens_median",
    "input_tokens_p95",
    "sample_sha256",
    "error_message",
]


def validate_artifacts(methods: list[str]) -> None:
    missing = []
    for method in methods:
        if method == "qwen3_lora_next_token":
            if not (QWEN_ADAPTER_PATH / "adapter_config.json").is_file():
                missing.append(str(QWEN_ADAPTER_PATH))
        elif not ARTIFACT_PATHS[method].exists():
            missing.append(str(ARTIFACT_PATHS[method]))
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing deployment artifacts:\n{formatted}")


def hardware_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "system_memory_gib": psutil.virtual_memory().total / (1024**3),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            }
        )
        if torch.cuda.is_available():
            payload["accelerator"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            payload["accelerator"] = "Apple Metal Performance Shaders"
        else:
            payload["accelerator"] = "CPU"
    except ImportError:
        payload["accelerator"] = "CPU"
    return payload


def run_worker(method: str, sample: Path, result_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        str(BENCHMARK_DIR / "benchmark_worker.py"),
        "--method",
        method,
        "--sample",
        str(sample),
        "--output",
        str(result_path),
    ]
    print(f"\n=== {method} ===", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode != 0:
        if result_path.is_file():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("status") == "completed":
                print("worker cleanup failed after a completed result was saved; preserving result", flush=True)
                return saved
        payload = {
            "method": method,
            "status": "failed",
            "rows": 1000,
            "started_at_utc": started,
            "error_message": completed.stderr.strip() or f"worker exited with {completed.returncode}",
        }
        write_json(result_path, payload)
        return payload
    return json.loads(result_path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    extra_columns = sorted({key for row in rows for key in row} - set(CSV_COLUMNS))
    columns = CSV_COLUMNS + extra_columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deployment cost on one shared 1000-message sample.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "inference_benchmark.csv")
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_directories()

    if args.force_sample or not args.sample.is_file():
        metadata = prepare_sample(args.sample)
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    validate_artifacts(args.methods)

    hardware_path = BENCHMARK_DIR / "hardware.json"
    write_json(hardware_path, hardware_metadata())
    rows = []
    for method in args.methods:
        result_path = WORKER_RESULTS_DIR / f"{method}.json"
        if result_path.is_file() and not args.overwrite:
            print(f"reusing={result_path}", flush=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_worker(method, args.sample, result_path)
        rows.append(result)
        write_csv(args.output, rows)

    print(f"\nresults={args.output}", flush=True)
    print(f"hardware={hardware_path}", flush=True)
    return 0 if all(row.get("status") == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
