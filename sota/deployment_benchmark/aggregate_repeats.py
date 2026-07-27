from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_benchmark import CSV_COLUMNS


RUN_PATTERN = re.compile(r"^(?P<method>.+)_run\d+\.json$")
MEDIAN_FIELDS = {
    "load_seconds",
    "preprocessing_seconds",
    "forward_seconds",
    "inference_seconds",
    "rss_after_data_mib",
    "rss_after_inference_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "peak_accelerator_allocated_mib",
    "peak_accelerator_driver_mib",
    "runner_seconds",
    "total_worker_seconds",
}
INVARIANT_FIELDS = {
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
    "true_positive_count",
    "true_negative_count",
    "batch_size",
    "device",
    "dtype",
    "artifact_size_mib",
    "adapter_size_mib",
    "parameter_count",
    "feature_count",
    "embedding_dimension",
    "max_seq_length",
    "sample_sha256",
}


def load_runs(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*_run*.json")):
        match = RUN_PATTERN.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise RuntimeError(f"Run did not complete: {path}")
        method = match.group("method")
        if payload.get("method") != method:
            raise RuntimeError(f"Method mismatch in {path}")
        grouped[method].append(payload)
    if not grouped:
        raise RuntimeError(f"No repeat results found in {input_dir}")
    return dict(grouped)


def assert_invariants(method: str, runs: list[dict[str, Any]]) -> None:
    first = runs[0]
    for field in INVARIANT_FIELDS:
        expected = first.get(field)
        for index, run in enumerate(runs[1:], start=2):
            if run.get(field) != expected:
                raise RuntimeError(
                    f"{method}: field {field!r} differs between run 1 and run {index}"
                )


def aggregate_method(method: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    assert_invariants(method, runs)
    result = dict(runs[0])
    for field in MEDIAN_FIELDS:
        values = [float(run[field]) for run in runs if field in run]
        if values:
            result[field] = statistics.median(values)

    inference_seconds = float(result["inference_seconds"])
    rows = int(result["rows"])
    result["throughput_samples_per_second"] = rows / inference_seconds
    result["latency_mean_ms_per_sample"] = inference_seconds * 1000.0 / rows
    result["repeat_count"] = len(runs)
    result["inference_seconds_min"] = min(float(run["inference_seconds"]) for run in runs)
    result["inference_seconds_max"] = max(float(run["inference_seconds"]) for run in runs)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    extra_columns = sorted({key for row in rows for key in row} - set(CSV_COLUMNS))
    columns = CSV_COLUMNS + extra_columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated deployment benchmarks using medians."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = load_runs(args.input)
    rows = [aggregate_method(method, runs) for method, runs in sorted(grouped.items())]
    write_csv(args.output, rows)
    print(f"results={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
