#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")


def load_baseline_module() -> Any:
    script_path = Path(__file__).resolve().with_name("qwen3_0.6b_unmodified_baseline.py")
    spec = importlib.util.spec_from_file_location("qwen3_unmodified_baseline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load baseline script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_baseline_module()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(dataset_name: str, records: list[dict[str, Any]], metrics: dict[str, float]) -> dict[str, Any]:
    rows = len(records)
    ham_count = sum(1 for row in records if int(row["label"]) == 0)
    spam_count = sum(1 for row in records if int(row["label"]) == 1)
    ham_predictions = sum(1 for row in records if int(row["prediction"]) == 0)
    spam_predictions = sum(1 for row in records if int(row["prediction"]) == 1)
    return {
        "dataset": dataset_name,
        "rows": rows,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_predictions": ham_predictions,
        "spam_predictions": spam_predictions,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "f1": round(float(metrics["f1"]), 6),
        "recall": round(float(metrics["recall"]), 6),
        "specificity": round(float(metrics["specificity"]), 6),
        "balanced_accuracy": round(float(metrics["balanced_accuracy"]), 6),
        "spam_prediction_rate": round(float(metrics["spam_prediction_rate"]), 6),
        "parse_failure_count": int(metrics["parse_failure_count"]),
        "parse_failure_rate": round(float(metrics["parse_failure_rate"]), 6),
    }


def print_summary(row: dict[str, Any]) -> None:
    print(
        " | ".join(
            [
                row["dataset"],
                f"rows={row['rows']}",
                f"labels ham/spam={row['ham_count']}/{row['spam_count']}",
                f"pred ham/spam={row['ham_predictions']}/{row['spam_predictions']}",
                f"accuracy={row['accuracy']:.3f}",
                f"f1={row['f1']:.3f}",
                f"spam_pred_rate={row['spam_prediction_rate']:.3f}",
                f"parse_fail_rate={row['parse_failure_rate']:.3f}",
            ]
        ),
        flush=True,
    )


def show_unparsed_examples(records: list[dict[str, Any]], limit: int) -> None:
    examples = [row for row in records if row.get("parse_failed")]
    if not examples:
        print("  no parse failures", flush=True)
        return
    print(f"  parse failure examples, first {min(limit, len(examples))}:", flush=True)
    for row in examples[:limit]:
        raw = str(row.get("raw_generation", "")).replace("\n", "\\n")
        print(
            f"  - row={row['row_index']} label={row['label_text']} "
            f"fallback_prediction={row['prediction_text']} raw={raw[:160]!r}",
            flush=True,
        )


def evaluate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    baseline.project_root()

    run_dir = Path(args.output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(baseline.MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device, dtype = baseline.device_and_dtype(torch)
    print(f"model={baseline.MODEL_ID}", flush=True)
    print(f"device={device}; dtype={dtype}", flush=True)
    print(f"prompt_style={args.prompt_style}; parse_failure_label={args.parse_failure_label}", flush=True)
    print(f"sample_limit={args.sample_limit}; batch_size={args.batch_size}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            baseline.MODEL_ID,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            baseline.MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    summary_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        started = time.perf_counter()
        print(f"\nPreparing {dataset_name}", flush=True)
        raw_dataset, metadata = baseline.build_dataset_sample(dataset_name, args.sample_limit, args.seed)
        dataset = baseline.tokenized_eval_dataset(tokenizer, raw_dataset, args.max_seq_length, args.prompt_style)
        eval_args = SimpleNamespace(
            batch_size=args.batch_size,
            current_dataset=dataset_name,
            max_new_tokens=args.max_new_tokens,
            parse_failure_label=args.parse_failure_label,
            prompt_style=args.prompt_style,
        )
        metrics, records = baseline.prediction_records_from_generation(
            model=model,
            torch_module=torch,
            tokenizer=tokenizer,
            dataset=dataset,
            args=eval_args,
        )
        row = summarize_records(dataset_name, records, metrics)
        row["runtime_seconds"] = round(time.perf_counter() - started, 4)
        row["source_rows"] = metadata["rows"]
        summary_rows.append(row)
        all_records.extend(records)
        print_summary(row)
        show_unparsed_examples(records, args.show_failures)

    write_csv(run_dir / "summary.csv", summary_rows)
    write_csv(run_dir / "predictions.csv", all_records)
    print(f"\nWrote {run_dir / 'summary.csv'}", flush=True)
    print(f"Wrote {run_dir / 'predictions.csv'}", flush=True)

    if all(row["spam_predictions"] == 0 for row in summary_rows):
        print("Result check: all tested datasets produced only ham predictions.", flush=True)
    else:
        print("Result check: at least one dataset produced spam predictions.", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small local smoke test for unmodified Qwen3-0.6B generation parsing on 50 samples."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=baseline.SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=baseline.DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=baseline.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--prompt-style",
        choices=["defined-labels", "decision-checklist", "thinking-structured", "training-compatible"],
        default=baseline.DEFAULT_PROMPT_STYLE,
    )
    parser.add_argument(
        "--parse-failure-label",
        choices=[baseline.NEGATIVE_LABEL_TEXT, baseline.POSITIVE_LABEL_TEXT],
        default=baseline.DEFAULT_PARSE_FAILURE_LABEL,
    )
    parser.add_argument("--show-failures", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results" / "simple_generation_50"),
    )
    return parser


def main() -> int:
    return evaluate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
