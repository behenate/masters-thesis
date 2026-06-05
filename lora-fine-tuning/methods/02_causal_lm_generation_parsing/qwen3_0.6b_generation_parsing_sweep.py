#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


AIM_EXPERIMENT_NAME = "qwen3-0.6b-spam-causal-lm-generation-parsing-sweep"
EVALUATION_METHOD = "causal_lm_generation_parsing"
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_PARSE_FAILURE_LABEL = "ham"
TOP_NEXT_TOKEN_CONFIG_INDEXES = (8, 11, 6, 9)

_TOKENIZER: Any | None = None
_GENERATION_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
_PARSE_FAILURE_LABEL = DEFAULT_PARSE_FAILURE_LABEL


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "lora-fine-tuning").is_dir():
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and lora-fine-tuning/.")


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def load_next_token_sweep_module() -> Any:
    root = project_root()
    candidates = [
        root
        / "lora-fine-tuning"
        / "methods"
        / "03_causal_lm_next_token"
        / "notebooks"
        / "qwen3_0.6b_casual_lm_sweep.py",
        root / "lora-fine-tuning" / "qwen3_0.6b_casual_lm_sweep.py",
    ]
    script_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if script_path is None:
        attempted = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find qwen3_0.6b_casual_lm_sweep.py. Tried:\n{attempted}")
    spec = importlib.util.spec_from_file_location("qwen3_next_token_sweep_helpers", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sweep helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_next_token_sweep_module()
_BASE_SWEEP_CONFIGS = {config.index: config for config in base.build_sweep_configs()}
base.AIM_EXPERIMENT_NAME = AIM_EXPERIMENT_NAME

MODEL_ID = base.MODEL_ID
SEED = base.SEED
POSITIVE_LABEL_TEXT = base.POSITIVE_LABEL_TEXT
NEGATIVE_LABEL_TEXT = base.NEGATIVE_LABEL_TEXT
IM_END_TOKEN = base.IM_END_TOKEN
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def default_results_root() -> Path:
    return method_dir() / "results"


def resolve_results_root(value: str | None) -> Path:
    if value is None:
        return default_results_root()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = method_dir() / path
    return path.resolve()


def make_sweep_id() -> str:
    return "qwen3_clm_generation_parsing_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_sweep_configs() -> list[Any]:
    return [_BASE_SWEEP_CONFIGS[index] for index in TOP_NEXT_TOKEN_CONFIG_INDEXES]


def checkpoint_settings_for_config(config: Any, max_steps: int | None = None) -> dict[str, Any]:
    checkpoint_save_steps = base.CHECKPOINT_SAVE_STEPS if config.max_seq_length < 600 else base.CHECKPOINT_SAVE_STEPS * 2
    if max_steps is not None:
        checkpoint_save_steps = max(1, min(checkpoint_save_steps, int(max_steps)))
    return {
        "checkpoint_save_steps": checkpoint_save_steps,
        "eval_steps": checkpoint_save_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
    }


def get_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER = tokenizer
    return _TOKENIZER


def strip_reasoning_and_special_tokens(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub(" ", text)
    cleaned = cleaned.replace(IM_END_TOKEN, " ")
    cleaned = cleaned.replace("<|endoftext|>", " ")
    return cleaned.strip()


def parse_generated_label(text: str) -> str | None:
    cleaned = strip_reasoning_and_special_tokens(text).lower()
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""

    if re.search(r"\bspam\b", first_line):
        return POSITIVE_LABEL_TEXT
    if re.search(r"\bham\b", first_line):
        return NEGATIVE_LABEL_TEXT
    if re.search(r"\bvalid\b", first_line):
        return NEGATIVE_LABEL_TEXT
    if re.search(r"\byes\b", first_line):
        return POSITIVE_LABEL_TEXT
    if re.search(r"\bno\b", first_line):
        return NEGATIVE_LABEL_TEXT
    return None


def label_to_id(label_text: str | None) -> int:
    if label_text == POSITIVE_LABEL_TEXT:
        return 1
    if label_text == NEGATIVE_LABEL_TEXT:
        return 0
    return 1 if _PARSE_FAILURE_LABEL == POSITIVE_LABEL_TEXT else 0


def compute_generation_metrics(
    predictions: list[int],
    labels: list[int],
    parse_failure_count: int,
) -> dict[str, float]:
    import numpy as np

    predictions_array = np.asarray(predictions)
    labels_array = np.asarray(labels)

    if len(labels_array) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "classification_failure_count": float(parse_failure_count),
            "classification_failure_rate": 0.0,
            "parse_failure_count": float(parse_failure_count),
            "parse_failure_rate": 0.0,
            "false_positive_count": 0.0,
            "false_negative_count": 0.0,
            "true_positive_count": 0.0,
            "true_negative_count": 0.0,
        }

    tp = int(((predictions_array == 1) & (labels_array == 1)).sum())
    fp = int(((predictions_array == 1) & (labels_array == 0)).sum())
    fn = int(((predictions_array == 0) & (labels_array == 1)).sum())
    tn = int(((predictions_array == 0) & (labels_array == 0)).sum())

    accuracy = float((predictions_array == labels_array).mean())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (recall + specificity) / 2
    parse_failure_rate = parse_failure_count / max(len(labels_array), 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "classification_failure_count": float(parse_failure_count),
        "classification_failure_rate": float(parse_failure_rate),
        "parse_failure_count": float(parse_failure_count),
        "parse_failure_rate": float(parse_failure_rate),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
    }


def pad_prompt_batch_left(torch_module: Any, prompt_batches: list[list[int]], pad_token_id: int, device: Any) -> dict[str, Any]:
    max_length = max(len(ids) for ids in prompt_batches)
    input_ids = torch_module.full((len(prompt_batches), max_length), pad_token_id, dtype=torch_module.long, device=device)
    attention_mask = torch_module.zeros((len(prompt_batches), max_length), dtype=torch_module.long, device=device)
    for row, ids in enumerate(prompt_batches):
        values = torch_module.tensor(ids, dtype=torch_module.long, device=device)
        input_ids[row, -len(ids) :] = values
        attention_mask[row, -len(ids) :] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def evaluate_generation_parser_classifier(
    *,
    model: Any,
    torch_module: Any,
    dataset_split: Any,
    label_token_ids: dict[str, int],
    pad_token_id: int,
    batch_size: int,
) -> dict[str, float]:
    del label_token_ids
    tokenizer = get_tokenizer()
    model.eval()
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    parse_failure_count = 0

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    stop_token_ids = [im_end_id] if im_end_id is not None and im_end_id >= 0 else []
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    if not stop_token_ids:
        stop_token_ids = None

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True

    try:
        with torch_module.inference_mode():
            for start in range(0, len(dataset_split), batch_size):
                rows = dataset_split.select(range(start, min(start + batch_size, len(dataset_split))))
                prompt_input_ids = rows["prompt_input_ids"]
                batch = pad_prompt_batch_left(torch_module, prompt_input_ids, pad_token_id, device)
                outputs = model.generate(
                    **batch,
                    max_new_tokens=_GENERATION_MAX_NEW_TOKENS,
                    do_sample=False,
                    eos_token_id=stop_token_ids,
                    pad_token_id=pad_token_id,
                )
                generated = outputs[:, batch["input_ids"].shape[1] :]
                decoded = tokenizer.batch_decode(
                    generated.detach().cpu().tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                for raw_generation, actual_label in zip(decoded, rows["label"], strict=False):
                    parsed_label = parse_generated_label(raw_generation)
                    if parsed_label is None:
                        parse_failure_count += 1
                    predictions.append(label_to_id(parsed_label))
                    labels.append(int(actual_label))
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    return compute_generation_metrics(predictions, labels, parse_failure_count)


def build_generation_metrics_callback(
    *,
    torch_module: Any,
    trainer_callback_cls: Any,
    eval_dataset: Any,
    label_token_ids: dict[str, int],
    pad_token_id: int,
    batch_size: int,
) -> Any:
    class GenerationParsingMetricsCallback(trainer_callback_cls):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if model is None or metrics is None:
                return
            classification_metrics = evaluate_generation_parser_classifier(
                model=model,
                torch_module=torch_module,
                dataset_split=eval_dataset,
                label_token_ids=label_token_ids,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
            )
            for key, value in classification_metrics.items():
                metrics[f"eval_{key}"] = value

    return GenerationParsingMetricsCallback()


def patch_base_evaluator() -> None:
    base.AIM_EXPERIMENT_NAME = AIM_EXPERIMENT_NAME
    base.build_sweep_configs = build_sweep_configs
    base.evaluate_next_token_classifier = evaluate_generation_parser_classifier
    base.build_next_token_metrics_callback = build_generation_metrics_callback


SUMMARY_COLUMNS = [
    "status",
    "config_index",
    "config_id",
    "group",
    "max_seq_length",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "train_batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "evaluation_method",
    "max_new_tokens",
    "parse_failure_default_label",
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_specificity",
    "validation_balanced_accuracy",
    "validation_parse_failure_count",
    "validation_parse_failure_rate",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_specificity",
    "test_balanced_accuracy",
    "test_parse_failure_count",
    "test_parse_failure_rate",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "global_step",
    "best_metric",
    "aim_run_hash",
    "run_name",
    "run_dir",
    "adapter_path",
    "error_type",
    "error_message",
]


def summary_row_from_metrics(config: Any, run_dir: Path) -> dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "config_index": config.index,
            "config_id": config.config_id,
            "group": config.group,
            "max_seq_length": config.max_seq_length,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "evaluation_method": EVALUATION_METHOD,
            "run_dir": str(run_dir),
        }
    )

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        row["status"] = "missing"
        return row

    try:
        metrics = base.read_json(metrics_path)
    except json.JSONDecodeError as exc:
        row["status"] = "metrics_json_invalid"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        return row

    row["status"] = metrics.get("status", "unknown")
    row["run_name"] = metrics.get("run_name", "")
    row["adapter_path"] = metrics.get("adapter_path", "")
    row["aim_run_hash"] = metrics.get("aim_run_hash", "")
    row["error_type"] = metrics.get("error_type", "")
    row["error_message"] = metrics.get("error_message", "")
    row["evaluation_method"] = metrics.get("evaluation_method", EVALUATION_METHOD)
    row["max_new_tokens"] = metrics.get("max_new_tokens", "")
    row["parse_failure_default_label"] = metrics.get("parse_failure_default_label", "")

    training = metrics.get("training_metrics", {}) or {}
    trainer_state = metrics.get("trainer_state", {}) or {}
    row["train_runtime"] = training.get("train_runtime", "")
    row["train_samples_per_second"] = training.get("train_samples_per_second", "")
    row["train_steps_per_second"] = training.get("train_steps_per_second", "")
    row["global_step"] = trainer_state.get("global_step", "")
    row["best_metric"] = trainer_state.get("best_metric", "")

    for prefix, metric_key in [("validation", "validation_metrics"), ("test", "test_metrics")]:
        split_metrics = metrics.get(metric_key, {}) or {}
        for name in ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"]:
            row[f"{prefix}_{name}"] = split_metrics.get(name, "")
        row[f"{prefix}_parse_failure_count"] = split_metrics.get(
            "parse_failure_count", split_metrics.get("classification_failure_count", "")
        )
        row[f"{prefix}_parse_failure_rate"] = split_metrics.get(
            "parse_failure_rate", split_metrics.get("classification_failure_rate", "")
        )

    return row


def write_summary(sweep_dir: Path, selected_configs: list[Any]) -> Path:
    rows = [summary_row_from_metrics(config, base.config_run_dir(sweep_dir, config)) for config in selected_configs]
    summary_path = sweep_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_manifest(sweep_dir: Path, sweep_id: str, selected_configs: list[Any], args: argparse.Namespace) -> Path:
    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest = {
        "sweep_id": sweep_id,
        "created_or_updated_at": dt.datetime.now(dt.UTC).isoformat(),
        "aim_experiment_name": AIM_EXPERIMENT_NAME,
        "evaluation_method": EVALUATION_METHOD,
        "parse_failure_default_label": args.parse_failure_label,
        "max_new_tokens": args.max_new_tokens,
        "model_id": MODEL_ID,
        "seed": SEED,
        "train_split": base.TRAIN_SPLIT,
        "validation_split": base.VALIDATION_SPLIT,
        "test_split": base.TEST_SPLIT,
        "num_train_epochs": base.NUM_TRAIN_EPOCHS,
        "warmup_ratio": base.WARMUP_RATIO,
        "weight_decay": base.WEIGHT_DECAY,
        "max_grad_norm": base.MAX_GRAD_NORM,
        "logging_steps": base.LOGGING_STEPS,
        "checkpoint_save_steps": base.CHECKPOINT_SAVE_STEPS,
        "checkpointing": {
            "base_checkpoint_save_steps": base.CHECKPOINT_SAVE_STEPS,
            "per_config": {
                str(config.index): checkpoint_settings_for_config(config, args.max_steps)
                for config in selected_configs
            },
        },
        "script": str(Path(__file__).resolve()),
        "project_root": str(project_root()),
        "sweep_dir": str(sweep_dir),
        "selected_config_indexes": [config.index for config in selected_configs],
        "configs": [config.to_dict() for config in selected_configs],
        "runner_args": vars(args),
    }
    base.write_json(manifest_path, manifest)
    return manifest_path


def run_sweep(args: argparse.Namespace) -> int:
    patch_base_evaluator()
    sweep_id = args.sweep_id or make_sweep_id()
    results_root = resolve_results_root(args.results_root)
    sweep_dir = results_root / "sweeps" / sweep_id
    selected_configs = [base.get_config(index) for index in base.parse_config_indexes(args.config_index)]

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(sweep_dir, sweep_id, selected_configs, args)
    summary_path = write_summary(sweep_dir, selected_configs)

    print(f"Sweep ID: {sweep_id}")
    print(f"Method: {EVALUATION_METHOD}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if args.dry_run:
        print("Dry run requested; no training subprocesses were launched.")
        return 0

    failures = 0
    python_bin = args.python or sys.executable
    root = project_root()

    for config in selected_configs:
        run_dir = base.config_run_dir(sweep_dir, config)
        if args.resume and base.metrics_completed(run_dir):
            print(f"[{config.index:02d}] Skipping completed config: {config.config_id}")
            continue

        command = [
            python_bin,
            str(Path(__file__).resolve()),
            "run-one",
            "--config-index",
            str(config.index),
            "--sweep-id",
            sweep_id,
            "--results-root",
            str(results_root),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--parse-failure-label",
            args.parse_failure_label,
        ]
        if args.allow_non_cuda:
            command.append("--allow-non-cuda")
        if args.resume:
            command.append("--resume-checkpoint")
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.train_limit is not None:
            command.extend(["--train-limit", str(args.train_limit)])
        if args.validation_limit is not None:
            command.extend(["--validation-limit", str(args.validation_limit)])
        if args.test_limit is not None:
            command.extend(["--test-limit", str(args.test_limit)])

        print(f"[{config.index:02d}] Starting {config.config_id}")
        try:
            return_code = base.run_child_process(command, cwd=root)
        except KeyboardInterrupt:
            base.write_subprocess_failure_metrics(
                run_dir=run_dir,
                sweep_id=sweep_id,
                config=config,
                status="interrupted",
                error_type="KeyboardInterrupt",
                error_message="Sweep interrupted; active training subprocess was terminated.",
            )
            write_summary(sweep_dir, selected_configs)
            print(f"[{config.index:02d}] Interrupted; active child process was terminated.")
            return 130

        if return_code != 0:
            failures += 1
            base.write_subprocess_failure_metrics(
                run_dir=run_dir,
                sweep_id=sweep_id,
                config=config,
                status="failed",
                error_type="SubprocessFailure",
                error_message=f"run-one exited with code {return_code} before writing metrics.json",
            )
            print(f"[{config.index:02d}] Failed with exit code {return_code}; continuing.")
            print(f"[{config.index:02d}] Failure details: {base.subprocess_failure_details(run_dir)}")
        else:
            print(f"[{config.index:02d}] Completed.")

        summary_path = write_summary(sweep_dir, selected_configs)
        print(f"Updated summary: {summary_path}")

        if args.cooldown_seconds > 0 and config.index != selected_configs[-1].index:
            print(f"Cooling down for {args.cooldown_seconds:g} seconds.")
            time.sleep(args.cooldown_seconds)

    summary_path = write_summary(sweep_dir, selected_configs)
    if failures:
        print(f"Sweep finished with {failures} failed config(s). Summary: {summary_path}")
    return 1 if failures else 0


def run_one(args: argparse.Namespace) -> int:
    global _GENERATION_MAX_NEW_TOKENS, _PARSE_FAILURE_LABEL
    if args.sweep_id is None:
        raise SystemExit("--sweep-id is required for run-one")
    _GENERATION_MAX_NEW_TOKENS = int(args.max_new_tokens)
    _PARSE_FAILURE_LABEL = args.parse_failure_label
    patch_base_evaluator()

    config = base.get_config(args.config_index)
    results_root = resolve_results_root(args.results_root)
    sweep_dir = results_root / "sweeps" / args.sweep_id
    run_dir = base.config_run_dir(sweep_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    base.write_json(
        run_dir / "config.json",
        {
            "sweep_id": args.sweep_id,
            "run_name": base.run_name(args.sweep_id, config),
            "run_dir": str(run_dir),
            "evaluation_method": EVALUATION_METHOD,
            "max_new_tokens": args.max_new_tokens,
            "parse_failure_default_label": args.parse_failure_label,
            "config": config.to_dict(),
            "checkpointing": checkpoint_settings_for_config(config, args.max_steps),
            "args": vars(args),
        },
    )

    try:
        metrics = base.run_training_config(args, config, args.sweep_id, run_dir)
        metrics["evaluation_method"] = EVALUATION_METHOD
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["parse_failure_default_label"] = args.parse_failure_label
        base.write_json(run_dir / "metrics.json", metrics)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "sweep_id": args.sweep_id,
            "run_name": base.run_name(args.sweep_id, config),
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "evaluation_method": EVALUATION_METHOD,
            "max_new_tokens": args.max_new_tokens,
            "parse_failure_default_label": args.parse_failure_label,
            "checkpointing": checkpoint_settings_for_config(config, args.max_steps),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        base.write_json(run_dir / "metrics.json", failure)
        print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(failure["traceback"], file=sys.stderr)
        return 1
    finally:
        base.cleanup_after_run()


def summarize_command(args: argparse.Namespace) -> int:
    patch_base_evaluator()
    results_root = resolve_results_root(args.results_root)
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir).resolve()
    elif args.sweep_id:
        sweep_dir = results_root / "sweeps" / args.sweep_id
    else:
        raise SystemExit("Pass --sweep-id or --sweep-dir.")

    selected_configs = base.build_sweep_configs()
    summary_path = write_summary(sweep_dir, selected_configs)
    print(f"Wrote summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3 causal LM generation+parsing hyperparameter sweeps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-configs", help="Print the selected top next-token configurations.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    sweep_parser = subparsers.add_parser("run-sweep", help="Launch one subprocess per selected config.")
    sweep_parser.add_argument("--sweep-id", default=None, help="Optional stable sweep id.")
    sweep_parser.add_argument("--results-root", default=None, help="Defaults to this method's results directory.")
    sweep_parser.add_argument("--resume", action="store_true", help="Skip configs with completed metrics.json.")
    sweep_parser.add_argument("--allow-non-cuda", action="store_true", help="Allow non-CUDA runs for tiny smoke tests.")
    sweep_parser.add_argument("--dry-run", action="store_true", help="Write manifest/summary but do not launch training.")
    sweep_parser.add_argument("--cooldown-seconds", type=float, default=10.0, help="Pause between subprocess runs.")
    sweep_parser.add_argument("--python", default=None, help="Python executable for child run-one processes.")
    sweep_parser.add_argument("--config-index", action="append", help="Config index or comma-separated indexes.")
    sweep_parser.add_argument("--max-steps", type=int, default=None, help="Optional Trainer max_steps override.")
    sweep_parser.add_argument("--train-limit", type=int, default=None, help="Optional train row limit for smoke tests.")
    sweep_parser.add_argument("--validation-limit", type=int, default=None, help="Optional validation row limit for smoke tests.")
    sweep_parser.add_argument("--test-limit", type=int, default=None, help="Optional test row limit for smoke tests.")
    sweep_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    sweep_parser.add_argument("--parse-failure-label", choices=[NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT], default=DEFAULT_PARSE_FAILURE_LABEL)

    one_parser = subparsers.add_parser("run-one", help="Run a single config. Used by run-sweep.")
    one_parser.add_argument("--config-index", type=int, required=True)
    one_parser.add_argument("--sweep-id", default=None)
    one_parser.add_argument("--results-root", default=None)
    one_parser.add_argument("--allow-non-cuda", action="store_true")
    one_parser.add_argument("--resume-checkpoint", action="store_true", help="Resume from the latest trainer checkpoint if present.")
    one_parser.add_argument("--dry-run", action="store_true", help="Prepare/tokenize data only; do not load model.")
    one_parser.add_argument("--max-steps", type=int, default=None)
    one_parser.add_argument("--train-limit", type=int, default=None)
    one_parser.add_argument("--validation-limit", type=int, default=None)
    one_parser.add_argument("--test-limit", type=int, default=None)
    one_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    one_parser.add_argument("--parse-failure-label", choices=[NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT], default=DEFAULT_PARSE_FAILURE_LABEL)

    summarize_parser = subparsers.add_parser("summarize", help="Rebuild summary.csv for a sweep.")
    summarize_parser.add_argument("--sweep-id", default=None)
    summarize_parser.add_argument("--sweep-dir", default=None)
    summarize_parser.add_argument("--results-root", default=None)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    base.validate_args(args)
    if getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS) < 1:
        raise SystemExit("--max-new-tokens must be >= 1.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patch_base_evaluator()
    validate_args(args)
    if args.command == "list-configs":
        base.list_configs(args.json)
        return 0
    if args.command == "run-sweep":
        return run_sweep(args)
    if args.command == "run-one":
        return run_one(args)
    if args.command == "summarize":
        return summarize_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
