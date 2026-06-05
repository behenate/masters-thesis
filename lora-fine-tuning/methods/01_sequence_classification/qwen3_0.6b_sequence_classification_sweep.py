#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


AIM_EXPERIMENT_NAME = "qwen3-0.6b-spam-sequence-classification-sweep"
EVALUATION_METHOD = "sequence_classification"
TOP_NEXT_TOKEN_CONFIG_INDEXES = (8, 11, 6, 9)
AIM_SYSTEM_TRACKING_INTERVAL = 10


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

MODEL_ID = base.MODEL_ID
SEED = base.SEED
POSITIVE_LABEL_TEXT = base.POSITIVE_LABEL_TEXT
NEGATIVE_LABEL_TEXT = base.NEGATIVE_LABEL_TEXT


def build_sweep_configs() -> list[Any]:
    return [_BASE_SWEEP_CONFIGS[index] for index in TOP_NEXT_TOKEN_CONFIG_INDEXES]


def patch_base_configs() -> None:
    base.AIM_EXPERIMENT_NAME = AIM_EXPERIMENT_NAME
    base.build_sweep_configs = build_sweep_configs


patch_base_configs()


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
    return "qwen3_sequence_classification_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_specificity",
    "validation_balanced_accuracy",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_specificity",
    "test_balanced_accuracy",
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
        "script": str(Path(__file__).resolve()),
        "project_root": str(project_root()),
        "sweep_dir": str(sweep_dir),
        "selected_config_indexes": [config.index for config in selected_configs],
        "configs": [config.to_dict() for config in selected_configs],
        "runner_args": vars(args),
    }
    base.write_json(manifest_path, manifest)
    return manifest_path


def text_from_sample(sample: dict[str, Any]) -> str:
    return base.build_email_text(sample["subject"], sample["body"])


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(int(limit), len(dataset))))


def label_counts(dataset: Any) -> tuple[int, int]:
    labels = [int(value) for value in dataset["label"]]
    spam = sum(1 for value in labels if value == 1)
    ham = sum(1 for value in labels if value == 0)
    return ham, spam


def token_stats(dataset_dict: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, split_dataset in dataset_dict.items():
        rows = len(split_dataset)
        lengths = split_dataset["token_length"] if rows else []
        was_trimmed = split_dataset["was_trimmed"] if rows else []
        stats[split] = {
            "rows": rows,
            "trimmed_rows": int(sum(bool(value) for value in was_trimmed)),
            "trimmed_rate": float(sum(bool(value) for value in was_trimmed) / max(rows, 1)),
            "max_token_length": int(max(lengths) if lengths else 0),
            "mean_token_length": float(sum(lengths) / max(rows, 1)),
        }
    return stats


def prepare_datasets(
    *,
    tokenizer: Any,
    config: Any,
    train_limit: int | None,
    validation_limit: int | None,
    test_limit: int | None,
) -> tuple[Any, dict[str, Any], str, str]:
    from datasets import ClassLabel, DatasetDict, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets
    from aim_tracking import summarize_text_classification_dataset

    disable_progress_bars()

    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    dataset_sha = base.sha256_file(data_path)
    raw_dataset = load_dataset("parquet", data_files=data_path, split="train")
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=[NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT]))
    raw_summary = summarize_text_classification_dataset(raw_dataset)

    holdout = raw_dataset.train_test_split(
        test_size=base.HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=SEED,
    )
    valid_test = holdout["test"].train_test_split(
        test_size=base.TEST_SPLIT / base.HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=SEED,
    )

    dataset = DatasetDict(
        {
            "train": select_limit(holdout["train"], train_limit),
            "validation": select_limit(valid_test["train"], validation_limit),
            "test": select_limit(valid_test["test"], test_limit),
        }
    )
    dataset = dataset.filter(lambda sample: bool(text_from_sample(sample)), desc="Filtering empty emails")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [
            base.build_email_text(subject, body)
            for subject, body in zip(batch["subject"], batch["body"], strict=False)
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )
        raw_lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
        token_lengths = [len(ids) for ids in encoded["input_ids"]]
        encoded["text"] = texts
        encoded["labels"] = [int(value) for value in batch["label"]]
        encoded["token_length"] = token_lengths
        encoded["raw_email_tokens"] = raw_lengths
        encoded["was_trimmed"] = [raw > tokenized for raw, tokenized in zip(raw_lengths, token_lengths, strict=False)]
        return encoded

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        desc=f"Tokenizing emails to <= {config.max_seq_length} tokens",
    )
    stats = token_stats(tokenized)

    ham, spam = label_counts(raw_dataset)
    metadata = {
        "path": data_path,
        "sha256": dataset_sha,
        "rows": len(raw_dataset),
        "spam": spam,
        "ham": ham,
        "sources": raw_summary["source_counts"],
        "avg_subject_chars": raw_summary["text_stats"]["avg_subject_chars"],
        "avg_body_chars": raw_summary["text_stats"]["avg_body_chars"],
        "train_rows": len(tokenized["train"]),
        "validation_rows": len(tokenized["validation"]),
        "test_rows": len(tokenized["test"]),
        "token_stats": stats,
    }
    return tokenized, metadata, data_path, dataset_sha


def _prediction_array(raw_predictions: Any) -> Any:
    predictions = raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions
    return predictions


def compute_classification_metrics_from_arrays(logits: Any, labels: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels).astype(int)
    if logits_array.ndim == 1:
        logits_array = np.stack([-logits_array, logits_array], axis=-1)
    predictions_array = np.argmax(logits_array, axis=-1).astype(int)

    if len(labels_array) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "matthews_corrcoef": 0.0,
            "false_positive_count": 0.0,
            "false_negative_count": 0.0,
            "true_positive_count": 0.0,
            "true_negative_count": 0.0,
        }

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
    accuracy = float((predictions_array == labels_array).mean())
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (float(per_class_recall[1]) + specificity) / 2

    return {
        "accuracy": accuracy,
        "precision": float(per_class_precision[1]),
        "recall": float(per_class_recall[1]),
        "f1": float(per_class_f1[1]),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "matthews_corrcoef": float(matthews_corrcoef(labels_array, predictions_array)) if len(set(labels_array)) > 1 else 0.0,
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
    }


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    return compute_classification_metrics_from_arrays(_prediction_array(eval_pred.predictions), eval_pred.label_ids)


def evaluate_split(trainer: Any, dataset_split: Any) -> dict[str, float]:
    predictions = trainer.predict(dataset_split)
    return compute_classification_metrics_from_arrays(
        _prediction_array(predictions.predictions),
        predictions.label_ids,
    )


def non_cuda_batch_settings(config: Any) -> dict[str, int]:
    del config
    return {
        "train_batch_size": base.TRAIN_BATCH_SIZE,
        "eval_batch_size": base.EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": base.GRADIENT_ACCUMULATION_STEPS,
    }


def run_training_config(args: argparse.Namespace, config: Any, sweep_id: str, run_dir: Path) -> dict[str, Any]:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from aim_tracking import create_aim_callbacks
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        PrinterCallback,
        ProgressCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    device = base.find_device(torch)
    cuda = device == "cuda"
    if device == "mps":
        base.configure_mps(torch)
    if not cuda and not args.allow_non_cuda and not args.dry_run:
        raise RuntimeError(
            f"CUDA is required for the sweep. Detected {device.upper()}. "
            "Pass --allow-non-cuda only for dry runs or tiny smoke tests."
        )

    if cuda:
        train_batch_size = config.train_batch_size
        eval_batch_size = config.eval_batch_size
        gradient_accumulation_steps = config.gradient_accumulation_steps
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        target_optim = "adamw_torch_fused"
        model_dtype = torch.bfloat16
        bf16 = True
        gradient_checkpointing = False
        gradient_checkpointing_kwargs = None
        dataloader_pin_memory = True
        dataloader_num_workers = 4
        logging_steps = base.LOGGING_STEPS
    else:
        fallback = non_cuda_batch_settings(config)
        train_batch_size = fallback["train_batch_size"]
        eval_batch_size = fallback["eval_batch_size"]
        gradient_accumulation_steps = fallback["gradient_accumulation_steps"]
        target_optim = "adamw_torch"
        model_dtype = torch.float32
        bf16 = False
        gradient_checkpointing = True
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        dataloader_pin_memory = False
        dataloader_num_workers = 0
        logging_steps = 1

    effective_batch_size = train_batch_size * gradient_accumulation_steps
    print(f"Using device: {device.upper()}")
    if cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Run: {base.run_name(sweep_id, config)}")
    print(f"Config: {config.to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset, dataset_metadata, data_path, dataset_sha = prepare_datasets(
        tokenizer=tokenizer,
        config=config,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )

    if args.dry_run:
        return {
            "status": "dry_run",
            "sweep_id": sweep_id,
            "run_name": base.run_name(sweep_id, config),
            "evaluation_method": EVALUATION_METHOD,
            "config": config.to_dict(),
            "dataset": dataset_metadata,
            "run_dir": str(run_dir),
        }

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            dtype=model_dtype,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            torch_dtype=model_dtype,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    checkpoint_save_steps = base.CHECKPOINT_SAVE_STEPS if config.max_seq_length < 600 else base.CHECKPOINT_SAVE_STEPS * 2
    if args.max_steps is not None:
        checkpoint_save_steps = max(1, min(checkpoint_save_steps, int(args.max_steps)))
    eval_steps = checkpoint_save_steps

    output_dir = run_dir / "trainer_output"
    adapter_path = run_dir / "adapter"
    tensorboard_dir = run_dir / "tensorboard"
    current_run_name = base.run_name(sweep_id, config)

    training_args_kwargs: dict[str, Any] = {
        "report_to": ["tensorboard"],
        "run_name": current_run_name,
        "output_dir": str(output_dir),
        "logging_dir": str(tensorboard_dir),
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "disable_tqdm": True,
        "seed": SEED,
        "data_seed": SEED,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": base.NUM_TRAIN_EPOCHS,
        "learning_rate": config.learning_rate,
        "warmup_ratio": base.WARMUP_RATIO,
        "weight_decay": base.WEIGHT_DECAY,
        "max_grad_norm": base.MAX_GRAD_NORM,
        "lr_scheduler_type": "cosine",
        "optim": target_optim,
        "max_steps": int(args.max_steps) if args.max_steps is not None else -1,
        "bf16": bf16,
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": checkpoint_save_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_num_workers": dataloader_num_workers,
        "remove_unused_columns": True,
    }
    if gradient_checkpointing_kwargs is not None:
        training_args_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs
    training_args = TrainingArguments(**training_args_kwargs)

    run_config = {
        "sweep_id": sweep_id,
        "config_id": config.config_id,
        "config_index": config.index,
        "group": config.group,
        "evaluation_method": EVALUATION_METHOD,
        "model_id": MODEL_ID,
        "seed": SEED,
        "max_seq_length": config.max_seq_length,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": base.NUM_TRAIN_EPOCHS,
        "learning_rate": config.learning_rate,
        "warmup_ratio": base.WARMUP_RATIO,
        "weight_decay": base.WEIGHT_DECAY,
        "max_grad_norm": base.MAX_GRAD_NORM,
        "optim": training_args.optim,
        "device": device,
        "bf16": bool(training_args.bf16),
        "fp16": bool(training_args.fp16),
        "gradient_checkpointing": bool(training_args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_num_workers": dataloader_num_workers,
        "eval_steps": training_args.eval_steps,
        "save_steps": training_args.save_steps,
        "checkpoint_save_steps": checkpoint_save_steps,
        "logging_steps": training_args.logging_steps,
        "tensorboard_log_dir": training_args.logging_dir,
        "output_dir": training_args.output_dir,
        "adapter_path": str(adapter_path),
        "resume_checkpoint_enabled": bool(args.resume_checkpoint),
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
        "dataset_path": data_path,
        "dataset_sha256": dataset_sha,
    }
    lora_metadata = {
        "r": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": peft_config.target_modules,
        "task_type": "SEQ_CLS",
        "modules_to_save": ["score"],
    }

    aim_callback, notebook_aim_callback = create_aim_callbacks(
        repo_path=str(root),
        experiment_name=AIM_EXPERIMENT_NAME,
        system_tracking_interval=AIM_SYSTEM_TRACKING_INTERVAL,
        run_config=run_config,
        dataset_metadata=dataset_metadata,
        lora_metadata=lora_metadata,
    )
    aim_run = aim_callback.experiment
    aim_run.name = current_run_name
    base.safe_aim_set(aim_run, "sweep_id", sweep_id)
    base.safe_aim_set(aim_run, "config_id", config.config_id)
    base.safe_aim_set(aim_run, "config_index", config.index)
    base.safe_aim_set(aim_run, "group", config.group)
    base.safe_aim_set(aim_run, "evaluation_method", EVALUATION_METHOD)
    base.safe_aim_set(aim_run, "run_name", current_run_name)
    base.safe_aim_set(aim_run, "run_dir", str(run_dir))
    aim_run_hash = getattr(aim_run, "hash", "")
    base.safe_aim_set(aim_run, "status", "running")

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                base.build_console_progress_callback(
                    trainer_callback_cls=TrainerCallback,
                    every_steps=base.CONSOLE_PROGRESS_STEPS,
                ),
                aim_callback,
                notebook_aim_callback,
            ],
        )
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)

        resume_checkpoint = base.latest_checkpoint_path(output_dir) if args.resume_checkpoint else None
        if resume_checkpoint is not None:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            base.safe_aim_set(aim_callback.experiment, "resume_from_checkpoint", str(resume_checkpoint))

        trainer_stats = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )

        validation_metrics = evaluate_split(trainer, dataset["validation"])
        test_metrics = evaluate_split(trainer, dataset["test"])

        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        aim_run = aim_callback.experiment
        if getattr(aim_run, "name", None) != current_run_name:
            aim_run.name = current_run_name
        aim_run_hash = getattr(aim_run, "hash", aim_run_hash)
        base.safe_aim_set(aim_run, "status", "completed")
        base.safe_aim_set(aim_run, "final_saved_model_path", str(adapter_path))
        base.safe_aim_set(aim_run, "saved_tokenizer_path", str(adapter_path))
        base.safe_aim_set(aim_run, "final_validation_metrics", validation_metrics)
        base.safe_aim_set(aim_run, "final_test_metrics", test_metrics)
        for metric_name, metric_value in validation_metrics.items():
            if isinstance(metric_value, (int, float)):
                base.safe_aim_track(
                    aim_run,
                    metric_value,
                    name=metric_name,
                    step=trainer.state.global_step,
                    epoch=trainer.state.epoch,
                    context={"subset": "validation_final"},
                )
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                base.safe_aim_track(
                    aim_run,
                    metric_value,
                    name=metric_name,
                    step=trainer.state.global_step,
                    epoch=trainer.state.epoch,
                    context={"subset": "test"},
                )

        return {
            "status": "completed",
            "sweep_id": sweep_id,
            "run_name": current_run_name,
            "evaluation_method": EVALUATION_METHOD,
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "adapter_path": str(adapter_path),
            "aim_run_hash": aim_run_hash,
            "dataset": dataset_metadata,
            "run_config": run_config,
            "lora": lora_metadata,
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "training_metrics": trainer_stats.metrics,
            "trainer_state": {
                "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch,
                "best_metric": trainer.state.best_metric,
                "best_global_step": trainer.state.best_global_step,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }
    except BaseException as exc:
        try:
            aim_run = aim_callback.experiment
        except Exception:
            pass
        base.safe_aim_set(aim_run, "status", "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        base.safe_aim_set(aim_run, "failure_type", type(exc).__name__)
        base.safe_aim_set(aim_run, "failure_message", str(exc))
        raise
    finally:
        base.close_aim_run(aim_run)


def cleanup_after_run() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def run_one(args: argparse.Namespace) -> int:
    if args.sweep_id is None:
        raise SystemExit("--sweep-id is required for run-one")
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
            "config": config.to_dict(),
            "args": vars(args),
        },
    )

    try:
        metrics = run_training_config(args, config, args.sweep_id, run_dir)
        base.write_json(run_dir / "metrics.json", metrics)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "sweep_id": args.sweep_id,
            "run_name": base.run_name(args.sweep_id, config),
            "evaluation_method": EVALUATION_METHOD,
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        base.write_json(run_dir / "metrics.json", failure)
        print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(failure["traceback"], file=sys.stderr)
        return 1
    finally:
        cleanup_after_run()


def run_sweep(args: argparse.Namespace) -> int:
    patch_base_configs()
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


def summarize_command(args: argparse.Namespace) -> int:
    patch_base_configs()
    results_root = resolve_results_root(args.results_root)
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir).resolve()
    elif args.sweep_id:
        sweep_dir = results_root / "sweeps" / args.sweep_id
    else:
        raise SystemExit("Pass --sweep-id or --sweep-dir.")

    selected_configs = build_sweep_configs()
    summary_path = write_summary(sweep_dir, selected_configs)
    print(f"Wrote summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3 sequence classification LoRA sweeps.")
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

    summarize_parser = subparsers.add_parser("summarize", help="Rebuild summary.csv for a sweep.")
    summarize_parser.add_argument("--sweep-id", default=None)
    summarize_parser.add_argument("--sweep-dir", default=None)
    summarize_parser.add_argument("--results-root", default=None)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    base.validate_args(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patch_base_configs()
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
