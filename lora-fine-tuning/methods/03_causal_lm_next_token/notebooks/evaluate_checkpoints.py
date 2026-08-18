#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
METHOD_ALIASES = {
    "01": "01_sequence_classification",
    "1": "01_sequence_classification",
    "sequence_classification": "01_sequence_classification",
    "01_sequence_classification": "01_sequence_classification",
    "02": "02_causal_lm_generation_parsing",
    "2": "02_causal_lm_generation_parsing",
    "generation_parsing": "02_causal_lm_generation_parsing",
    "02_causal_lm_generation_parsing": "02_causal_lm_generation_parsing",
    "03": "03_causal_lm_next_token",
    "3": "03_causal_lm_next_token",
    "next_token": "03_causal_lm_next_token",
    "03_causal_lm_next_token": "03_causal_lm_next_token",
    "04": "04_causal_lm_structured_generation",
    "4": "04_causal_lm_structured_generation",
    "structured_generation": "04_causal_lm_structured_generation",
    "04_causal_lm_structured_generation": "04_causal_lm_structured_generation",
}
SUMMARY_COLUMNS = [
    "method",
    "evaluation_method",
    "dataset",
    "dataset_role",
    "dataset_source",
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
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "checkpoint_path",
    "runtime_seconds",
    "status",
    "error_message",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class CheckpointInfo:
    method: str
    evaluation_method: str
    config_index: int
    config_id: str
    checkpoint_type: str
    checkpoint_step: str
    checkpoint_path: str
    run_dir: str
    max_seq_length: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float


@dataclass(frozen=True)
class MethodSpec:
    name: str
    evaluation_method: str
    evaluation_kind: str
    script_path: Path
    results_root: Path
    output_root: Path


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


def methods_root() -> Path:
    return project_root() / "lora-fine-tuning" / "methods"


def normalize_method(value: str) -> str:
    key = value.strip()
    try:
        return METHOD_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(METHOD_ALIASES))
        raise SystemExit(f"Unsupported --method {value!r}. Choices/aliases: {choices}") from exc


def method_spec(method_value: str) -> MethodSpec:
    method = normalize_method(method_value)
    root = methods_root()
    if method == "01_sequence_classification":
        script_path = root / method / "qwen3_0.6b_sequence_classification_sweep.py"
        return MethodSpec(
            name=method,
            evaluation_method="sequence_classification",
            evaluation_kind="sequence_classification",
            script_path=script_path,
            results_root=root / method / "results",
            output_root=root / method,
        )
    if method == "02_causal_lm_generation_parsing":
        script_path = root / method / "qwen3_0.6b_generation_parsing_sweep.py"
        return MethodSpec(
            name=method,
            evaluation_method="causal_lm_generation_parsing",
            evaluation_kind="generation_parsing",
            script_path=script_path,
            results_root=root / method / "results",
            output_root=root / method,
        )
    if method == "03_causal_lm_next_token":
        script_path = root / method / "notebooks" / "qwen3_0.6b_casual_lm_sweep.py"
        return MethodSpec(
            name=method,
            evaluation_method="causal_lm_next_token",
            evaluation_kind="next_token",
            script_path=script_path,
            results_root=root / method / "notebooks" / "results",
            output_root=root / method / "notebooks",
        )
    if method == "04_causal_lm_structured_generation":
        script_path = root / method / "qwen3_0.6b_structured_generation_sweep.py"
        return MethodSpec(
            name=method,
            evaluation_method="causal_lm_structured_json_generation_parsing",
            evaluation_kind="generation_parsing",
            script_path=script_path,
            results_root=root / method / "results",
            output_root=root / method,
        )
    raise AssertionError(f"Unhandled method: {method}")


def base_sweep_path() -> Path:
    script_dir = method_dir()
    root = project_root()
    candidates = [
        script_dir / "qwen3_0.6b_casual_lm_sweep.py",
        script_dir / "notebooks" / "qwen3_0.6b_casual_lm_sweep.py",
        script_dir.parent / "notebooks" / "qwen3_0.6b_casual_lm_sweep.py",
        root / "lora-fine-tuning" / "methods" / "03_causal_lm_next_token" / "notebooks" / "qwen3_0.6b_casual_lm_sweep.py",
        root / "lora-fine-tuning" / "qwen3_0.6b_casual_lm_sweep.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find qwen3_0.6b_casual_lm_sweep.py. Tried:\n{attempted}")


def default_results_root() -> Path:
    return method_spec("03").results_root


def load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module() -> Any:
    return load_module(
        "qwen3_next_token_sweep_helpers",
        base_sweep_path(),
    )


def load_method_module(spec: MethodSpec, max_new_tokens: int | None, parse_failure_label: str) -> Any:
    if not spec.script_path.exists():
        raise FileNotFoundError(f"Could not find sweep script for {spec.name}: {spec.script_path}")
    module = load_module(f"checkpoint_eval_{spec.name}", spec.script_path)
    if spec.name == "02_causal_lm_generation_parsing":
        module._GENERATION_MAX_NEW_TOKENS = int(max_new_tokens or module.DEFAULT_MAX_NEW_TOKENS)
        module._PARSE_FAILURE_LABEL = parse_failure_label
    elif spec.name == "04_causal_lm_structured_generation":
        module.apply_structured_patches()
        module.gen._GENERATION_MAX_NEW_TOKENS = int(max_new_tokens or module.DEFAULT_MAX_NEW_TOKENS)
        module.gen._PARSE_FAILURE_LABEL = parse_failure_label
    return module


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 LoRA checkpoints from methods 01/02/03/04 on train and external datasets."
    )
    parser.add_argument(
        "--method",
        default="03",
        choices=sorted(METHOD_ALIASES),
        help="Training method to evaluate. Aliases: 01/02/03/04, sequence_classification, generation_parsing, next_token, structured_generation.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Directory searched recursively for adapter/ and trainer_output/checkpoint-* directories. Defaults to selected method results/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary.csv and metrics.json. Defaults to checkpoint_eval_<timestamp> under this method.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to evaluate separately.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Use frozen Parquet splits from a manifest created by dataset/prepare_evaluation_splits.py.",
    )
    parser.add_argument(
        "--split-roles",
        nargs="+",
        choices=["external_validation", "final_test"],
        default=["final_test"],
        help="Manifest split roles to evaluate. Used only with --dataset-manifest.",
    )
    parser.add_argument("--sample-limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Smallest batch size to retry after CUDA OOM.",
    )
    parser.add_argument(
        "--checkpoints-per-run",
        type=int,
        default=10,
        help="Evaluate at most N checkpoints/adapters per training run. Uses first, last, and evenly spaced middle points. Use 0 for all.",
    )
    parser.add_argument(
        "--config-indices",
        nargs="+",
        type=int,
        default=None,
        help="Only evaluate checkpoints whose config_index is in this list, e.g. --config-indices 1 2 3 5.",
    )
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation-only methods: max tokens generated for label parsing. Defaults to method default.",
    )
    parser.add_argument(
        "--parse-failure-label",
        choices=["ham", "spam"],
        default="ham",
        help="Generation-only methods: label assigned when generated output cannot be parsed.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for torch/BLAS-style work.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress logs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print batch progress every N batches per dataset when not quiet. Use 0 to disable batch logs.",
    )
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def resolve_results_root(value: str | None, spec: MethodSpec) -> Path:
    if value is None:
        return spec.results_root
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def resolve_output_dir(value: str | None, spec: MethodSpec) -> Path:
    if value is None:
        return spec.output_root / f"checkpoint_eval_{timestamp()}"
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def config_from_run_dir(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for run: {run_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError(f"Unsupported config.json shape: {config_path}")
    return config


def checkpoint_info_from_path(path: Path, spec: MethodSpec) -> CheckpointInfo | None:
    if path.name == "adapter":
        checkpoint_type = "final_adapter"
        checkpoint_step = "final"
        run_dir = path.parent
    elif path.name.startswith("checkpoint-") and path.parent.name == "trainer_output":
        checkpoint_type = "trainer_checkpoint"
        checkpoint_step = path.name.removeprefix("checkpoint-")
        run_dir = path.parent.parent
    else:
        return None

    config = config_from_run_dir(run_dir)
    config_index = int(config["index"])
    config_id = str(config.get("config_id") or run_dir.name)
    return CheckpointInfo(
        method=spec.name,
        evaluation_method=spec.evaluation_method,
        config_index=config_index,
        config_id=config_id,
        checkpoint_type=checkpoint_type,
        checkpoint_step=checkpoint_step,
        checkpoint_path=str(path.resolve()),
        run_dir=str(run_dir.resolve()),
        max_seq_length=int(config["max_seq_length"]),
        learning_rate=float(config["learning_rate"]),
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
    )


def checkpoint_sort_key(info: CheckpointInfo) -> tuple[int, int, float, str]:
    type_rank = 1 if info.checkpoint_type == "final_adapter" else 0
    if info.checkpoint_step == "final":
        step_value = math.inf
    else:
        step_value = float(info.checkpoint_step)
    return (info.config_index, type_rank, step_value, info.checkpoint_path)


def checkpoint_run_sort_key(info: CheckpointInfo) -> tuple[int, str, int, float, str]:
    base_key = checkpoint_sort_key(info)
    return (info.config_index, Path(info.run_dir).parent.name, base_key[1], base_key[2], info.checkpoint_path)


def discover_checkpoints(results_root: Path, spec: MethodSpec) -> list[CheckpointInfo]:
    seen: set[str] = set()
    checkpoints: list[CheckpointInfo] = []
    for adapter_config in sorted(results_root.rglob("adapter_config.json")):
        checkpoint_dir = adapter_config.parent
        resolved = str(checkpoint_dir.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        info = checkpoint_info_from_path(checkpoint_dir, spec)
        if info is not None:
            checkpoints.append(info)
    checkpoints.sort(key=checkpoint_sort_key)
    return checkpoints


def select_evenly_spaced(items: list[CheckpointInfo], limit: int) -> list[CheckpointInfo]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    selected_indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in selected_indices]


def limit_checkpoints_per_run(checkpoints: list[CheckpointInfo], limit: int) -> tuple[list[CheckpointInfo], dict[str, Any]]:
    if limit <= 0:
        return checkpoints, {
            "strategy": "all",
            "checkpoints_per_run": limit,
            "original_checkpoint_count": len(checkpoints),
            "selected_checkpoint_count": len(checkpoints),
            "runs": [],
        }

    grouped: dict[str, list[CheckpointInfo]] = {}
    for checkpoint in checkpoints:
        grouped.setdefault(checkpoint.run_dir, []).append(checkpoint)

    selected: list[CheckpointInfo] = []
    run_summaries: list[dict[str, Any]] = []
    for run_dir in sorted(grouped, key=lambda value: (grouped[value][0].config_index, Path(value).parent.name, value)):
        run_checkpoints = sorted(grouped[run_dir], key=checkpoint_sort_key)
        run_selected = select_evenly_spaced(run_checkpoints, limit)
        selected.extend(run_selected)
        run_summaries.append(
            {
                "run_dir": run_dir,
                "config_index": run_checkpoints[0].config_index,
                "config_id": run_checkpoints[0].config_id,
                "available_checkpoint_count": len(run_checkpoints),
                "selected_checkpoint_count": len(run_selected),
                "selected_checkpoint_steps": [checkpoint.checkpoint_step for checkpoint in run_selected],
            }
        )

    selected.sort(key=checkpoint_run_sort_key)
    return selected, {
        "strategy": "evenly_spaced_per_run",
        "checkpoints_per_run": limit,
        "original_checkpoint_count": len(checkpoints),
        "selected_checkpoint_count": len(selected),
        "runs": run_summaries,
    }


def label_counts(dataset: Any) -> tuple[int, int]:
    labels = [int(value) for value in dataset["label"]]
    spam = sum(1 for value in labels if value == 1)
    ham = sum(1 for value in labels if value == 0)
    return ham, spam


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


def build_dataset_sample(dataset_name: str, sample_limit: int, seed: int, output_path: Path) -> dict[str, Any]:
    base = load_base_module()
    from datasets import ClassLabel, disable_progress_bars, load_dataset
    from dataset.combine import combine_datasets

    disable_progress_bars()

    if dataset_name == "train_subset":
        data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
        raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["ham", "spam"]))
        holdout = raw_dataset.train_test_split(
            test_size=base.HOLDOUT_SPLIT,
            stratify_by_column="label",
            seed=seed,
        )
        dataset = holdout["train"]
        dataset = stratified_sample(dataset, sample_limit, seed)
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        dataset = dataset.filter(lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])))
        if dataset_name == "spam_ham":
            dataset = stratified_sample(dataset, sample_limit, seed)
        else:
            dataset = plain_sample(dataset, sample_limit, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    ham, spam = label_counts(dataset)
    return {
        "dataset": dataset_name,
        "path": str(output_path),
        "source_path": str(data_path),
        "rows": len(dataset),
        "ham_count": ham,
        "spam_count": spam,
    }


def prepare_dataset_samples(dataset_names: list[str], sample_limit: int, seed: int, output_dir: Path) -> dict[str, dict[str, Any]]:
    samples_dir = output_dir / "dataset_samples"
    metadata: dict[str, dict[str, Any]] = {}
    for dataset_name in dataset_names:
        sample_path = samples_dir / f"{dataset_name}.parquet"
        print(f"Preparing dataset sample: {dataset_name} -> {sample_path}")
        metadata[dataset_name] = build_dataset_sample(dataset_name, sample_limit, seed, sample_path)
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_dataset_samples(
    manifest_value: str,
    split_roles: list[str],
) -> tuple[dict[str, dict[str, Any]], Path, str]:
    from datasets import load_dataset

    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (project_root() / manifest_path).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Dataset manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit(
            f"Unsupported dataset manifest schema: {manifest.get('schema_version')!r}"
        )

    selected_roles = set(split_roles)
    samples: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("splits", []):
        role = str(entry.get("role", ""))
        if role not in selected_roles:
            continue

        name = str(entry.get("name", "")).strip()
        if not name or name in samples:
            raise SystemExit(f"Invalid or duplicate split name in manifest: {name!r}")

        split_path = Path(str(entry.get("path", "")))
        if not split_path.is_absolute():
            split_path = (manifest_path.parent / split_path).resolve()
        if not split_path.is_file():
            raise SystemExit(f"Manifest split does not exist: {split_path}")

        expected_sha256 = str(entry.get("sha256", ""))
        actual_sha256 = sha256_file(split_path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise SystemExit(
                f"SHA-256 mismatch for {name}: expected {expected_sha256}, got {actual_sha256}"
            )

        dataset = load_dataset("parquet", data_files=str(split_path), split="train")
        ham, spam = label_counts(dataset)
        expected_rows = int(entry.get("rows", len(dataset)))
        expected_ham = int(entry.get("ham_count", ham))
        expected_spam = int(entry.get("spam_count", spam))
        if (len(dataset), ham, spam) != (expected_rows, expected_ham, expected_spam):
            raise SystemExit(
                f"Manifest counts do not match {name}: "
                f"expected rows/ham/spam={expected_rows}/{expected_ham}/{expected_spam}, "
                f"got {len(dataset)}/{ham}/{spam}"
            )

        samples[name] = {
            "dataset": name,
            "role": role,
            "source": str(entry.get("source", "")),
            "path": str(split_path),
            "rows": len(dataset),
            "ham_count": ham,
            "spam_count": spam,
            "sha256": actual_sha256,
            "content_sha256": entry.get("content_sha256", ""),
        }

    if not samples:
        raise SystemExit(
            f"Manifest contains no splits for roles: {', '.join(sorted(selected_roles))}"
        )
    return samples, manifest_path, sha256_file(manifest_path)


def pretokenize_dataset_samples(
    *,
    dataset_samples: dict[str, dict[str, Any]],
    max_seq_lengths: list[int],
    output_dir: Path,
    spec: MethodSpec,
) -> dict[str, dict[str, Any]]:
    base = load_base_module()
    from datasets import disable_progress_bars, load_dataset
    from transformers import AutoTokenizer

    disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenized_root = output_dir / "tokenized_samples"
    for dataset_name, metadata in dataset_samples.items():
        tokenized_paths: dict[str, str] = {}
        for max_seq_length in sorted(set(max_seq_lengths)):
            cache_dir = tokenized_root / f"{dataset_name}_seq{max_seq_length}"
            if cache_dir.exists():
                tokenized_paths[str(max_seq_length)] = str(cache_dir)
                continue
            print(f"Tokenizing dataset sample: {dataset_name}, max_seq_length={max_seq_length} -> {cache_dir}")
            dataset = load_dataset("parquet", data_files=str(metadata["path"]), split="train")
            dataset = dataset.filter(lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])))
            if spec.evaluation_kind == "sequence_classification":
                def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
                    texts = [
                        base.build_email_text(subject, body)
                        for subject, body in zip(batch["subject"], batch["body"], strict=False)
                    ]
                    encoded = tokenizer(
                        texts,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=False,
                    )
                    encoded["labels"] = [int(value) for value in batch["label"]]
                    encoded["token_length"] = [len(ids) for ids in encoded["input_ids"]]
                    return encoded

                tokenized = dataset.map(
                    tokenize_batch,
                    batched=True,
                    desc=f"Tokenizing {dataset_name} emails to <= {max_seq_length} tokens",
                )
            else:
                tokenized = dataset.map(
                    lambda sample: base.build_tokenized_sample(tokenizer, sample, max_seq_length),
                    desc=f"Formatting {dataset_name} prompts to <= {max_seq_length} tokens",
                )
            if "prompt_token_length" in tokenized.column_names:
                tokenized = tokenized.sort("prompt_token_length")
            elif "token_length" in tokenized.column_names:
                tokenized = tokenized.sort("token_length")
            tokenized.save_to_disk(str(cache_dir))
            tokenized_paths[str(max_seq_length)] = str(cache_dir)
        metadata["tokenized_paths"] = tokenized_paths
    return dataset_samples


def pad_prompt_batch(torch_module: Any, prompt_batches: list[list[int]], pad_token_id: int, device: Any) -> dict[str, Any]:
    max_length = max(len(ids) for ids in prompt_batches)
    batch_size = len(prompt_batches)
    input_ids = torch_module.full((batch_size, max_length), pad_token_id, dtype=torch_module.long, device=device)
    attention_mask = torch_module.zeros((batch_size, max_length), dtype=torch_module.long, device=device)
    for row, prompt_ids in enumerate(prompt_batches):
        values = torch_module.tensor(prompt_ids, dtype=torch_module.long, device=device)
        start = max_length - len(prompt_ids)
        input_ids[row, start:] = values
        attention_mask[row, start:] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def forward_last_token_logits(model: Any, batch: dict[str, Any], use_logits_to_keep: bool | None) -> tuple[Any, bool]:
    if use_logits_to_keep is not False:
        try:
            outputs = model(**batch, use_cache=False, logits_to_keep=1)
            return outputs.logits[:, -1, :], True
        except TypeError:
            if use_logits_to_keep is True:
                raise

    outputs = model(**batch, use_cache=False)
    return outputs.logits[:, -1, :], False


def evaluate_with_predictions(
    *,
    model: Any,
    torch_module: Any,
    dataset_split: Any,
    label_token_ids: dict[str, int],
    pad_token_id: int,
    batch_size: int,
    write_predictions_path: Path | None,
    log_prefix: str,
    quiet: bool,
    progress_every: int,
) -> dict[str, Any]:
    base = load_base_module()

    model.eval()
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    failure_count = 0
    prediction_rows: list[dict[str, Any]] = []
    label_ids = torch_module.tensor(
        [label_token_ids[base.NEGATIVE_LABEL_TEXT], label_token_ids[base.POSITIVE_LABEL_TEXT]],
        dtype=torch_module.long,
        device=device,
    )
    use_logits_to_keep: bool | None = None

    total_batches = math.ceil(len(dataset_split) / max(batch_size, 1))
    with torch_module.inference_mode():
        for start in range(0, len(dataset_split), batch_size):
            batch_index = start // batch_size + 1
            if not quiet and progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches):
                log(f"{log_prefix} batch {batch_index}/{total_batches} rows {start}-{min(start + batch_size, len(dataset_split))}")
            rows = dataset_split.select(range(start, min(start + batch_size, len(dataset_split))))
            prompt_input_ids = rows["prompt_input_ids"]
            batch = pad_prompt_batch(torch_module, prompt_input_ids, pad_token_id, device)
            next_token_logits, logits_to_keep_was_used = forward_last_token_logits(model, batch, use_logits_to_keep)
            if use_logits_to_keep is None:
                use_logits_to_keep = logits_to_keep_was_used
                if not quiet:
                    if logits_to_keep_was_used:
                        log(f"{log_prefix} using logits_to_keep=1 to reduce GPU memory")
                    else:
                        log(f"{log_prefix} logits_to_keep unsupported; falling back to full-sequence logits")
            label_logits = next_token_logits.index_select(dim=-1, index=label_ids)
            finite = torch_module.isfinite(label_logits).all(dim=-1)
            safe_logits = torch_module.nan_to_num(label_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            batch_probabilities = torch_module.softmax(safe_logits, dim=-1)
            batch_predictions = batch_probabilities.argmax(dim=-1)

            batch_probabilities_list = batch_probabilities.detach().cpu().float().tolist()
            batch_predictions_list = batch_predictions.detach().cpu().int().tolist()
            batch_labels = [int(value) for value in rows["label"]]

            failure_count += int((~finite).sum().item())
            probabilities.extend(batch_probabilities_list)
            predictions.extend(batch_predictions_list)
            labels.extend(batch_labels)

            if write_predictions_path is not None:
                for offset, label in enumerate(batch_labels):
                    prediction = int(batch_predictions_list[offset])
                    probability = batch_probabilities_list[offset]
                    prediction_rows.append(
                        {
                            "row_index": start + offset,
                            "source": rows["source"][offset] if "source" in rows.column_names else "",
                            "label": label,
                            "prediction": prediction,
                            "label_text": base.label_to_text(label),
                            "prediction_text": base.label_to_text(prediction),
                            "ham_probability": probability[0],
                            "spam_probability": probability[1],
                            "subject": rows["subject"][offset] if "subject" in rows.column_names else "",
                        }
                    )
            del batch, next_token_logits, label_logits, safe_logits, batch_probabilities, batch_predictions

    metrics = base.compute_metrics(predictions, labels, probabilities, failure_count)
    spam_predictions = float(sum(1 for value in predictions if int(value) == 1))
    tp = float(metrics.get("true_positive_count", 0.0))
    fp = float(metrics.get("false_positive_count", 0.0))
    fn = float(metrics.get("false_negative_count", 0.0))
    tn = float(metrics.get("true_negative_count", 0.0))
    metrics["false_positive_rate"] = fp / max(fp + tn, 1.0)
    metrics["false_negative_rate"] = fn / max(fn + tp, 1.0)
    metrics["spam_prediction_rate"] = spam_predictions / max(len(predictions), 1)

    if write_predictions_path is not None:
        write_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with write_predictions_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "row_index",
                "source",
                "label",
                "prediction",
                "label_text",
                "prediction_text",
                "ham_probability",
                "spam_probability",
                "subject",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)
        metrics["predictions_path"] = str(write_predictions_path)

    return metrics


def add_common_error_rates(metrics: dict[str, Any], predictions: list[int]) -> dict[str, Any]:
    spam_predictions = float(sum(1 for value in predictions if int(value) == 1))
    tp = float(metrics.get("true_positive_count", 0.0))
    fp = float(metrics.get("false_positive_count", 0.0))
    fn = float(metrics.get("false_negative_count", 0.0))
    tn = float(metrics.get("true_negative_count", 0.0))
    metrics["false_positive_rate"] = fp / max(fp + tn, 1.0)
    metrics["false_negative_rate"] = fn / max(fn + tp, 1.0)
    metrics["spam_prediction_rate"] = spam_predictions / max(len(predictions), 1)
    return metrics


def write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "source",
        "label",
        "prediction",
        "label_text",
        "prediction_text",
        "ham_probability",
        "spam_probability",
        "generation",
        "subject",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def evaluate_sequence_classification_with_predictions(
    *,
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    dataset_split: Any,
    batch_size: int,
    write_predictions_path: Path | None,
    log_prefix: str,
    quiet: bool,
    progress_every: int,
) -> dict[str, Any]:
    base = load_base_module()
    model.eval()
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(dataset_split) / max(batch_size, 1))

    with torch_module.inference_mode():
        for start in range(0, len(dataset_split), batch_size):
            batch_index = start // batch_size + 1
            if not quiet and progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches):
                log(f"{log_prefix} batch {batch_index}/{total_batches} rows {start}-{min(start + batch_size, len(dataset_split))}")
            rows = dataset_split.select(range(start, min(start + batch_size, len(dataset_split))))
            encoded = tokenizer.pad(
                {"input_ids": rows["input_ids"], "attention_mask": rows["attention_mask"]},
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**batch).logits
            safe_logits = torch_module.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            batch_probabilities = torch_module.softmax(safe_logits, dim=-1)
            batch_predictions = batch_probabilities.argmax(dim=-1)

            batch_probabilities_list = batch_probabilities.detach().cpu().float().tolist()
            batch_predictions_list = batch_predictions.detach().cpu().int().tolist()
            batch_labels = [int(value) for value in rows["label"]]
            probabilities.extend(batch_probabilities_list)
            predictions.extend(batch_predictions_list)
            labels.extend(batch_labels)

            if write_predictions_path is not None:
                for offset, label in enumerate(batch_labels):
                    prediction = int(batch_predictions_list[offset])
                    probability = batch_probabilities_list[offset]
                    prediction_rows.append(
                        {
                            "row_index": start + offset,
                            "source": rows["source"][offset] if "source" in rows.column_names else "",
                            "label": label,
                            "prediction": prediction,
                            "label_text": base.label_to_text(label),
                            "prediction_text": base.label_to_text(prediction),
                            "ham_probability": probability[0],
                            "spam_probability": probability[1],
                            "subject": rows["subject"][offset] if "subject" in rows.column_names else "",
                        }
                    )
            del batch, logits, safe_logits, batch_probabilities, batch_predictions

    metrics = base.compute_metrics(predictions, labels, probabilities, 0)
    metrics = add_common_error_rates(metrics, predictions)
    if write_predictions_path is not None:
        write_prediction_rows(write_predictions_path, prediction_rows)
        metrics["predictions_path"] = str(write_predictions_path)
    return metrics


def generation_label_to_id(label_text: str | None, parse_failure_label: str) -> int:
    base = load_base_module()
    if label_text == base.POSITIVE_LABEL_TEXT:
        return 1
    if label_text == base.NEGATIVE_LABEL_TEXT:
        return 0
    return 1 if parse_failure_label == base.POSITIVE_LABEL_TEXT else 0


def evaluate_generation_with_predictions(
    *,
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    dataset_split: Any,
    parse_generated_label: Any,
    compute_generation_metrics: Any,
    max_new_tokens: int,
    parse_failure_label: str,
    pad_token_id: int,
    batch_size: int,
    write_predictions_path: Path | None,
    log_prefix: str,
    quiet: bool,
    progress_every: int,
) -> dict[str, Any]:
    base = load_base_module()
    model.eval()
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    parse_failure_count = 0
    prediction_rows: list[dict[str, Any]] = []

    im_end_id = tokenizer.convert_tokens_to_ids(base.IM_END_TOKEN)
    stop_token_ids = [im_end_id] if im_end_id is not None and im_end_id >= 0 else []
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    if not stop_token_ids:
        stop_token_ids = None

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True

    total_batches = math.ceil(len(dataset_split) / max(batch_size, 1))
    try:
        with torch_module.inference_mode():
            for start in range(0, len(dataset_split), batch_size):
                batch_index = start // batch_size + 1
                if not quiet and progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches):
                    log(f"{log_prefix} batch {batch_index}/{total_batches} rows {start}-{min(start + batch_size, len(dataset_split))}")
                rows = dataset_split.select(range(start, min(start + batch_size, len(dataset_split))))
                prompt_input_ids = rows["prompt_input_ids"]
                batch = pad_prompt_batch(torch_module, prompt_input_ids, pad_token_id, device)
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
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
                batch_labels = [int(value) for value in rows["label"]]
                for offset, (raw_generation, actual_label) in enumerate(zip(decoded, batch_labels, strict=False)):
                    parsed_label = parse_generated_label(raw_generation)
                    if parsed_label is None:
                        parse_failure_count += 1
                    prediction = generation_label_to_id(parsed_label, parse_failure_label)
                    predictions.append(prediction)
                    labels.append(int(actual_label))
                    if write_predictions_path is not None:
                        prediction_rows.append(
                            {
                                "row_index": start + offset,
                                "source": rows["source"][offset] if "source" in rows.column_names else "",
                                "label": int(actual_label),
                                "prediction": prediction,
                                "label_text": base.label_to_text(int(actual_label)),
                                "prediction_text": base.label_to_text(prediction),
                                "generation": raw_generation,
                                "subject": rows["subject"][offset] if "subject" in rows.column_names else "",
                            }
                        )
                del batch, outputs, generated
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    metrics = compute_generation_metrics(predictions, labels, parse_failure_count)
    metrics = add_common_error_rates(metrics, predictions)
    if write_predictions_path is not None:
        write_prediction_rows(write_predictions_path, prediction_rows)
        metrics["predictions_path"] = str(write_predictions_path)
    return metrics


def is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and exc.__class__.__name__ in {
        "OutOfMemoryError",
        "RuntimeError",
    }


def evaluate_with_oom_retries(
    *,
    evaluator: Any,
    batch_size: int,
    min_batch_size: int,
    log_prefix: str,
    quiet: bool,
) -> tuple[dict[str, Any], int]:
    current_batch_size = max(1, int(batch_size))
    min_batch_size = max(1, int(min_batch_size))
    while True:
        try:
            metrics = evaluator(current_batch_size, f"{log_prefix} batch_size={current_batch_size}")
            metrics["eval_batch_size"] = float(current_batch_size)
            return metrics, current_batch_size
        except Exception as exc:
            if not is_cuda_oom(exc) or current_batch_size <= min_batch_size:
                raise
            next_batch_size = max(min_batch_size, current_batch_size // 2)
            if next_batch_size >= current_batch_size:
                raise
            if not quiet:
                log(
                    f"{log_prefix} CUDA OOM at batch_size={current_batch_size}; "
                    f"clearing cache and retrying with batch_size={next_batch_size}"
                )
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
            current_batch_size = next_batch_size


def tokenize_dataset(dataset_path: str, tokenizer: Any, max_seq_length: int) -> Any:
    base = load_base_module()
    from datasets import disable_progress_bars, load_dataset

    disable_progress_bars()
    dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
    dataset = dataset.filter(lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])))
    return dataset.map(
        lambda sample: base.build_tokenized_sample(tokenizer, sample, max_seq_length),
        desc=f"Formatting prompts to <= {max_seq_length} tokens",
    )


def load_tokenized_dataset(dataset_metadata: dict[str, Any], max_seq_length: int, tokenizer: Any) -> Any:
    from datasets import load_from_disk

    tokenized_paths = dataset_metadata.get("tokenized_paths", {})
    tokenized_path = tokenized_paths.get(str(max_seq_length))
    if tokenized_path:
        return load_from_disk(str(tokenized_path))
    return tokenize_dataset(dataset_metadata["path"], tokenizer, max_seq_length)


def result_row(
    *,
    dataset_name: str,
    checkpoint: CheckpointInfo,
    dataset_metadata: dict[str, Any],
    metrics: dict[str, Any],
    runtime_seconds: float,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "method": checkpoint.method,
            "evaluation_method": checkpoint.evaluation_method,
            "dataset": dataset_name,
            "dataset_role": dataset_metadata.get("role", "legacy_sample"),
            "dataset_source": dataset_metadata.get("source", dataset_name),
            "config_index": checkpoint.config_index,
            "config_id": checkpoint.config_id,
            "checkpoint_type": checkpoint.checkpoint_type,
            "checkpoint_step": checkpoint.checkpoint_step,
            "rows": dataset_metadata.get("rows", ""),
            "ham_count": dataset_metadata.get("ham_count", ""),
            "spam_count": dataset_metadata.get("spam_count", ""),
            "max_seq_length": checkpoint.max_seq_length,
            "learning_rate": checkpoint.learning_rate,
            "lora_r": checkpoint.lora_r,
            "lora_alpha": checkpoint.lora_alpha,
            "lora_dropout": checkpoint.lora_dropout,
            "checkpoint_path": checkpoint.checkpoint_path,
            "runtime_seconds": runtime_seconds,
            "status": status,
            "error_message": error_message,
        }
    )
    for key in [
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
    ]:
        row[key] = metrics.get(key, "")
    return row


def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    dataset_samples: dict[str, dict[str, Any]],
    dataset_names: list[str],
    output_dir: str,
    spec: MethodSpec,
    method_module: Any,
    max_new_tokens: int | None,
    parse_failure_label: str,
    batch_size: int,
    min_batch_size: int,
    write_predictions: bool,
    torch_threads: int,
    quiet: bool,
    progress_every: int,
) -> list[dict[str, Any]]:
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch_threads))
    project_root()
    log_prefix = f"[config={checkpoint.config_index:02d} step={checkpoint.checkpoint_step}]"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        if not quiet:
            log(f"{log_prefix} start path={checkpoint.checkpoint_path}")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        stage_started = time.perf_counter()
        base = load_base_module()
        torch.set_num_threads(max(1, int(torch_threads)))
        try:
            torch.set_num_interop_threads(max(1, int(torch_threads)))
        except RuntimeError:
            pass
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            torch.cuda.set_device(0)
        model_dtype = torch.bfloat16 if cuda else torch.float32
        if not quiet:
            device_name = torch.cuda.get_device_name(0) if cuda else "cpu"
            log(f"{log_prefix} imports/base ready in {time.perf_counter() - stage_started:.1f}s device={device_name}")

        stage_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" if spec.evaluation_kind == "sequence_classification" else "left"
        if not quiet:
            log(f"{log_prefix} tokenizer loaded in {time.perf_counter() - stage_started:.1f}s")

        label_token_ids: dict[str, int] = {}
        if spec.evaluation_kind == "next_token":
            label_token_id_lists = {
                base.NEGATIVE_LABEL_TEXT: base.encode_text(tokenizer, base.NEGATIVE_LABEL_TEXT),
                base.POSITIVE_LABEL_TEXT: base.encode_text(tokenizer, base.POSITIVE_LABEL_TEXT),
            }
            for label_text, token_ids in label_token_id_lists.items():
                if len(token_ids) != 1:
                    raise ValueError(f"Expected {label_text!r} to be a single token, got {token_ids}.")
            label_token_ids = {label_text: token_ids[0] for label_text, token_ids in label_token_id_lists.items()}

        stage_started = time.perf_counter()
        if spec.evaluation_kind == "sequence_classification":
            try:
                model = AutoModelForSequenceClassification.from_pretrained(base.MODEL_ID, num_labels=2, dtype=model_dtype)
            except TypeError:
                model = AutoModelForSequenceClassification.from_pretrained(base.MODEL_ID, num_labels=2, torch_dtype=model_dtype)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(base.MODEL_ID, dtype=model_dtype)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(base.MODEL_ID, torch_dtype=model_dtype)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if not quiet:
            log(f"{log_prefix} base model loaded in {time.perf_counter() - stage_started:.1f}s")

        stage_started = time.perf_counter()
        model = PeftModel.from_pretrained(model, checkpoint.checkpoint_path)
        model.to(device)
        model.eval()
        if not quiet:
            log(f"{log_prefix} adapter loaded and moved to device in {time.perf_counter() - stage_started:.1f}s")

        for dataset_name in dataset_names:
            dataset_started = time.perf_counter()
            dataset_metadata = dataset_samples[dataset_name]
            try:
                if not quiet:
                    log(f"{log_prefix} dataset={dataset_name} load tokenized start rows={dataset_metadata.get('rows')}")
                tokenized = load_tokenized_dataset(dataset_metadata, checkpoint.max_seq_length, tokenizer)
                if not quiet:
                    log(f"{log_prefix} dataset={dataset_name} tokenized loaded in {time.perf_counter() - dataset_started:.1f}s")
                predictions_path = None
                if write_predictions:
                    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", checkpoint.config_id)
                    safe_run = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(checkpoint.run_dir).parent.name)
                    predictions_path = (
                        Path(output_dir)
                        / "predictions"
                        / f"{dataset_name}__{safe_run}__{checkpoint.config_index:02d}__{safe_id}__{checkpoint.checkpoint_step}.csv"
                    )
                if spec.evaluation_kind == "next_token":
                    def evaluator(current_batch_size: int, current_log_prefix: str) -> dict[str, Any]:
                        return evaluate_with_predictions(
                            model=model,
                            torch_module=torch,
                            dataset_split=tokenized,
                            label_token_ids=label_token_ids,
                            pad_token_id=tokenizer.pad_token_id,
                            batch_size=current_batch_size,
                            write_predictions_path=predictions_path,
                            log_prefix=current_log_prefix,
                            quiet=quiet,
                            progress_every=progress_every,
                        )
                elif spec.evaluation_kind == "sequence_classification":
                    def evaluator(current_batch_size: int, current_log_prefix: str) -> dict[str, Any]:
                        return evaluate_sequence_classification_with_predictions(
                            model=model,
                            torch_module=torch,
                            tokenizer=tokenizer,
                            dataset_split=tokenized,
                            batch_size=current_batch_size,
                            write_predictions_path=predictions_path,
                            log_prefix=current_log_prefix,
                            quiet=quiet,
                            progress_every=progress_every,
                        )
                else:
                    generation_module = method_module.gen if spec.name == "04_causal_lm_structured_generation" else method_module
                    parse_generated_label = (
                        method_module.parse_structured_generated_label
                        if spec.name == "04_causal_lm_structured_generation"
                        else generation_module.parse_generated_label
                    )
                    generation_max_new_tokens = int(
                        max_new_tokens
                        or getattr(method_module, "DEFAULT_MAX_NEW_TOKENS", getattr(generation_module, "DEFAULT_MAX_NEW_TOKENS", 4))
                    )

                    def evaluator(current_batch_size: int, current_log_prefix: str) -> dict[str, Any]:
                        return evaluate_generation_with_predictions(
                            model=model,
                            torch_module=torch,
                            tokenizer=tokenizer,
                            dataset_split=tokenized,
                            parse_generated_label=parse_generated_label,
                            compute_generation_metrics=generation_module.compute_generation_metrics,
                            max_new_tokens=generation_max_new_tokens,
                            parse_failure_label=parse_failure_label,
                            pad_token_id=tokenizer.pad_token_id,
                            batch_size=current_batch_size,
                            write_predictions_path=predictions_path,
                            log_prefix=current_log_prefix,
                            quiet=quiet,
                            progress_every=progress_every,
                        )

                metrics, effective_batch_size = evaluate_with_oom_retries(
                    evaluator=evaluator,
                    batch_size=batch_size,
                    min_batch_size=min_batch_size,
                    log_prefix=f"{log_prefix} dataset={dataset_name}",
                    quiet=quiet,
                )
                if not quiet:
                    log(
                        f"{log_prefix} dataset={dataset_name} done in {time.perf_counter() - dataset_started:.1f}s "
                        f"batch_size={effective_batch_size} f1={metrics.get('f1', '')} acc={metrics.get('accuracy', '')}"
                    )
                rows.append(
                    result_row(
                        dataset_name=dataset_name,
                        checkpoint=checkpoint,
                        dataset_metadata=dataset_metadata,
                        metrics=metrics,
                        runtime_seconds=round(time.perf_counter() - dataset_started, 4),
                        status="completed",
                    )
                )
            except Exception as exc:
                if not quiet:
                    log(f"{log_prefix} dataset={dataset_name} failed after {time.perf_counter() - dataset_started:.1f}s: {type(exc).__name__}: {exc}")
                rows.append(
                    result_row(
                        dataset_name=dataset_name,
                        checkpoint=checkpoint,
                        dataset_metadata=dataset_metadata,
                        metrics={},
                        runtime_seconds=round(time.perf_counter() - dataset_started, 4),
                        status="failed",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )

        del model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if not quiet:
            log(f"{log_prefix} done in {time.perf_counter() - started:.1f}s")
        return rows
    except Exception as exc:
        if not quiet:
            log(f"{log_prefix} failed after {time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}")
        error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        if "model" in locals():
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        for dataset_name in dataset_names:
            rows.append(
                result_row(
                    dataset_name=dataset_name,
                    checkpoint=checkpoint,
                    dataset_metadata=dataset_samples.get(dataset_name, {}),
                    metrics={},
                    runtime_seconds=round(time.perf_counter() - started, 4),
                    status="failed",
                    error_message=error_message,
                )
            )
        return rows


def write_summary_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()


def append_summary_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    spec = method_spec(args.method)
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.min_batch_size < 1:
        raise SystemExit("--min-batch-size must be >= 1")
    if args.min_batch_size > args.batch_size:
        raise SystemExit("--min-batch-size must be <= --batch-size")
    if args.checkpoints_per_run < 0:
        raise SystemExit("--checkpoints-per-run must be >= 0")
    if args.config_indices is not None and any(index < 1 for index in args.config_indices):
        raise SystemExit("--config-indices values must be >= 1")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))

    root = project_root()
    method_module = load_method_module(spec, args.max_new_tokens, args.parse_failure_label)
    results_root = resolve_results_root(args.results_root, spec)
    output_dir = resolve_output_dir(args.output_dir, spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(results_root, spec)
    if not checkpoints:
        raise SystemExit(f"No adapter checkpoints found under {results_root}")
    source_checkpoint_count = len(checkpoints)
    available_config_indices = sorted({checkpoint.config_index for checkpoint in checkpoints})
    requested_config_indices = sorted(set(args.config_indices or []))
    filtered_source_checkpoint_count = source_checkpoint_count
    if requested_config_indices:
        missing_config_indices = [
            index for index in requested_config_indices if index not in available_config_indices
        ]
        if missing_config_indices:
            raise SystemExit(
                "--config-indices contains values with no discovered checkpoints: "
                f"{missing_config_indices}. Available config indices: {available_config_indices}"
            )
        checkpoints = [
            checkpoint for checkpoint in checkpoints if checkpoint.config_index in requested_config_indices
        ]
        filtered_source_checkpoint_count = len(checkpoints)
        if not checkpoints:
            raise SystemExit(
                f"No checkpoints matched --config-indices {requested_config_indices} under {results_root}"
            )
    checkpoints, checkpoint_selection = limit_checkpoints_per_run(checkpoints, args.checkpoints_per_run)

    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    if args.dataset_manifest:
        dataset_samples, manifest_path, manifest_sha256 = load_manifest_dataset_samples(
            args.dataset_manifest,
            args.split_roles,
        )
    else:
        dataset_samples = prepare_dataset_samples(args.datasets, args.sample_limit, args.seed, output_dir)
    dataset_names = list(dataset_samples)
    dataset_samples = pretokenize_dataset_samples(
        dataset_samples=dataset_samples,
        max_seq_lengths=sorted({checkpoint.max_seq_length for checkpoint in checkpoints}),
        output_dir=output_dir,
        spec=spec,
    )
    summary_path = output_dir / "summary.csv"
    metrics_path = output_dir / "metrics.json"
    write_summary_header(summary_path)

    all_rows: list[dict[str, Any]] = []
    metrics_payload: dict[str, Any] = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(root),
        "method": spec.name,
        "evaluation_method": spec.evaluation_method,
        "evaluation_kind": spec.evaluation_kind,
        "method_script": str(spec.script_path),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "datasets": dataset_names,
        "dataset_manifest": str(manifest_path) if manifest_path else None,
        "dataset_manifest_sha256": manifest_sha256,
        "split_roles": args.split_roles if manifest_path else None,
        "sample_limit": args.sample_limit,
        "batch_size": args.batch_size,
        "min_batch_size": args.min_batch_size,
        "checkpoints_per_run": args.checkpoints_per_run,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "parse_failure_label": args.parse_failure_label,
        "execution_strategy": "sequential",
        "torch_threads": args.torch_threads,
        "config_indices": requested_config_indices or None,
        "available_config_indices": available_config_indices,
        "source_checkpoint_count": source_checkpoint_count,
        "filtered_source_checkpoint_count": filtered_source_checkpoint_count,
        "checkpoint_count": len(checkpoints),
        "checkpoint_selection": checkpoint_selection,
        "quiet": bool(args.quiet),
        "progress_every": args.progress_every,
        "dataset_samples": dataset_samples,
        "checkpoints": [checkpoint.__dict__ for checkpoint in checkpoints],
        "rows": all_rows,
    }
    write_metrics(metrics_path, metrics_payload)

    log(f"Method: {spec.name} ({spec.evaluation_method})")
    log(f"Found {source_checkpoint_count} checkpoints/adapters under {results_root}")
    if requested_config_indices:
        log(
            f"Filtered to config indices {requested_config_indices}: "
            f"{filtered_source_checkpoint_count} checkpoints/adapters before per-run limiting"
        )
    log(
        f"Selected {len(checkpoints)} checkpoints/adapters "
        f"using checkpoints_per_run={args.checkpoints_per_run}"
    )
    log(f"Evaluating datasets: {', '.join(dataset_names)}")
    if manifest_path:
        log(f"Using frozen dataset manifest: {manifest_path} ({manifest_sha256})")
    log("Running sequentially: one checkpoint at a time")
    log(f"Writing summary to: {summary_path}")

    if not args.quiet:
        for checkpoint in checkpoints[:10]:
            log(f"Queue checkpoint config={checkpoint.config_index:02d} step={checkpoint.checkpoint_step} type={checkpoint.checkpoint_type}")
        if len(checkpoints) > 10:
            log(f"Queue checkpoint ... plus {len(checkpoints) - 10} more")
    completed = 0
    def record_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        completed += 1
        append_summary_rows(summary_path, rows)
        all_rows.extend(rows)
        metrics_payload["rows"] = all_rows
        metrics_payload["completed_checkpoint_jobs"] = completed
        metrics_payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_metrics(metrics_path, metrics_payload)
        labels = ", ".join(f"{row['dataset']}={row['status']}" for row in rows)
        first = rows[0] if rows else {}
        log(
            f"[{completed}/{len(checkpoints)}] "
            f"config={first.get('config_index', '?')} "
            f"checkpoint={first.get('checkpoint_step', '?')} "
            f"{labels}"
        )

    try:
        for checkpoint in checkpoints:
            rows = evaluate_checkpoint(
                checkpoint,
                dataset_samples,
                dataset_names,
                str(output_dir),
                spec,
                method_module,
                args.max_new_tokens,
                args.parse_failure_label,
                args.batch_size,
                args.min_batch_size,
                bool(args.write_predictions),
                args.torch_threads,
                bool(args.quiet),
                args.progress_every,
            )
            record_rows(rows)
    except KeyboardInterrupt:
        log("Stopping after current partial state...")
        metrics_payload["interrupted_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metrics_payload["completed_checkpoint_jobs"] = completed
        metrics_payload["rows"] = all_rows
        write_metrics(metrics_path, metrics_payload)
        log(f"Interrupted. Partial summary: {summary_path}")
        log(f"Interrupted. Partial metrics: {metrics_path}")
        return 130

    metrics_payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_metrics(metrics_path, metrics_payload)
    log(f"Done. Summary: {summary_path}")
    log(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
