#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from enum import Enum
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 67
TRAIN_SPLIT = 0.95
VALIDATION_SPLIT = 0.006
TEST_SPLIT = 0.04
HOLDOUT_SPLIT = VALIDATION_SPLIT + TEST_SPLIT
NUM_TRAIN_EPOCHS = 0.5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5
LOGGING_STEPS = 25
POSITIVE_LABEL_TEXT = "spam"
NEGATIVE_LABEL_TEXT = "ham"
IM_END_TOKEN = "<|im_end|>"
TRUNCATION_MARKER = "\n\n[...]\n\n"
AIM_EXPERIMENT_NAME = "qwen3-0.6b-spam-causal-lm-hparam-sweep"
AIM_SYSTEM_TRACKING_INTERVAL = 10


@dataclass(frozen=True)
class SweepConfig:
    index: int
    group: str
    max_seq_length: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    @property
    def train_batch_size(self) -> int:
        return 24 if self.max_seq_length == 512 else 12

    @property
    def eval_batch_size(self) -> int:
        return 24 if self.max_seq_length == 512 else 12

    @property
    def gradient_accumulation_steps(self) -> int:
        return 1 if self.max_seq_length == 512 else 2

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps

    @property
    def config_id(self) -> str:
        group_token = self.group.lower().replace("/", "_").replace(" ", "_")
        return (
            f"{group_token}_seq{self.max_seq_length}"
            f"_lr{float_token(self.learning_rate)}"
            f"_r{self.lora_r}_a{self.lora_alpha}"
            f"_do{float_token(self.lora_dropout)}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "config_id": self.config_id,
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
            }
        )
        return data


def float_token(value: float) -> str:
    explicit = {
        1e-6: "1e-6",
        5e-5: "5e-5",
        1e-4: "1e-4",
        0.0: "0p0",
        0.1: "0p1",
        0.4: "0p4",
    }
    for key, token in explicit.items():
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1e-12):
            return token
    return f"{value:g}".replace(".", "p")


def build_sweep_configs() -> list[SweepConfig]:
    return [
        SweepConfig(1, "LR/context", 512, 1e-6, 16, 32, 0.1),
        SweepConfig(2, "LR/context", 512, 5e-5, 16, 32, 0.1),
        SweepConfig(3, "LR/context", 512, 1e-4, 16, 32, 0.1),
        SweepConfig(4, "LR/context", 1024, 1e-6, 16, 32, 0.1),
        SweepConfig(5, "LR/context", 1024, 5e-5, 16, 32, 0.1),
        SweepConfig(6, "LR/context", 1024, 1e-4, 16, 32, 0.1),
        SweepConfig(7, "LoRA capacity", 512, 5e-5, 8, 16, 0.1),
        SweepConfig(8, "LoRA capacity", 512, 5e-5, 32, 64, 0.1),
        SweepConfig(9, "LoRA capacity", 512, 5e-5, 64, 128, 0.1),
        SweepConfig(10, "LoRA capacity", 1024, 5e-5, 8, 16, 0.1),
        SweepConfig(11, "LoRA capacity", 1024, 5e-5, 32, 64, 0.1),
        SweepConfig(12, "LoRA capacity", 1024, 5e-5, 64, 128, 0.1),
        SweepConfig(13, "Dropout", 512, 5e-5, 16, 32, 0.0),
        SweepConfig(14, "Dropout", 512, 5e-5, 16, 32, 0.4),
        SweepConfig(15, "Dropout", 1024, 5e-5, 16, 32, 0.0),
        SweepConfig(16, "Dropout", 1024, 5e-5, 16, 32, 0.4),
    ]


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    candidates = [Path.cwd().resolve(), script_dir(), script_dir().parent]
    for candidate in candidates:
        if (candidate / "dataset").is_dir() and (candidate / "aim_tracking").is_dir():
            return candidate
    return script_dir().parent


def default_results_root() -> Path:
    return script_dir() / "results"


def resolve_results_root(value: str | None) -> Path:
    if value is None:
        return default_results_root()
    path = Path(value)
    if not path.is_absolute():
        path = script_dir() / path
    return path.resolve()


def make_sweep_id() -> str:
    return "qwen3_clm_sweep_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_config_indexes(values: list[str] | None) -> list[int]:
    if not values:
        return [config.index for config in build_sweep_configs()]

    indexes: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            indexes.append(int(part))

    available = {config.index for config in build_sweep_configs()}
    unknown = sorted(set(indexes) - available)
    if unknown:
        raise SystemExit(f"Unknown config indexes: {unknown}")
    return indexes


def get_config(index: int) -> SweepConfig:
    configs = {config.index: config for config in build_sweep_configs()}
    try:
        return configs[index]
    except KeyError as exc:
        raise SystemExit(f"Unknown config index: {index}") from exc


def config_run_dir(sweep_dir: Path, config: SweepConfig) -> Path:
    return sweep_dir / f"{config.index:02d}_{config.config_id}"


def run_name(sweep_id: str, config: SweepConfig) -> str:
    return f"{sweep_id}__{config.index:02d}__{config.config_id}"


def latest_checkpoint_path(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return None
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(inner) for inner in value]
    if isinstance(value, set):
        return sorted(json_safe(inner) for inner in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def close_aim_run(aim_run: Any | None) -> None:
    if aim_run is None or not hasattr(aim_run, "close"):
        return
    try:
        aim_run.close()
    except Exception:
        pass


def safe_aim_set(aim_run: Any | None, key: str, value: Any) -> bool:
    if aim_run is None:
        return False
    try:
        aim_run[key] = value
        return True
    except Exception as exc:
        print(f"Warning: could not write Aim metadata {key!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def safe_aim_track(
    aim_run: Any | None,
    value: float,
    *,
    name: str,
    step: int | None,
    epoch: float | None,
    context: dict[str, str],
) -> bool:
    if aim_run is None:
        return False
    try:
        aim_run.track(float(value), name=name, step=step, epoch=epoch, context=context)
        return True
    except Exception as exc:
        print(f"Warning: could not track Aim metric {name!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_configs(json_output: bool) -> None:
    configs = [config.to_dict() for config in build_sweep_configs()]
    if json_output:
        print(json.dumps(configs, indent=2, sort_keys=True))
        return

    headers = ["#", "Group", "Max Seq", "LR", "LoRA", "Dropout", "Batch", "Grad Accum", "Config ID"]
    rows = [
        [
            str(config.index),
            config.group,
            str(config.max_seq_length),
            f"{config.learning_rate:g}",
            f"{config.lora_r}/{config.lora_alpha}",
            f"{config.lora_dropout:g}",
            str(config.train_batch_size),
            str(config.gradient_accumulation_steps),
            config.config_id,
        ]
        for config in build_sweep_configs()
    ]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


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
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_specificity",
    "validation_balanced_accuracy",
    "validation_classification_failure_count",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_specificity",
    "test_balanced_accuracy",
    "test_classification_failure_count",
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


def summary_row_from_metrics(config: SweepConfig, run_dir: Path) -> dict[str, Any]:
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
            "run_dir": str(run_dir),
        }
    )

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        row["status"] = "missing"
        return row

    try:
        metrics = read_json(metrics_path)
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

    training = metrics.get("training_metrics", {}) or {}
    trainer_state = metrics.get("trainer_state", {}) or {}
    row["train_runtime"] = training.get("train_runtime", "")
    row["train_samples_per_second"] = training.get("train_samples_per_second", "")
    row["train_steps_per_second"] = training.get("train_steps_per_second", "")
    row["global_step"] = trainer_state.get("global_step", "")
    row["best_metric"] = trainer_state.get("best_metric", "")

    for prefix, metric_key in [("validation", "validation_metrics"), ("test", "test_metrics")]:
        split_metrics = metrics.get(metric_key, {}) or {}
        for name in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "classification_failure_count",
        ]:
            row[f"{prefix}_{name}"] = split_metrics.get(name, "")

    return row


def write_summary(sweep_dir: Path, selected_configs: list[SweepConfig]) -> Path:
    rows = [summary_row_from_metrics(config, config_run_dir(sweep_dir, config)) for config in selected_configs]
    summary_path = sweep_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_manifest(sweep_dir: Path, sweep_id: str, selected_configs: list[SweepConfig], args: argparse.Namespace) -> Path:
    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest = {
        "sweep_id": sweep_id,
        "created_or_updated_at": dt.datetime.now(dt.UTC).isoformat(),
        "aim_experiment_name": AIM_EXPERIMENT_NAME,
        "model_id": MODEL_ID,
        "seed": SEED,
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "logging_steps": LOGGING_STEPS,
        "script": str(Path(__file__).resolve()),
        "project_root": str(project_root()),
        "sweep_dir": str(sweep_dir),
        "selected_config_indexes": [config.index for config in selected_configs],
        "configs": [config.to_dict() for config in selected_configs],
        "runner_args": vars(args),
    }
    write_json(manifest_path, manifest)
    return manifest_path


def metrics_completed(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        return read_json(metrics_path).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def write_subprocess_failure_metrics(
    *,
    run_dir: Path,
    sweep_id: str,
    config: SweepConfig,
    status: str,
    error_type: str,
    error_message: str,
) -> None:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return
    write_json(
        metrics_path,
        {
            "status": status,
            "sweep_id": sweep_id,
            "run_name": run_name(sweep_id, config),
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "error_type": error_type,
            "error_message": error_message,
        },
    )


def subprocess_failure_details(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return "metrics.json was not written"
    try:
        metrics = read_json(metrics_path)
    except Exception as exc:
        return f"could not read metrics.json: {type(exc).__name__}: {exc}"

    status = metrics.get("status", "unknown")
    error_type = metrics.get("error_type") or "unknown_error"
    error_message = metrics.get("error_message") or "no error_message recorded"
    return f"{status}: {error_type}: {error_message}"


def terminate_process_tree(process: subprocess.Popen[Any], *, timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        except Exception:
            process.send_signal(signal.SIGINT)

    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()
    process.wait()


def run_child_process(command: list[str], *, cwd: Path) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        start_new_session=(os.name != "nt"),
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        terminate_process_tree(process)
        raise


def run_sweep(args: argparse.Namespace) -> int:
    sweep_id = args.sweep_id or make_sweep_id()
    results_root = resolve_results_root(args.results_root)
    sweep_dir = results_root / "sweeps" / sweep_id
    selected_configs = [get_config(index) for index in parse_config_indexes(args.config_index)]

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(sweep_dir, sweep_id, selected_configs, args)
    summary_path = write_summary(sweep_dir, selected_configs)

    print(f"Sweep ID: {sweep_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if args.dry_run:
        print("Dry run requested; no training subprocesses were launched.")
        return 0

    failures = 0
    python_bin = args.python or sys.executable
    root = project_root()

    for config in selected_configs:
        run_dir = config_run_dir(sweep_dir, config)
        if args.resume and metrics_completed(run_dir):
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
            return_code = run_child_process(command, cwd=root)
        except KeyboardInterrupt:
            write_subprocess_failure_metrics(
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
            write_subprocess_failure_metrics(
                run_dir=run_dir,
                sweep_id=sweep_id,
                config=config,
                status="failed",
                error_type="SubprocessFailure",
                error_message=f"run-one exited with code {return_code} before writing metrics.json",
            )
            print(f"[{config.index:02d}] Failed with exit code {return_code}; continuing.")
            print(f"[{config.index:02d}] Failure details: {subprocess_failure_details(run_dir)}")
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


def find_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_email_text(subject: str | None, body: str | None) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def build_user_prompt(email_text: str) -> str:
    return (
        "You are an email spam classifier.\n"
        f"Classify the following email as {POSITIVE_LABEL_TEXT} or {NEGATIVE_LABEL_TEXT}.\n"
        f"Return exactly one lowercase word: {POSITIVE_LABEL_TEXT} or {NEGATIVE_LABEL_TEXT}.\n\n"
        "Email:\n"
        f"{email_text}"
    )


def apply_qwen_chat_template(tokenizer: Any, email_text: str) -> str:
    messages = [{"role": "user", "content": build_user_prompt(email_text.strip())}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def trim_ids_head_tail(token_ids: list[int], token_budget: int, marker_ids: list[int]) -> list[int]:
    if token_budget <= 0:
        return []
    if len(token_ids) <= token_budget:
        return list(token_ids)

    marker_budget = token_budget - len(marker_ids)
    if marker_budget >= 8:
        head_count = max(1, math.ceil(marker_budget * 0.67))
        tail_count = max(1, marker_budget - head_count)
        return token_ids[:head_count] + marker_ids + token_ids[-tail_count:]

    head_count = max(1, math.ceil(token_budget * 0.67))
    tail_count = max(0, token_budget - head_count)
    return token_ids[:head_count] + (token_ids[-tail_count:] if tail_count else [])


def trim_email_to_fit(
    *,
    tokenizer: Any,
    email_text: str,
    completion_text: str,
    max_seq_length: int,
) -> dict[str, Any]:
    completion_ids = encode_text(tokenizer, completion_text)
    empty_prompt_ids = encode_text(tokenizer, apply_qwen_chat_template(tokenizer, ""))
    email_ids = encode_text(tokenizer, email_text)
    available_email_tokens = max_seq_length - len(empty_prompt_ids) - len(completion_ids)

    if available_email_tokens < 1:
        raise ValueError(
            f"max_seq_length={max_seq_length} is too small; prompt overhead plus completion uses "
            f"{len(empty_prompt_ids) + len(completion_ids)} tokens."
        )

    marker_ids = encode_text(tokenizer, TRUNCATION_MARKER)
    candidate_budget = min(len(email_ids), available_email_tokens)
    was_trimmed = len(email_ids) > available_email_tokens

    while candidate_budget >= 0:
        candidate_ids = trim_ids_head_tail(email_ids, candidate_budget, marker_ids)
        candidate_text = decode_ids(tokenizer, candidate_ids)
        prompt_ids = encode_text(tokenizer, apply_qwen_chat_template(tokenizer, candidate_text))
        total_length = len(prompt_ids) + len(completion_ids)
        if total_length <= max_seq_length:
            return {
                "text": candidate_text,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "raw_email_tokens": len(email_ids),
                "trimmed_email_tokens": len(candidate_ids),
                "token_length": total_length,
                "was_trimmed": was_trimmed,
            }
        candidate_budget -= max(1, total_length - max_seq_length)

    raise ValueError("Could not trim email to fit the configured max sequence length.")


def label_to_text(label: int) -> str:
    return POSITIVE_LABEL_TEXT if int(label) == 1 else NEGATIVE_LABEL_TEXT


def build_tokenized_sample(tokenizer: Any, sample: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    email_text = build_email_text(sample["subject"], sample["body"])
    label_text = label_to_text(int(sample["label"]))
    completion_text = f"{label_text}{IM_END_TOKEN}"
    trimmed = trim_email_to_fit(
        tokenizer=tokenizer,
        email_text=email_text,
        completion_text=completion_text,
        max_seq_length=max_seq_length,
    )

    prompt_text = apply_qwen_chat_template(tokenizer, trimmed["text"])
    prompt_ids = trimmed["prompt_ids"]
    completion_ids = trimmed["completion_ids"]
    input_ids = prompt_ids + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

    if len(input_ids) > max_seq_length:
        raise ValueError(f"Tokenized sample length {len(input_ids)} exceeds max_seq_length={max_seq_length}.")
    if not any(completion_mask):
        raise ValueError("Completion mask has no supervised tokens.")

    return {
        "text": email_text,
        "trimmed_text": trimmed["text"],
        "prompt": prompt_text,
        "completion": completion_text,
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "prompt_input_ids": prompt_ids,
        "label_text": label_text,
        "raw_email_tokens": trimmed["raw_email_tokens"],
        "trimmed_email_tokens": trimmed["trimmed_email_tokens"],
        "token_length": trimmed["token_length"],
        "prompt_token_length": len(prompt_ids),
        "completion_token_length": len(completion_ids),
        "was_trimmed": trimmed["was_trimmed"],
    }


def select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(int(limit), len(dataset))))


def dataset_trim_stats(dataset_dict: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, split_dataset in dataset_dict.items():
        rows = len(split_dataset)
        was_trimmed = split_dataset["was_trimmed"] if rows else []
        token_lengths = split_dataset["token_length"] if rows else []
        raw_lengths = split_dataset["raw_email_tokens"] if rows else []
        stats[split] = {
            "rows": rows,
            "trimmed_rows": int(sum(bool(value) for value in was_trimmed)),
            "trimmed_rate": float(sum(bool(value) for value in was_trimmed) / max(rows, 1)),
            "max_token_length": int(max(token_lengths) if token_lengths else 0),
            "mean_token_length": float(sum(token_lengths) / max(rows, 1)),
            "max_raw_email_tokens": int(max(raw_lengths) if raw_lengths else 0),
        }
    return stats


def prepare_datasets(
    *,
    tokenizer: Any,
    config: SweepConfig,
    train_limit: int | None,
    validation_limit: int | None,
    test_limit: int | None,
) -> tuple[Any, dict[str, Any], str, str]:
    from datasets import ClassLabel, DatasetDict, load_dataset
    from dataset.combine import combine_datasets
    from aim_tracking import summarize_text_classification_dataset

    data_path = combine_datasets(["trec_2007", "ceas_2008"], spam_ham_ratio=0.5)
    dataset_sha = sha256_file(data_path)
    raw_dataset = load_dataset("parquet", data_files=data_path, split="train")
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["valid", "spam"]))
    raw_summary = summarize_text_classification_dataset(raw_dataset)

    holdout = raw_dataset.train_test_split(
        test_size=HOLDOUT_SPLIT,
        stratify_by_column="label",
        seed=SEED,
    )
    valid_test = holdout["test"].train_test_split(
        test_size=TEST_SPLIT / HOLDOUT_SPLIT,
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

    dataset = dataset.filter(
        lambda sample: bool(build_email_text(sample["subject"], sample["body"])),
        desc="Filtering empty emails",
    )
    dataset = dataset.map(
        lambda sample: build_tokenized_sample(tokenizer, sample, config.max_seq_length),
        desc=f"Formatting prompts to <= {config.max_seq_length} tokens",
    )

    trim_stats = dataset_trim_stats(dataset)
    for split_name, split_stats in trim_stats.items():
        if split_stats["max_token_length"] > config.max_seq_length:
            raise ValueError(
                f"{split_name} max token length {split_stats['max_token_length']} exceeds "
                f"max_seq_length={config.max_seq_length}."
            )

    metadata = {
        "path": data_path,
        "sha256": dataset_sha,
        "rows": len(raw_dataset),
        "spam": raw_summary["label_counts"]["spam"],
        "ham": raw_summary["label_counts"]["ham"],
        "sources": raw_summary["source_counts"],
        "avg_subject_chars": raw_summary["text_stats"]["avg_subject_chars"],
        "avg_body_chars": raw_summary["text_stats"]["avg_body_chars"],
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]),
        "test_rows": len(dataset["test"]),
        "trim_stats": trim_stats,
    }
    return dataset, metadata, data_path, dataset_sha


def compute_metrics(predictions: list[int], labels: list[int], probabilities: list[list[float]], failure_count: int) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score

    predictions_array = np.asarray(predictions)
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    if probabilities_array.size == 0:
        probabilities_array = np.zeros((0, 2), dtype=float)

    if len(labels_array) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "ham_precision": 0.0,
            "ham_recall": 0.0,
            "ham_f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "matthews_corrcoef": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "cross_entropy": 0.0,
            "classification_failure_count": float(failure_count),
            "classification_failure_rate": 0.0,
            "false_positive_count": 0.0,
            "false_negative_count": 0.0,
            "true_positive_count": 0.0,
            "true_negative_count": 0.0,
        }

    row_sums = probabilities_array.sum(axis=1, keepdims=True)
    probabilities_array = np.divide(
        probabilities_array,
        np.clip(row_sums, 1e-12, None),
        out=np.full_like(probabilities_array, 0.5, dtype=float),
        where=row_sums > 0,
    )

    accuracy = float((predictions_array == labels_array).mean())
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=[0, 1],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        average="macro",
        zero_division=0,
    )

    tp = int(((predictions_array == 1) & (labels_array == 1)).sum())
    fp = int(((predictions_array == 1) & (labels_array == 0)).sum())
    fn = int(((predictions_array == 0) & (labels_array == 1)).sum())
    tn = int(((predictions_array == 0) & (labels_array == 0)).sum())

    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (float(per_class_recall[1]) + specificity) / 2
    spam_probabilities = probabilities_array[:, 1] if len(probabilities_array) else np.asarray([])

    try:
        roc_auc = float(roc_auc_score(labels_array, spam_probabilities))
    except ValueError:
        roc_auc = 0.0
    try:
        pr_auc = float(average_precision_score(labels_array, spam_probabilities))
    except ValueError:
        pr_auc = 0.0
    try:
        cross_entropy = float(log_loss(labels_array, probabilities_array, labels=[0, 1]))
    except ValueError:
        cross_entropy = 0.0

    return {
        "accuracy": accuracy,
        "precision": float(per_class_precision[1]),
        "recall": float(per_class_recall[1]),
        "f1": float(per_class_f1[1]),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "ham_precision": float(per_class_precision[0]),
        "ham_recall": float(per_class_recall[0]),
        "ham_f1": float(per_class_f1[0]),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "matthews_corrcoef": float(matthews_corrcoef(labels_array, predictions_array)) if len(set(labels)) > 1 else 0.0,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cross_entropy": cross_entropy,
        "classification_failure_count": float(failure_count),
        "classification_failure_rate": float(failure_count / max(len(labels_array), 1)),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
    }


def pad_prompt_batch(torch_module: Any, prompt_batches: list[list[int]], pad_token_id: int, device: Any) -> dict[str, Any]:
    max_length = max(len(ids) for ids in prompt_batches)
    input_ids = torch_module.full((len(prompt_batches), max_length), pad_token_id, dtype=torch_module.long, device=device)
    attention_mask = torch_module.zeros((len(prompt_batches), max_length), dtype=torch_module.long, device=device)
    for row, ids in enumerate(prompt_batches):
        values = torch_module.tensor(ids, dtype=torch_module.long, device=device)
        input_ids[row, : len(ids)] = values
        attention_mask[row, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def evaluate_next_token_classifier(
    *,
    model: Any,
    torch_module: Any,
    dataset_split: Any,
    label_token_ids: dict[str, int],
    pad_token_id: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    failure_count = 0
    label_ids = torch_module.tensor(
        [label_token_ids[NEGATIVE_LABEL_TEXT], label_token_ids[POSITIVE_LABEL_TEXT]],
        dtype=torch_module.long,
        device=device,
    )

    with torch_module.inference_mode():
        for start in range(0, len(dataset_split), batch_size):
            rows = dataset_split.select(range(start, min(start + batch_size, len(dataset_split))))
            prompt_input_ids = rows["prompt_input_ids"]
            batch = pad_prompt_batch(torch_module, prompt_input_ids, pad_token_id, device)
            outputs = model(**batch)
            last_positions = batch["attention_mask"].sum(dim=1) - 1
            next_token_logits = outputs.logits[torch_module.arange(len(prompt_input_ids), device=device), last_positions]
            label_logits = next_token_logits.index_select(dim=-1, index=label_ids)
            finite = torch_module.isfinite(label_logits).all(dim=-1)
            safe_logits = torch_module.nan_to_num(label_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            batch_probabilities = torch_module.softmax(safe_logits, dim=-1)
            batch_predictions = batch_probabilities.argmax(dim=-1)

            failure_count += int((~finite).sum().item())
            probabilities.extend(batch_probabilities.detach().cpu().float().tolist())
            predictions.extend(batch_predictions.detach().cpu().int().tolist())
            labels.extend([int(value) for value in rows["label"]])

    return compute_metrics(predictions, labels, probabilities, failure_count)


def build_next_token_metrics_callback(
    *,
    torch_module: Any,
    trainer_callback_cls: Any,
    eval_dataset: Any,
    label_token_ids: dict[str, int],
    pad_token_id: int,
    batch_size: int,
) -> Any:
    class NextTokenClassificationMetricsCallback(trainer_callback_cls):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if model is None or metrics is None:
                return
            classification_metrics = evaluate_next_token_classifier(
                model=model,
                torch_module=torch_module,
                dataset_split=eval_dataset,
                label_token_ids=label_token_ids,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
            )
            for key, value in classification_metrics.items():
                metrics[f"eval_{key}"] = value

    return NextTokenClassificationMetricsCallback()


def non_cuda_batch_settings(config: SweepConfig) -> dict[str, int]:
    train_batch_size = min(2, config.train_batch_size)
    return {
        "train_batch_size": train_batch_size,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": max(1, math.ceil(config.effective_batch_size / train_batch_size)),
    }


def run_training_config(args: argparse.Namespace, config: SweepConfig, sweep_id: str, run_dir: Path) -> dict[str, Any]:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import numpy as np
    import torch
    from aim_tracking import create_aim_callbacks
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(SEED)
    device = find_device(torch)
    cuda = device == "cuda"
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

    effective_batch_size = train_batch_size * gradient_accumulation_steps
    print(f"Using device: {device.upper()}")
    if cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Run: {run_name(sweep_id, config)}")
    print(f"Config: {config.to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    label_token_id_lists = {
        NEGATIVE_LABEL_TEXT: encode_text(tokenizer, NEGATIVE_LABEL_TEXT),
        POSITIVE_LABEL_TEXT: encode_text(tokenizer, POSITIVE_LABEL_TEXT),
    }
    for label_text, token_ids in label_token_id_lists.items():
        if len(token_ids) != 1:
            raise ValueError(f"Expected {label_text!r} to be a single token, got {token_ids}.")
    label_token_ids = {label_text: token_ids[0] for label_text, token_ids in label_token_id_lists.items()}

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
            "run_name": run_name(sweep_id, config),
            "config": config.to_dict(),
            "dataset": dataset_metadata,
            "run_dir": str(run_dir),
        }

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=model_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=model_dtype)

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_batches_per_epoch = int(np.ceil(len(dataset["train"]) / train_batch_size))
    optimizer_steps_per_epoch = int(np.ceil(train_batches_per_epoch / gradient_accumulation_steps))
    eval_steps = max(1, optimizer_steps_per_epoch // 4)
    if args.max_steps is not None:
        eval_steps = max(1, min(eval_steps, int(args.max_steps)))

    output_dir = run_dir / "trainer_output"
    adapter_path = run_dir / "adapter"
    tensorboard_dir = run_dir / "tensorboard"
    current_run_name = run_name(sweep_id, config)

    training_args_kwargs: dict[str, Any] = {
        "report_to": ["tensorboard"],
        "run_name": current_run_name,
        "output_dir": str(output_dir),
        "logging_dir": str(tensorboard_dir),
        "logging_strategy": "steps",
        "logging_steps": LOGGING_STEPS,
        "logging_first_step": True,
        "save_total_limit": 2,
        "seed": SEED,
        "data_seed": SEED,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": config.learning_rate,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lr_scheduler_type": "cosine",
        "optim": target_optim,
        "max_steps": int(args.max_steps) if args.max_steps is not None else -1,
        "bf16": bf16,
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "max_length": config.max_seq_length,
        "completion_only_loss": True,
        "packing": False,
        "eos_token": IM_END_TOKEN,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_num_workers": dataloader_num_workers,
        "remove_unused_columns": False,
    }
    if gradient_checkpointing_kwargs is not None:
        training_args_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

    training_args = SFTConfig(**training_args_kwargs)
    run_config = {
        "sweep_id": sweep_id,
        "config_id": config.config_id,
        "config_index": config.index,
        "group": config.group,
        "model_id": MODEL_ID,
        "seed": SEED,
        "max_seq_length": config.max_seq_length,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": config.learning_rate,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": training_args.optim,
        "device": device,
        "bf16": bool(training_args.bf16),
        "fp16": bool(training_args.fp16),
        "gradient_checkpointing": bool(training_args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_num_workers": dataloader_num_workers,
        "cuda_a100_optimized": cuda,
        "non_cuda_fallback": not cuda,
        "eval_steps": training_args.eval_steps,
        "save_steps": training_args.save_steps,
        "logging_steps": training_args.logging_steps,
        "tensorboard_log_dir": training_args.logging_dir,
        "output_dir": training_args.output_dir,
        "adapter_path": str(adapter_path),
        "resume_checkpoint_enabled": bool(args.resume_checkpoint),
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
        "positive_label_text": POSITIVE_LABEL_TEXT,
        "negative_label_text": NEGATIVE_LABEL_TEXT,
        "label_token_ids": label_token_ids,
        "dataset_path": data_path,
        "dataset_sha256": dataset_sha,
    }
    lora_metadata = {
        "r": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": peft_config.target_modules,
        "task_type": "CAUSAL_LM",
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
    safe_aim_set(aim_run, "sweep_id", sweep_id)
    safe_aim_set(aim_run, "config_id", config.config_id)
    safe_aim_set(aim_run, "config_index", config.index)
    safe_aim_set(aim_run, "group", config.group)
    safe_aim_set(aim_run, "run_name", current_run_name)
    safe_aim_set(aim_run, "run_dir", str(run_dir))
    aim_run_hash = getattr(aim_run, "hash", "")
    safe_aim_set(aim_run, "status", "running")

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            callbacks=[
                build_next_token_metrics_callback(
                    torch_module=torch,
                    trainer_callback_cls=TrainerCallback,
                    eval_dataset=dataset["validation"],
                    label_token_ids=label_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    batch_size=eval_batch_size,
                ),
                aim_callback,
                notebook_aim_callback,
            ],
        )

        resume_checkpoint = latest_checkpoint_path(output_dir) if args.resume_checkpoint else None
        if resume_checkpoint is not None:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            safe_aim_set(aim_callback.experiment, "resume_from_checkpoint", str(resume_checkpoint))

        trainer_stats = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        model = trainer.model
        model.config.use_cache = True

        validation_metrics = evaluate_next_token_classifier(
            model=model,
            torch_module=torch,
            dataset_split=dataset["validation"],
            label_token_ids=label_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=eval_batch_size,
        )
        test_metrics = evaluate_next_token_classifier(
            model=model,
            torch_module=torch,
            dataset_split=dataset["test"],
            label_token_ids=label_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=eval_batch_size,
        )

        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # AimCallback closes its run during Trainer.on_train_end. Reopen by hash
        # before writing final sweep metadata and metrics.
        aim_run = aim_callback.experiment
        if getattr(aim_run, "name", None) != current_run_name:
            aim_run.name = current_run_name
        aim_run_hash = getattr(aim_run, "hash", aim_run_hash)
        safe_aim_set(aim_run, "status", "completed")
        safe_aim_set(aim_run, "final_saved_model_path", str(adapter_path))
        safe_aim_set(aim_run, "saved_tokenizer_path", str(adapter_path))
        safe_aim_set(aim_run, "final_validation_metrics", validation_metrics)
        safe_aim_set(aim_run, "final_test_metrics", test_metrics)
        for metric_name, metric_value in validation_metrics.items():
            if isinstance(metric_value, (int, float)):
                safe_aim_track(
                    aim_run,
                    metric_value,
                    name=metric_name,
                    step=trainer.state.global_step,
                    epoch=trainer.state.epoch,
                    context={"subset": "validation_final"},
                )
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                safe_aim_track(
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
        safe_aim_set(aim_run, "status", "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        safe_aim_set(aim_run, "failure_type", type(exc).__name__)
        safe_aim_set(aim_run, "failure_message", str(exc))
        raise
    finally:
        close_aim_run(aim_run)


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
    config = get_config(args.config_index)
    results_root = resolve_results_root(args.results_root)
    sweep_dir = results_root / "sweeps" / args.sweep_id
    run_dir = config_run_dir(sweep_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "config.json",
        {
            "sweep_id": args.sweep_id,
            "run_name": run_name(args.sweep_id, config),
            "run_dir": str(run_dir),
            "config": config.to_dict(),
            "args": vars(args),
        },
    )

    try:
        metrics = run_training_config(args, config, args.sweep_id, run_dir)
        write_json(run_dir / "metrics.json", metrics)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "sweep_id": args.sweep_id,
            "run_name": run_name(args.sweep_id, config),
            "config": config.to_dict(),
            "run_dir": str(run_dir),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "metrics.json", failure)
        print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(failure["traceback"], file=sys.stderr)
        return 1
    finally:
        cleanup_after_run()


def summarize_command(args: argparse.Namespace) -> int:
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
    parser = argparse.ArgumentParser(description="Run Qwen3 causal LM LoRA hyperparameter sweeps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-configs", help="Print the 16 sweep configurations.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    sweep_parser = subparsers.add_parser("run-sweep", help="Launch one subprocess per selected config.")
    sweep_parser.add_argument("--sweep-id", default=None, help="Optional stable sweep id.")
    sweep_parser.add_argument("--results-root", default=None, help="Defaults to lora-fine-tuning/results.")
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
    for attr in ["max_steps", "train_limit", "validation_limit", "test_limit"]:
        value = getattr(args, attr, None)
        if value is not None and value < 1:
            option = "--" + attr.replace("_", "-")
            raise SystemExit(f"{option} must be >= 1 when provided.")

    cooldown_seconds = getattr(args, "cooldown_seconds", None)
    if cooldown_seconds is not None and cooldown_seconds < 0:
        raise SystemExit("--cooldown-seconds must be >= 0.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.command == "list-configs":
        list_configs(args.json)
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
