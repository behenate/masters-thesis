#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


AIM_EXPERIMENT_NAME = "qwen3-0.6b-spam-causal-lm-structured-generation-sweep"
EVALUATION_METHOD = "causal_lm_structured_json_generation_parsing"
COMPLETION_FORMAT = "{\"label\":\"<ham_or_spam>\"}<|im_end|>"
DEFAULT_MAX_NEW_TOKENS = 12
DEFAULT_PARSE_FAILURE_LABEL = "ham"


def project_root() -> Path:
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "lora-fine-tuning").is_dir():
            return candidate
    raise RuntimeError("Could not find project root containing dataset/ and lora-fine-tuning/.")


def method_dir() -> Path:
    return Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generation_script = method_dir().parent / "02_causal_lm_generation_parsing" / "qwen3_0.6b_generation_parsing_sweep.py"
gen = load_module("qwen3_generation_parsing_helpers", generation_script)
base = gen.base

MODEL_ID = base.MODEL_ID
SEED = base.SEED
POSITIVE_LABEL_TEXT = base.POSITIVE_LABEL_TEXT
NEGATIVE_LABEL_TEXT = base.NEGATIVE_LABEL_TEXT
IM_END_TOKEN = base.IM_END_TOKEN


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
    return "qwen3_clm_structured_generation_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_sweep_configs() -> list[Any]:
    cfg = base.SweepConfig
    return [
        cfg(1, "Structured LR", 1024, 7e-5, 16, 32, 0.1),
        cfg(2, "Structured LR", 1024, 1e-4, 16, 32, 0.1),
        cfg(3, "Structured LR", 1024, 1.5e-4, 16, 32, 0.1),
        cfg(4, "Structured context", 512, 1e-4, 16, 32, 0.1),
        cfg(5, "Structured context", 512, 7e-5, 16, 32, 0.1),
        cfg(6, "Structured LoRA", 1024, 1e-4, 8, 16, 0.1),
        cfg(7, "Structured LoRA", 1024, 1e-4, 32, 64, 0.1),
        cfg(8, "Structured dropout", 1024, 1e-4, 16, 32, 0.0),
    ]


def structured_completion(label_text: str) -> str:
    return f'{{"label":"{label_text}"}}{IM_END_TOKEN}'


def build_structured_user_prompt(email_text: str) -> str:
    return (
        "You are an email spam classifier.\n"
        "Classify the following email as spam or ham.\n"
        "Return exactly one JSON object and no other text.\n"
        f'Allowed outputs: {{"label":"{POSITIVE_LABEL_TEXT}"}} or {{"label":"{NEGATIVE_LABEL_TEXT}"}}.\n\n'
        "Email:\n"
        f"{email_text}"
    )


def build_structured_tokenized_sample(tokenizer: Any, sample: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    email_text = base.build_email_text(sample["subject"], sample["body"])
    label_text = base.label_to_text(int(sample["label"]))
    completion_text = structured_completion(label_text)
    trimmed = base.trim_email_to_fit(
        tokenizer=tokenizer,
        email_text=email_text,
        completion_text=completion_text,
        max_seq_length=max_seq_length,
    )

    prompt_text = base.apply_qwen_chat_template(tokenizer, trimmed["text"])
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
        "label_text": label_text,
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "prompt_input_ids": prompt_ids,
        "raw_email_tokens": trimmed["raw_email_tokens"],
        "trimmed_email_tokens": trimmed["trimmed_email_tokens"],
        "token_length": trimmed["token_length"],
        "prompt_token_length": len(prompt_ids),
        "completion_token_length": len(completion_ids),
        "was_trimmed": trimmed["was_trimmed"],
    }


def parse_structured_generated_label(text: str) -> str | None:
    cleaned = gen.strip_reasoning_and_special_tokens(text)
    json_match = re.search(r"\{[^{}]*\}", cleaned)
    if json_match:
        try:
            import json

            payload = json.loads(json_match.group(0))
            label = str(payload.get("label", "")).strip().lower()
            if label in {POSITIVE_LABEL_TEXT, NEGATIVE_LABEL_TEXT}:
                return label
        except Exception:
            pass

    label_match = re.search(r'["\']?label["\']?\s*:\s*["\']?(spam|ham)["\']?', cleaned, flags=re.IGNORECASE)
    if label_match:
        return label_match.group(1).lower()

    # Last-resort parsing keeps the method comparable with the naive generation
    # baseline while still recording failures when neither label is recoverable.
    first_line = cleaned.lower().splitlines()[0].strip() if cleaned.strip() else ""
    if re.search(r"\bspam\b", first_line):
        return POSITIVE_LABEL_TEXT
    if re.search(r"\bham\b", first_line):
        return NEGATIVE_LABEL_TEXT
    return None


def apply_structured_patches() -> None:
    base.AIM_EXPERIMENT_NAME = AIM_EXPERIMENT_NAME
    base.build_sweep_configs = build_sweep_configs
    base.build_user_prompt = build_structured_user_prompt
    base.build_tokenized_sample = build_structured_tokenized_sample

    gen.AIM_EXPERIMENT_NAME = AIM_EXPERIMENT_NAME
    gen.EVALUATION_METHOD = EVALUATION_METHOD
    gen.DEFAULT_MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
    gen.DEFAULT_PARSE_FAILURE_LABEL = DEFAULT_PARSE_FAILURE_LABEL
    gen.default_results_root = default_results_root
    gen.resolve_results_root = resolve_results_root
    gen.make_sweep_id = make_sweep_id
    gen.parse_generated_label = parse_structured_generated_label


apply_structured_patches()


def write_manifest(sweep_dir: Path, sweep_id: str, selected_configs: list[Any], args: argparse.Namespace) -> Path:
    manifest_path = gen.write_manifest(sweep_dir, sweep_id, selected_configs, args)
    manifest = base.read_json(manifest_path)
    manifest["completion_format"] = COMPLETION_FORMAT
    manifest["structured_prompt"] = True
    base.write_json(manifest_path, manifest)
    return manifest_path


def run_sweep(args: argparse.Namespace) -> int:
    sweep_id = args.sweep_id or make_sweep_id()
    results_root = resolve_results_root(args.results_root)
    sweep_dir = results_root / "sweeps" / sweep_id
    selected_configs = [base.get_config(index) for index in base.parse_config_indexes(args.config_index)]

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(sweep_dir, sweep_id, selected_configs, args)
    summary_path = gen.write_summary(sweep_dir, selected_configs)

    print(f"Sweep ID: {sweep_id}")
    print(f"Method: {EVALUATION_METHOD}")
    print(f"Completion format: {COMPLETION_FORMAT}")
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
            gen.write_summary(sweep_dir, selected_configs)
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

        summary_path = gen.write_summary(sweep_dir, selected_configs)
        print(f"Updated summary: {summary_path}")

        if args.cooldown_seconds > 0 and config.index != selected_configs[-1].index:
            print(f"Cooling down for {args.cooldown_seconds:g} seconds.")
            time.sleep(args.cooldown_seconds)

    summary_path = gen.write_summary(sweep_dir, selected_configs)
    if failures:
        print(f"Sweep finished with {failures} failed config(s). Summary: {summary_path}")
    return 1 if failures else 0


def run_one(args: argparse.Namespace) -> int:
    apply_structured_patches()
    result = gen.run_one(args)
    if result == 0 and args.sweep_id:
        config = base.get_config(args.config_index)
        run_dir = base.config_run_dir(resolve_results_root(args.results_root) / "sweeps" / args.sweep_id, config)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = base.read_json(metrics_path)
            metrics["completion_format"] = COMPLETION_FORMAT
            metrics["structured_prompt"] = True
            base.write_json(metrics_path, metrics)
    return result


def summarize_command(args: argparse.Namespace) -> int:
    return gen.summarize_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = gen.build_parser()
    parser.description = "Run Qwen3 causal LM structured JSON generation sweeps."
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_structured_patches()
    args = build_parser().parse_args(argv)
    gen.validate_args(args)
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
