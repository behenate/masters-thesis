#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


EVALUATION_METHOD = "unmodified_zero_shot"
DEFAULT_TRAIN_SPLIT = 0.93
DEFAULT_VALIDATION_SPLIT = 0.02
DEFAULT_TEST_SPLIT = 0.05
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 8
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
gen = load_module("qwen3_unmodified_generation_helpers", generation_script)
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


def make_run_id() -> str:
    return "qwen3_0p6b_unmodified_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_raw_dataset(args: argparse.Namespace) -> tuple[Any, str]:
    from datasets import ClassLabel, load_dataset
    from dataset.combine import combine_datasets

    data_path = combine_datasets("training_all", combination_mode="mixed_50_50")
    raw_dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=["valid", "spam"]))
    return raw_dataset, str(data_path)


def select_split(raw_dataset: Any, args: argparse.Namespace) -> Any:
    holdout_split = args.validation_split + args.test_split
    holdout = raw_dataset.train_test_split(
        test_size=holdout_split,
        stratify_by_column="label",
        seed=args.seed,
    )
    valid_test = holdout["test"].train_test_split(
        test_size=args.test_split / holdout_split,
        stratify_by_column="label",
        seed=args.seed,
    )
    splits = {
        "train": holdout["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"],
    }
    return splits[args.split]


def tokenized_eval_dataset(tokenizer: Any, raw_split: Any, args: argparse.Namespace) -> Any:
    dataset = raw_split.filter(
        lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])),
        desc="Filtering empty emails",
    )
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = dataset.map(
        lambda sample: base.build_tokenized_sample(tokenizer, sample, args.max_seq_length),
        desc=f"Formatting prompts to <= {args.max_seq_length} tokens",
    )
    return dataset


def label_token_ids(tokenizer: Any) -> dict[str, int]:
    ids = {
        NEGATIVE_LABEL_TEXT: base.encode_text(tokenizer, NEGATIVE_LABEL_TEXT),
        POSITIVE_LABEL_TEXT: base.encode_text(tokenizer, POSITIVE_LABEL_TEXT),
    }
    for label_text, token_ids in ids.items():
        if len(token_ids) != 1:
            raise ValueError(f"Expected {label_text!r} to be one token, got {token_ids}")
    return {label_text: token_ids[0] for label_text, token_ids in ids.items()}


def device_and_dtype(torch_module: Any) -> tuple[Any, Any]:
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), torch_module.bfloat16
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps"), torch_module.float16
    return torch_module.device("cpu"), torch_module.float32


def prediction_records_from_generation(
    *,
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gen._TOKENIZER = tokenizer
    gen._GENERATION_MAX_NEW_TOKENS = args.max_new_tokens
    gen._PARSE_FAILURE_LABEL = args.parse_failure_label

    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    parse_failure_count = 0
    label_names = [NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT]

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    stop_token_ids = [im_end_id] if im_end_id is not None and im_end_id >= 0 else []
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    with torch_module.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            rows = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            batch = gen.pad_prompt_batch_left(torch_module, rows["prompt_input_ids"], tokenizer.pad_token_id, device)
            outputs = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = outputs[:, batch["input_ids"].shape[1] :]
            raw_generations = tokenizer.batch_decode(
                generated.detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for offset, (row, raw_generation) in enumerate(zip(rows, raw_generations, strict=False)):
                parsed_label = gen.parse_generated_label(raw_generation)
                parse_failed = parsed_label is None
                if parse_failed:
                    parse_failure_count += 1
                    parsed_label = args.parse_failure_label
                prediction = 1 if parsed_label == POSITIVE_LABEL_TEXT else 0
                actual = int(row["label"])
                predictions.append(prediction)
                labels.append(actual)
                records.append(
                    {
                        "sample_index": start + offset,
                        "method": "generation_parsing",
                        "actual": label_names[actual],
                        "predicted": label_names[prediction],
                        "correct": prediction == actual,
                        "parse_failed": parse_failed,
                        "raw_generation": raw_generation,
                        "subject": row.get("subject") or "",
                        "was_trimmed": bool(row["was_trimmed"]),
                        "token_length": int(row["token_length"]),
                    }
                )

    metrics = gen.compute_generation_metrics(predictions, labels, parse_failure_count)
    return metrics, records


def prediction_records_from_next_token(
    *,
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    dataset: Any,
    label_ids: dict[str, int],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[list[float]] = []
    records: list[dict[str, Any]] = []
    failure_count = 0
    label_names = [NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT]
    label_id_tensor = torch_module.tensor(
        [label_ids[NEGATIVE_LABEL_TEXT], label_ids[POSITIVE_LABEL_TEXT]],
        dtype=torch_module.long,
        device=device,
    )

    with torch_module.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            rows = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            batch = base.pad_prompt_batch(torch_module, rows["prompt_input_ids"], tokenizer.pad_token_id, device)
            outputs = model(**batch)
            last_positions = batch["attention_mask"].sum(dim=1) - 1
            next_token_logits = outputs.logits[torch_module.arange(len(rows), device=device), last_positions]
            label_logits = next_token_logits.index_select(dim=-1, index=label_id_tensor)
            finite = torch_module.isfinite(label_logits).all(dim=-1)
            safe_logits = torch_module.nan_to_num(label_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            batch_probabilities = torch_module.softmax(safe_logits, dim=-1)
            batch_predictions = batch_probabilities.argmax(dim=-1)

            failure_count += int((~finite).sum().item())
            batch_probabilities_list = batch_probabilities.detach().cpu().float().tolist()
            batch_predictions_list = batch_predictions.detach().cpu().int().tolist()
            for offset, row in enumerate(rows):
                actual = int(row["label"])
                prediction = int(batch_predictions_list[offset])
                prob_ham, prob_spam = batch_probabilities_list[offset]
                predictions.append(prediction)
                labels.append(actual)
                probabilities.append([prob_ham, prob_spam])
                records.append(
                    {
                        "sample_index": start + offset,
                        "method": "next_token",
                        "actual": label_names[actual],
                        "predicted": label_names[prediction],
                        "correct": prediction == actual,
                        "parse_failed": False,
                        "raw_generation": "",
                        "p_ham": prob_ham,
                        "p_spam": prob_spam,
                        "subject": row.get("subject") or "",
                        "was_trimmed": bool(row["was_trimmed"]),
                        "token_length": int(row["token_length"]),
                    }
                )

    metrics = base.compute_metrics(predictions, labels, probabilities, failure_count)
    return metrics, records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> int:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    results_root = resolve_results_root(args.results_root)
    run_id = args.run_id or make_run_id()
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ids = label_token_ids(tokenizer)

    raw_dataset, data_path = load_raw_dataset(args)
    dataset = tokenized_eval_dataset(tokenizer, select_split(raw_dataset, args), args)

    device, dtype = device_and_dtype(torch)
    print(f"Using device: {device}; dtype: {dtype}")
    print(f"Run ID: {run_id}")
    print(f"Samples: {len(dataset)}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.to(device)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    metrics: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    if args.mode in {"generation", "both"}:
        generation_metrics, generation_records = prediction_records_from_generation(
            model=model,
            torch_module=torch,
            tokenizer=tokenizer,
            dataset=dataset,
            args=args,
        )
        metrics["generation_parsing"] = generation_metrics
        all_records.extend(generation_records)
    if args.mode in {"next-token", "both"}:
        next_token_metrics, next_token_records = prediction_records_from_next_token(
            model=model,
            torch_module=torch,
            tokenizer=tokenizer,
            dataset=dataset,
            label_ids=ids,
            args=args,
        )
        metrics["next_token"] = next_token_metrics
        all_records.extend(next_token_records)

    payload = {
        "status": "completed",
        "run_id": run_id,
        "evaluation_method": EVALUATION_METHOD,
        "model_id": MODEL_ID,
        "mode": args.mode,
        "split": args.split,
        "limit": args.limit,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "test_split": args.test_split,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "parse_failure_default_label": args.parse_failure_label,
        "dataset_path": data_path,
        "sample_count": len(dataset),
        "label_token_ids": ids,
        "metrics": metrics,
        "run_dir": str(run_dir),
    }
    base.write_json(run_dir / "metrics.json", payload)
    write_csv(run_dir / "predictions.csv", all_records)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote: {run_dir / 'metrics.json'}")
    print(f"Wrote: {run_dir / 'predictions.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate unmodified Qwen3-0.6B as a zero-shot spam classifier.")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["generation", "next-token", "both"], default="both")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--test-split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--parse-failure-label", choices=[NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT], default=DEFAULT_PARSE_FAILURE_LABEL)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_seq_length < 64:
        raise SystemExit("--max-seq-length must be >= 64")
    total = args.train_split + args.validation_split + args.test_split
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("--train-split + --validation-split + --test-split must equal 1.0")
    if args.validation_split <= 0 or args.test_split <= 0:
        raise SystemExit("--validation-split and --test-split must be > 0")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
