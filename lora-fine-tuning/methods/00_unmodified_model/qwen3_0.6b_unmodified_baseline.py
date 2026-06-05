#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


EVALUATION_METHOD = "unmodified_zero_shot"
DEFAULT_DATASETS = ("train_subset", "enron", "fraudulent_email_corpus", "spam_ham")
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_PARSE_FAILURE_LABEL = "ham"
DEFAULT_PROMPT_STYLE = "defined-labels"
SUMMARY_COLUMNS = [
    "dataset",
    "method",
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
    "parse_failure_count",
    "parse_failure_rate",
    "eval_batch_size",
    "max_seq_length",
    "model_id",
    "checkpoint_path",
    "runtime_seconds",
    "status",
    "error_message",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


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


def base_sweep_path() -> Path:
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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find qwen3_0.6b_casual_lm_sweep.py. Tried:\n{attempted}")


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
    return load_module("qwen3_next_token_sweep_helpers", base_sweep_path())


base = load_base_module()
MODEL_ID = base.MODEL_ID
SEED = base.SEED
POSITIVE_LABEL_TEXT = base.POSITIVE_LABEL_TEXT
NEGATIVE_LABEL_TEXT = base.NEGATIVE_LABEL_TEXT
IM_END_TOKEN = base.IM_END_TOKEN
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def default_results_root() -> Path:
    return method_dir() / "results"


def resolve_results_root(value: str | None) -> Path:
    if value is None:
        return default_results_root()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def make_run_id() -> str:
    return "qwen3_0p6b_unmodified_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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


def build_dataset_sample(dataset_name: str, sample_limit: int, seed: int) -> tuple[Any, dict[str, Any]]:
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
        dataset = stratified_sample(holdout["train"], sample_limit, seed)
    else:
        data_path = combine_datasets(dataset_name, duplicate_detection="high")
        dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        dataset = dataset.filter(lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])))
        if dataset_name == "spam_ham":
            dataset = stratified_sample(dataset, sample_limit, seed)
        else:
            dataset = plain_sample(dataset, sample_limit, seed)

    ham, spam = label_counts(dataset)
    metadata = {
        "dataset": dataset_name,
        "source_path": str(data_path),
        "rows": len(dataset),
        "ham_count": ham,
        "spam_count": spam,
    }
    return dataset, metadata


def build_baseline_user_prompt(email_text: str, prompt_style: str) -> str:
    if prompt_style == "training-compatible":
        return base.build_user_prompt(email_text)
    if prompt_style != "defined-labels":
        raise ValueError(f"Unknown prompt_style={prompt_style!r}")
    return (
        "You are an email security classifier.\n"
        "Classify the email as exactly one of two labels.\n\n"
        "Definitions:\n"
        "- spam: unsolicited advertising, scam, phishing, fraud, suspicious offer, malware, or otherwise unwanted email.\n"
        "- ham: legitimate non-spam email.\n\n"
        "Return only one lowercase word: spam or ham.\n\n"
        "Email:\n"
        f"{email_text}"
    )


def apply_baseline_chat_template(tokenizer: Any, email_text: str, prompt_style: str) -> str:
    messages = [{"role": "user", "content": build_baseline_user_prompt(email_text.strip(), prompt_style)}]
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


def trim_email_to_fit(
    *,
    tokenizer: Any,
    email_text: str,
    completion_text: str,
    max_seq_length: int,
    prompt_style: str,
) -> dict[str, Any]:
    completion_ids = base.encode_text(tokenizer, completion_text)
    empty_prompt_ids = base.encode_text(tokenizer, apply_baseline_chat_template(tokenizer, "", prompt_style))
    email_ids = base.encode_text(tokenizer, email_text)
    available_email_tokens = max_seq_length - len(empty_prompt_ids) - len(completion_ids)

    if available_email_tokens < 1:
        raise ValueError(
            f"max_seq_length={max_seq_length} is too small; prompt overhead plus completion uses "
            f"{len(empty_prompt_ids) + len(completion_ids)} tokens."
        )

    marker_text = getattr(base, "TRUNCATION_MARKER", "\n[...]\n")
    marker_ids = base.encode_text(tokenizer, marker_text)
    candidate_budget = min(len(email_ids), available_email_tokens)
    was_trimmed = len(email_ids) > available_email_tokens

    while candidate_budget >= 0:
        candidate_ids = base.trim_ids_head_tail(email_ids, candidate_budget, marker_ids)
        candidate_text = base.decode_ids(tokenizer, candidate_ids)
        prompt_ids = base.encode_text(tokenizer, apply_baseline_chat_template(tokenizer, candidate_text, prompt_style))
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


def build_tokenized_sample(
    tokenizer: Any,
    sample: dict[str, Any],
    max_seq_length: int,
    prompt_style: str,
) -> dict[str, Any]:
    email_text = base.build_email_text(sample["subject"], sample["body"])
    label_text = base.label_to_text(int(sample["label"]))
    completion_text = f"{label_text}{IM_END_TOKEN}"
    trimmed = trim_email_to_fit(
        tokenizer=tokenizer,
        email_text=email_text,
        completion_text=completion_text,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
    )

    prompt_text = apply_baseline_chat_template(tokenizer, trimmed["text"], prompt_style)
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


def tokenized_eval_dataset(tokenizer: Any, dataset: Any, max_seq_length: int, prompt_style: str) -> Any:
    dataset = dataset.filter(lambda sample: bool(base.build_email_text(sample["subject"], sample["body"])))
    return dataset.map(
        lambda sample: build_tokenized_sample(tokenizer, sample, max_seq_length, prompt_style),
        desc=f"Formatting prompts to <= {max_seq_length} tokens",
    )


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


def pad_prompt_batch_left(torch_module: Any, prompt_batches: list[list[int]], pad_token_id: int, device: Any) -> dict[str, Any]:
    max_length = max(len(ids) for ids in prompt_batches)
    input_ids = torch_module.full((len(prompt_batches), max_length), pad_token_id, dtype=torch_module.long, device=device)
    attention_mask = torch_module.zeros((len(prompt_batches), max_length), dtype=torch_module.long, device=device)
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


def strip_generation(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub(" ", text)
    cleaned = cleaned.replace(IM_END_TOKEN, " ")
    cleaned = cleaned.replace("<|endoftext|>", " ")
    cleaned = cleaned.replace("```json", " ")
    cleaned = cleaned.replace("```", " ")
    return cleaned.strip()


def normalize_label(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", value.lower()).strip()
    if normalized in {"spam", "junk", "phishing", "scam", "malicious"}:
        return POSITIVE_LABEL_TEXT
    if normalized in {"ham", "valid", "legitimate", "not spam", "non spam", "safe", "clean"}:
        return NEGATIVE_LABEL_TEXT
    return None


def parse_json_label(text: str) -> tuple[str | None, str | None]:
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        candidate = match.group(0)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            for key in ("label", "classification", "class", "result", "answer"):
                if key in payload:
                    label = normalize_label(str(payload[key]))
                    if label is not None:
                        return label, f"json.{key}"
    return None, None


def parse_generated_label(text: str) -> tuple[str | None, str]:
    cleaned = strip_generation(text)
    lowered = cleaned.lower()
    first_line = lowered.splitlines()[0].strip() if lowered else ""
    label, rule = parse_json_label(cleaned)
    if label is not None:
        return label, rule or "json"

    field_match = re.search(
        r"\b(?:label|classification|class|result|answer)\b\s*[:=\-]\s*[\"']?([a-z][a-z\-\s]{0,24})",
        lowered,
    )
    if field_match:
        label = normalize_label(field_match.group(1))
        if label is not None:
            return label, "field"

    negative_spam_patterns = [
        r"\bnot\s+spam\b",
        r"\bnon[-\s]?spam\b",
        r"\bno\s+spam\b",
        r"\bnot\s+a\s+spam\b",
        r"\blegitimate\b",
        r"\bvalid\b",
        r"\bsafe\b",
        r"\bclean\b",
    ]
    positive_spam_patterns = [
        r"\bspam\b",
        r"\bjunk\b",
        r"\bphishing\b",
        r"\bscam\b",
        r"\bmalicious\b",
        r"\bunsolicited\b",
    ]
    ham_patterns = [r"\bham\b", r"\bnot\s+phishing\b", r"\bnot\s+malicious\b"]

    for pattern in negative_spam_patterns:
        if re.search(pattern, first_line) or re.search(pattern, lowered[:400]):
            return NEGATIVE_LABEL_TEXT, "negative_spam_phrase"
    for pattern in ham_patterns:
        if re.search(pattern, first_line) or re.search(pattern, lowered[:400]):
            return NEGATIVE_LABEL_TEXT, "ham_phrase"
    for pattern in positive_spam_patterns:
        if re.search(pattern, first_line) or re.search(pattern, lowered[:400]):
            return POSITIVE_LABEL_TEXT, "spam_phrase"

    if re.search(r"\byes\b", first_line):
        return POSITIVE_LABEL_TEXT, "yes_no"
    if re.search(r"\bno\b", first_line):
        return NEGATIVE_LABEL_TEXT, "yes_no"
    return None, "unparsed"


def binary_metrics(
    predictions: list[int],
    labels: list[int],
    failure_count: int,
    probabilities: list[list[float]] | None = None,
) -> dict[str, float]:
    del probabilities
    rows = len(labels)
    tp = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels, strict=False) if pred == 0 and label == 0)
    accuracy = (tp + tn) / max(rows, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (recall + specificity) / 2
    spam_predictions = sum(1 for value in predictions if value == 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "spam_prediction_rate": float(spam_predictions / max(rows, 1)),
        "classification_failure_count": float(failure_count),
        "classification_failure_rate": float(failure_count / max(rows, 1)),
    }


def prediction_records_from_generation(
    *,
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
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
    if not stop_token_ids:
        stop_token_ids = None

    total_batches = math.ceil(len(dataset) / max(args.batch_size, 1))
    with torch_module.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            batch_index = start // args.batch_size + 1
            log(f"{args.current_dataset}/generation batch {batch_index}/{total_batches}")
            rows = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            batch = pad_prompt_batch_left(torch_module, rows["prompt_input_ids"], tokenizer.pad_token_id, device)
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
                parsed_label, parse_rule = parse_generated_label(raw_generation)
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
                        "dataset": args.current_dataset,
                        "row_index": start + offset,
                        "method": "generation_parsing",
                        "label": actual,
                        "prediction": prediction,
                        "label_text": label_names[actual],
                        "prediction_text": label_names[prediction],
                        "correct": prediction == actual,
                        "parse_failed": parse_failed,
                        "parse_rule": parse_rule,
                        "raw_generation": raw_generation,
                        "subject": row.get("subject") or "",
                        "was_trimmed": bool(row["was_trimmed"]),
                        "token_length": int(row["token_length"]),
                    }
                )
            del batch, outputs, generated

    metrics = binary_metrics(predictions, labels, parse_failure_count)
    metrics["parse_failure_count"] = float(parse_failure_count)
    metrics["parse_failure_rate"] = float(parse_failure_count / max(len(labels), 1))
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
    use_logits_to_keep: bool | None = None

    total_batches = math.ceil(len(dataset) / max(args.batch_size, 1))
    with torch_module.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            batch_index = start // args.batch_size + 1
            log(f"{args.current_dataset}/next-token batch {batch_index}/{total_batches}")
            rows = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            batch = pad_prompt_batch_left(torch_module, rows["prompt_input_ids"], tokenizer.pad_token_id, device)
            next_token_logits, logits_to_keep_was_used = forward_last_token_logits(model, batch, use_logits_to_keep)
            if use_logits_to_keep is None:
                use_logits_to_keep = logits_to_keep_was_used
                if logits_to_keep_was_used:
                    log("Using logits_to_keep=1 for next-token baseline.")
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
                        "dataset": args.current_dataset,
                        "row_index": start + offset,
                        "method": "next_token",
                        "label": actual,
                        "prediction": prediction,
                        "label_text": label_names[actual],
                        "prediction_text": label_names[prediction],
                        "correct": prediction == actual,
                        "parse_failed": False,
                        "parse_rule": "",
                        "raw_generation": "",
                        "ham_probability": prob_ham,
                        "spam_probability": prob_spam,
                        "subject": row.get("subject") or "",
                        "was_trimmed": bool(row["was_trimmed"]),
                        "token_length": int(row["token_length"]),
                    }
                )
            del batch, next_token_logits, label_logits, safe_logits, batch_probabilities, batch_predictions

    metrics = base.compute_metrics(predictions, labels, probabilities, failure_count)
    extra = binary_metrics(predictions, labels, failure_count, probabilities)
    metrics.update(
        {
            "false_positive_rate": extra["false_positive_rate"],
            "false_negative_rate": extra["false_negative_rate"],
            "spam_prediction_rate": extra["spam_prediction_rate"],
            "parse_failure_count": 0.0,
            "parse_failure_rate": 0.0,
        }
    )
    return metrics, records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_row(
    *,
    dataset_name: str,
    method: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    runtime_seconds: float,
    status: str = "completed",
    error_message: str = "",
) -> dict[str, Any]:
    row = {
        "dataset": dataset_name,
        "method": method,
        "config_index": 0,
        "config_id": "unmodified_model",
        "checkpoint_type": "base_model",
        "checkpoint_step": "none",
        "rows": metadata.get("rows", 0),
        "ham_count": metadata.get("ham_count", 0),
        "spam_count": metadata.get("spam_count", 0),
        "eval_batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "model_id": MODEL_ID,
        "checkpoint_path": MODEL_ID,
        "runtime_seconds": round(runtime_seconds, 4),
        "status": status,
        "error_message": error_message,
    }
    for key in SUMMARY_COLUMNS:
        row.setdefault(key, metrics.get(key, ""))
    return row


def evaluate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    project_root()
    set_seed(args.seed)
    results_root = resolve_results_root(args.results_root)
    run_id = args.run_id or make_run_id()
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ids = label_token_ids(tokenizer)

    device, dtype = device_and_dtype(torch)
    log(f"Using device: {device}; dtype: {dtype}")
    log(f"Run ID: {run_id}")
    log(f"Datasets: {', '.join(args.datasets)}")
    log(f"Sample limit per dataset: {args.sample_limit}")
    log(f"Prompt style: {args.prompt_style}")

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
    dataset_metadata: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        log(f"Preparing dataset sample: {dataset_name}")
        raw_dataset, metadata = build_dataset_sample(dataset_name, args.sample_limit, args.seed)
        dataset_metadata[dataset_name] = metadata
        dataset = tokenized_eval_dataset(tokenizer, raw_dataset, args.max_seq_length, args.prompt_style)
        metadata["rows"] = len(dataset)
        metadata["ham_count"], metadata["spam_count"] = label_counts(dataset)
        metrics[dataset_name] = {}

        args.current_dataset = dataset_name
        for method in (["generation_parsing"] if args.mode == "generation" else ["next_token"] if args.mode == "next-token" else ["generation_parsing", "next_token"]):
            started = time.perf_counter()
            try:
                if method == "generation_parsing":
                    method_metrics, records = prediction_records_from_generation(
                        model=model,
                        torch_module=torch,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        args=args,
                    )
                else:
                    method_metrics, records = prediction_records_from_next_token(
                        model=model,
                        torch_module=torch,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        label_ids=ids,
                        args=args,
                    )
                runtime_seconds = time.perf_counter() - started
                metrics[dataset_name][method] = method_metrics
                summary_rows.append(
                    summary_row(
                        dataset_name=dataset_name,
                        method=method,
                        metadata=metadata,
                        metrics=method_metrics,
                        args=args,
                        runtime_seconds=runtime_seconds,
                    )
                )
                if args.write_predictions:
                    write_csv(run_dir / "predictions" / f"{dataset_name}_{method}.csv", records)
            except Exception as exc:
                runtime_seconds = time.perf_counter() - started
                summary_rows.append(
                    summary_row(
                        dataset_name=dataset_name,
                        method=method,
                        metadata=metadata,
                        metrics={},
                        args=args,
                        runtime_seconds=runtime_seconds,
                        status="failed",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                if not args.keep_going:
                    raise

    payload = {
        "status": "completed",
        "run_id": run_id,
        "evaluation_method": EVALUATION_METHOD,
        "model_id": MODEL_ID,
        "mode": args.mode,
        "datasets": list(args.datasets),
        "sample_limit": args.sample_limit,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "prompt_template": build_baseline_user_prompt("{email}", args.prompt_style),
        "parse_failure_default_label": args.parse_failure_label,
        "dataset_metadata": dataset_metadata,
        "label_token_ids": ids,
        "metrics": metrics,
        "summary_path": str(run_dir / "summary.csv"),
        "run_dir": str(run_dir),
    }
    base.write_json(run_dir / "metrics.json", payload)
    write_csv(run_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    log(f"Wrote: {run_dir / 'metrics.json'}")
    log(f"Wrote: {run_dir / 'summary.csv'}")
    if args.write_predictions:
        log(f"Wrote predictions to: {run_dir / 'predictions'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate unmodified Qwen3-0.6B as a zero-shot spam classifier.")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["generation", "next-token", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-limit", "--limit", dest="sample_limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--prompt-style",
        choices=["defined-labels", "training-compatible"],
        default=DEFAULT_PROMPT_STYLE,
        help="Prompt used for the unmodified model. Use training-compatible to reproduce the method-03 prompt.",
    )
    parser.add_argument("--parse-failure-label", choices=[NEGATIVE_LABEL_TEXT, POSITIVE_LABEL_TEXT], default=DEFAULT_PARSE_FAILURE_LABEL)
    parser.add_argument("--write-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_seq_length < 64:
        raise SystemExit("--max-seq-length must be >= 64")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
