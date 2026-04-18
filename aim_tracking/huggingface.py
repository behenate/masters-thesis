from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from aim.hugging_face import AimCallback
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import TrainerCallback


def _normalize_aim_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_aim_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_aim_value(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_text_classification_dataset(dataset) -> dict[str, Any]:
    label_counts = {"spam": 0, "ham": 0}
    source_counts: dict[str, int] = {}
    subject_char_total = 0
    body_char_total = 0

    for sample in dataset:
        label_counts["spam" if sample["label"] == 1 else "ham"] += 1
        source = sample.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        subject_char_total += len(sample.get("subject", ""))
        body_char_total += len(sample.get("body", ""))

    row_count = len(dataset)
    return {
        "label_counts": label_counts,
        "source_counts": source_counts,
        "text_stats": {
            "avg_subject_chars": subject_char_total / max(row_count, 1),
            "avg_body_chars": body_char_total / max(row_count, 1),
        },
    }


def compute_binary_classification_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    shifted_logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    spam_probabilities = probabilities[:, 1]
    predictions = np.argmax(probabilities, axis=-1)

    accuracy = float((predictions == labels).mean())
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())

    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = (per_class_recall[1] + specificity) / 2
    prediction_entropy = float(
        -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=-1).mean()
    )
    spam_prediction_rate = float((predictions == 1).mean())
    ham_prediction_rate = float((predictions == 0).mean())
    mean_spam_probability = float(spam_probabilities.mean())
    mean_spam_probability_on_spam = float(spam_probabilities[labels == 1].mean()) if np.any(labels == 1) else 0.0
    mean_spam_probability_on_ham = float(spam_probabilities[labels == 0].mean()) if np.any(labels == 0) else 0.0
    brier_score = float(np.mean((spam_probabilities - labels) ** 2))
    absolute_probability_error = float(np.mean(np.abs(spam_probabilities - labels)))
    matthews = float(matthews_corrcoef(labels, predictions)) if len(np.unique(labels)) > 1 else 0.0

    try:
        roc_auc = float(roc_auc_score(labels, spam_probabilities))
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(labels, spam_probabilities))
    except ValueError:
        pr_auc = 0.0

    try:
        cross_entropy = float(log_loss(labels, probabilities, labels=[0, 1]))
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
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "ham_precision": float(per_class_precision[0]),
        "ham_recall": float(per_class_recall[0]),
        "ham_f1": float(per_class_f1[0]),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "matthews_corrcoef": matthews,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cross_entropy": cross_entropy,
        "brier_score": brier_score,
        "mean_abs_probability_error": absolute_probability_error,
        "prediction_entropy": prediction_entropy,
        "spam_prediction_rate": spam_prediction_rate,
        "ham_prediction_rate": ham_prediction_rate,
        "mean_spam_probability": mean_spam_probability,
        "mean_spam_probability_on_spam": mean_spam_probability_on_spam,
        "mean_spam_probability_on_ham": mean_spam_probability_on_ham,
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "true_positive_count": float(tp),
        "true_negative_count": float(tn),
    }


class NotebookAimCallback(TrainerCallback):
    def __init__(
        self,
        hf_aim_callback: AimCallback,
        *,
        run_config: dict[str, Any],
        dataset_metadata: dict[str, Any],
        lora_metadata: dict[str, Any],
    ) -> None:
        self.hf_aim_callback = hf_aim_callback
        self.run_config = _normalize_aim_value(run_config)
        self.dataset_metadata = _normalize_aim_value(dataset_metadata)
        self.lora_metadata = _normalize_aim_value(lora_metadata)
        self.checkpoint_history: list[str] = []

    @property
    def run(self):
        return self.hf_aim_callback.experiment

    def on_train_begin(self, args, state, control, **kwargs):
        self.run["run_config"] = self.run_config
        self.run["dataset"] = self.dataset_metadata
        self.run["lora"] = self.lora_metadata

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        for name, value in logs.items():
            if name.startswith("eval_") or not isinstance(value, (int, float)):
                continue
            self.run.track(
                float(value),
                name=name,
                step=state.global_step,
                epoch=state.epoch,
                context={"subset": "train"},
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            metric_name = name[5:] if name.startswith("eval_") else name
            self.run.track(
                float(value),
                name=metric_name,
                step=state.global_step,
                epoch=state.epoch,
                context={"subset": "validation"},
            )

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.checkpoint_history.append(checkpoint_path)
        self.run["latest_checkpoint_path"] = checkpoint_path
        self.run["checkpoint_history"] = self.checkpoint_history
        self.run.track(
            float(len(self.checkpoint_history)),
            name="checkpoint_count",
            step=state.global_step,
            epoch=state.epoch,
            context={"subset": "artifacts"},
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.run["training_summary"] = _normalize_aim_value(
            {
                "best_metric": state.best_metric,
                "best_global_step": state.best_global_step,
                "best_model_checkpoint": state.best_model_checkpoint,
                "global_step": state.global_step,
                "log_history_entries": len(state.log_history),
                "output_dir": args.output_dir,
            }
        )


def create_aim_callbacks(
    *,
    repo_path: str,
    experiment_name: str,
    system_tracking_interval: int,
    run_config: dict[str, Any],
    dataset_metadata: dict[str, Any],
    lora_metadata: dict[str, Any],
) -> tuple[AimCallback, NotebookAimCallback]:
    aim_callback = AimCallback(
        repo=repo_path,
        experiment=experiment_name,
        system_tracking_interval=system_tracking_interval,
        log_system_params=True,
        capture_terminal_logs=False,
    )
    notebook_callback = NotebookAimCallback(
        aim_callback,
        run_config=run_config,
        dataset_metadata=dataset_metadata,
        lora_metadata=lora_metadata,
    )
    return aim_callback, notebook_callback
