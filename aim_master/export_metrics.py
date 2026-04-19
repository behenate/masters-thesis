from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from aim import Repo

from .common import (
    DEFAULT_METRIC_NAMES,
    context_to_dict,
    context_to_label,
    ensure_exports_dir,
    ensure_master_repo_initialized,
    flatten_mapping,
    list_run_hashes_via_cli,
    load_manifest,
    metric_identity,
)
from .ui import choose_many, choose_one


DEFAULT_EXPORT_COLUMNS = [
    "run_hash",
    "metric_name",
    "context_label",
    "step",
    "epoch",
    "time",
    "value",
]


@dataclass(frozen=True)
class MetricChoice:
    identity: str
    label: str
    metric_name: str
    context_dict: dict[str, Any]
    metric: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected metrics from a run in the Aim master repo.")
    parser.add_argument("--master-repo", default=None, help="Override the default master repo path.")
    parser.add_argument("--run-hash", default=None, help="Skip interactive run selection and export this run hash.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument(
        "--include-run-metadata",
        action="store_true",
        help="Include expanded run metadata columns in the CSV. By default exports stay narrow and metric-focused.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Export every metric for the selected run without asking for extras.",
    )
    return parser.parse_args()


def choose_run(master_repo: Path, manifest: dict[str, dict[str, Any]]):
    run_hashes = list_run_hashes_via_cli(master_repo)
    if not run_hashes:
        raise RuntimeError("The master Aim repo does not contain any runs yet.")

    options: list[tuple[str, str]] = []
    for run_hash in run_hashes:
        source_repo = manifest.get(run_hash, {}).get("source_repo")
        label_parts = [run_hash]
        if source_repo:
            label_parts.append(f"source={source_repo}")
        options.append((run_hash, " | ".join(label_parts)))

    selected_hash = choose_one(
        "Select Run",
        "Choose a run from the master repo for CSV export.",
        options,
        default=options[0][0],
    )
    if selected_hash is None:
        raise RuntimeError("No run was selected.")
    return selected_hash


def collect_metric_choices(repo: Repo, run_hash: str) -> list[MetricChoice]:
    metrics = list(repo.query_metrics(f'run.hash == "{run_hash}"'))
    choices: list[MetricChoice] = []
    for metric in metrics:
        context_dict = context_to_dict(metric.context)
        context_label = context_to_label(metric.context)
        label = f"{metric.name} [{context_label}]"
        choices.append(
            MetricChoice(
                identity=metric_identity(metric.name, metric.context),
                label=label,
                metric_name=metric.name,
                context_dict=context_dict,
                metric=metric,
            )
        )
    return sorted(choices, key=lambda choice: (choice.metric_name, json.dumps(choice.context_dict, sort_keys=True)))


def choose_metric_set(metric_choices: list[MetricChoice], *, include_all_metrics: bool) -> tuple[list[MetricChoice], list[MetricChoice]]:
    default_choices = [choice for choice in metric_choices if choice.metric_name in DEFAULT_METRIC_NAMES]
    if not default_choices:
        default_choices = list(metric_choices)

    if include_all_metrics:
        return default_choices, list(metric_choices)

    default_ids = {choice.identity for choice in default_choices}
    extra_options = [(choice.identity, choice.label) for choice in metric_choices if choice.identity not in default_ids]
    selected_extra_ids = choose_many(
        "Extra Metrics",
        "The exporter will always include the default metric set below.\nUse Space to select any extra metrics to add.",
        extra_options,
    )

    selected_lookup = {choice.identity: choice for choice in metric_choices}
    selected_choices = list(default_choices)
    for identity in selected_extra_ids:
        choice = selected_lookup.get(identity)
        if choice and choice not in selected_choices:
            selected_choices.append(choice)

    return default_choices, selected_choices


def build_run_metadata(run, source_repo: Path) -> dict[str, Any]:
    metadata = {
        "run_hash": run.hash,
        "run_name": run.name,
        "experiment": run.experiment,
        "source_repo": str(source_repo),
    }

    for key in [
        "run_config",
        "dataset",
        "lora",
        "training_summary",
        "latest_checkpoint_path",
        "final_saved_model_path",
        "saved_tokenizer_path",
    ]:
        value = run.get(key, default=None, strict=False, resolve_objects=True)
        if value is not None:
            metadata[key] = value

    flattened = flatten_mapping("", metadata)
    return {key: flattened[key] for key in sorted(flattened)}


def export_metrics(
    run,
    source_repo: Path,
    metric_choices: list[MetricChoice],
    default_choices: list[MetricChoice],
    output_path: Path,
    *,
    include_run_metadata: bool,
) -> Path:
    run_metadata = build_run_metadata(run, source_repo) if include_run_metadata else {}
    frames: list[pd.DataFrame] = []

    for choice in metric_choices:
        frame = choice.metric.dataframe().copy()
        frame = frame.rename(columns={"idx": "sequence_index"})
        frame["run_hash"] = run.hash
        frame["metric_name"] = choice.metric_name
        frame["context_label"] = context_to_label(choice.context_dict)
        if include_run_metadata:
            frame["context_json"] = json.dumps(choice.context_dict, sort_keys=True)
            frame["exported_by_default"] = choice.identity in {c.identity for c in default_choices}
        for key, value in run_metadata.items():
            frame[key] = value
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    ordered_columns = [column for column in DEFAULT_EXPORT_COLUMNS if column in combined.columns]
    if include_run_metadata:
        remaining_columns = [column for column in combined.columns if column not in ordered_columns]
        combined = combined[ordered_columns + remaining_columns]
    else:
        combined = combined[ordered_columns]
    combined.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    args = parse_args()
    master_repo = ensure_master_repo_initialized(Path(args.master_repo) if args.master_repo else None)
    exports_dir = ensure_exports_dir()
    manifest = load_manifest()

    if args.run_hash:
        run_hash = args.run_hash
        if run_hash not in set(list_run_hashes_via_cli(master_repo)):
            raise RuntimeError(f"Run {run_hash} was not found in {master_repo}.")
    else:
        run_hash = choose_run(master_repo, manifest)

    run_entry = manifest.get(run_hash)
    if not run_entry or "source_repo" not in run_entry:
        raise RuntimeError(
            f"Run {run_hash} is present in the master repo, but no source repo mapping was found in the manifest."
        )

    source_repo = Path(run_entry["source_repo"]).expanduser().resolve()
    if not source_repo.exists():
        raise RuntimeError(
            f"The source repo for run {run_hash} is missing: {source_repo}. "
            "The exporter currently reads metrics from the original downloaded repo."
        )

    repo = Repo(str(source_repo))
    run = next((candidate for candidate in repo.iter_runs() if candidate.hash == run_hash), None)
    if run is None:
        raise RuntimeError(f"Run {run_hash} was not found in its recorded source repo: {source_repo}")

    metric_choices = collect_metric_choices(repo, run.hash)
    if not metric_choices:
        raise RuntimeError(f"Run {run.hash} does not contain any metrics.")

    default_choices, selected_choices = choose_metric_set(metric_choices, include_all_metrics=args.all_metrics)
    if not selected_choices:
        raise RuntimeError("No metrics were selected for export.")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = exports_dir / f"{run.hash}_{timestamp}.csv"

    exported_path = export_metrics(
        run,
        source_repo,
        selected_choices,
        default_choices,
        output_path,
        include_run_metadata=args.include_run_metadata,
    )

    print(f"Exported {len(selected_choices)} metric streams from run {run.hash}")
    print("Included by default:")
    for choice in default_choices:
        print(f"- {choice.label}")
    print("Final metric selection:")
    for choice in selected_choices:
        print(f"- {choice.label}")
    print(f"CSV written to: {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
