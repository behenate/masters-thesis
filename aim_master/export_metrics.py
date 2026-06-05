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
from .ui import choose_many


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


@dataclass(frozen=True)
class RunExport:
    run: Any
    source_repo: Path
    default_choices: list[MetricChoice]
    selected_choices: list[MetricChoice]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected metrics from Aim runs in the master repo.")
    parser.add_argument("--master-repo", default=None, help="Override the default master repo path.")
    parser.add_argument(
        "--run-hash",
        action="append",
        default=[],
        help=(
            "Skip interactive run selection and export this run hash. "
            "Repeat the flag or pass comma-separated hashes to export multiple runs."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument(
        "--include-run-metadata",
        action="store_true",
        help="Include expanded run metadata columns in the CSV. By default exports stay narrow and metric-focused.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Export every metric for the selected run(s) without asking for extras.",
    )
    return parser.parse_args()


def expand_run_hash_args(raw_values: list[str]) -> list[str]:
    run_hashes: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for chunk in raw_value.split(","):
            run_hash = chunk.strip()
            if run_hash and run_hash not in seen:
                seen.add(run_hash)
                run_hashes.append(run_hash)
    return run_hashes


def collect_run_picker_metadata(run_hashes: list[str], manifest: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    hashes_by_source: dict[Path, set[str]] = {}
    for run_hash in run_hashes:
        source_repo = manifest.get(run_hash, {}).get("source_repo")
        if source_repo:
            hashes_by_source.setdefault(Path(source_repo).expanduser().resolve(), set()).add(run_hash)

    metadata: dict[str, dict[str, Any]] = {}
    for source_repo, source_hashes in hashes_by_source.items():
        if not source_repo.exists():
            continue
        try:
            repo = Repo(str(source_repo))
            for run in repo.iter_runs():
                if run.hash not in source_hashes:
                    continue
                metadata[run.hash] = {
                    "name": str(run.name) if run.name else "",
                    "creation_time": float(getattr(run, "creation_time", 0.0) or 0.0),
                }
        except Exception:
            continue
    return metadata


def build_run_options(master_repo: Path, manifest: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    run_hashes = list_run_hashes_via_cli(master_repo)
    if not run_hashes:
        raise RuntimeError("The master Aim repo does not contain any runs yet.")

    run_metadata = collect_run_picker_metadata(run_hashes, manifest)
    sorted_run_hashes = sorted(
        run_hashes,
        key=lambda run_hash: (run_metadata.get(run_hash, {}).get("creation_time", 0.0), run_hash),
        reverse=True,
    )
    options: list[tuple[str, str]] = []
    for run_hash in sorted_run_hashes:
        source_repo = manifest.get(run_hash, {}).get("source_repo")
        label_parts = [run_hash]
        run_name = run_metadata.get(run_hash, {}).get("name")
        if run_name:
            label_parts.append(f"name={run_name}")
        if source_repo:
            label_parts.append(f"source={source_repo}")
        options.append((run_hash, " | ".join(label_parts)))
    return options


def choose_runs(master_repo: Path, manifest: dict[str, dict[str, Any]]) -> list[str]:
    options = build_run_options(master_repo, manifest)
    selected_hashes = choose_many(
        "Select Runs",
        "Choose one or more runs from the master repo for CSV export.",
        options,
    )
    if not selected_hashes:
        raise RuntimeError("No runs were selected.")
    return selected_hashes


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


def choose_metric_identities(
    choices_by_run: dict[str, list[MetricChoice]],
    *,
    include_all_metrics: bool,
) -> tuple[set[str], set[str]]:
    all_choices = [choice for choices in choices_by_run.values() for choice in choices]
    default_ids = {choice.identity for choice in all_choices if choice.metric_name in DEFAULT_METRIC_NAMES}
    if not default_ids:
        default_ids = {choice.identity for choice in all_choices}

    if include_all_metrics:
        return default_ids, {choice.identity for choice in all_choices}

    option_labels: dict[str, str] = {}
    for choice in all_choices:
        if choice.identity not in default_ids:
            option_labels.setdefault(choice.identity, choice.label)

    extra_options = sorted(option_labels.items(), key=lambda item: item[1])
    selected_extra_ids = choose_many(
        "Extra Metrics",
        "The exporter will always include the default metric set below.\nUse Space to select any extra metrics to add.",
        extra_options,
    )
    return default_ids, default_ids | set(selected_extra_ids)


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


def export_run_frames(
    run,
    source_repo: Path,
    metric_choices: list[MetricChoice],
    default_choices: list[MetricChoice],
    *,
    include_run_metadata: bool,
) -> list[pd.DataFrame]:
    run_metadata = build_run_metadata(run, source_repo) if include_run_metadata else {}
    frames: list[pd.DataFrame] = []
    default_ids = {c.identity for c in default_choices}

    for choice in metric_choices:
        frame = choice.metric.dataframe().copy()
        frame = frame.rename(columns={"idx": "sequence_index"})
        frame["run_hash"] = run.hash
        frame["metric_name"] = choice.metric_name
        frame["context_label"] = context_to_label(choice.context_dict)
        if include_run_metadata:
            frame["context_json"] = json.dumps(choice.context_dict, sort_keys=True)
            frame["exported_by_default"] = choice.identity in default_ids
        for key, value in run_metadata.items():
            frame[key] = value
        frames.append(frame)

    return frames


def export_metrics(
    run_exports: list[RunExport],
    output_path: Path,
    *,
    include_run_metadata: bool,
) -> Path:
    frames: list[pd.DataFrame] = []
    for run_export in run_exports:
        frames.extend(
            export_run_frames(
                run_export.run,
                run_export.source_repo,
                run_export.selected_choices,
                run_export.default_choices,
                include_run_metadata=include_run_metadata,
            )
        )

    if not frames:
        raise RuntimeError("No metric data was collected for export.")

    combined = pd.concat(frames, ignore_index=True)
    ordered_columns = [column for column in DEFAULT_EXPORT_COLUMNS if column in combined.columns]
    if include_run_metadata:
        remaining_columns = [column for column in combined.columns if column not in ordered_columns]
        combined = combined[ordered_columns + remaining_columns]
    else:
        combined = combined[ordered_columns]
    combined.to_csv(output_path, index=False)
    return output_path


def load_run_from_source(run_hash: str, manifest: dict[str, dict[str, Any]]) -> tuple[Any, Path, Repo]:
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

    return run, source_repo, repo


def main() -> int:
    args = parse_args()
    master_repo = ensure_master_repo_initialized(Path(args.master_repo) if args.master_repo else None)
    exports_dir = ensure_exports_dir()
    manifest = load_manifest()

    available_run_hashes = set(list_run_hashes_via_cli(master_repo))
    if args.run_hash:
        run_hashes = expand_run_hash_args(args.run_hash)
        if not run_hashes:
            raise RuntimeError("No run hashes were provided.")
        missing_hashes = [run_hash for run_hash in run_hashes if run_hash not in available_run_hashes]
        if missing_hashes:
            raise RuntimeError(f"Run(s) not found in {master_repo}: {', '.join(missing_hashes)}")
    else:
        run_hashes = choose_runs(master_repo, manifest)

    loaded_runs: dict[str, tuple[Any, Path, Repo]] = {}
    choices_by_run: dict[str, list[MetricChoice]] = {}
    for run_hash in run_hashes:
        run, source_repo, repo = load_run_from_source(run_hash, manifest)
        loaded_runs[run_hash] = (run, source_repo, repo)
        metric_choices = collect_metric_choices(repo, run.hash)
        if not metric_choices:
            if len(run_hashes) == 1:
                raise RuntimeError(f"Run {run.hash} does not contain any metrics.")
            print(f"Skipping run {run.hash}: no metrics found.")
            continue
        choices_by_run[run_hash] = metric_choices

    if not choices_by_run:
        raise RuntimeError("None of the selected runs contain metrics.")

    if len(choices_by_run) == 1:
        run_hashes_with_metrics = list(choices_by_run)
        run_hash = run_hashes_with_metrics[0]
        default_choices, selected_choices = choose_metric_set(
            choices_by_run[run_hash],
            include_all_metrics=args.all_metrics,
        )
        if not selected_choices:
            raise RuntimeError("No metrics were selected for export.")
        run, source_repo, _ = loaded_runs[run_hash]
        run_exports = [
            RunExport(
                run=run,
                source_repo=source_repo,
                default_choices=default_choices,
                selected_choices=selected_choices,
            )
        ]
    else:
        default_ids, selected_ids = choose_metric_identities(
            choices_by_run,
            include_all_metrics=args.all_metrics,
        )
        run_exports = []
        for run_hash in choices_by_run:
            run, source_repo, _ = loaded_runs[run_hash]
            metric_choices = choices_by_run[run_hash]
            default_choices = [choice for choice in metric_choices if choice.identity in default_ids]
            selected_choices = [choice for choice in metric_choices if choice.identity in selected_ids]
            if selected_choices:
                run_exports.append(
                    RunExport(
                        run=run,
                        source_repo=source_repo,
                        default_choices=default_choices,
                        selected_choices=selected_choices,
                    )
                )
        if not run_exports:
            raise RuntimeError("No metrics were selected for export.")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(run_hashes) == 1:
            output_path = exports_dir / f"{run_hashes[0]}_{timestamp}.csv"
        else:
            output_path = exports_dir / f"runs_{len(run_hashes)}_{timestamp}.csv"

    exported_path = export_metrics(
        run_exports,
        output_path,
        include_run_metadata=args.include_run_metadata,
    )

    print(f"Exported metrics from {len(run_exports)} run(s)")
    for run_export in run_exports:
        print(f"- {run_export.run.hash}: {len(run_export.selected_choices)} metric stream(s)")
    print("Included by default:")
    for label in sorted({choice.label for run_export in run_exports for choice in run_export.default_choices}):
        print(f"- {label}")
    print("Final metric selection:")
    for label in sorted({choice.label for run_export in run_exports for choice in run_export.selected_choices}):
        print(f"- {label}")
    print(f"CSV written to: {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
