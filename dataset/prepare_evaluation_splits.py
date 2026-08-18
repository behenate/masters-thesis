#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, disable_progress_bars, load_dataset

try:
    import combine
    from analyze_trec_2006_duplicates import (
        MIN_NORMALIZED_BODY_LENGTH,
        _canonical_exact_text,
        analyze as analyze_trec_2006,
    )
except ImportError:  # pragma: no cover - allows package-style imports
    from . import combine
    from .analyze_trec_2006_duplicates import (
        MIN_NORMALIZED_BODY_LENGTH,
        _canonical_exact_text,
        analyze as analyze_trec_2006,
    )


SEED = 67
EXT_VALID_SIZE = 1000
FINAL_TEST_SIZE = 1000
EXTERNAL_SOURCES = ("enron", "fraudulent_email_corpus", "spam_ham")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CHECKSUMS_PATH = Path(__file__).resolve().parent / "raw_datasets" / "SHA256SUMS"
EXPECTED_SPLIT_HASHES_PATH = Path(__file__).resolve().parent / "evaluation_split_expected_hashes.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_splits" / f"seed_{SEED}"
STANDARD_COLUMNS = [
    "subject",
    "body",
    "label",
    "source",
    "source_row_index",
    "message_sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def verify_raw_dataset_checksums() -> list[dict[str, str]]:
    if not RAW_CHECKSUMS_PATH.is_file():
        raise RuntimeError(f"Missing raw dataset checksum file: {RAW_CHECKSUMS_PATH}")

    verified = []
    raw_root = RAW_CHECKSUMS_PATH.parent
    for line in RAW_CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_sha256, relative_name = line.split(maxsplit=1)
        source_path = raw_root / relative_name
        if not source_path.is_file():
            raise RuntimeError(f"Expected downloaded dataset file is missing: {source_path}")
        actual_sha256 = sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Raw dataset checksum mismatch for {relative_name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        verified.append({"path": f"dataset/raw_datasets/{relative_name}", "sha256": actual_sha256})
    return verified


def verify_expected_split_hashes(
    split_entries: list[dict[str, Any]],
    seed: int,
    ext_valid_size: int,
    test_size: int,
) -> None:
    if (seed, ext_valid_size, test_size) != (SEED, EXT_VALID_SIZE, FINAL_TEST_SIZE):
        return
    expected = json.loads(EXPECTED_SPLIT_HASHES_PATH.read_text(encoding="utf-8"))
    actual = {entry["name"]: entry["content_sha256"] for entry in split_entries}
    if actual != expected["content_sha256"]:
        raise RuntimeError(
            "Generated split content does not match the checked-in seed-67 reference hashes"
        )


def message_fingerprint(subject: object, body: object) -> str:
    exact_text = _canonical_exact_text(subject, body)
    exact_sha256 = hashlib.sha256(exact_text.encode("utf-8", errors="replace")).hexdigest()
    normalized_body = combine._normalized_body_fingerprint_high(str(body))
    if len(normalized_body) < MIN_NORMALIZED_BODY_LENGTH:
        return f"subject_body:{exact_sha256}"
    return hashlib.sha256(normalized_body.encode("utf-8", errors="replace")).hexdigest()


def row_fingerprint(row: dict[str, Any]) -> str:
    existing = row.get("message_sha256") or row.get("dedup_sha256")
    if existing:
        return str(existing)
    return message_fingerprint(row.get("subject", ""), row.get("body", ""))


def add_message_ids(dataset: Dataset) -> Dataset:
    if "message_sha256" in dataset.column_names:
        return dataset
    fingerprints = [
        message_fingerprint(subject, body)
        for subject, body in zip(dataset["subject"], dataset["body"], strict=True)
    ]
    return dataset.add_column("message_sha256", fingerprints)


def normalize_columns(dataset: Dataset) -> Dataset:
    if "source_row_index" not in dataset.column_names:
        dataset = dataset.add_column("source_row_index", list(range(len(dataset))))
    dataset = add_message_ids(dataset)
    missing = [column for column in STANDARD_COLUMNS if column not in dataset.column_names]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")
    return dataset.select_columns(STANDARD_COLUMNS)


def load_source_dataset(source: str) -> tuple[Dataset, Path]:
    source_path = Path(combine.combine_datasets(source, duplicate_detection="high")).resolve()
    dataset = load_dataset("parquet", data_files=str(source_path), split="train")
    dataset = dataset.filter(
        lambda row: bool(str(row.get("subject") or "").strip() or str(row.get("body") or "").strip())
    )
    return normalize_columns(dataset), source_path


def label_counts(dataset: Dataset) -> dict[str, int]:
    labels = [int(label) for label in dataset["label"]]
    return {
        "ham": sum(label == 0 for label in labels),
        "spam": sum(label == 1 for label in labels),
    }


def stratified_take_counts(dataset: Dataset, limit: int) -> dict[int, int]:
    labels = [int(label) for label in dataset["label"]]
    total = len(labels)
    remaining = limit
    counts: dict[int, int] = {}
    label_values = sorted(set(labels))
    for offset, label in enumerate(label_values):
        available = labels.count(label)
        if offset == len(label_values) - 1:
            take = min(remaining, available)
        else:
            take = int(round(limit * available / total))
            take = max(0, min(take, available, remaining))
        counts[label] = take
        remaining -= take
    if remaining:
        raise RuntimeError(f"Could not allocate {limit} stratified rows; {remaining} remain")
    return counts


def historical_plain_validation(dataset: Dataset, size: int, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) < size * 2:
        raise RuntimeError(f"Need at least {size * 2} rows, found {len(dataset)}")
    ordered = dataset.shuffle(seed=seed)
    validation = ordered.select(range(size))
    remaining = ordered.select(range(size, len(ordered)))
    return validation, remaining


def historical_stratified_validation(
    dataset: Dataset,
    size: int,
    seed: int,
) -> tuple[Dataset, dict[int, Dataset], dict[int, int]]:
    labels = [int(label) for label in dataset["label"]]
    take_counts = stratified_take_counts(dataset, size)
    selected_parts = []
    remaining_by_label: dict[int, Dataset] = {}

    for label in sorted(take_counts):
        indices = [index for index, value in enumerate(labels) if value == label]
        ordered = dataset.select(indices).shuffle(seed=seed + label)
        take = take_counts[label]
        selected_parts.append(ordered.select(range(take)))
        remaining_by_label[label] = ordered.select(range(take, len(ordered)))

    validation = concatenate_datasets(selected_parts).shuffle(seed=seed)
    return validation, remaining_by_label, take_counts


def select_disjoint(
    candidates: Dataset,
    count: int,
    excluded_fingerprints: set[str],
) -> Dataset:
    selected_indices = []
    selected_fingerprints: set[str] = set()
    for index, row in enumerate(candidates):
        fingerprint = row_fingerprint(row)
        if fingerprint in excluded_fingerprints or fingerprint in selected_fingerprints:
            continue
        selected_indices.append(index)
        selected_fingerprints.add(fingerprint)
        if len(selected_indices) == count:
            break
    if len(selected_indices) != count:
        raise RuntimeError(f"Requested {count} disjoint rows but found {len(selected_indices)}")
    return candidates.select(selected_indices)


def select_stratified_disjoint(
    candidates_by_label: dict[int, Dataset],
    take_counts: dict[int, int],
    excluded_fingerprints: set[str],
    seed: int,
) -> Dataset:
    parts = []
    local_excluded = set(excluded_fingerprints)
    for label in sorted(take_counts):
        part = select_disjoint(candidates_by_label[label], take_counts[label], local_excluded)
        parts.append(part)
        local_excluded.update(part["message_sha256"])
    return concatenate_datasets(parts).shuffle(seed=seed)


def content_sha256(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for label, fingerprint in zip(dataset["label"], dataset["message_sha256"], strict=True):
        digest.update(f"{int(label)}\0{fingerprint}\n".encode("utf-8"))
    return digest.hexdigest()


def save_split(
    dataset: Dataset,
    *,
    output_dir: Path,
    relative_path: Path,
    name: str,
    role: str,
    source: str,
    selection: str,
    source_path: Path,
) -> dict[str, Any]:
    dataset = normalize_columns(dataset)
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    counts = label_counts(dataset)
    return {
        "name": name,
        "role": role,
        "source": source,
        "path": relative_path.as_posix(),
        "rows": len(dataset),
        "ham_count": counts["ham"],
        "spam_count": counts["spam"],
        "selection": selection,
        "source_path": portable_path(source_path),
        "source_sha256": sha256_file(source_path),
        "sha256": sha256_file(path),
        "content_sha256": content_sha256(dataset),
    }


def fingerprint_set(dataset: Dataset) -> set[str]:
    return {str(value) for value in dataset["message_sha256"]}


def pairwise_overlap(splits: dict[str, Dataset]) -> dict[str, int]:
    names = sorted(splits)
    overlaps = {}
    for left_index, left_name in enumerate(names):
        left = fingerprint_set(splits[left_name])
        for right_name in names[left_index + 1:]:
            overlap = len(left & fingerprint_set(splits[right_name]))
            overlaps[f"{left_name}__{right_name}"] = overlap
    return overlaps


def prepare(output_dir: Path, seed: int, ext_valid_size: int, test_size: int) -> Path:
    disable_progress_bars()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing the training exclusion fingerprint set...", flush=True)
    training_path = Path(
        combine.combine_datasets("training_all", combination_mode="mixed_50_50")
    ).resolve()
    training = normalize_columns(
        load_dataset("parquet", data_files=str(training_path), split="train")
    )
    training_fingerprints = fingerprint_set(training)

    ext_valid: dict[str, Dataset] = {}
    test_candidates: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}

    for source in EXTERNAL_SOURCES:
        print(f"Preparing historical external validation: {source}", flush=True)
        dataset, source_path = load_source_dataset(source)
        source_paths[source] = source_path
        if source == "spam_ham":
            validation, remaining_by_label, _ = historical_stratified_validation(
                dataset, ext_valid_size, seed
            )
            test_candidates[source] = (
                remaining_by_label,
                stratified_take_counts(dataset, test_size),
            )
        else:
            validation, remaining = historical_plain_validation(dataset, ext_valid_size, seed)
            test_candidates[source] = remaining
        ext_valid[source] = validation

    ext_valid_fingerprints = set().union(*(fingerprint_set(split) for split in ext_valid.values()))
    excluded_for_tests = training_fingerprints | ext_valid_fingerprints

    final_tests: dict[str, Dataset] = {}
    for source in EXTERNAL_SOURCES:
        print(f"Preparing disjoint final test: {source}", flush=True)
        if source == "spam_ham":
            remaining_by_label, take_counts = test_candidates[source]
            test = select_stratified_disjoint(
                remaining_by_label,
                take_counts,
                excluded_for_tests,
                seed + 10_000,
            )
        else:
            test = select_disjoint(test_candidates[source], test_size, excluded_for_tests)
            test = test.shuffle(seed=seed + 10_000)
        final_tests[source] = test
        test_fingerprints = fingerprint_set(test)
        if test_fingerprints & excluded_for_tests:
            raise RuntimeError(f"Leakage detected while preparing final test for {source}")
        excluded_for_tests.update(test_fingerprints)

    print("Preparing cleaned TREC 2006 final test...", flush=True)
    trec_analysis_dir = output_dir / "trec_2006_overlap_analysis"
    analyze_trec_2006(trec_analysis_dir)
    trec_eligible_path = (trec_analysis_dir / "eligible_pool.parquet").resolve()
    trec_eligible = normalize_columns(
        load_dataset("parquet", data_files=str(trec_eligible_path), split="train")
    )
    trec_labels = [int(label) for label in trec_eligible["label"]]
    trec_parts = []
    trec_take_counts = {0: test_size // 2, 1: test_size - (test_size // 2)}
    for label, count in trec_take_counts.items():
        indices = [index for index, value in enumerate(trec_labels) if value == label]
        ordered = trec_eligible.select(indices).shuffle(seed=seed + label)
        part = select_disjoint(ordered, count, excluded_for_tests)
        trec_parts.append(part)
        excluded_for_tests.update(fingerprint_set(part))
    final_tests["trec_2006"] = concatenate_datasets(trec_parts).shuffle(seed=seed + 20_000)

    final_overlap = pairwise_overlap(final_tests)
    if any(final_overlap.values()):
        raise RuntimeError(f"Final test sets overlap: {final_overlap}")

    split_entries = []
    for source, dataset in ext_valid.items():
        split_entries.append(save_split(
            dataset,
            output_dir=output_dir,
            relative_path=Path("external_validation") / f"{source}.parquet",
            name=f"ext_valid_{source}",
            role="external_validation",
            source=source,
            selection="historical deterministic sample used for model selection",
            source_path=source_paths[source],
        ))

    for source, dataset in final_tests.items():
        source_path = trec_eligible_path if source == "trec_2006" else source_paths[source]
        split_entries.append(save_split(
            dataset,
            output_dir=output_dir,
            relative_path=Path("final_test") / f"{source}.parquet",
            name=f"test_{source}",
            role="final_test",
            source=source,
            selection=(
                f"balanced {trec_take_counts[0]} ham / {trec_take_counts[1]} spam from deduplicated overlap-free pool"
                if source == "trec_2006"
                else "next deterministic sample after validation, excluding training and validation fingerprints"
            ),
            source_path=source_path,
        ))

    ext_overlap = pairwise_overlap(ext_valid)
    ext_vs_training = {
        name: len(fingerprint_set(dataset) & training_fingerprints)
        for name, dataset in ext_valid.items()
    }
    test_vs_training = {
        name: len(fingerprint_set(dataset) & training_fingerprints)
        for name, dataset in final_tests.items()
    }
    test_vs_validation = {
        name: len(fingerprint_set(dataset) & ext_valid_fingerprints)
        for name, dataset in final_tests.items()
    }
    if any(test_vs_training.values()) or any(test_vs_validation.values()):
        raise RuntimeError("Final test leakage verification failed")
    verify_expected_split_hashes(split_entries, seed, ext_valid_size, test_size)
    raw_sources = verify_raw_dataset_checksums()

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "external_validation_size_per_source": ext_valid_size,
        "final_test_size_per_source": test_size,
        "fingerprint": {
            "minimum_normalized_body_length": MIN_NORMALIZED_BODY_LENGTH,
            "labels_included": False,
            "short_message_fallback": "NFKC subject + body exact SHA-256",
            "long_message_normalization": "dataset.combine high body normalization",
        },
        "training_reference": {
            "path": portable_path(training_path),
            "sha256": sha256_file(training_path),
            "rows": len(training),
        },
        "raw_sources": raw_sources,
        "checks": {
            "external_validation_pairwise_overlap": ext_overlap,
            "external_validation_overlap_with_training": ext_vs_training,
            "final_test_pairwise_overlap": final_overlap,
            "final_test_overlap_with_training": test_vs_training,
            "final_test_overlap_with_external_validation": test_vs_validation,
        },
        "splits": split_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved evaluation manifest: {manifest_path}", flush=True)
    print(f"Manifest SHA-256: {sha256_file(manifest_path)}", flush=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download source corpora and prepare frozen external-validation and final-test splits."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ext-valid-size", type=int, default=EXT_VALID_SIZE)
    parser.add_argument("--test-size", type=int, default=FINAL_TEST_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.ext_valid_size < 1 or arguments.test_size < 1:
        raise SystemExit("Split sizes must be positive")
    prepare(
        arguments.output_dir.resolve(),
        arguments.seed,
        arguments.ext_valid_size,
        arguments.test_size,
    )
