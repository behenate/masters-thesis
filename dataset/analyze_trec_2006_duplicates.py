from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    import combine
except ImportError:  # pragma: no cover - allows package-style imports
    from . import combine


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "trec_2006_overlap"
NEW_DATASET = "trec_2006"
WHITESPACE_RE = re.compile(r"\s+")
MIN_NORMALIZED_BODY_LENGTH = 50


def _canonical_exact_text(subject: object, body: object) -> str:
    normalized_subject = unicodedata.normalize("NFKC", str(subject)).replace("\r\n", "\n").strip()
    normalized_body = unicodedata.normalize("NFKC", str(body)).replace("\r\n", "\n").strip()
    normalized_subject = WHITESPACE_RE.sub(" ", normalized_subject)
    normalized_body = WHITESPACE_RE.sub(" ", normalized_body)
    return f"{normalized_subject}\n\n{normalized_body}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=frame.index)
    keys["exact_sha256"] = [
        _sha256(_canonical_exact_text(subject, body))
        for subject, body in zip(frame["subject"], frame["body"], strict=True)
    ]
    keys["normalized_body"] = frame["body"].map(combine._normalized_body_fingerprint_high)
    keys["normalized_body_length"] = keys["normalized_body"].str.len()
    keys["normalized_sha256"] = keys["normalized_body"].map(_sha256)
    short_body = keys["normalized_body_length"] < MIN_NORMALIZED_BODY_LENGTH
    keys.loc[short_body, "normalized_sha256"] = ""
    keys["dedup_sha256"] = keys["normalized_sha256"]
    keys.loc[short_body, "dedup_sha256"] = (
        "subject_body:" + keys.loc[short_body, "exact_sha256"]
    )
    return keys.drop(columns="normalized_body")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = combine._drop_empty_subject_and_body(frame)
    prepared = prepared.reset_index(names="source_row_index")
    prepared["label"] = prepared["label"].astype(int)
    return pd.concat([prepared, _fingerprints(prepared)], axis=1)


def _index_new_dataset(frame: pd.DataFrame, column: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, fingerprint in enumerate(frame[column]):
        if fingerprint:
            index[str(fingerprint)].append(row_index)
    return index


def _internal_duplicate_stats(frame: pd.DataFrame, column: str) -> dict[str, int]:
    fingerprints = frame[column]
    fingerprints = fingerprints[fingerprints != ""]
    counts = fingerprints.value_counts()
    duplicate_groups = counts[counts > 1]
    return {
        "duplicate_groups": int(len(duplicate_groups)),
        "rows_in_duplicate_groups": int(duplicate_groups.sum()),
        "redundant_rows": int((duplicate_groups - 1).sum()),
    }


def analyze(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {NEW_DATASET}...", flush=True)
    combine.ATOMIC_DATASET_DOWNLOADERS[NEW_DATASET]()
    trec = _prepare_frame(combine.ATOMIC_DATASET_BUILDERS[NEW_DATASET]())
    exact_index = _index_new_dataset(trec, "exact_sha256")
    normalized_index = _index_new_dataset(trec, "normalized_sha256")

    matches: dict[tuple[int, str], dict[str, object]] = {}
    source_summaries = []

    comparison_sources = sorted(name for name in combine.ATOMIC_DATASET_BUILDERS if name != NEW_DATASET)
    for source in comparison_sources:
        print(f"Comparing against {source}...", flush=True)
        combine.ATOMIC_DATASET_DOWNLOADERS[source]()
        current = _prepare_frame(combine.ATOMIC_DATASET_BUILDERS[source]())

        source_exact_trec_rows: set[int] = set()
        source_normalized_trec_rows: set[int] = set()
        source_matched_dedup_fingerprints: set[str] = set()
        matched_current_rows: set[int] = set()

        for current_position, row in current.iterrows():
            exact_candidates = exact_index.get(str(row["exact_sha256"]), [])
            normalized_candidates = normalized_index.get(str(row["normalized_sha256"]), [])

            candidate_types: dict[int, str] = {candidate: "normalized" for candidate in normalized_candidates}
            candidate_types.update({candidate: "exact" for candidate in exact_candidates})

            for trec_position, match_type in candidate_types.items():
                if match_type == "exact":
                    source_exact_trec_rows.add(trec_position)
                source_normalized_trec_rows.add(trec_position)
                source_matched_dedup_fingerprints.add(str(trec.iloc[trec_position]["dedup_sha256"]))
                matched_current_rows.add(current_position)

                key = (trec_position, source)
                entry = matches.setdefault(key, {
                    "trec_row_index": int(trec.iloc[trec_position]["source_row_index"]),
                    "trec_label": int(trec.iloc[trec_position]["label"]),
                    "trec_subject": str(trec.iloc[trec_position]["subject"]),
                    "trec_exact_sha256": str(trec.iloc[trec_position]["exact_sha256"]),
                    "trec_normalized_sha256": str(trec.iloc[trec_position]["normalized_sha256"]),
                    "trec_dedup_sha256": str(trec.iloc[trec_position]["dedup_sha256"]),
                    "trec_normalized_body_length": int(trec.iloc[trec_position]["normalized_body_length"]),
                    "matched_source": source,
                    "match_type": match_type,
                    "matched_current_rows": 0,
                    "matched_current_labels": set(),
                    "first_current_row_index": int(row["source_row_index"]),
                    "first_current_subject": str(row["subject"]),
                })
                if match_type == "exact":
                    entry["match_type"] = "exact"
                entry["matched_current_rows"] = int(entry["matched_current_rows"]) + 1
                entry["matched_current_labels"].add(int(row["label"]))

        source_summaries.append({
            "source": source,
            "source_rows": int(len(current)),
            "trec_rows_exact_match": int(len(source_exact_trec_rows)),
            "trec_rows_normalized_match": int(len(source_normalized_trec_rows)),
            "trec_rows_normalized_only": int(len(source_normalized_trec_rows - source_exact_trec_rows)),
            "unique_trec_messages_matched": int(len(source_matched_dedup_fingerprints)),
            "current_rows_involved": int(len(matched_current_rows)),
        })
        del current

    detail_rows = []
    for entry in matches.values():
        entry = dict(entry)
        entry["matched_current_labels"] = json.dumps(sorted(entry["matched_current_labels"]))
        detail_rows.append(entry)

    details = pd.DataFrame(detail_rows)
    if not details.empty:
        details = details.sort_values(["match_type", "matched_source", "trec_row_index"]).reset_index(drop=True)

    summary = pd.DataFrame(source_summaries)
    matched_exact = set(details.loc[details["match_type"] == "exact", "trec_row_index"]) if not details.empty else set()
    matched_normalized = set(details["trec_row_index"]) if not details.empty else set()

    matched_dedup_fingerprints = (
        set(details["trec_dedup_sha256"])
        if not details.empty
        else set()
    )
    label_counts_by_fingerprint = trec.groupby("dedup_sha256")["label"].nunique()
    conflicting_internal_fingerprints = set(
        label_counts_by_fingerprint[label_counts_by_fingerprint > 1].index
    )
    conflicting_internal_rows = trec[
        trec["dedup_sha256"].isin(conflicting_internal_fingerprints)
    ]
    deduplicated_trec = trec.drop_duplicates(subset=["dedup_sha256"], keep="first")
    unambiguous_trec = deduplicated_trec[
        ~deduplicated_trec["dedup_sha256"].isin(conflicting_internal_fingerprints)
    ]
    eligible_pool = unambiguous_trec[
        ~unambiguous_trec["dedup_sha256"].isin(matched_dedup_fingerprints)
    ].copy()

    label_conflicts = 0
    for entry in matches.values():
        if any(
            int(current_label) != int(entry["trec_label"])
            for current_label in entry["matched_current_labels"]
        ):
            label_conflicts += 1

    metrics = {
        "new_dataset": NEW_DATASET,
        "comparison_sources": comparison_sources,
        "trec_2006": {
            "rows": int(len(trec)),
            "ham": int((trec["label"] == 0).sum()),
            "spam": int((trec["label"] == 1).sum()),
            "internal_exact_duplicates": _internal_duplicate_stats(trec, "exact_sha256"),
            "internal_deduplicated_duplicates": _internal_duplicate_stats(trec, "dedup_sha256"),
            "conflicting_label_groups": len(conflicting_internal_fingerprints),
            "rows_in_conflicting_label_groups": int(len(conflicting_internal_rows)),
        },
        "cross_dataset_overlap": {
            "trec_rows_exact_match": len(matched_exact),
            "trec_rows_normalized_match": len(matched_normalized),
            "trec_rows_normalized_only": len(matched_normalized - matched_exact),
            "trec_rows_without_normalized_match": int(len(trec) - len(matched_normalized)),
            "normalized_overlap_rate": len(matched_normalized) / len(trec) if len(trec) else 0.0,
            "unique_dedup_fingerprints_matched": len(matched_dedup_fingerprints),
            "matches_with_conflicting_label": label_conflicts,
        },
        "eligible_pool": {
            "rows_after_internal_deduplication": int(len(deduplicated_trec)),
            "rows_after_conflicting_label_removal": int(len(unambiguous_trec)),
            "rows_after_cross_dataset_overlap_removal": int(len(eligible_pool)),
            "ham": int((eligible_pool["label"] == 0).sum()),
            "spam": int((eligible_pool["label"] == 1).sum()),
        },
        "normalization": {
            "exact": "NFKC, normalized line endings and collapsed whitespace over subject + body; label ignored",
            "normalized": (
                "existing high body normalization from dataset/combine.py; label ignored; "
                f"minimum normalized body length {MIN_NORMALIZED_BODY_LENGTH} characters"
            ),
            "short_messages": "subject + body exact fingerprint used below the normalized-length threshold",
        },
    }

    summary_path = output_dir / "summary.csv"
    details_path = output_dir / "matches.csv"
    metrics_path = output_dir / "metrics.json"
    eligible_path = output_dir / "eligible_pool.parquet"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    eligible_pool[[
        "source_row_index",
        "subject",
        "body",
        "label",
        "source",
        "exact_sha256",
        "normalized_sha256",
        "dedup_sha256",
        "normalized_body_length",
    ]].to_parquet(eligible_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved matches: {details_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Saved eligible pool: {eligible_path}", flush=True)
    print(json.dumps(metrics["cross_dataset_overlap"], indent=2), flush=True)
    return summary_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TREC 2006 with all datasets already registered in dataset/combine.py."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(arguments.output_dir.resolve())
