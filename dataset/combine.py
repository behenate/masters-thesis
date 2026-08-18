import email
import html
import hashlib
import json
import math
import mailbox
import os
import re
import unicodedata
from collections.abc import Iterable, Sequence

import pandas as pd

try:
    from combiners import spam_2007_2008, spam_detection
    from downloaders import (
        ceas_2008,
        enron,
        fraudulent_email_corpus,
        ling,
        nazario,
        spam_assassin,
        spam_ham,
        trec_2006,
        trec_2007,
    )
except ImportError:  # pragma: no cover - allows package-style imports
    from .combiners import spam_2007_2008, spam_detection
    from .downloaders import (
        ceas_2008,
        enron,
        fraudulent_email_corpus,
        ling,
        nazario,
        spam_assassin,
        spam_ham,
        trec_2006,
        trec_2007,
    )


SEED = 67
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "combined_datasets",
    "generated",
)
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_datasets")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
DUPLICATE_DETECTION_LEVELS = {"basic", "medium", "high"}
EMPTY_EMAIL_FILTER = "subject_or_body"
COMBINATION_MODES = {
    "mixed",
    "mixed_50_50",
    "source_aware_50_50",
    "balanced_source_50_50",
}
DEFAULT_SOURCE_AWARE_MAX_SMALLEST_MULTIPLIER = 8.0
TRAINING_EXCLUDED_DATASETS = frozenset({
    "enron",
    "fraudulent_email_corpus",
    "spam_ham",
    "trec_2006",
})


def _extract_subject_and_body_from_message(msg: email.message.Message) -> tuple[str, str]:
    subject = msg.get("Subject", "") or ""
    body_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() != "text/plain":
                continue

            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            if payload:
                try:
                    body_parts.append(payload.decode(charset, errors="replace"))
                except LookupError:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
            elif isinstance(part.get_payload(), str):
                body_parts.append(part.get_payload())
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        if payload:
            try:
                body_parts.append(payload.decode(charset, errors="replace"))
            except LookupError:
                body_parts.append(payload.decode("utf-8", errors="replace"))
        elif isinstance(msg.get_payload(), str):
            body_parts.append(msg.get_payload())

    body = "\n".join(body_parts).strip()
    return subject.strip(), body


def _extract_subject_and_body_from_raw_email(raw: str) -> tuple[str, str]:
    try:
        message = email.message_from_string(raw)
        subject, body = _extract_subject_and_body_from_message(message)
        if subject or body:
            return subject, body
    except Exception:
        pass

    return "", raw.strip()


def build_spam_ham() -> pd.DataFrame:
    return spam_detection.extract_spam_ham()


def build_spam_assassin() -> pd.DataFrame:
    return spam_detection.extract_spam_assassin()


def build_trec_2007() -> pd.DataFrame:
    return spam_2007_2008.extract_trec_2007()


def build_trec_2006() -> pd.DataFrame:
    return spam_2007_2008.extract_trec_2006()


def build_ceas_2008() -> pd.DataFrame:
    return spam_2007_2008.extract_ceas_2008()


def build_enron() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "enron", "emails.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        subject, body = _extract_subject_and_body_from_raw_email(str(row["message"]))
        records.append({
            "subject": subject,
            "body": body,
            "label": 0,
            "source": "enron",
        })

    return pd.DataFrame(records)


def build_fraudulent_email_corpus() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "fraudulent_email_corpus", "fradulent_emails.txt")

    records = []
    for message in mailbox.mbox(path):
        subject, body = _extract_subject_and_body_from_message(message)
        records.append({
            "subject": subject,
            "body": body,
            "label": 1,
            "source": "fraudulent_email_corpus",
        })

    return pd.DataFrame(records)


def build_ling() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "ling", "Ling.csv")
    df = pd.read_csv(path)
    body_column = "body" if "body" in df.columns else "message"

    records = []
    for _, row in df.iterrows():
        records.append({
            "subject": str(row["subject"]) if pd.notna(row["subject"]) else "",
            "body": str(row[body_column]) if pd.notna(row[body_column]) else "",
            "label": int(row["label"]),
            "source": "ling",
        })

    return pd.DataFrame(records)


def build_nazario() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "nazario", "Nazario.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "subject": str(row["subject"]) if pd.notna(row["subject"]) else "",
            "body": str(row["body"]) if pd.notna(row["body"]) else "",
            "label": int(row["label"]),
            "source": "nazario",
        })

    return pd.DataFrame(records)


ATOMIC_DATASET_BUILDERS = {
    "ceas_2008": build_ceas_2008,
    "enron": build_enron,
    "fraudulent_email_corpus": build_fraudulent_email_corpus,
    "ling": build_ling,
    "nazario": build_nazario,
    "spam_assassin": build_spam_assassin,
    "spam_ham": build_spam_ham,
    "trec_2006": build_trec_2006,
    "trec_2007": build_trec_2007,
}

ATOMIC_DATASET_DOWNLOADERS = {
    "ceas_2008": ceas_2008.download,
    "enron": enron.download,
    "fraudulent_email_corpus": fraudulent_email_corpus.download,
    "ling": ling.download,
    "nazario": nazario.download,
    "spam_assassin": spam_assassin.download,
    "spam_ham": spam_ham.download,
    "trec_2006": trec_2006.download,
    "trec_2007": trec_2007.download,
}

DATASET_ALIASES = {
    "all": tuple(sorted(ATOMIC_DATASET_BUILDERS)),
    "training_all": tuple(
        name for name in sorted(ATOMIC_DATASET_BUILDERS)
        if name not in TRAINING_EXCLUDED_DATASETS
    ),
    "spam_2007_2008": ("ceas_2008", "trec_2007"),
    "spam_detection": ("spam_assassin", "spam_ham"),
}


def _concat_atomic_datasets(dataset_names: Iterable[str]) -> pd.DataFrame:
    frames = [ATOMIC_DATASET_BUILDERS[name]() for name in dataset_names]
    return pd.concat(frames, ignore_index=True)


def build_spam_detection() -> pd.DataFrame:
    return _concat_atomic_datasets(DATASET_ALIASES["spam_detection"])


def build_spam_2007_2008() -> pd.DataFrame:
    return _concat_atomic_datasets(DATASET_ALIASES["spam_2007_2008"])


def build_all() -> pd.DataFrame:
    return _concat_atomic_datasets(DATASET_ALIASES["all"])


def build_training_all() -> pd.DataFrame:
    return _concat_atomic_datasets(DATASET_ALIASES["training_all"])


DATASET_FUNCTIONS = {
    **ATOMIC_DATASET_BUILDERS,
    "all": build_all,
    "spam_2007_2008": build_spam_2007_2008,
    "spam_detection": build_spam_detection,
    "training_all": build_training_all,
}


def _normalize_requested_datasets(dataset_names: str | Sequence[str]) -> list[str]:
    if isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        names = list(dataset_names)

    normalized = []
    for name in names:
        dataset_name = str(name).strip()
        if not dataset_name:
            continue
        if dataset_name not in DATASET_FUNCTIONS:
            available = ", ".join(sorted(DATASET_FUNCTIONS))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")
        normalized.append(dataset_name)

    if not normalized:
        raise ValueError("At least one dataset name must be provided.")

    return normalized


def _resolve_atomic_dataset_names(dataset_names: str | Sequence[str]) -> list[str]:
    requested_names = _normalize_requested_datasets(dataset_names)
    resolved = set()
    pending = list(requested_names)

    while pending:
        name = pending.pop()
        if name in DATASET_ALIASES:
            pending.extend(DATASET_ALIASES[name])
            continue
        resolved.add(name)

    return sorted(resolved)


def _validate_spam_ham_ratio(spam_ham_ratio: float) -> None:
    if spam_ham_ratio == -1:
        return
    if not 0.0 <= spam_ham_ratio <= 1.0:
        raise ValueError("spam_ham_ratio must be between 0.0 and 1.0, or -1 to disable balancing.")


def _validate_duplicate_detection(duplicate_detection: str) -> str:
    normalized = str(duplicate_detection).strip().lower()
    if normalized not in DUPLICATE_DETECTION_LEVELS:
        available = ", ".join(sorted(DUPLICATE_DETECTION_LEVELS))
        raise ValueError(f"duplicate_detection must be one of: {available}")
    return normalized


def _validate_combination_mode(combination_mode: str) -> str:
    normalized = str(combination_mode).strip().lower()
    if normalized not in COMBINATION_MODES:
        available = ", ".join(sorted(COMBINATION_MODES))
        raise ValueError(f"combination_mode must be one of: {available}")
    return normalized


def _ratio_token(spam_ham_ratio: float) -> str:
    if spam_ham_ratio == -1:
        return "all_samples"
    return f"spam_{str(spam_ham_ratio).replace('.', '_')}"


def _mode_token(combination_mode: str) -> str:
    return f"mode_{combination_mode}"


def _multiplier_token(value: float) -> str:
    return f"max{float(value):g}x".replace(".", "_")


def _combined_dataset_path(
    dataset_names: list[str],
    spam_ham_ratio: float,
    duplicate_detection: str,
    combination_mode: str,
    source_aware_max_multiplier: float,
) -> str:
    spec = {
        "combination_mode": combination_mode,
        "dataset_names": dataset_names,
        "duplicate_detection": duplicate_detection,
        "empty_email_filter": EMPTY_EMAIL_FILTER,
        "spam_ham_ratio": spam_ham_ratio,
    }
    if combination_mode == "source_aware_50_50":
        spec["source_aware_max_multiplier"] = source_aware_max_multiplier
    if combination_mode == "mixed":
        spec.pop("combination_mode")
    digest = hashlib.sha1(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    stem = "__".join(dataset_names)
    mode_part = "" if combination_mode == "mixed" else f"__{_mode_token(combination_mode)}"
    if combination_mode == "source_aware_50_50":
        mode_part += f"__{_multiplier_token(source_aware_max_multiplier)}"
    filename = f"{stem}__dedupe_{duplicate_detection}{mode_part}__{_ratio_token(spam_ham_ratio)}__{digest}.parquet"
    return os.path.join(OUTPUT_DIR, filename)


def _shuffle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def _stable_seed(*parts: object) -> int:
    payload = json.dumps([SEED, *parts], sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


def _normalized_body_fingerprint_medium(body: str) -> str:
    return re.sub(r"\s+", "", str(body))


def _normalized_body_fingerprint_high(body: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(body))
    normalized = html.unescape(normalized)
    normalized = normalized.lower()
    normalized = HTML_TAG_RE.sub(" ", normalized)
    normalized = URL_RE.sub(" ", normalized)
    normalized = EMAIL_RE.sub(" ", normalized)
    normalized = NON_ALNUM_RE.sub("", normalized)
    return normalized


def _duplicate_key_frame(df: pd.DataFrame, duplicate_detection: str) -> pd.DataFrame:
    if duplicate_detection == "basic":
        keyed = df[["subject", "body", "label"]].copy()
        keyed.columns = ["key_subject", "key_body", "key_label"]
        return keyed

    if duplicate_detection == "medium":
        return pd.DataFrame({
            "key_body": df["body"].map(_normalized_body_fingerprint_medium),
        })

    return pd.DataFrame({
        "key_body": df["body"].map(_normalized_body_fingerprint_high),
    })


def _drop_empty_subject_and_body(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["subject"] = cleaned["subject"].fillna("").astype(str).str.strip()
    cleaned["body"] = cleaned["body"].fillna("").astype(str).str.strip()
    filtered = cleaned[(cleaned["subject"] != "") | (cleaned["body"] != "")].reset_index(drop=True)
    removed = len(df) - len(filtered)
    print(f"Removed rows with empty subject and body: {removed}")
    return filtered


def _drop_duplicate_messages(df: pd.DataFrame, duplicate_detection: str) -> pd.DataFrame:
    keyed = df.reset_index(drop=True).copy()
    duplicate_keys = _duplicate_key_frame(keyed, duplicate_detection)
    deduplicated = pd.concat([keyed, duplicate_keys], axis=1)
    deduplicated = deduplicated.drop_duplicates(subset=list(duplicate_keys.columns)).reset_index(drop=True)
    deduplicated = deduplicated[df.columns.tolist()]
    removed = len(df) - len(deduplicated)
    print(f"Removed duplicate rows: {removed}")
    return deduplicated


def _duplicate_report_path(
    dataset_names: list[str],
    spam_ham_ratio: float,
    duplicate_detection: str,
) -> str:
    spec = {
        "dataset_names": dataset_names,
        "duplicate_detection": duplicate_detection,
        "empty_email_filter": EMPTY_EMAIL_FILTER,
        "spam_ham_ratio": spam_ham_ratio,
        "report": "duplicates",
    }
    digest = hashlib.sha1(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    stem = "__".join(dataset_names)
    filename = f"{stem}__dedupe_{duplicate_detection}__{_ratio_token(spam_ham_ratio)}__{digest}__duplicates.csv"
    return os.path.join(REPORTS_DIR, filename)


def _generate_duplicate_report(
    df: pd.DataFrame,
    *,
    dataset_names: list[str],
    spam_ham_ratio: float,
    duplicate_detection: str,
) -> str:
    report_path = _duplicate_report_path(dataset_names, spam_ham_ratio, duplicate_detection)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    indexed = df.reset_index(names="original_row_index").copy()
    duplicate_keys = _duplicate_key_frame(indexed, duplicate_detection)
    duplicate_keys = duplicate_keys.rename(columns={column: f"_{column}" for column in duplicate_keys.columns})
    indexed = pd.concat([indexed, duplicate_keys], axis=1)
    key_columns = list(duplicate_keys.columns)

    grouped_rows = []
    duplicate_groups = indexed.groupby(key_columns, sort=False, dropna=False)
    duplicate_group_id = 0

    for _, group in duplicate_groups:
        if len(group) <= 1:
            continue
        duplicate_group_id += 1
        group = group.reset_index(drop=True)
        kept = group.iloc[0]
        all_rows = []
        for _, row in group.iterrows():
            all_rows.append({
                "original_row_index": int(row["original_row_index"]),
                "subject": str(row["subject"]),
                "body": str(row["body"]),
                "label": int(row["label"]),
                "source": str(row["source"]),
            })

        kept_row = all_rows[0]
        grouped_row = {
            "duplicate_group_id": duplicate_group_id,
            "duplicate_detection": duplicate_detection,
            "duplicate_count": len(group),
            "labels": json.dumps(sorted({row["label"] for row in all_rows})),
            "sources": json.dumps(sorted({row["source"] for row in all_rows}), ensure_ascii=False),
            "kept_original_row_index": kept_row["original_row_index"],
            "kept_subject": kept_row["subject"],
            "kept_source": kept_row["source"],
            "kept_label": kept_row["label"],
            "all_original_row_indexes": json.dumps([row["original_row_index"] for row in all_rows]),
        }

        for sample_index, sample in enumerate(all_rows):
            grouped_row[f"sample_{sample_index}_original_row_index"] = sample["original_row_index"]
            grouped_row[f"sample_{sample_index}_subject"] = sample["subject"]
            grouped_row[f"sample_{sample_index}_body"] = sample["body"]
            grouped_row[f"sample_{sample_index}_label"] = sample["label"]
            grouped_row[f"sample_{sample_index}_source"] = sample["source"]

        grouped_rows.append(grouped_row)

    report = pd.DataFrame(grouped_rows)
    if not report.empty:
        report = report.sort_values(["duplicate_count", "duplicate_group_id"], ascending=[False, True]).reset_index(drop=True)
    report.to_csv(report_path, index=False)
    print(f"Duplicate report saved to: {report_path}")
    return report_path


def _apply_spam_ham_ratio(df: pd.DataFrame, spam_ham_ratio: float) -> pd.DataFrame:
    if spam_ham_ratio == -1:
        return df

    spam = df[df["label"] == 1]
    ham = df[df["label"] == 0]

    if spam_ham_ratio == 1.0:
        if spam.empty:
            raise ValueError("Requested 100 percent spam, but no spam samples are available.")
        return spam.copy()

    if spam_ham_ratio == 0.0:
        if ham.empty:
            raise ValueError("Requested 0 percent spam, but no ham samples are available.")
        return ham.copy()

    if spam.empty or ham.empty:
        raise ValueError("Balancing requires both spam and ham samples to be present.")

    spam_count = len(spam)
    ham_count = len(ham)

    if spam_count / spam_ham_ratio <= ham_count / (1.0 - spam_ham_ratio):
        selected_spam = spam_count
        selected_ham = int(selected_spam * (1.0 - spam_ham_ratio) / spam_ham_ratio)
    else:
        selected_ham = ham_count
        selected_spam = int(selected_ham * spam_ham_ratio / (1.0 - spam_ham_ratio))

    if selected_spam <= 0 or selected_ham <= 0:
        raise ValueError("Requested spam_ham_ratio leaves no samples in one of the classes.")

    spam_df = spam.sample(n=selected_spam, random_state=SEED) if selected_spam < spam_count else spam.copy()
    ham_df = ham.sample(n=selected_ham, random_state=SEED) if selected_ham < ham_count else ham.copy()
    return pd.concat([spam_df, ham_df], ignore_index=True)


def _sample_rows(df: pd.DataFrame, n: int, *seed_parts: object) -> pd.DataFrame:
    if n < 0:
        raise ValueError("sample size must be non-negative.")
    if n == 0:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=_stable_seed(*seed_parts))


def _largest_remainder_capped_allocation(
    capacities: dict[str, int],
    target_total: int,
    caps: dict[str, int] | None = None,
) -> dict[str, int]:
    capacities = {source: int(count) for source, count in capacities.items() if int(count) > 0}
    if caps is not None:
        capacities = {
            source: min(count, int(caps.get(source, count)))
            for source, count in capacities.items()
            if int(caps.get(source, count)) > 0
        }
    if not capacities:
        raise ValueError("Cannot allocate samples without any positive source capacity.")

    available_total = sum(capacities.values())
    if target_total > available_total:
        raise ValueError(f"Requested {target_total} samples, but only {available_total} are available.")
    if target_total == available_total:
        return capacities.copy()

    allocations = {source: 0 for source in capacities}
    remaining_capacity = capacities.copy()
    remaining_total = int(target_total)

    if remaining_total >= len(remaining_capacity):
        for source in remaining_capacity:
            allocations[source] = 1
            remaining_capacity[source] -= 1
        remaining_total -= len(remaining_capacity)

    while remaining_total > 0:
        eligible = {
            source: capacity
            for source, capacity in remaining_capacity.items()
            if capacity > 0
        }
        if not eligible:
            break

        total_weight = sum(math.sqrt(capacities[source]) for source in eligible)
        raw_increments = {
            source: remaining_total * math.sqrt(capacities[source]) / total_weight
            for source in eligible
        }
        increments = {
            source: min(eligible[source], int(math.floor(raw_increments[source])))
            for source in eligible
        }
        assigned = sum(increments.values())
        leftover = remaining_total - assigned

        for source in sorted(
            eligible,
            key=lambda item: (
                raw_increments[item] - math.floor(raw_increments[item]),
                eligible[item],
                item,
            ),
            reverse=True,
        ):
            if leftover <= 0:
                break
            if increments[source] < eligible[source]:
                increments[source] += 1
                leftover -= 1

        increment_total = sum(increments.values())
        if increment_total <= 0:
            raise ValueError("Could not allocate source-aware samples.")

        for source, increment in increments.items():
            allocations[source] += increment
            remaining_capacity[source] -= increment
        remaining_total -= increment_total

    return allocations


def _source_aware_caps(
    capacities: dict[str, int],
    source_aware_max_multiplier: float,
) -> dict[str, int]:
    if source_aware_max_multiplier < 1:
        raise ValueError("source_aware_max_multiplier must be >= 1.")
    positive_counts = [count for count in capacities.values() if count > 0]
    if not positive_counts:
        return {}
    cap = max(1, int(math.ceil(min(positive_counts) * source_aware_max_multiplier)))
    return {source: min(count, cap) for source, count in capacities.items() if count > 0}


def _apply_source_aware_50_50(
    df: pd.DataFrame,
    source_aware_max_multiplier: float,
) -> pd.DataFrame:
    spam = df[df["label"] == 1]
    ham = df[df["label"] == 0]
    if spam.empty or ham.empty:
        raise ValueError("source_aware_50_50 requires both spam and ham samples.")

    capped_capacities_by_label = {}
    for label in [0, 1]:
        label_df = df[df["label"] == label]
        capacities = label_df.groupby("source", sort=True).size().astype(int).to_dict()
        capped_capacities_by_label[label] = _source_aware_caps(capacities, source_aware_max_multiplier)

    target_per_class = min(sum(caps.values()) for caps in capped_capacities_by_label.values())
    selected_frames = []

    for label, label_name in [(0, "ham"), (1, "spam")]:
        label_df = df[df["label"] == label]
        capacities = label_df.groupby("source", sort=True).size().astype(int).to_dict()
        caps = capped_capacities_by_label[label]
        allocations = _largest_remainder_capped_allocation(capacities, target_per_class, caps)
        print(f"Source-aware {label_name} allocation: {allocations}")
        for source, count in allocations.items():
            source_df = label_df[label_df["source"] == source]
            selected_frames.append(_sample_rows(source_df, count, "source_aware_50_50", label_name, source))

    return pd.concat(selected_frames, ignore_index=True)


def _apply_balanced_source_50_50(df: pd.DataFrame) -> pd.DataFrame:
    source_label_counts = (
        df.groupby(["source", "label"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    for label in [0, 1]:
        if label not in source_label_counts.columns:
            source_label_counts[label] = 0

    missing_sources = source_label_counts[
        (source_label_counts[0] <= 0) | (source_label_counts[1] <= 0)
    ]
    if not missing_sources.empty:
        missing = ", ".join(str(source) for source in missing_sources.index)
        raise ValueError(
            "balanced_source_50_50 requires every source to contain both labels. "
            f"Missing one class in: {missing}"
        )

    samples_per_source_per_class = int(source_label_counts[[0, 1]].min(axis=1).min())
    if samples_per_source_per_class <= 0:
        raise ValueError("balanced_source_50_50 leaves no samples per source/class.")

    selected_frames = []
    for source in sorted(source_label_counts.index):
        for label, label_name in [(0, "ham"), (1, "spam")]:
            source_label_df = df[(df["source"] == source) & (df["label"] == label)]
            selected_frames.append(
                _sample_rows(
                    source_label_df,
                    samples_per_source_per_class,
                    "balanced_source_50_50",
                    source,
                    label_name,
                )
            )

    print(
        "Balanced source/class allocation: "
        f"{samples_per_source_per_class} ham and {samples_per_source_per_class} spam per source"
    )
    return pd.concat(selected_frames, ignore_index=True)


def _apply_combination_mode(
    df: pd.DataFrame,
    combination_mode: str,
    spam_ham_ratio: float,
    source_aware_max_multiplier: float,
) -> pd.DataFrame:
    if combination_mode == "mixed":
        return _apply_spam_ham_ratio(df, spam_ham_ratio)
    if combination_mode == "mixed_50_50":
        return _apply_spam_ham_ratio(df, 0.5)
    if combination_mode == "source_aware_50_50":
        return _apply_source_aware_50_50(df, source_aware_max_multiplier)
    if combination_mode == "balanced_source_50_50":
        return _apply_balanced_source_50_50(df)
    raise ValueError(f"Unsupported combination_mode: {combination_mode}")


def combine_datasets(
    dataset_names: str | Sequence[str],
    spam_ham_ratio: float = -1,
    duplicate_detection: str = "high",
    generate_duplicate_report: bool = False,
    combination_mode: str = "mixed",
    source_aware_max_multiplier: float = DEFAULT_SOURCE_AWARE_MAX_SMALLEST_MULTIPLIER,
) -> str:
    _validate_spam_ham_ratio(spam_ham_ratio)
    duplicate_detection = _validate_duplicate_detection(duplicate_detection)
    combination_mode = _validate_combination_mode(combination_mode)
    source_aware_max_multiplier = float(source_aware_max_multiplier)
    if source_aware_max_multiplier < 1:
        raise ValueError("source_aware_max_multiplier must be >= 1.")
    if combination_mode != "mixed" and spam_ham_ratio not in {-1, 0.5}:
        raise ValueError(
            "Fixed 50/50 combination modes ignore spam_ham_ratio; "
            "use spam_ham_ratio=-1 or 0.5, or combination_mode='mixed'."
        )
    effective_spam_ham_ratio = 0.5 if combination_mode != "mixed" else spam_ham_ratio
    atomic_dataset_names = _resolve_atomic_dataset_names(dataset_names)
    output_path = _combined_dataset_path(
        atomic_dataset_names,
        effective_spam_ham_ratio,
        duplicate_detection,
        combination_mode,
        source_aware_max_multiplier,
    )
    report_path = _duplicate_report_path(atomic_dataset_names, spam_ham_ratio, duplicate_detection)

    if os.path.exists(output_path):
        print(f"Combined dataset already exists: {output_path}")
        if generate_duplicate_report:
            if os.path.exists(report_path):
                print("Duplicate report requested; regenerating it to ensure the latest report format.")
            else:
                print("Duplicate report requested and missing; rebuilding inputs to generate it.")
        else:
            return output_path
    elif not generate_duplicate_report:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if generate_duplicate_report:
        os.makedirs(REPORTS_DIR, exist_ok=True)

    if os.path.exists(output_path) and not generate_duplicate_report:
        return output_path

    print("=== Downloading required raw datasets ===")
    for name in atomic_dataset_names:
        print(f"--- {name} ---")
        ATOMIC_DATASET_DOWNLOADERS[name]()

    print("\n=== Building combined dataset ===")
    frames = []
    for name in atomic_dataset_names:
        print(f"--- {name} ---")
        frame = ATOMIC_DATASET_BUILDERS[name]()
        print(f"  Loaded {len(frame)} rows")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined rows before balancing: {len(combined)}")

    combined = _drop_empty_subject_and_body(combined)
    if generate_duplicate_report:
        _generate_duplicate_report(
            combined,
            dataset_names=atomic_dataset_names,
            spam_ham_ratio=spam_ham_ratio,
            duplicate_detection=duplicate_detection,
        )
    combined = _drop_duplicate_messages(combined, duplicate_detection)
    print(f"Combination mode: {combination_mode}")
    combined = _apply_combination_mode(
        combined,
        combination_mode,
        spam_ham_ratio,
        source_aware_max_multiplier,
    )
    combined = _shuffle_dataframe(combined)
    combined.to_parquet(output_path, index=False)

    print("=== Final dataset ===")
    print(f"  Rows: {len(combined)}")
    print(f"  Spam: {int(combined['label'].sum())}")
    print(f"  Ham: {int((combined['label'] == 0).sum())}")
    print(f"  Sources:\n{combined['source'].value_counts().to_string()}")
    print(f"  Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    combine_datasets("all", spam_ham_ratio=-1)
