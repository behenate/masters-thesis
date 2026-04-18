import email
import hashlib
import json
import mailbox
import os
from collections.abc import Iterable, Sequence

import pandas as pd

try:
    from combiners import spam_2007_2008, spam_detection
    from downloaders import (
        ceas_2008,
        enron,
        fraud_email,
        fraudulent_email_corpus,
        phishing_email,
        spam_assassin,
        spam_ham,
        spam_mail,
        trec_2007,
    )
except ImportError:  # pragma: no cover - allows package-style imports
    from .combiners import spam_2007_2008, spam_detection
    from .downloaders import (
        ceas_2008,
        enron,
        fraud_email,
        fraudulent_email_corpus,
        phishing_email,
        spam_assassin,
        spam_ham,
        spam_mail,
        trec_2007,
    )


SEED = 67
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "combined_datasets",
    "generated",
)
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_datasets")


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


def build_spam_mail() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "spam_mail", "spam_dataset.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "subject": str(row["title"]) if pd.notna(row["title"]) else "",
            "body": str(row["text"]) if pd.notna(row["text"]) else "",
            "label": 1 if str(row["type"]).strip().lower() == "spam" else 0,
            "source": "spam_mail",
        })

    return pd.DataFrame(records)


def build_trec_2007() -> pd.DataFrame:
    return spam_2007_2008.extract_trec_2007()


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


def build_fraud_email() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "fraud_email", "fraud_email_.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        subject, body = _extract_subject_and_body_from_raw_email(str(row["Text"]))
        records.append({
            "subject": subject,
            "body": body,
            "label": int(row["Class"]),
            "source": "fraud_email",
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


def build_phishing_email() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "phishing_email", "phishing_email.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "subject": "",
            "body": str(row["text_combined"]) if pd.notna(row["text_combined"]) else "",
            "label": int(row["label"]),
            "source": "phishing_email",
        })

    return pd.DataFrame(records)


ATOMIC_DATASET_BUILDERS = {
    "ceas_2008": build_ceas_2008,
    "enron": build_enron,
    "fraud_email": build_fraud_email,
    "fraudulent_email_corpus": build_fraudulent_email_corpus,
    "phishing_email": build_phishing_email,
    "spam_assassin": build_spam_assassin,
    "spam_ham": build_spam_ham,
    "spam_mail": build_spam_mail,
    "trec_2007": build_trec_2007,
}

ATOMIC_DATASET_DOWNLOADERS = {
    "ceas_2008": ceas_2008.download,
    "enron": enron.download,
    "fraud_email": fraud_email.download,
    "fraudulent_email_corpus": fraudulent_email_corpus.download,
    "phishing_email": phishing_email.download,
    "spam_assassin": spam_assassin.download,
    "spam_ham": spam_ham.download,
    "spam_mail": spam_mail.download,
    "trec_2007": trec_2007.download,
}

DATASET_ALIASES = {
    "all": tuple(sorted(ATOMIC_DATASET_BUILDERS)),
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


DATASET_FUNCTIONS = {
    **ATOMIC_DATASET_BUILDERS,
    "all": build_all,
    "spam_2007_2008": build_spam_2007_2008,
    "spam_detection": build_spam_detection,
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


def _ratio_token(spam_ham_ratio: float) -> str:
    if spam_ham_ratio == -1:
        return "all_samples"
    return f"spam_{str(spam_ham_ratio).replace('.', '_')}"


def _combined_dataset_path(dataset_names: list[str], spam_ham_ratio: float) -> str:
    spec = {
        "dataset_names": dataset_names,
        "spam_ham_ratio": spam_ham_ratio,
    }
    digest = hashlib.sha1(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    stem = "__".join(dataset_names)
    filename = f"{stem}__{_ratio_token(spam_ham_ratio)}__{digest}.parquet"
    return os.path.join(OUTPUT_DIR, filename)


def _shuffle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def _drop_empty_subject_or_body(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["subject"] = cleaned["subject"].fillna("").astype(str).str.strip()
    cleaned["body"] = cleaned["body"].fillna("").astype(str).str.strip()
    filtered = cleaned[(cleaned["subject"] != "") & (cleaned["body"] != "")].reset_index(drop=True)
    removed = len(df) - len(filtered)
    print(f"Removed rows with empty subject or body: {removed}")
    return filtered


def _drop_duplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    deduplicated = df.drop_duplicates(subset=["subject", "body", "label"]).reset_index(drop=True)
    removed = len(df) - len(deduplicated)
    print(f"Removed duplicate rows: {removed}")
    return deduplicated


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


def combine_datasets(dataset_names: str | Sequence[str], spam_ham_ratio: float = -1) -> str:
    _validate_spam_ham_ratio(spam_ham_ratio)
    atomic_dataset_names = _resolve_atomic_dataset_names(dataset_names)
    output_path = _combined_dataset_path(atomic_dataset_names, spam_ham_ratio)

    if os.path.exists(output_path):
        print(f"Combined dataset already exists: {output_path}")
        return output_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    combined = _drop_empty_subject_or_body(combined)
    combined = _drop_duplicate_messages(combined)
    combined = _apply_spam_ham_ratio(combined, spam_ham_ratio)
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
