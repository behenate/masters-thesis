import os
import sys
import email
import re
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_root, "raw_datasets")
OUTPUT = os.path.join(_root, "combined_datasets", "spam_detection", "dataset.parquet")


def extract_spam_ham() -> pd.DataFrame:
    path = os.path.join(RAW, "spam_ham", "spam_ham_dataset.csv")
    df = pd.read_csv(path, index_col=0)

    records = []
    for _, row in df.iterrows():
        text = str(row["text"])
        first_newline = text.find("\n")
        if first_newline != -1:
            subject = text[:first_newline].replace("Subject:", "", 1).strip()
            body = text[first_newline + 1:].strip()
        else:
            subject = text.replace("Subject:", "", 1).strip()
            body = ""
        label = 1 if row["label"] == "spam" else 0
        records.append({"subject": subject, "body": body, "label": label, "source": "spam_ham"})

    return pd.DataFrame(records)


def extract_spam_assassin() -> pd.DataFrame:
    path = os.path.join(RAW, "spam_assassin", "spam_assassin.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        raw = str(row["text"])
        try:
            msg = email.message_from_string(raw)
            subject = msg.get("Subject", "") or ""

            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(payload.decode("utf-8", errors="replace"))
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
                elif isinstance(msg.get_payload(), str):
                    body_parts.append(msg.get_payload())

            body = "\n".join(body_parts).strip()
        except Exception:
            subject = ""
            body = raw

        if not subject or not body:
            fallback_subject, fallback_body = extract_flattened_email(raw)
            if not subject:
                subject = fallback_subject
            if not body:
                body = fallback_body

        records.append({"subject": subject, "body": body, "label": int(row["target"]), "source": "spam_assassin"})

    return pd.DataFrame(records)


def extract_flattened_email(raw: str) -> tuple[str, str]:
    subject_match = re.search(
        r"\bSubject:\s*(.*?)\s+(?=(?:Date:|MIME-Version:|Content-Type:|Content-Transfer-Encoding:|"
        r"Message-Id:|X-[A-Za-z-]+:|Status:|To:|From:|Cc:|Reply-To:|$))",
        raw,
        flags=re.IGNORECASE,
    )
    subject = subject_match.group(1).strip() if subject_match else ""

    body = ""
    body_patterns = [
        r"\bContent-Transfer-Encoding:\s*[^\s]+\s+(.*)$",
        r"\bContent-Type:\s*[^:]+?\s+(.*)$",
        r"\bMIME-Version:\s*[^\s]+\s+(.*)$",
    ]

    for pattern in body_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            body = match.group(1).strip()
            break

    if not body:
        body = raw.strip()

    return subject, body


def build_dataset() -> pd.DataFrame:
    print("Processing spam_ham...")
    spam_ham_df = extract_spam_ham()
    print(f"  {len(spam_ham_df)} rows — spam: {spam_ham_df['label'].sum()}, ham: {(spam_ham_df['label']==0).sum()}")

    print("Processing spam_assassin...")
    spam_assassin_df = extract_spam_assassin()
    print(f"  {len(spam_assassin_df)} rows — spam: {spam_assassin_df['label'].sum()}, ham: {(spam_assassin_df['label']==0).sum()}")

    combined = pd.concat([spam_ham_df, spam_assassin_df], ignore_index=True)
    print(f"\nCombined: {len(combined)} rows — spam: {combined['label'].sum()}, ham: {(combined['label']==0).sum()}")
    return combined


def combine() -> str:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    combined = build_dataset()
    combined.to_parquet(OUTPUT, index=False)

    print(f"Saved to: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    combine()
