import os
import email
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_root, "raw_datasets")
OUTPUT = os.path.join(_root, "combined_datasets", "spam_2007_2008", "dataset.parquet")

TARGET_PER_CLASS = 7500
SEED = 42


def extract_trec_2007() -> pd.DataFrame:
    path = os.path.join(RAW, "trec_2007", "email_origin.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        raw = str(row["origin"])
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

        records.append({"subject": subject, "body": body, "label": int(row["label"]), "source": "trec_2007"})

    return pd.DataFrame(records)


def extract_ceas_2008() -> pd.DataFrame:
    path = os.path.join(RAW, "ceas_2008", "CEAS_08.csv")
    df = pd.read_csv(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "subject": str(row["subject"]) if pd.notna(row["subject"]) else "",
            "body": str(row["body"]) if pd.notna(row["body"]) else "",
            "label": int(row["label"]),
            "source": "ceas_2008",
        })

    return pd.DataFrame(records)


def combine() -> str:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print("Processing trec_2007...")
    trec_df = extract_trec_2007()
    print(f"  {len(trec_df)} rows — spam: {trec_df['label'].sum()}, ham: {(trec_df['label']==0).sum()}")

    print("Processing ceas_2008...")
    ceas_df = extract_ceas_2008()
    print(f"  {len(ceas_df)} rows — spam: {ceas_df['label'].sum()}, ham: {(ceas_df['label']==0).sum()}")

    combined = pd.concat([trec_df, ceas_df], ignore_index=True)

    spam = combined[combined["label"] == 1].sample(n=TARGET_PER_CLASS, random_state=SEED)
    ham = combined[combined["label"] == 0].sample(n=TARGET_PER_CLASS, random_state=SEED)
    balanced = pd.concat([spam, ham], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    balanced.to_parquet(OUTPUT, index=False)

    print(f"\nBalanced: {len(balanced)} rows — spam: {balanced['label'].sum()}, ham: {(balanced['label']==0).sum()}")
    print(f"Source breakdown:\n{balanced['source'].value_counts().to_string()}")
    print(f"Saved to: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    combine()
