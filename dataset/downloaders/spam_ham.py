import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataset_downloader import download_and_unzip_dataset, get_existing_dataset_path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/venky73/spam-mails-dataset"
DATASET_DIR = os.path.join(_root, "raw_datasets", "spam_ham")
EXPECTED_PATH = os.path.join(DATASET_DIR, "spam_ham_dataset.csv")


def download() -> str | None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    existing = get_existing_dataset_path(EXPECTED_PATH)
    if existing:
        return existing
    return download_and_unzip_dataset(DATASET_URL, os.path.join(DATASET_DIR, "spam_ham_dataset"))


if __name__ == "__main__":
    download()
