import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataset_downloader import download_and_unzip_dataset

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/harshalpatil3558/spam-mail-prediction-dataset"
DATASET_DIR = os.path.join(_root, "raw_datasets", "spam_mail")


def download() -> str | None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    return download_and_unzip_dataset(DATASET_URL, os.path.join(DATASET_DIR, "spam_mail"))


if __name__ == "__main__":
    download()
