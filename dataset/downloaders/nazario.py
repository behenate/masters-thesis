import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataset_downloader import download_and_unzip_dataset, get_existing_dataset_path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/zeyadkarimfcai/nazario-5-datatest"
DATASET_DIR = os.path.join(_root, "raw_datasets", "nazario")
EXPECTED_PATH = os.path.join(DATASET_DIR, "Nazario.csv")


def _normalize_downloaded_csv(path: str | None) -> str | None:
    if path is None:
        return None

    full_path = os.path.abspath(path)
    if full_path == os.path.abspath(EXPECTED_PATH):
        return EXPECTED_PATH

    os.replace(full_path, EXPECTED_PATH)
    print(f"Renamed downloaded CSV to: {EXPECTED_PATH}")
    return EXPECTED_PATH


def download() -> str | None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    existing = get_existing_dataset_path(EXPECTED_PATH)
    if existing:
        return existing

    downloaded = download_and_unzip_dataset(DATASET_URL, os.path.join(DATASET_DIR, "nazario"))
    return _normalize_downloaded_csv(downloaded)


if __name__ == "__main__":
    download()
