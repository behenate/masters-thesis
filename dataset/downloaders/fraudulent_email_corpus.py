import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataset_downloader import download_dataset, _kaggle_auth
import zipfile

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/rtatman/fraudulent-email-corpus"
DATASET_DIR = os.path.join(_root, "raw_datasets", "fraudulent_email_corpus")


def download() -> str | None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    archive_path = os.path.join(DATASET_DIR, "fraudulent_email_corpus.zip")

    try:
        archive_path = download_dataset(DATASET_URL, archive_path)

        print(f"Extracting to {DATASET_DIR}...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
            members = zip_ref.namelist()

        os.remove(archive_path)
        print(f"Cleaned up: Removed {archive_path}")

        for member in members:
            full_path = os.path.abspath(os.path.join(DATASET_DIR, member))
            print(f"Extracted: {full_path}")
            return full_path

        return None
    except Exception as e:
        print(f"Process failed: {e}")
        return None


if __name__ == "__main__":
    download()
