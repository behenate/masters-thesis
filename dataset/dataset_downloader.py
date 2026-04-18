import os
import json
import requests
import zipfile


def _kaggle_auth():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        with open(kaggle_json) as f:
            creds = json.load(f)
        return (creds["username"], creds["key"])
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return (username, key)
    return None


def download_dataset(url: str, save_path: str) -> str:
    if not save_path.endswith(".zip"):
        save_path = save_path + ".zip"
    full_path = os.path.abspath(os.path.expanduser(save_path))
    directory = os.path.dirname(full_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    auth = _kaggle_auth() if "kaggle.com" in url else None

    print(f"Downloading to {full_path}...")
    try:
        with requests.get(url, stream=True, auth=auth) as response:
            response.raise_for_status()
            with open(full_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return full_path

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        raise


def download_and_unzip_dataset(url: str, save_path: str) -> str:
    try:
        archive_path = download_dataset(url, save_path)
        extract_dir = os.path.dirname(archive_path)

        largest_csv_path = None
        max_size = 0

        print(f"Extracting to {extract_dir}...")
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

                for member in zip_ref.infolist():
                    if member.filename.lower().endswith(".csv"):
                        if member.file_size > max_size:
                            max_size = member.file_size
                            largest_csv_path = os.path.join(extract_dir, member.filename)

            print("Extraction complete.")

            os.remove(archive_path)
            print(f"Cleaned up: Removed {archive_path}")

            if largest_csv_path:
                full_csv_path = os.path.abspath(largest_csv_path)
                print(f"Found largest CSV in zip: {full_csv_path} ({max_size / 1024 / 1024:.2f} MB)")
                return full_csv_path
            else:
                print("No CSV files found inside the downloaded archive.")
                return None

        else:
            print("Error: The file is not a valid zip archive.")
            return None

    except Exception as e:
        print(f"Process failed: {e}")
        return None
