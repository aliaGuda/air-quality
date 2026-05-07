from pathlib import Path
import zipfile

import requests


DATA_DIR = Path("data/raw")
ZIP_PATH = DATA_DIR / "AirQualityUCI.zip"
CSV_PATH = DATA_DIR / "AirQualityUCI.csv"

URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        print(f"Dataset already exists: {CSV_PATH}")
        return

    print("Downloading UCI Air Quality dataset...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    ZIP_PATH.write_bytes(response.content)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Expected dataset not found: {CSV_PATH}")

    print(f"Dataset downloaded successfully: {CSV_PATH}")


if __name__ == "__main__":
    main()