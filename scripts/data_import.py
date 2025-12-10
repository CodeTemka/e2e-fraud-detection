"""Download the Kaggle credit card fraud dataset if it is missing."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.config import get_settings  # noqa: E402
from fraud_detection.utils.logging import get_logger  # noqa: E402

settings = get_settings()
logger = get_logger(__name__)

os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
os.environ["KAGGLE_KEY"] = settings.kaggle_key

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "creditcard.csv"


def main() -> None:
    if DATA_FILE.exists():
        logger.info("Dataset already exists. Skipping download.")
        return

    logger.info("Dataset not found. Downloading...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files("mlg-ulb/creditcardfraud", path=DATA_DIR, unzip=True)


if __name__ == "__main__":
    main()
