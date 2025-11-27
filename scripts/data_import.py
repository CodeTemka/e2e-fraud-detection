# Downloading dataset from kaggle using kaggle api
# Dataset:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Make sure to have kaggle credentials set in environment variables or kaggle.json file

import os
from pathlib import Path

from fraud_detection.config import get_settings

settings = get_settings()

os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
os.environ["KAGGLE_KEY"] = settings.kaggle_key

path = Path('data')

try:
    os.path.exists('creditcard.csv')
    print('Dataset already exists. Skipping download.')
except FileNotFoundError as e:
    print('Dataset not found. Downloading...')
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('mlg-ulb/creditcardfraud', path=path, unzip=True)