# Downloading dataset from kaggle using kaggle api
# Dataset:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Make sure to have kaggle credentials set in environment variables or kaggle.json file

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

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