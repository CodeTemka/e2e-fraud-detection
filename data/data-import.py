# downloading dataset from kaggle using kaggle api, not manually

import os
import json

with open('kaggle-token.json') as f:
    creds = json.load(f)
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']

try:
    os.path.exists('creditcard.csv')
    print('Dataset already exists. Skipping download.')
except FileNotFoundError as e:
    print('Dataset not found. Downloading...')
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('mlg-ulb/creditcardfraud', path='.', unzip=True)