.PHONY: install lint test format

install:
pip install -r requirements-dev.txt

lint:
ruff check src tests train/auto_ml/train_recall.py train/custom_train/train_script.py azure_connection/ml_client_setup.py

format:
ruff format src train deploy azure_connection

test:
pytest
