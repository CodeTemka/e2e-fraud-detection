# 🚀 End-to-End Fraud Detection (Azure ML)

A production-ready template for training and deploying fraud-detection models with **Azure Machine Learning**. The repository is organized for clarity, testability, and CI/CD automation.

## 📂 Repository Structure

```
e2e-fraud-detection/
├── src/fraud_detection/        # Reusable Python package
│   ├── azure/                  # Azure ML client helpers
│   └── training/               # AutoML + LightGBM pipelines
├── train/                      # CLI entrypoints for training
│   ├── auto_ml/train_recall.py
│   └── custom_train/train_script.py
├── deploy/                     # Deployment artifacts (endpoints, environments)
├── data/                       # Data assets (not tracked)
├── tests/                      # Unit tests
├── requirements-dev.txt        # Dev/test dependencies
├── Makefile                    # Automation for linting/tests
└── .github/workflows/ci.yaml   # CI pipeline (lint + tests)
```

## 🧠 Key Components

- **Azure ML connectivity**: Centralized client factory using environment variables for subscription, resource group, and workspace configuration.
- **Training pipelines**:
  - AutoML classification job builder focused on recall-oriented metrics.
  - Local LightGBM training pipeline with MLflow logging and optional model registration.
- **Quality gates**: Ruff-based linting and Pytest-driven unit tests, enforced through GitHub Actions.

## 🛠️ Getting Started

### 1. Set up a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Configure Azure ML credentials

Create a `.env` file with the required settings:

```
SUBSCRIPTION_ID=<your-azure-subscription>
RESOURCE_GROUP=<your-resource-group>
WORKSPACE_NAME=<your-aml-workspace>
```

### 3. Run quality checks locally

```bash
make lint
make test
```

### 4. Submit an AutoML job

```bash
python train/auto_ml/train_recall.py \
  --training-data azureml:mltable_creditcard_data:1 \
  --compute automated-ml-cluster
```

### 5. Train LightGBM locally

```bash
python train/custom_train/train_script.py \
  --data-path data/raw/creditcard.csv \
  --output-dir artifacts
```

## 📈 CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yaml`) installs dependencies, runs Ruff linting, and executes unit tests on pushes and pull requests targeting `main` or `master`. Extend this workflow with deployment steps (e.g., Azure ML job submission) to complete your MLOps pipeline.

## 📌 Future Enhancements

- SHAP-based explainability
- Data validation and drift monitoring
- Batch inference pipeline
- Automated endpoint deployment from CI

## 📄 License

MIT License.
