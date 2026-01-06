# End-to-End Fraud Detection

Production-ready scaffolding for Azure Machine Learning experiments that detect credit card fraud.
The repository now follows a `src/` layout, ships with a typed CLI, and comes with CI-ready linting
and unit tests.

## What you get
- Azure ML AutoML presets for recall-first and accuracy-first LightGBM models.
- Reusable utilities for Azure authentication and MLflow model registration.
- Typer-based CLI (`fraud-cli`) for submitting jobs and registering the best child runs.
- Conda environment + `pyproject.toml` for reproducible installs.
- GitHub Actions workflow for linting (Ruff) and testing (Pytest).

## Repository structure
```
├── .github/workflows/ci.yml     # CI pipeline
├── .env.example                 # Copy to .env with your Azure credentials
├── data/                        # (Optional) local data assets
├── notebooks/                   # EDA and training notebooks
├── scripts/                     # CLI entry points
├── src/fraud_detection/         # Application code (Azure + training utilities)
└── tests/                       # Unit tests
```

## Setup
```bash
# 1) Create environment (Conda recommended)
conda env create -f environment.yml
conda activate fraud-detection

# 2) Provide credentials
cp .env.example .env
# edit the values for your credentials

# 3) Install pinned dependencies (CI/reproducible installs)
pip install -r requirements.lock

# 4) Install the project (already handled by the env file, but works standalone too)
pip install -e .[dev]
```

To refresh the lockfile after updating dependencies, install the desired versions and run:
```bash
pip freeze > requirements.lock
```

## Usage
Quickstart (CLI flows):

1. Prep Azure resources and secrets.
2. Register a dataset.
3. Submit training jobs.
4. Register + promote the best model.
5. Deploy and invoke an endpoint.

See the detailed runbook and architecture docs in `docs/`:
- [Architecture overview](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Troubleshooting guide](docs/TROUBLESHOOTING.md)

Prepare required resources for Azure ML:
```bash
python scripts/az_prep.py
```

Ensure the required GitHub secret exists:
```bash
python scripts/git_add_secret.py
```

Register the local credit card dataset as an Azure ML data asset (defaults to `data/creditcard.csv`):
```bash
python scripts/register_local_data.py
```

Validate a local dataset before uploading/training:
```bash
fraud-cli validate-data --sample-rows 2000
```

Submit an AutoML job (recall-optimized by default):
```bash
fraud-cli submit-automl --metric recall
```

Force an accuracy-focused LightGBM-only search:
```bash
fraud-cli submit-automl --metric accuracy --algorithms lightgbm
```

Submit an XGBoost hyperparameter sweep (uses the registered data asset by default):
```bash
fraud-cli submit-xgb-sweep --metric average_precision
```

Optional dry-run to validate the job definition (still registers the XGBoost environment if missing):
```bash
fraud-cli submit-xgb-sweep --dry-run
```

Register the best model for the current AutoML experiment:
```bash
fraud-cli register-best-automl-model --best-by-metric --metric norm-macro-recall
```

Register the best child run for each completed AutoML job:
```bash
fraud-cli register-best-automl-model --best-child-run
```

Register and (if better) promote the best custom-trained model, then deploy with top-K alerting:
```bash
fraud-cli register-best-custom-model --metric average_precision_score_macro
```

Custom scoring payloads (online endpoint) accept raw feature values and optional alert caps:
```json
{
  "data": [{"Time": 0, "Amount": 123.4, "V1": 0.1, "...": 0.0}],
  "alert_cap": 100
}
```

Create an online endpoint, deploy the production model, and invoke it:
```bash
fraud-cli endpoint-create
fraud-cli deploy
fraud-cli serve-invoke
```

## Local quickstart (no Azure)
Train an XGBoost model locally using the sample CSV:
```bash
python src/fraud_detection/training/train_xgb.py --train_data data/creditcard.csv
```

## One-command test run
```bash
pytest
```

## Testing & Linting
```bash
ruff check .
pytest
```

## Notes
- The CLI relies on Azure credentials available to `DefaultAzureCredential` (environment variables,
  managed identity, or developer login).
- XGBoost sweep jobs use the Conda spec at `src/fraud_detection/training/xgb_env.yaml`. Override the
  registered environment via `XGB_ENV_NAME` / `XGB_ENV_VERSION` if needed.
- Large raw datasets should stay out of version control; the `.gitignore` excludes common artifacts.
- `.env.example` is a safe template checked into Git so collaborators know which variables to set.
  Copy it to `.env` locally (which is ignored by Git) and fill in your real subscription, resource
  group, workspace, compute, and dataset names—personal tokens stay on your machine only.
