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

# 3) Install the project (already handled by the env file, but works standalone too)
pip install -e .[dev]
```

## Usage
Ensure must resources for azure service:
```bash
python srcipts/az_prep.py
```

Ensure must github secret is created:
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

Register the best model for the current AutoML experiment:
```bash
fraud-cli register-best-automl-model --best-by-metric --metric norm-macro-recall
```

Register the best child run for each completed AutoML job:
```bash
fraud-cli register-best-automl-model --best-child-run
```

Create an online endpoint, deploy the production model, and invoke it:
```bash
fraud-cli endpoint-create
fraud-cli deploy
fraud-cli serve-invoke
```

## Testing & Linting
```bash
ruff check .
pytest
```

## Notes
- The CLI relies on Azure credentials available to `DefaultAzureCredential` (environment variables,
  managed identity, or developer login).
- Large raw datasets should stay out of version control; the `.gitignore` excludes common artifacts.
- `.env.example` is a safe template checked into Git so collaborators know which variables to set.
  Copy it to `.env` locally (which is ignored by Git) and fill in your real subscription, resource
  group, workspace, compute, and dataset names—personal tokens stay on your machine only.
