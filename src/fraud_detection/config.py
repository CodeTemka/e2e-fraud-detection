from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import subprocess
from typing import ClassVar

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud_detection.data.data_validation import ValidationOptions, validate_creditcard_data

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SUBSCRIPTION_ID = "aa4acb09-3611-4169-8504-1e68ad04a40f"
DEFAULT_RESOURCE_GROUP = "e2e-fraud-detection"
DEFAULT_WORKSPACE_NAME = "e2e-fraud-detection-ws"
DEFAULT_LOCATION = "eastasia"
# TODO: Define default budget limits and tag names.
DEFAULT_COST_CENTER_TAG_KEY = "cost_center"
DEFAULT_PROJECT_TAG_KEY = "project"
DEFAULT_COST_CENTER_TAG_VALUE = "unknown"
DEFAULT_PROJECT_TAG_VALUE = "e2e-fraud-detection"


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Azure (defaults in code; can be overridden by env vars)
    subscription_id: str = Field(
        default=DEFAULT_SUBSCRIPTION_ID,
        validation_alias=AliasChoices("SUBSCRIPTION_ID", "AZURE_SUBSCRIPTION_ID"),
    )
    resource_group: str = Field(
        default=DEFAULT_RESOURCE_GROUP,
        validation_alias=AliasChoices("RESOURCE_GROUP", "AZURE_RESOURCE_GROUP"),
    )
    workspace_name: str = Field(
        default=DEFAULT_WORKSPACE_NAME,
        validation_alias=AliasChoices("WORKSPACE_NAME", "AZURE_WORKSPACE_NAME"),
    )
    location: str = Field(
        default=DEFAULT_LOCATION,
        validation_alias=AliasChoices("LOCATION", "AZURE_LOCATION"),
    )

    # Github
    github_token: str | None = Field(None, alias="GITHUB_TOKEN")
    github_owner: str | None = Field(None, alias="GITHUB_OWNER")
    github_repo: str | None = Field(None, alias="GITHUB_REPO")

    # Kaggle
    kaggle_username: str | None = Field(None, alias="KAGGLE_USERNAME")
    kaggle_key: str | None = Field(None, alias="KAGGLE_KEY")

    # Default dataset
    dataset_name: str = Field(
        default="creditcard",
        validation_alias=AliasChoices("AML_DATASET", "DATASET_NAME"),
    )
    local_data_path: Path = ROOT_DIR / "data" / "creditcard.csv"
    registered_data_path: str = "azureml:creditcard:1"

    data_path_mltable: Path = ROOT_DIR / "data"

    # Default model
    default_metric: str = "norm_macro_recall"
    prod_model_name: str = "production-model"

    # Default experiments
    custom_train_exp: str = "custom-train"
    automl_train_exp: str = "automl-train"

    # Default compute
    training_compute: str = Field(
        default="cpu-cluster",
        validation_alias=AliasChoices(
            "AML_COMPUTE_TRAIN",
            "AML_COMPUTE",
            "COMPUTE_CLUSTER",
        ),
    )
    deployment_instance_type: str = Field(
        default="Standard_D2a_v4",
        validation_alias=AliasChoices("DEPLOYMENT_INSTANCE_TYPE"),
    )
    instance_type: str = "Standard_D4s_v3"
    instance_count: int = 1
    compute_min_nodes: int = 0
    compute_max_nodes: int = 2
    compute_idle_time_before_scale_down: int = 900

    cost_center_tag_key: str = Field(
        default=DEFAULT_COST_CENTER_TAG_KEY,
        validation_alias=AliasChoices("COST_CENTER_TAG_KEY"),
    )
    project_tag_key: str = Field(
        default=DEFAULT_PROJECT_TAG_KEY,
        validation_alias=AliasChoices("PROJECT_TAG_KEY"),
    )
    cost_center_tag_value: str = Field(
        default=DEFAULT_COST_CENTER_TAG_VALUE,
        validation_alias=AliasChoices("COST_CENTER"),
    )
    project_tag_value: str = Field(
        default=DEFAULT_PROJECT_TAG_VALUE,
        validation_alias=AliasChoices("PROJECT_NAME"),
    )

    # XGBoost sweep environment
    xgb_env_name: str = Field(
        default="xgb-sweep-env",
        validation_alias=AliasChoices("XGB_ENV_NAME"),
    )
    xgb_env_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("XGB_ENV_VERSION"),
    )

    # LightGBM sweep environment
    lgbm_env_name: str = Field(
        default="lgbm-sweep-env",
        validation_alias=AliasChoices("LGBM_ENV_NAME"),
    )
    lgbm_env_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LGBM_ENV_VERSION"),
    )

    # Scoring environment (online endpoints)
    scoring_env_name: str = Field(
        default="fraud-scoring-env",
        validation_alias=AliasChoices("SCORING_ENV_NAME"),
    )
    scoring_env_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SCORING_ENV_VERSION"),
    )
    scoring_env_file: Path = ROOT_DIR / "src" / "fraud_detection" / "serving" / "scoring_env.yaml"

    # Default endpoint
    endpoint_name: str = "fraud-detection"

    # Default deployment
    deployment_name: str = "blue"

    # Alert capacity defaults for scoring
    default_alert_cap: int = Field(
        default=100,
        validation_alias=AliasChoices("DEFAULT_ALERT_CAP"),
    )

    # Promotion config
    promotion_metric_epsilon: float = Field(
        default=1e-6,
        validation_alias=AliasChoices("PROMOTION_METRIC_EPSILON"),
    )
    promotion_min_metric: float | None = Field(
        default=None,
        validation_alias=AliasChoices("PROMOTION_MIN_METRIC"),
    )
    promotion_metric_delta: float = Field(
        default=0.0,
        validation_alias=AliasChoices("PROMOTION_METRIC_DELTA"),
    )

    # Default smoke test json
    smoke_test: Path = ROOT_DIR / "sample_request.json"

    def _require(self, required: dict[str, str], *, context: str) -> Settings:
        missing = [env for attr, env in required.items() if not getattr(self, attr)]
        if missing:
            raise ValueError(
                f"Missing required environment variables for {context}: {', '.join(missing)}"
            )
        return self

    def require_github(self) -> Settings:
        return self._require(
            {
                "github_token": "GITHUB_TOKEN",
                "github_owner": "GITHUB_OWNER",
                "github_repo": "GITHUB_REPO",
            },
            context="Github",
        )

    def require_kaggle(self) -> Settings:
        return self._require(
            {
                "kaggle_username": "KAGGLE_USERNAME",
                "kaggle_key": "KAGGLE_KEY",
            },
            context="Kaggle",
        )

    @property
    def compute_cluster(self) -> str:
        """Backward-compatible alias for training_compute."""
        return self.training_compute

    def get_training_compute(self) -> str:
        compute = (self.training_compute or "").strip()
        if not compute:
            raise ValueError("Training compute is not configured. Set TRAINING_COMPUTE.")
        return compute

    def get_deployment_compute(self) -> str:
        compute = (self.deployment_compute or "").strip()
        if not compute:
            raise ValueError(
                "Deployment compute is not configured. Set DEPLOYMENT_COMPUTE."
            )
        return compute

    def get_deployment_instance_type(self) -> str:
        instance_type = (self.deployment_instance_type or "").strip()
        if not instance_type:
            raise ValueError(
                "Deployment instance type is not configured. Set DEPLOYMENT_INSTANCE_TYPE."
            )
        return instance_type

    def get_resource_tags(self) -> dict[str, str]:
        return {
            self.project_tag_key: self.project_tag_value,
            self.cost_center_tag_key: self.cost_center_tag_value,
        }


def slugify(value: str) -> str:
    """Make a string safe for Azure ML asset names (lowercase, hyphen-separated)."""
    text = (value or "").lower().strip()
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "job"


def get_git_sha(*, short: bool = True) -> str:
    """Return the current git commit SHA (best-effort)."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    sha = cp.stdout.strip() or "unknown"
    return sha[:7] if short else sha


def build_idempotency_key(experiment: str, metric: str, *, git_sha: str | None = None) -> str:
    """Build a canonical idempotency key for job/model submissions."""
    resolved_sha = (git_sha or get_git_sha(short=True) or "unknown").strip() or "unknown"
    return slugify(f"{experiment}-{metric}-{resolved_sha}")


def build_job_name(prefix: str, idempotency_key: str) -> str:
    """Build a deterministic job name using a prefix and idempotency key."""
    return slugify(f"{prefix}-{idempotency_key}")


def _load_local_table(path: str, *, sample_rows: int | None = None):
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        import pandas as pd

        return pd.read_csv(p, nrows=sample_rows)

    if p.is_dir() and (p / "MLTable").is_file():
        import mltable

        tbl = mltable.load(str(p))
        if sample_rows is not None:
            tbl = tbl.take(int(sample_rows))
        return tbl.to_pandas_dataframe()

    raise ValueError("Unsupported path. Provide a .csv file or a folder containing an MLTable file.")


def preflight_validate_training_data(
    training_data: str,
    *,
    sample_rows: int = 1000,
    check_balance: bool = True,
) -> bool:
    """Validate local training data before job submission.

    Returns True if validation ran, False if skipped (non-local or remote dataset).
    """
    if not training_data:
        raise ValueError("training_data is empty. Provide an MLTable asset path/ID/URI.")

    if training_data.startswith("azureml:") or re.match(r"^[a-z]+://", training_data):
        return False

    data_path = Path(training_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_data}")

    df = _load_local_table(training_data, sample_rows=sample_rows)
    options = ValidationOptions(check_balance=check_balance)
    validate_creditcard_data(df, options=options)
    return True



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
