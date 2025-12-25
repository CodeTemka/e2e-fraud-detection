from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SUBSCRIPTION_ID = "aa4acb09-3611-4169-8504-1e68ad04a40f"
DEFAULT_RESOURCE_GROUP = "e2e-fraud-detection"
DEFAULT_WORKSPACE_NAME = "e2e-fraud-detection-ws"
DEFAULT_LOCATION = "eastasia"


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Azure (defaults in code; can be overridden by env vars in CI)
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
            "TRAINING_COMPUTE",
            "AML_COMPUTE_TRAIN",
            "AML_COMPUTE",
            "COMPUTE_CLUSTER",
        ),
    )
    deployment_compute: str = Field(
        default="deploy-cluster",
        validation_alias=AliasChoices("DEPLOYMENT_COMPUTE", "AML_COMPUTE_DEPLOY"),
    )
    deployment_instance_type: str = Field(
        default="Standard_D3_v2",
        validation_alias=AliasChoices("DEPLOYMENT_INSTANCE_TYPE"),
    )
    instance_type: str = "Standard_DS3_v2"
    instance_count: int = 1
    compute_min_nodes: int = 0
    compute_max_nodes: int = 2

    # Default endpoint
    endpoint_name: str = "fraud-detection"

    # Default deployment
    deployment_name: str = "blue"

    # Default smoke test json
    smoke_test: Path = ROOT_DIR / "sample_request.json"

    def _require(self, required: dict[str, str], *, context: str) -> "Settings":
        missing = [env for attr, env in required.items() if not getattr(self, attr)]
        if missing:
            raise ValueError(
                f"Missing required environment variables for {context}: {', '.join(missing)}"
            )
        return self

    def require_azure(self) -> "Settings":
        # With defaults set above, this will pass in CI even without .env
        return self._require(
            {
                "subscription_id": "SUBSCRIPTION_ID",
                "resource_group": "RESOURCE_GROUP",
                "workspace_name": "WORKSPACE_NAME",
                "location": "LOCATION",
            },
            context="Azure",
        )

    def require_github(self) -> "Settings":
        return self._require(
            {
                "github_token": "GITHUB_TOKEN",
                "github_owner": "GITHUB_OWNER",
                "github_repo": "GITHUB_REPO",
            },
            context="Github",
        )

    def require_kaggle(self) -> "Settings":
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



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
