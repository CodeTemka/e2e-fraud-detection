""" Configuration settings for the fraud detection module. """
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """ Application wide settings loaded from environment variables. """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars instead of erroring
    )
    
    # Azure
    subscription_id: str | None = Field(None, alias="SUBSCRIPTION_ID")
    resource_group: str | None = Field(None, alias="RESOURCE_GROUP")
    workspace_name: str | None = Field(None, alias="WORKSPACE_NAME")
    location: str | None = Field(None, alias="LOCATION")

    # Github
    github_token: str | None = Field(None, alias="GITHUB_TOKEN")
    github_owner: str | None = Field(None, alias="GITHUB_OWNER")
    github_repo: str | None = Field(None, alias="GITHUB_REPO")

    # Kaggle
    kaggle_username: str | None = Field(None, alias="KAGGLE_USERNAME")
    kaggle_key: str | None = Field(None, alias="KAGGLE_KEY")

    # Default dataset
    dataset_name: str = "credit-card-data"
    local_data_path: Path = ROOT_DIR / "data" / "creditcard.csv"

    # Default model
    default_metric: str = "norm-macro-recall"
    prod_model_name: str = "production-model"

    # Default experiments
    custom_train_exp = 'custom-train'
    automl_train_exp = 'automl-train'

    # Default compute
    instance_type = "Standard_DS3_v2"
    instance_count = 1
    compute_cluster = ''
    compute_min_nodes = 0
    compute_max_nodes = 2

    # Default endpoint
    endpoint_name = "fraud-detection"

    # Default deployment
    deployment_name = 'blue'

    # Default smoke test json
    smoke_test = ROOT_DIR / "sample_request.json"


    def _require(self, required: dict[str, str], *, context: str) -> Settings:
        """ Internal helper to enforce required settings. """

        missing = [env for attr, env in required.items() if not getattr(self, attr)]

        if missing:
            raise ValueError(
                f"Missing required environment variables for {context}: {', '.join(missing)}"
            )
        return self
    
    
    def require_azure(self) -> Settings:
        """ Ensure Azure-specific settings are provided. """
        return self._require(
            {
                "subscription_id": "SUBSCRIPTION_ID",
                "resource_group": "RESOURCE_GROUP",
                "workspace_name": "WORKSPACE_NAME",
                "location": "LOCATION",
            },
            context="Azure",
        )
    

    def require_github(self) -> Settings:
        """ Ensure Github-specific settings are provided. """
        return self._require(
            {
                "github_token": "GITHUB_TOKEN",
                "github_owner": "GITHUB_OWNER",
                "github_repo": "GITHUB_REPO",
            },
            context="Github",
        )
    
    
    def require_kaggle(self) -> Settings:
        """ Ensure Kaggle-specific settings are provided. """
        return self._require(
            {
                "kaggle_username": "KAGGLE_USERNAME",
                "kaggle_key": "KAGGLE_KEY",
            },
            context="Kaggle",
        )
    
    
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """ Get the application settings with caching. """
    return Settings()


__all__ = ["get_settings", "Settings"]
