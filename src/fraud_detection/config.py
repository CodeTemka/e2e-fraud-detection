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
        env_file=[ROOT_DIR / ".env", ROOT_DIR / ".env.generated"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars instead of erroring
    )
    
    # Azure
    subscription_id: str | None = Field(None, alias="SUBSCRIPTION_ID")
    resource_group: str | None = Field(None, alias="RESOURCE_GROUP")
    workspace_name: str | None = Field(None, alias="WORKSPACE_NAME")
    location: str | None = Field(None, alias="LOCATION")
    aml_compute: str | None = Field(None, alias="AML_COMPUTE")
    aml_dataset: str | None = Field(None, alias="AML_DATASET")

    # Github
    github_token: str | None = Field(None, alias="GITHUB_TOKEN")
    github_owner: str = Field(None, alias="GITHUB_OWNER")
    github_repo: str = Field(None, alias="GITHUB_REPO")

    # Kaggle
    kaggle_username: str | None = Field(None, alias="KAGGLE_USERNAME")
    kaggle_key: str | None = Field(None, alias="KAGGLE_KEY")

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
    
    def require_training(self) -> Settings:
        """ Ensure all settings required for training are provided. """
        return self._require(
            {
                "aml_compute": "AML_COMPUTE",
                "aml_dataset": "AML_DATASET",
            },
            context="Training",
        )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """ Get the application settings with caching. """
    return Settings()


__all__ = ["get_settings", "Settings"]
