"""Configuration management for the fraud detection project."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    subscription_id: str = Field(..., alias="SUBSCRIPTION_ID")
    resource_group: str = Field(..., alias="RESOURCE_GROUP")
    workspace_name: str = Field(..., alias="WORKSPACE_NAME")
    default_compute: str = Field(
        "automated-ml-cluster",
        alias="AML_COMPUTE",
        description="Default compute target used for Azure ML jobs.",
    )
    default_dataset: str | None = Field(
        "azureml:mltable_creditcard_data:1",
        alias="AML_DATASET",
        description="Default MLTable asset for training jobs.",
    )
    github_token: str | None = Field(
        None,
        alias="GITHUB_TOKEN",
        description="Optional GitHub token used by helper scripts.",
    )
    github_owner: str | None = Field(
        None,
        alias="GITHUB_OWNER",
        description="Optional GitHub repository owner for automation scripts.",
    )
    github_repo: str | None = Field(
        None,
        alias="GITHUB_REPO",
        description="Optional GitHub repository name for automation scripts.",
    )
    kaggle_username: str | None = Field(
        None,
        alias="KAGGLE_USERNAME",
        description="Optional Kaggle username when downloading datasets.",
    )
    kaggle_key: str | None = Field(
        None,
        alias="KAGGLE_KEY",
        description="Optional Kaggle API key when downloading datasets.",
    )
    location: str | None = Field(
        None,
        alias="LOCATION",
        description="Optional Azure region to target for resource creation.",
    )

    @model_validator(mode="after")
    def validate_dataset(self) -> "Settings":
        """Ensure a default dataset is configured for training jobs."""

        if not self.default_dataset:
            msg = "Set AML_DATASET to the MLTable asset used for training jobs."
            raise ValueError(msg)
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Load settings once and cache the result."""

    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "get_settings"]
