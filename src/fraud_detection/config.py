"""Configuration management for the fraud detection project."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    github_token: str = Field(..., alias="GITHUB_TOKEN")
    github_owner: str = Field(..., alias="GITHUB_OWNER")
    github_repo: str = Field(..., alias="GITHUB_REPO")
    kaggle_username: str = Field(..., alias="KAGGLE_USERNAME")
    kaggle_key: str = Field(..., alias="KAGGLE_KEY")
    location: str = Field(..., alias="LOCATION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Load settings once and cache the result."""

    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "get_settings"]
