"""Configuration utilities for the fraud detection project."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        loc_by_alias=False,
    )

    subscription_id: str = Field(
        ..., validation_alias=AliasChoices("SUBSCRIPTION_ID", "AZURE_SUBSCRIPTION_ID")
    )
    resource_group: str = Field(
        ..., validation_alias=AliasChoices("RESOURCE_GROUP", "AZURE_RESOURCE_GROUP")
    )
    workspace_name: str = Field(
        ..., validation_alias=AliasChoices("WORKSPACE_NAME", "AZURE_WORKSPACE_NAME")
    )
    location: str = Field(..., validation_alias=AliasChoices("LOCATION", "AZURE_LOCATION"))

    training_compute: str = Field(
        ..., validation_alias=AliasChoices("AML_COMPUTE", "TRAINING_COMPUTE")
    )
    compute_cluster: str | None = Field(default=None, validation_alias=AliasChoices("COMPUTE_CLUSTER"))

    deployment_instance_type: str = Field(
        ..., validation_alias=AliasChoices("DEPLOYMENT_INSTANCE_TYPE")
    )
    instance_type: str = Field(
        default="Standard_DS3_v2", validation_alias=AliasChoices("INSTANCE_TYPE")
    )

    dataset_name: str = Field(..., validation_alias=AliasChoices("AML_DATASET", "DATASET_NAME"))
    registered_data_path: str | None = Field(
        default=None, validation_alias=AliasChoices("REGISTERED_DATA_PATH")
    )
    local_data_path: Path = Field(
        default=ROOT_DIR / "data" / "creditcard.csv",
        validation_alias=AliasChoices("LOCAL_DATA_PATH"),
    )
    smoke_test: Path = Field(
        default=ROOT_DIR / "sample_request.json",
        validation_alias=AliasChoices("SMOKE_TEST"),
    )

    default_metric: str = Field(
        default="norm_macro_recall", validation_alias=AliasChoices("DEFAULT_METRIC")
    )
    automl_train_exp: str = Field(
        default="automl-fraud-accuracy", validation_alias=AliasChoices("AUTOML_TRAIN_EXP")
    )
    custom_train_exp: str = Field(
        default="custom-fraud-train", validation_alias=AliasChoices("CUSTOM_TRAIN_EXP")
    )
    prod_model_name: str = Field(
        default="fraud-detection-model", validation_alias=AliasChoices("PROD_MODEL_NAME")
    )

    endpoint_name: str = Field(
        default="fraud-endpoint", validation_alias=AliasChoices("ENDPOINT_NAME")
    )
    deployment_name: str = Field(
        default="blue", validation_alias=AliasChoices("DEPLOYMENT_NAME")
    )
    instance_count: int = Field(default=1, validation_alias=AliasChoices("INSTANCE_COUNT"))
    default_alert_cap: int = Field(
        default=100, validation_alias=AliasChoices("DEFAULT_ALERT_CAP")
    )
    promotion_metric_epsilon: float = Field(
        default=0.0, validation_alias=AliasChoices("PROMOTION_METRIC_EPSILON")
    )

    xgb_env_name: str = Field(
        default="xgb-env", validation_alias=AliasChoices("XGB_ENV_NAME")
    )
    xgb_env_version: str | None = Field(
        default=None, validation_alias=AliasChoices("XGB_ENV_VERSION")
    )
    lgbm_env_name: str = Field(
        default="lgbm-env", validation_alias=AliasChoices("LGBM_ENV_NAME")
    )
    lgbm_env_version: str | None = Field(
        default=None, validation_alias=AliasChoices("LGBM_ENV_VERSION")
    )

    scoring_env_name: str = Field(
        default="fraud-scoring-env", validation_alias=AliasChoices("SCORING_ENV_NAME")
    )
    scoring_env_version: str | None = Field(
        default=None, validation_alias=AliasChoices("SCORING_ENV_VERSION")
    )
    scoring_env_file: Path = Field(
        default=ROOT_DIR / "src" / "fraud_detection" / "serving" / "scoring_env.yaml",
        validation_alias=AliasChoices("SCORING_ENV_FILE"),
    )

    compute_min_nodes: int = Field(
        default=0, validation_alias=AliasChoices("COMPUTE_MIN_NODES")
    )
    compute_max_nodes: int = Field(
        default=2, validation_alias=AliasChoices("COMPUTE_MAX_NODES")
    )
    compute_idle_time_before_scale_down: int = Field(
        default=120, validation_alias=AliasChoices("COMPUTE_IDLE_TIME_BEFORE_SCALE_DOWN")
    )

    resource_tags: str | None = Field(
        default=None, validation_alias=AliasChoices("RESOURCE_TAGS")
    )

    github_owner: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_OWNER"))
    github_repo: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_REPO"))
    github_token: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_TOKEN"))

    def model_post_init(self, __context: Any) -> None:
        if not self.compute_cluster:
            object.__setattr__(self, "compute_cluster", self.training_compute)
        if not self.registered_data_path:
            object.__setattr__(self, "registered_data_path", self.dataset_name)

    def get_training_compute(self) -> str:
        return (self.compute_cluster or self.training_compute).strip() or self.training_compute

    def get_deployment_instance_type(self) -> str:
        resolved = (self.deployment_instance_type or "").strip()
        return resolved or self.instance_type

    def get_resource_tags(self) -> dict[str, str]:
        tags: dict[str, str] = {"project": "fraud-detection"}
        if not self.resource_tags:
            return tags
        raw = self.resource_tags.strip()
        if not raw:
            return tags
        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return tags
            if isinstance(payload, dict):
                tags.update({str(k): str(v) for k, v in payload.items()})
            return tags
        for pair in raw.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                tags[key] = value
        return tags

    def require_github(self) -> "Settings":
        missing = [
            name
            for name, value in {
                "GITHUB_OWNER": self.github_owner,
                "GITHUB_REPO": self.github_repo,
                "GITHUB_TOKEN": self.github_token,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required GitHub settings: {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def build_idempotency_key(*parts: str, git_sha: str | None = None) -> str:
    pieces = [p.strip() for p in parts if p]
    if git_sha:
        pieces.append(git_sha)
    digest = hashlib.sha1("::".join(pieces).encode("utf-8")).hexdigest()
    return digest[:12]


def build_job_name(prefix: str, key: str) -> str:
    return f"{prefix}-{key}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "job"


def preflight_validate_training_data(path: str) -> bool:
    if not path:
        return False
    if path.startswith("azureml:"):
        return False
    data_path = Path(path)
    if data_path.is_dir():
        return (data_path / "MLTable").exists()
    return data_path.exists()


def get_git_sha(short: bool = True) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = [
    "ROOT_DIR",
    "Settings",
    "build_job_name",
    "build_idempotency_key",
    "get_git_sha",
    "get_settings",
    "preflight_validate_training_data",
    "slugify",
]
