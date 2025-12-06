"""Opinionated helpers to submit Azure AutoML jobs."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
import subprocess

from azure.ai.ml import Input, MLClient, automl
from azure.ai.ml.automl import ClassificationPrimaryMetrics
from azure.ai.ml.constants import AssetTypes

from fraud_detection.config import Settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AutoMLJobConfig:
    """Configuration for submitting a classification AutoML job."""

    experiment_name: str
    target_column: str
    primary_metric: ClassificationPrimaryMetrics | str
    compute: str
    training_data: str
    cross_validations: int = 5
    tags: dict[str, str] = field(default_factory=dict)
    allowed_algorithms: Iterable[str] | None = None
    timeout_minutes: int = 480
    trial_timeout_minutes: int = 60
    max_trials: int = 20
    max_concurrent_trials: int = 2
    enable_early_termination: bool = True


def create_automl_job(config: AutoMLJobConfig):
    """Create a configured AutoML classification job."""

    logger.info("Preparing AutoML job", extra={"experiment": config.experiment_name})

    data_input = Input(type=AssetTypes.MLTABLE, path=config.training_data)
    job = automl.classification(
        compute=config.compute,
        experiment_name=config.experiment_name,
        training_data=data_input,
        target_column_name=config.target_column,
        primary_metric=config.primary_metric,
        n_cross_validations=config.cross_validations,
        enable_model_explainability=True,
        tags=config.tags,
    )

    job.set_limits(
        timeout_minutes=config.timeout_minutes,
        trial_timeout_minutes=config.trial_timeout_minutes,
        max_trials=config.max_trials,
        enable_early_termination=config.enable_early_termination,
        max_concurrent_trials=config.max_concurrent_trials,
    )

    if config.allowed_algorithms:
        job.set_training(allowed_training_algorithms=list(config.allowed_algorithms))

    return job


def submit_job(ml_client: MLClient, job) -> str:
    """Submit a job and return the created job name."""

    returned_job = ml_client.jobs.create_or_update(job)
    logger.info("Submitted job", extra={"job_name": returned_job.name})
    return returned_job.name


def lightgbm_recall_job(settings: Settings) -> AutoMLJobConfig:
    """Preconfigured job optimized for recall on the fraud dataset."""

    return AutoMLJobConfig(
        experiment_name=_build_experiment_name(metric="recall"),
        target_column="Class",
        primary_metric="norm_macro_recall",
        compute=settings.default_compute,
        training_data=settings.default_dataset or "",
        cross_validations=10,
        tags={
            "project": "e2e-fraud-detection",
            "metric": "recall",
            "git_sha": _get_git_sha(),
        },
        max_trials=40,
        max_concurrent_trials=4,
    )


def lightgbm_accuracy_job(settings: Settings) -> AutoMLJobConfig:
    """Preconfigured job optimized for accuracy with only LightGBM allowed."""

    return AutoMLJobConfig(
        experiment_name=_build_experiment_name(metric="accuracy"),
        target_column="Class",
        primary_metric=ClassificationPrimaryMetrics.ACCURACY,
        compute=settings.default_compute,
        training_data=settings.default_dataset or "",
        cross_validations=5,
        allowed_algorithms=["LightGBM"],
        tags={
            "project": "e2e-fraud-detection",
            "metric": "accuracy",
            "git_sha": _get_git_sha(),
            "model": "LightGBM",
        },
        max_trials=80,
        max_concurrent_trials=4,
    )


def _build_experiment_name(metric: str) -> str:
    """Create a descriptive experiment name for Azure AutoML runs."""

    date_stamp = datetime.now().strftime("%Y%m%d")
    short_git_commit = _get_git_sha()[:7]
    return f"fraud-automl-{metric}-{date_stamp}-{short_git_commit}"


def _get_git_sha() -> str:
    """Return the current git commit SHA for tagging experiments."""

    completed_process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed_process.stdout.strip()


__all__ = [
    "AutoMLJobConfig",
    "create_automl_job",
    "submit_job",
    "lightgbm_recall_job",
    "lightgbm_accuracy_job",
]
