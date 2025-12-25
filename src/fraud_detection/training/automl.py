"""Opinionated helpers to submit Azure AutoML jobs."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from azure.ai.ml import Input, MLClient, automl
from azure.ai.ml.automl import ClassificationPrimaryMetrics
from azure.ai.ml.constants import AssetTypes

from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


# Supported metrics for Azure AutoML classification jobs (your strict list)
SUPPORTED_CLASSIFICATION_METRICS: set[str] = {
    "accuracy",
    "AUC_weighted",
    "average_precision_score_weighted",
    "norm_macro_recall",
    "precision_score_weighted",
}


def _slug(s: str) -> str:
    """Make a string safe for Azure ML names (lowercase, hyphen-separated)."""
    s = (s or "").lower().strip()
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "job"


def _get_git_sha() -> str:
    """Return the current git commit SHA for tagging experiments (best-effort)."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = cp.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _build_job_name(metric: str) -> str:
    """Build a unique job name based on metric, timestamp, and git sha."""
    stamp = datetime.now(UTC).strftime("%m%d-%H%M")
    short_sha = _get_git_sha()[:5]
    return _slug(f"{metric}-{stamp}-{short_sha}")


@dataclass
class AutoMLJobConfig:
    """Configuration for an AutoML classification job."""

    experiment_name: str
    primary_metric: ClassificationPrimaryMetrics | str
    compute: str | None
    training_data: str

    target_column: str = "Class"
    cross_validations: int = 5

    tags: dict[str, str] = field(default_factory=dict)
    allowed_algorithms: Iterable[str] | None = None  # None => AutoML chooses

    timeout_minutes: int = 480
    trial_timeout_minutes: int = 60
    max_trials: int = 20
    max_concurrent_trials: int = 2
    enable_early_termination: bool = True

    job_name: str | None = None


def create_automl_job(config: AutoMLJobConfig) -> Any:
    """Create a configured AutoML classification job."""
    if not config.training_data:
        raise ValueError("config.training_data is empty. Provide an MLTable asset path/ID/URI.")
    if not config.experiment_name:
        raise ValueError("config.experiment_name is empty.")
    if not config.target_column:
        raise ValueError("config.target_column is empty.")

    settings = get_settings()
    compute_target = (config.compute or "").strip() or settings.get_training_compute()
    logger.info("Preparing AutoML job", extra={"experiment": config.experiment_name})

    data_input = Input(type=AssetTypes.MLTABLE, path=config.training_data)

    job = automl.classification(
        compute=compute_target,
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
        max_concurrent_trials=config.max_concurrent_trials,
        enable_early_termination=config.enable_early_termination,
    )

    # If allowed_algorithms is provided, restrict training algorithms.
    # If None/empty => AutoML tries its default algorithm set.
    if config.allowed_algorithms:
        job.set_training(allowed_training_algorithms=list(config.allowed_algorithms))

    desired_name = config.job_name or _build_job_name(metric=str(config.primary_metric))
    job.name = _slug(desired_name)
    return job


def submit_job(ml_client: MLClient, job: Any) -> str:
    """Submit a job and return the created job name."""
    returned_job = ml_client.jobs.create_or_update(job)
    logger.info("Submitted AutoML job", extra={"job_name": returned_job.name})
    return returned_job.name


def _metric_check(metric: str | ClassificationPrimaryMetrics) -> str:
    metric_str = metric.value if isinstance(metric, ClassificationPrimaryMetrics) else str(metric)
    metric_str = metric_str.strip().replace("-", "_")  # accept hyphen or underscore

    if metric_str not in SUPPORTED_CLASSIFICATION_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_CLASSIFICATION_METRICS))
        raise ValueError(f"Unsupported metric '{metric_str}'. Choose one of: {allowed}")

    return metric_str



def automl_job_builder(
    *,
    metric: str | ClassificationPrimaryMetrics,
    training_data: str,
    compute: str | None = None,
    allowed_algorithms: list[str] | None = None,
) -> AutoMLJobConfig:
    """Generic job builder based on specified metric (optionally restrict algorithms)."""
    resolved_metric = _metric_check(metric)
    settings = get_settings()
    # Use metric string for experiment/job naming (clean + stable)
    metric_slug = _slug(resolved_metric)
    compute_target = (compute or "").strip() or settings.get_training_compute()

    return AutoMLJobConfig(
        experiment_name=settings.automl_train_exp,
        primary_metric=resolved_metric,
        compute=compute_target,
        training_data=training_data,
        cross_validations=5,
        allowed_algorithms=allowed_algorithms,
        tags={
            "project": "fraud-detection",
            "metric": resolved_metric,
            "git_sha": _get_git_sha(),
            "allowed_algorithms": ",".join(allowed_algorithms) if allowed_algorithms else "auto",
        },
        max_trials=80,
        job_name=_build_job_name(metric=metric_slug),
    )


__all__ = [
    "AutoMLJobConfig",
    "create_automl_job",
    "submit_job",
    "automl_job_builder",
    "SUPPORTED_CLASSIFICATION_METRICS",
]
