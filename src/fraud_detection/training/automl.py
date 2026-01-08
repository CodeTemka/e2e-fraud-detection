"""Opinionated helpers to submit Azure AutoML jobs."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from azure.ai.ml import Input, MLClient, automl
from azure.ai.ml.automl import ClassificationPrimaryMetrics
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.config import (
    build_idempotency_key,
    build_job_name,
    get_git_sha,
    get_settings,
    preflight_validate_training_data,
    slugify,
)
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
    idempotency_key: str | None = None


def create_automl_job(config: AutoMLJobConfig) -> Any:
    """Create a configured AutoML classification job."""
    if not config.training_data:
        raise ValueError("config.training_data is empty. Provide an MLTable asset path/ID/URI.")
    if not config.experiment_name:
        raise ValueError("config.experiment_name is empty.")
    if not config.target_column:
        raise ValueError("config.target_column is empty.")

    validated = preflight_validate_training_data(config.training_data)
    if validated:
        logger.info("Preflight data validation passed", extra={"dataset": config.training_data})
    else:
        logger.info("Preflight data validation skipped", extra={"dataset": config.training_data})

    compute_target = (config.compute or "").strip()
    if not compute_target:
        settings = get_settings()
        compute_target = settings.get_training_compute()
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

    idempotency_key = config.idempotency_key or build_idempotency_key(
        config.experiment_name,
        str(config.primary_metric),
    )
    desired_name = config.job_name or build_job_name("automl", idempotency_key)
    job.name = slugify(desired_name)
    return job


def submit_job(ml_client: MLClient, job: Any) -> str:
    """Submit a job and return the created job name."""
    job_name = getattr(job, "name", None)
    if job_name and hasattr(ml_client.jobs, "get"):
        try:
            existing = ml_client.jobs.get(job_name)
        except ResourceNotFoundError:
            existing = None
        else:
            status = getattr(existing, "status", None)
            logger.info(
                "Existing AutoML job reused",
                extra={"job_name": existing.name, "status": status},
            )
            return existing.name
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
    compute_target = (compute or "").strip() or settings.get_training_compute()

    idempotency_key = build_idempotency_key(settings.automl_train_exp, resolved_metric)

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
            "git_sha": get_git_sha(short=False),
            "idempotency_key": idempotency_key,
            "allowed_algorithms": ",".join(allowed_algorithms) if allowed_algorithms else "auto",
        },
        max_trials=80,
        job_name=build_job_name("automl", idempotency_key),
        idempotency_key=idempotency_key,
    )


__all__ = [
    "AutoMLJobConfig",
    "create_automl_job",
    "submit_job",
    "automl_job_builder",
    "SUPPORTED_CLASSIFICATION_METRICS",
]
