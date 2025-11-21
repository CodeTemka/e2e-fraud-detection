"""Utilities for submitting Azure AutoML jobs."""
from __future__ import annotations

from azure.ai.ml import Input, automl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Job

from fraud_detection.azure.client import create_ml_client

DEFAULT_PRIMARY_METRIC = "norm_macro_recall"


def build_automl_job(
    *,
    training_data: str,
    target_column: str = "Class",
    experiment_name: str = "automl-fraud-classification",
    compute: str = "cpu-cluster",
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    tags: dict[str, str] | None = None,
    n_cross_validations: int = 5,
    timeout_minutes: int = 120,
    trial_timeout_minutes: int = 30,
    max_trials: int = 20,
    max_concurrent_trials: int = 4,
) -> Job:
    """Create an Azure AutoML classification job definition."""

    data_input = Input(type=AssetTypes.MLTABLE, path=training_data)
    job = automl.classification(
        compute=compute,
        experiment_name=experiment_name,
        training_data=data_input,
        target_column_name=target_column,
        primary_metric=primary_metric,
        n_cross_validations=n_cross_validations,
        enable_model_explainability=True,
        tags=tags or {"project": "fraud-detection", "stage": "automl"},
    )

    job.set_limits(
        timeout_minutes=timeout_minutes,
        trial_timeout_minutes=trial_timeout_minutes,
        max_trials=max_trials,
        enable_early_termination=True,
        max_concurrent_trials=max_concurrent_trials,
    )
    return job


def submit_automl_job(job: Job):
    """Submit an AutoML job to Azure ML and return the created job."""

    client = create_ml_client()
    return client.jobs.create_or_update(job)
