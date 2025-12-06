"""Utilities to register the best AutoML child runs as models."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def list_completed_jobs(ml_client: MLClient, experiments: Iterable[str]) -> dict[str, str]:
    """Return a mapping of job name to experiment for completed jobs in the provided experiments."""

    completed = {}
    for job in ml_client.jobs.list(list_view_type="All"):
        if job.experiment_name in experiments and job.status == "Completed":
            completed[job.name] = job.experiment_name
    logger.info("Found completed jobs", extra={"jobs": list(completed.keys())})
    return completed


def register_best_child_run(
    ml_client: MLClient,
    *,
    job_name: str,
    experiment_name: str,
    skip_on_missing_best_child: bool = True,
    skip_on_model_error: bool = True,
) -> Optional[Model]:
    """Register the best child run from an AutoML parent job as an MLflow model.

    Parameters
    ----------
    ml_client:
        Azure ML client instance.
    job_name:
        The AutoML parent job name to inspect.
    experiment_name:
        The experiment that produced the job.
    skip_on_missing_best_child:
        If True, log and skip registration when the best child run tag is missing.
        If False, propagate the error.
    skip_on_model_error:
        If True, log and skip registration when model creation fails.
        If False, propagate the error.
    """

    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(
        "Registering best child run",
        extra={"job_name": job_name, "experiment": experiment_name},
    )

    parent_run = mlflow.get_run(job_name)
    best_child_run_id = parent_run.data.tags.get("automl_best_child_run_id")
    if not best_child_run_id:
        message = "Best child run id not found on parent run"
        logger.warning(
            message,
            extra={"job_name": job_name, "experiment": experiment_name, "tags": parent_run.data.tags},
        )
        if skip_on_missing_best_child:
            return None
        raise KeyError(message)
    model_name = f"{experiment_name.replace(' ', '_').lower()}_{job_name}_model"

    try:
        model = ml_client.models.create_or_update(
            Model(
                path=f"azureml://jobs/{best_child_run_id}/outputs/artifacts/outputs/mlflow-model/",
                name=model_name,
                description="AutoML classification model registered from best child run",
                type=AssetTypes.MLFLOW_MODEL,
            )
        )
    except Exception:
        logger.exception(
            "Failed to register model from child run",
            extra={
                "job_name": job_name,
                "experiment": experiment_name,
                "best_child_run_id": best_child_run_id,
                "model_name": model_name,
            },
        )
        if skip_on_model_error:
            return None
        raise
    logger.info("Registered model", extra={"model_name": model.name, "version": model.version})
    return model


__all__ = ["list_completed_jobs", "register_best_child_run"]
