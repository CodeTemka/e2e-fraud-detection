"""Utilities to register the best AutoML child run as Azure ML models.
Used for registering best child runs from automl job,
Not recommended for a model to be deployed.
Instead use best_runs_by_metric.py to register a model to be deployed.
"""
from __future__ import annotations

from collections.abc import Iterable

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _model_name(job_run: str) -> str:
    return f"model_{job_run}"


def mlflow_tracking_uri(ml_client: MLClient) -> None:
    """Set MLflow tracking URI from workspace and return it."""
    uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)


def list_completed_jobs(ml_client: MLClient, experiments: Iterable[str] | str) -> dict[str, list[str]]:
    """Return mapping: {experiment_name -> [job_name, ...]} for completed jobs."""

    exp_list = [experiments] if isinstance(experiments, str) else list(experiments)
    exp_set = set(exp_list)

    completed: dict[str, list[str]] = {exp: [] for exp in exp_list}

    for job in ml_client.jobs.list():
        exp = getattr(job, "experiment_name", None)
        status = getattr(job, "status", None)
        name = getattr(job, "name", None)

        if exp in exp_set and status == "Completed" and name:
            completed.setdefault(exp, []).append(name)

    logger.info(
        "Found completed jobs",
        extra={"experiments": exp_list, "jobs": completed},
    )
    return completed


def _get_best_child_run_id(parent_run_id: str) -> str | None:
    """Get best child run id from parent run tags (tries common keys)."""
    run = mlflow.get_run(parent_run_id)
    tags = dict(run.data.tags or {})

    for key in ("automl_best_child_run_id", "azureml.best_child_run_id"):
        value = tags.get(key)
        if value:
            return value
    return None

def register_model_from_run(
    ml_client: MLClient,
    *,
    model_name: str,
    best_child_run: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    artifact_path: str = "model",
) -> Model:
    """Register a model from a best-child run.

    This uses the AzureML job artifacts URI pattern.
    Adjust `artifact_path` to match where your MLflow model is logged.
    """
    registered_model = Model(
        path=f"azureml://jobs/{best_child_run}/outputs/artifacts/paths/{artifact_path}",
        name=model_name or _model_name(best_child_run),
        description=description,
        type=AssetTypes.MLFLOW_MODEL,
        tags=tags or {},
    )

    return ml_client.models.create_or_update(registered_model)


def register_best_child_run(
    ml_client: MLClient,
    *,
    experiment: str | None = None,
    job_name: str,
    settings: Settings | None = None,
) -> Model | None:
    """Register the best child run from an AutoML parent job as an MLflow model."""
    experiment_name = experiment or (settings or get_settings()).automl_train_exp
    if not experiment_name:
        raise ValueError("experiment is required when no default experiment is configured.")

    # Configure MLflow tracking URI
    mlflow_tracking_uri(ml_client)

    # Verify job belongs to experiment and is completed
    completed_jobs = list_completed_jobs(ml_client, experiment_name)
    exp_jobs = completed_jobs.get(experiment_name, [])

    if job_name not in exp_jobs:
        logger.error(
            "Input job doesn't exist in experiment (or is not Completed)",
            extra={"experiment_name": experiment_name, "job_name": job_name, "completed_jobs": exp_jobs},
        )
        raise RuntimeError(
            f"Job '{job_name}' not found among Completed jobs for '{experiment_name}'. "
            f"Completed jobs: {exp_jobs}"
        )

    # Find best child run id
    best_child_run = _get_best_child_run_id(job_name)
    if not best_child_run:
        msg = f"Best child run couldn't be found for parent job '{job_name}'"
        logger.warning(msg)
        raise KeyError(msg)
    

    resolved_model_name = _model_name(best_child_run)

    model = register_model_from_run(
        ml_client,
        model_name=resolved_model_name,
        best_child_run=best_child_run,
        description="AutoML classification model registered from best child run",
        tags={
            "experiment_name": experiment_name,
            "parent_job_name": job_name,
            "best_child_run_id": best_child_run,
        },
    )

    return model


__all__ = [
    "list_completed_jobs",
    "register_best_child_run",
    "register_model_from_run",
    "mlflow_tracking_uri"
]
