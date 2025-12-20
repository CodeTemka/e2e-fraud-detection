"""Utilities to register the best AutoML child runs as Azure ML models."""
from __future__ import annotations

import re
from collections.abc import Iterable

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

from fraud_detection.registry.register_from_run import (
    DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES,
    register_model_from_run,
)
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _slug(s: str) -> str:
    """Make a stable, Azure-safe-ish name."""
    s = (s or "").lower().strip()
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def list_completed_jobs(ml_client: MLClient, experiments: Iterable[str]) -> dict[str, str]:
    """Return mapping: {job_name -> experiment_name} for completed jobs in given experiments."""
    exp_set = set(experiments)
    completed: dict[str, str] = {}

    for job in ml_client.jobs.list(list_view_type="All"):
        if job.experiment_name in exp_set and job.status == "Completed":
            completed[job.name] = job.experiment_name

    logger.info("Found completed jobs", extra={"jobs": list(completed.keys())})
    return completed


def _get_best_child_run_id(parent_run_tags: dict[str, str]) -> str | None:
    """Try known tag keys for best child run id (docs key first)."""
    # Common keys seen in practice / docs:
    for key in ("automl_best_child_run_id", "azureml.best_child_run_id"):
        value = parent_run_tags.get(key)
        if value:
            return value
    return None


def register_best_child_run(
    ml_client: MLClient,
    *,
    job_name: str,
    experiment_name: str,
    model_name: str | None = None,
    skip_on_missing_best_child: bool = True,
    skip_on_model_error: bool = True,
    artifact_paths: list[str] | None = None,
) -> Model | None:
    """Register the best child run from an AutoML parent job as an MLflow model.

    Strategy
    --------
    - Use MLflow tracking URI from the Azure ML workspace
    - Read the *parent* run (often the parent job name == run id in AzureML MLflow)
    - Extract best-child-run id from known tag keys
    - Register the MLflow model artifact from that best child run as an Azure ML Model asset
      using fallback artifact paths.

    Returns
    -------
    Model | None
        Model if registered, else None (when skipped).
    """
    # Configure MLflow to point at the workspace tracking server
    try:
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as exc:
        logger.exception(
            "Failed to configure MLflow tracking URI from workspace",
            extra={"job_name": job_name, "experiment": experiment_name},
        )
        if skip_on_model_error:
            return None
        raise RuntimeError("Failed to configure MLflow tracking URI") from exc
    

    logger.info(
        "Registering best child run",
        extra={"job_name": job_name, "experiment": experiment_name},
    )

    # Parent run lookup (job_name is often the run_id in AzureML MLflow)
    try:
        parent_run = tracking_client.get_run(job_name)
    except Exception as exc:
        logger.exception(
            "Failed to fetch parent run from MLflow",
            extra={"job_name": job_name, "experiment": experiment_name},
        )
        if skip_on_model_error:
            return None
        raise RuntimeError(f"Could not fetch MLflow run for job_name '{job_name}'") from exc

    tags = dict(parent_run.data.tags or {})
    best_child_run_id = _get_best_child_run_id(tags)

    if not best_child_run_id:
        message = f"Best child run id tag missing for parent job '{job_name}'"
        logger.warning(
            message,
            extra={"job_name": job_name, "experiment": experiment_name, "tags": tags},
        )
        if skip_on_missing_best_child:
            return None
        raise KeyError(message)

    # Stable name per experiment => versions grow over time
    resolved_model_name = _slug(model_name) if model_name else _slug(experiment_name)

    candidates = artifact_paths or list(DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES)

    try:
        model = register_model_from_run(
            ml_client,
            model_name=resolved_model_name,
            run_id=best_child_run_id,
            artifact_paths=candidates,
            description="AutoML classification model registered from best child run",
            tags={
                "experiment_name": experiment_name,
                "parent_job_name": job_name,
                "best_child_run_id": best_child_run_id,
            },
        )
    except Exception:
        logger.exception(
            "Failed to register model from best child run",
            extra={
                "job_name": job_name,
                "experiment": experiment_name,
                "best_child_run_id": best_child_run_id,
                "model_name": resolved_model_name,
                "artifact_paths_tried": candidates,
            },
        )
        if skip_on_model_error:
            return None
        raise

    logger.info(
        "Registered model",
        extra={"model_name": model.name, "version": model.version, "job_name": job_name},
    )
    return model


__all__ = ["list_completed_jobs", "register_best_child_run"]
