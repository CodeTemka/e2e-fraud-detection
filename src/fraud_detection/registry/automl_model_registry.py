"""Utilities to register the best AutoML child runs as models."""
from __future__ import annotations

import re
from collections.abc import Iterable

import mlflow
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _slug(s: str) -> str:
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
    # Microsoft docs commonly show: "automl_best_child_run_id"
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
) -> Model | None:
    """Register the best child run from an AutoML parent job as an MLflow model.

    Returns:
        Model if registered, else None (when skipped).
    Raises:
        KeyError / Exception if skip flags are False.
    """
    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    logger.info("Registering best child run", extra={"job_name": job_name, "experiment": experiment_name})

    # Docs-style approach: parent run id == automl parent job name
    parent_run = client.get_run(job_name)
    tags = dict(parent_run.data.tags)

    best_child_run_id = _get_best_child_run_id(tags)
    if not best_child_run_id:
        message = f"Best child run id tag missing for parent job '{job_name}'"
        logger.warning(message, extra={"job_name": job_name, "experiment": experiment_name, "tags": tags})
        if skip_on_missing_best_child:
            return None
        raise KeyError(message)

    # Stable name per experiment => versions grow over time
    resolved_model_name = model_name or _slug(experiment_name)

    try:
        model = ml_client.models.create_or_update(
            Model(
                # Best child job outputs contain an MLflow model artifact path
                path=f"azureml://jobs/{best_child_run_id}/outputs/artifacts/outputs/mlflow-model/",
                name=resolved_model_name,
                description="AutoML classification model registered from best child run",
                type=AssetTypes.MLFLOW_MODEL,
                tags={
                    "experiment_name": experiment_name,
                    "parent_job_name": job_name,
                    "best_child_run_id": best_child_run_id,
                },
            )
        )
    except Exception:
        logger.exception(
            "Failed to register model from best child run",
            extra={
                "job_name": job_name,
                "experiment": experiment_name,
                "best_child_run_id": best_child_run_id,
                "model_name": resolved_model_name,
            },
        )
        if skip_on_model_error:
            return None
        raise

    logger.info("Registered model", extra={"model_name": model.name, "version": model.version})
    return model


__all__ = ["list_completed_jobs", "register_best_child_run"]
