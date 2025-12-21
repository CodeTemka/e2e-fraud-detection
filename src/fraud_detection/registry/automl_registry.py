"""Utilities to register the best AutoML child run as Azure ML models."""
from __future__ import annotations

import re
from collections.abc import Iterable

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.training.automl import EXPERIMENT_ACCURACY, EXPERIMENT_RECALL
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Keep yours (legacy + new)
DEFAULT_EXPERIMENTS = [
    "automated-ml-classification-recall-experiment",
    "automated-ml-classification-experiment",
]
NEW_EXPERIMENTS = [EXPERIMENT_RECALL, EXPERIMENT_ACCURACY]


def _slug(s: str) -> str:
    """Make a stable, Azure-safe-ish name."""
    s = (s or "").lower().strip()  # FIX: strip()
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def _make_model_name(*, experiment_name: str, job_name: str | None = None) -> str:
    """Choose a model name when user doesn't provide one.

    Recommended default (stable): one model name per experiment, versions increase over time.
    Optional: include job_name if you want a unique name per job (many model names).
    """
    base = _slug(experiment_name)

    # Stable name per experiment (recommended)
    if not job_name:
        return base

    # If you *prefer* unique model names per job, uncomment this and return it instead:
    # return _slug(f"{base}-{job_name}")

    return base


def mlflow_tracking_uri(ml_client: MLClient) -> None:
    """Set MLflow tracking URI from workspace and return it."""
    uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)


def list_completed_jobs(ml_client: MLClient, experiments: Iterable[str] | str) -> dict[str, list[str]]:
    """Return mapping: {experiment_name -> [job_name, ...]} for completed jobs.

    NOTE: Your older test expected {job_name -> experiment_name}.
    If you still want that, see helper `list_completed_jobs_flat()` below.
    """
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


def list_completed_jobs_flat(ml_client: MLClient, experiments: Iterable[str]) -> dict[str, str]:
    """Compatibility helper: {job_name -> experiment_name} (matches your old unit test)."""
    exp_set = set(experiments)
    out: dict[str, str] = {}

    for job in ml_client.jobs.list(list_view_type="All"):
        exp = getattr(job, "experiment_name", None)
        status = getattr(job, "status", None)
        name = getattr(job, "name", None)

        if exp in exp_set and status == "Completed" and name:
            out[name] = exp

    return out


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
        name=_slug(model_name),
        description=description,
        type=AssetTypes.MLFLOW_MODEL,  # FIX: type (not types)
        tags=tags or {},
    )

    # In Azure ML, same model name => new version. No need to pre-check existence.
    return ml_client.models.create_or_update(registered_model)


def register_best_child_run(
    ml_client: MLClient,
    *,
    job_name: str,
    experiment_name: str,
    model_name: str | None = None,
    skip_on_missing_best_child: bool = False,
    skip_on_model_error: bool = False,
) -> Model | None:
    """Register the best child run from an AutoML parent job as an MLflow model."""
    allowed_experiments = DEFAULT_EXPERIMENTS + NEW_EXPERIMENTS
    if experiment_name not in allowed_experiments:
        logger.error(
            "Input experiment name is not available",
            extra={"experiment_name": experiment_name, "allowed": allowed_experiments},
        )
        raise RuntimeError(
            f"Experiment '{experiment_name}' is not available. Choose one of: {allowed_experiments}"
        )

    # Configure MLflow tracking URI
    try:
        mlflow_tracking_uri(ml_client)
    except Exception as ex:
        logger.exception("Failed to configure MLflow tracking uri for Azure ML")
        raise RuntimeError("Failed to configure MLflow tracking uri") from ex

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
        if skip_on_missing_best_child:
            return None
        raise KeyError(msg)

    resolved_model_name = _slug(model_name) if model_name else _make_model_name(
        experiment_name=experiment_name,
        job_name=job_name,
    )

    # FIX: call register_model_from_run (not register_best_child_run recursively)
    try:
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
    except Exception:
        logger.exception("Failed to register model from best child run")
        if skip_on_model_error:
            return None
        raise

    return model


__all__ = [
    "DEFAULT_EXPERIMENTS",
    "NEW_EXPERIMENTS",
    "list_completed_jobs",
    "list_completed_jobs_flat",
    "register_best_child_run",
    "register_model_from_run",
    "mlflow_tracking_uri"
]
