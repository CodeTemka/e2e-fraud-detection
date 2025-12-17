from __future__ import annotations

from typing import Iterable

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES: tuple[str, ...] = (
    # Your previous AutoML registration path
    "outputs/artifacts/outputs/mlflow-model/",
    # Common alternatives some jobs produce
    "outputs/mlflow-model/",
    "outputs/artifacts/model/",
)


def _normalize_artifact_path(p: str) -> str:
    p = (p or "").strip().lstrip("/")
    if not p:
        raise ValueError("artifact path is empty")
    return p


def _azureml_job_uri(run_id: str, artifact_path: str) -> str:
    # IMPORTANT: run_id must be an Azure ML job id/name for this URI to work.
    return f"azureml://jobs/{run_id}/{_normalize_artifact_path(artifact_path)}"


def find_existing_version_for_run(ml_client: MLClient, *, model_name: str, run_id: str) -> str | None:
    """If we already registered this run before, reuse that version."""
    for m in ml_client.models.list(name=model_name):
        tags = getattr(m, "tags", None) or {}
        if tags.get("source_run_id") == run_id:
            return str(m.version)
    return None


def register_model_from_run(
    ml_client: MLClient,
    *,
    model_name: str,
    run_id: str,
    artifact_paths: Iterable[str] | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> Model:
    """Register an Azure ML model from a job/run's MLflow model artifacts.

    Tries multiple artifact paths until one works.
    """
    existing = find_existing_version_for_run(ml_client, model_name=model_name, run_id=run_id)
    if existing is not None:
        logger.info("Model version already exists for run", extra={"model_name": model_name, "version": existing, "run_id": run_id})
        return ml_client.models.get(name=model_name, version=existing)

    candidates = list(artifact_paths) if artifact_paths else list(DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES)

    last_exc: Exception | None = None
    for ap in candidates:
        uri = _azureml_job_uri(run_id, ap)
        try:
            model = ml_client.models.create_or_update(
                Model(
                    name=model_name,
                    path=uri,
                    type=AssetTypes.MLFLOW_MODEL,
                    description=description or "Registered from best MLflow run",
                    tags={
                        **(tags or {}),
                        "source_run_id": run_id,
                        "source_artifact_path": ap,
                    },
                )
            )
            logger.info("Registered model", extra={"model_name": model.name, "version": model.version, "run_id": run_id, "artifact_path": ap})
            return model
        except Exception as exc:
            last_exc = exc
            logger.warning("Failed to register using artifact path; trying next", extra={"run_id": run_id, "artifact_path": ap}, exc_info=exc)

    raise RuntimeError(
        f"Could not register model '{model_name}' from run/job '{run_id}'. "
        f"Tried artifact paths: {candidates}"
    ) from last_exc


__all__ = ["register_model_from_run", "DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES"]
