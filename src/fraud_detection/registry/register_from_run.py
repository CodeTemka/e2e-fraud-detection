"""Register Azure ML model assets from MLflow model artifacts produced by a job/run."""
from __future__ import annotations

import os
from collections.abc import Iterable

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Default candidate locations we've seen for MLflow models in AzureML jobs.
# You can override these defaults with env var:
#   FRAUD_MLFLOW_MODEL_ARTIFACT_PATHS="outputs/mlflow-model/,outputs/artifacts/model/"
DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES: tuple[str, ...] = (
    # Common for AutoML child jobs (many examples use this)
    "outputs/artifacts/outputs/mlflow-model/",
    # Common alternative
    "outputs/mlflow-model/",
    # Another common layout for some pipelines
    "outputs/artifacts/model/",
)

_ENV_PRIMARY = "FRAUD_MLFLOW_MODEL_ARTIFACT_PATHS"
_ENV_FALLBACK = "AML_MLFLOW_MODEL_ARTIFACT_PATHS"


def _split_paths(value: str) -> list[str]:
    # Accept commas/semicolons/newlines.
    raw = value.replace(";", ",").replace("\n", ",")
    parts = [p.strip() for p in raw.split(",")]
    # Normalize and drop empties
    cleaned = []
    for p in parts:
        if not p:
            continue
        cleaned.append(p)
    return cleaned


def get_default_artifact_paths() -> list[str]:
    """Return default artifact path candidates, allowing env override.

    Env override (comma-separated):
      FRAUD_MLFLOW_MODEL_ARTIFACT_PATHS
      AML_MLFLOW_MODEL_ARTIFACT_PATHS (fallback)
    """
    override = os.getenv(_ENV_PRIMARY) or os.getenv(_ENV_FALLBACK)
    if override:
        paths = _split_paths(override)
        if paths:
            logger.info(
                "Using artifact path candidates from environment",
                extra={"env": _ENV_PRIMARY if os.getenv(_ENV_PRIMARY) else _ENV_FALLBACK, "paths": paths},
            )
            return paths
    return list(DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES)


def _normalize_artifact_path(p: str) -> str:
    p = (p or "").strip().lstrip("/")
    if not p:
        raise ValueError("artifact path is empty")
    return p


def _azureml_job_uri(run_id: str, artifact_path: str) -> str:
    # NOTE: In AzureML, the "run_id" here is typically a job name/id.
    return f"azureml://jobs/{run_id}/{_normalize_artifact_path(artifact_path)}"


def find_existing_version_for_run(ml_client: MLClient, *, model_name: str, run_id: str) -> str | None:
    """If we already registered this run before, reuse that version (idempotency)."""
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

    This is best-effort across multiple artifact paths:
      - uses `artifact_paths` if provided
      - otherwise uses env override paths (if set)
      - otherwise uses DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES

    Tags:
      - Adds `source_run_id` and `source_artifact_path` automatically.
    """
    existing = find_existing_version_for_run(ml_client, model_name=model_name, run_id=run_id)
    if existing is not None:
        logger.info(
            "Model version already exists for run; reusing",
            extra={"model_name": model_name, "version": existing, "run_id": run_id},
        )
        return ml_client.models.get(name=model_name, version=existing)

    candidates = list(artifact_paths) if artifact_paths else get_default_artifact_paths()
    if not candidates:
        raise ValueError("No artifact path candidates provided or configured.")

    last_exc: Exception | None = None
    for ap in candidates:
        uri = _azureml_job_uri(run_id, ap)
        try:
            model = ml_client.models.create_or_update(
                Model(
                    name=model_name,
                    path=uri,
                    type=AssetTypes.MLFLOW_MODEL,
                    description=description or "Registered from MLflow run/job artifacts",
                    tags={
                        **(tags or {}),
                        "source_run_id": run_id,
                        "source_artifact_path": ap,
                        "source_uri": uri,
                    },
                )
            )
            logger.info(
                "Registered model",
                extra={
                    "model_name": model.name,
                    "version": model.version,
                    "run_id": run_id,
                    "artifact_path": ap,
                },
            )
            return model
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Failed to register using artifact path; trying next",
                extra={"run_id": run_id, "artifact_path": ap},
                exc_info=exc,
            )

    raise RuntimeError(
        f"Could not register model '{model_name}' from run/job '{run_id}'. "
        f"Tried artifact paths: {candidates}"
    ) from last_exc


__all__ = [
    "DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES",
    "get_default_artifact_paths",
    "register_model_from_run",
]