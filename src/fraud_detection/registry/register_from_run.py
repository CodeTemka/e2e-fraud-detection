"""Register a model version directly from an MLflow run artifact."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES: Sequence[str] = (
    "model",
    "artifacts/model",
    "outputs/model",
)


def register_model_from_run(
    ml_client: MLClient,
    *,
    model_name: str,
    run_id: str,
    artifact_paths: Iterable[str] | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> Model:
    """Register a model version from an MLflow run artifact path.

    The first candidate artifact path is used to construct the AzureML artifact URI.
    """

    candidates = list(artifact_paths) if artifact_paths else list(DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES)
    if not candidates:
        raise ValueError("artifact_paths must be a non-empty list")

    artifact_path = candidates[0]
    registered_model = Model(
        path=f"azureml://jobs/{run_id}/outputs/artifacts/paths/{artifact_path}",
        name=model_name,
        description=description,
        type=AssetTypes.MLFLOW_MODEL,
        tags=tags or {},
    )

    return ml_client.models.create_or_update(registered_model)


__all__ = [
    "DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES",
    "register_model_from_run",
]
