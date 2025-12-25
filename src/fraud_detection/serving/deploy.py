"""Helpers for deploying registered models to online endpoints."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_version(v: Optional[str]) -> int:
    """AzureML model versions are typically numeric strings; be defensive."""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _get_latest_model_version(ml_client: MLClient, model_name: str) -> str:
    """Return the latest (highest numeric) version for a given model name."""
    models = list(ml_client.models.list(name=model_name))
    if not models:
        raise ResourceNotFoundError(f"No model found with name '{model_name}'.")

    latest = max(models, key=lambda m: _parse_version(getattr(m, "version", None)))
    version = getattr(latest, "version", None)
    if not version:
        raise RuntimeError(f"Found model '{model_name}' but it has no version field.")
    return str(version)


def deploy_model(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str = "blue",
    model_name: str,
    model_version: str | None = None,
    instance_type: str | None = "Standard_D2a_v4",
    instance_count: int = 1,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineDeployment:
    """Deploy a registered model to an existing managed online endpoint."""
    if instance_count < 1:
        raise ValueError("instance_count must be >= 1")

    settings = get_settings()
    resolved_instance_type = (instance_type or "").strip() or settings.get_deployment_instance_type()

    resolved_version = (model_version or "").strip() or _get_latest_model_version(ml_client, model_name)
    model = ml_client.models.get(name=model_name, version=resolved_version)

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=resolved_instance_type,
        instance_count=instance_count,
        tags=dict(tags) if tags else {"project": "fraud-detection"},
    )

    logger.info(
        "Creating/updating deployment",
        extra={
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "model_name": model_name,
            "model_version": resolved_version,
            "instance_type": resolved_instance_type,
            "instance_count": instance_count,
        },
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()


__all__ = ["deploy_model"]
