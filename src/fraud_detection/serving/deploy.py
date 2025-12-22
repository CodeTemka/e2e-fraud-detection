"""Helpers for deploying registered models to online endpoints."""
from __future__ import annotations

from collections.abc import Mapping

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _get_model_version(ml_client: MLClient, model_name: str) -> str:
    """Get the latest version string for a registered model."""

    model = ml_client.models.get(name=model_name)
    return str(model.version)


def deploy_model(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str = "blue",
    model_name: str,
    instance_type: str = "Standard_DS3_v2",
    instance_count: int = 1,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineDeployment:
    """Deploy a registered model to an existing managed online endpoint."""

    if instance_count < 1:
        raise ValueError("instance_count must be >= 1")

    model_version = _get_model_version(ml_client, model_name)
    model = ml_client.models.get(name=model_name, version=model_version)

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=instance_type,
        instance_count=instance_count,
        tags=dict(tags) if tags else {"project": "fraud-detection"},
    )

    logger.info(
        "Creating/updating deployment",
        extra={
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "model_name": model_name,
            "model_version": model_version,
            "instance_type": instance_type,
            "instance_count": instance_count,
        },
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()


__all__ = ["deploy_model"]
