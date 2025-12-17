"""Serving utilities for Azure ML managed online endpoints."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_AUTH_MODE = "key"


@dataclass(frozen=True)
class DeploymentSpec:
    endpoint_name: str
    deployment_name: str
    model_name: str
    model_version: str
    instance_type: str
    instance_count: int = 1


def get_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    """Fetch an endpoint or raise ResourceNotFoundError."""
    return ml_client.online_endpoints.get(endpoint_name)


def create_endpoint(
    ml_client: MLClient,
    *,
    name: str,
    description: str | None = None,
    auth_mode: str = DEFAULT_AUTH_MODE,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineEndpoint:
    """Create (or update) a managed online endpoint."""
    endpoint = ManagedOnlineEndpoint(
        name=name,
        description=description or "Online endpoint for fraud detection model",
        auth_mode=auth_mode,
        tags=dict(tags) if tags else {"project": "fraud-detection"},
    )
    logger.info("Creating/updating endpoint", extra={"endpoint_name": name})
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def ensure_endpoint(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    description: str | None = None,
    auth_mode: str = DEFAULT_AUTH_MODE,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineEndpoint:
    """Ensure an endpoint exists; create if missing."""
    try:
        return get_endpoint(ml_client, endpoint_name)
    except ResourceNotFoundError:
        logger.info("Endpoint not found; creating", extra={"endpoint_name": endpoint_name})
        return create_endpoint(
            ml_client,
            name=endpoint_name,
            description=description,
            auth_mode=auth_mode,
            tags=tags,
        )


def delete_endpoint(ml_client: MLClient, *, endpoint_name: str) -> None:
    """Delete an endpoint and its deployments."""
    logger.info("Deleting endpoint", extra={"endpoint_name": endpoint_name})
    ml_client.online_endpoints.begin_delete(name=endpoint_name).result()


def deploy_model(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    instance_type: str,
    instance_count: int = 1,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineDeployment:
    """Deploy a registered model to an existing managed online endpoint."""
    if instance_count < 1:
        raise ValueError("instance_count must be >= 1")

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


def update_traffic(ml_client: MLClient, *, endpoint_name: str, traffic: dict[str, int]) -> None:
    """Update traffic allocation for deployments on an endpoint."""
    _validate_traffic(traffic)

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = traffic

    logger.info("Updating endpoint traffic", extra={"endpoint_name": endpoint_name, "traffic": traffic})
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def _validate_traffic(traffic: dict[str, int]) -> None:
    if not traffic:
        raise ValueError("traffic mapping is empty")

    for dep, pct in traffic.items():
        if not dep or not isinstance(dep, str):
            raise ValueError("traffic keys must be deployment names (non-empty strings)")
        if not isinstance(pct, int):
            raise ValueError(f"traffic percentage for '{dep}' must be an int")
        if pct < 0:
            raise ValueError(f"traffic percentage for '{dep}' must be >= 0")

    total = sum(traffic.values())
    if total != 100:
        raise ValueError(f"traffic must sum to 100, got {total}")


__all__ = [
    "DeploymentSpec",
    "get_endpoint",
    "create_endpoint",
    "ensure_endpoint",
    "delete_endpoint",
    "deploy_model",
    "update_traffic",
    "DEFAULT_AUTH_MODE",
]
