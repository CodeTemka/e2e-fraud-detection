"""Helpers for Azure ML managed online endpoints."""
from __future__ import annotations

from collections.abc import Mapping
from math import floor
from dataclasses import dataclass

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
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


def update_traffic(ml_client: MLClient, *, endpoint_name: str, traffic: dict[str, int]) -> None:
    """Update traffic allocation for deployments on an endpoint."""
    _validate_traffic(traffic)

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = traffic

    logger.info("Updating endpoint traffic", extra={"endpoint_name": endpoint_name, "traffic": traffic})
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def set_initial_traffic(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str,
    low_traffic_percent: int = 10,
) -> dict[str, int]:
    """Set initial low traffic for a new deployment and return the final mapping."""
    if low_traffic_percent <= 0 or low_traffic_percent >= 100:
        raise ValueError("low_traffic_percent must be between 1 and 99")

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    current_traffic = dict(endpoint.traffic or {})
    traffic = _build_initial_traffic(
        current_traffic=current_traffic,
        deployment_name=deployment_name,
        low_traffic_percent=low_traffic_percent,
    )
    update_traffic(ml_client, endpoint_name=endpoint_name, traffic=traffic)
    return traffic


def _build_initial_traffic(
    *,
    current_traffic: Mapping[str, int],
    deployment_name: str,
    low_traffic_percent: int,
) -> dict[str, int]:
    remaining = 100 - low_traffic_percent
    existing = {name: pct for name, pct in current_traffic.items() if name != deployment_name}
    if not existing:
        return {deployment_name: 100}

    total_existing = sum(existing.values())
    traffic = _allocate_remaining_traffic(existing=existing, remaining=remaining, total_existing=total_existing)
    traffic[deployment_name] = low_traffic_percent
    return traffic


def _allocate_remaining_traffic(
    *,
    existing: Mapping[str, int],
    remaining: int,
    total_existing: int,
) -> dict[str, int]:
    if total_existing <= 0:
        base = remaining // len(existing)
        traffic = {name: base for name in existing}
        remainder = remaining - base * len(existing)
        for name in sorted(existing)[:remainder]:
            traffic[name] += 1
        return traffic

    allocations: list[tuple[str, int, float]] = []
    for name, pct in existing.items():
        raw = pct * remaining / total_existing
        base = floor(raw)
        allocations.append((name, base, raw - base))

    traffic = {name: base for name, base, _ in allocations}
    remainder = remaining - sum(base for _, base, _ in allocations)
    allocations.sort(key=lambda item: (-item[2], item[0]))
    for name, _, _ in allocations[:remainder]:
        traffic[name] += 1
    return traffic


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
    "update_traffic",
    "set_initial_traffic",
    "DEFAULT_AUTH_MODE",
]
