"""Utilities to manage Azure ML compute targets for training and deployment."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _is_quota_error(exc: HttpResponseError) -> bool:
    """Detect whether an HttpResponseError is likely quota-related."""

    message = (exc.message or str(exc)).lower()
    return "quota" in message or "exceeded" in message


def delete_compute(ml_client: MLClient, *, name: str, ignore_missing: bool = True) -> bool:
    """Delete a compute target.

    Returns True if a delete was issued, False when the compute was not found
    and ``ignore_missing`` is True.
    """

    try:
        logger.info("Deleting compute", extra={"compute_name": name})
        ml_client.compute.begin_delete(name).result()
        return True
    except ResourceNotFoundError:
        if ignore_missing:
            logger.info("Compute not found; nothing to delete", extra={"compute_name": name})
            return False
        raise


def _try_delete_existing_computes(
    ml_client: MLClient,
    *,
    exclude: Iterable[str] | None = None,
    limit: int = 1,
) -> list[str]:
    """Best-effort cleanup to free quota by deleting other computes."""

    deleted: list[str] = []
    excluded = {c.lower() for c in (exclude or [])}

    for compute in ml_client.compute.list():
        if compute.name.lower() in excluded:
            continue

        try:
            if delete_compute(ml_client, name=compute.name, ignore_missing=True):
                deleted.append(compute.name)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning(
                "Failed to delete compute during quota cleanup",
                extra={"compute_name": compute.name, "error": str(exc)},
            )

        if len(deleted) >= limit:
            break

    return deleted


def ensure_compute(
    ml_client: MLClient,
    *,
    name: str,
    size: str,
    min_instances: int = 0,
    max_instances: int = 1,
    idle_time_before_scale_down: int = 300,
    tags: dict[str, str] | None = None,
    allow_quota_cleanup: bool = True,
) -> Any:
    """Ensure an Azure ML compute cluster exists, creating it if needed.

    When creation fails due to quota, the function attempts to delete other
    compute clusters (best effort) to free capacity before retrying once.
    """

    try:
        return ml_client.compute.get(name)
    except ResourceNotFoundError:
        logger.info(
            "Compute not found; creating",
            extra={
                "compute_name": name,
                "size": size,
                "min_instances": min_instances,
                "max_instances": max_instances,
            },
        )

    cluster = AmlCompute(
        name=name,
        size=size,
        min_instances=min_instances,
        max_instances=max_instances,
        idle_time_before_scale_down=idle_time_before_scale_down,
        tags=tags or {"project": "fraud-detection"},
    )

    def _create() -> Any:
        return ml_client.compute.begin_create_or_update(cluster).result()

    try:
        return _create()
    except HttpResponseError as exc:
        if not (allow_quota_cleanup and _is_quota_error(exc)):
            raise

        logger.warning(
            "Compute creation failed due to quota; attempting cleanup",
            extra={"compute_name": name, "error": str(exc)},
        )

        deleted = _try_delete_existing_computes(ml_client, exclude=[name])
        logger.info("Deleted computes to free quota", extra={"deleted": deleted})
        return _create()


def ensure_training_compute(
    ml_client: MLClient,
    *,
    name: str,
    size: str = "Standard_D4s_v3",
    min_instances: int = 0,
    max_instances: int = 2,
    tags: dict[str, str] | None = None,
) -> Any:
    """Ensure compute for training jobs is present."""

    return ensure_compute(
        ml_client,
        name=name,
        size=size,
        min_instances=min_instances,
        max_instances=max_instances,
        tags=tags,
    )


__all__ = [
    "delete_compute",
    "ensure_compute",
    "ensure_training_compute",
]
