"""Helpers for working with Azure ML clients."""
from __future__ import annotations

from typing import Optional

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def build_default_credential() -> DefaultAzureCredential:
    """Create a default Azure credential suitable for CI and local use."""

    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def get_ml_client(
    *, settings: Optional[Settings] = None, credential: Optional[DefaultAzureCredential] = None
) -> MLClient:
    """Instantiate an :class:`~azure.ai.ml.MLClient` using project settings."""

    resolved_settings = settings or get_settings()
    resolved_credential = credential or build_default_credential()

    logger.info(
        "Connecting to Azure ML workspace",
        extra={
            "subscription_id": resolved_settings.subscription_id,
            "resource_group": resolved_settings.resource_group,
            "workspace": resolved_settings.workspace_name,
        },
    )

    return MLClient(  # type: ignore[call-arg]
        credential=resolved_credential,
        subscription_id=resolved_settings.subscription_id,
        resource_group_name=resolved_settings.resource_group,
        workspace_name=resolved_settings.workspace_name,
    )


__all__ = ["get_ml_client", "build_default_credential"]
