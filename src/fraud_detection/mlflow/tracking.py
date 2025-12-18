from __future__ import annotations

import mlflow
from azure.ai.ml import MLClient
from urllib.parse import urlparse, urlunparse

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_tracking_uri(tracking_uri: str) -> str:
    """Return a tracking URI compatible with MLflow clients.

    AzureML workspaces can return tracking URIs with the ``azureml`` scheme,
    which require the ``azureml-mlflow`` plugin. In environments where that
    plugin is not available (e.g., local CI), MLflow clients will raise
    ``UnsupportedModelRegistryStoreURIException``. We gracefully downgrade the
    scheme to ``https`` so standard MLflow clients can connect to the workspace
    endpoint.
    """
    if tracking_uri.startswith("azureml://"):
        parsed = urlparse(tracking_uri)
        normalized = urlunparse(parsed._replace(scheme="https"))
        logger.warning(
            "Downgrading AzureML tracking URI scheme for MLflow compatibility",
            extra={"original_uri": tracking_uri, "normalized_uri": normalized},
        )
        return normalized
    return tracking_uri


def set_tracking_uri_from_workspace(ml_client: MLClient) -> str:
    """Configure MLflow to use the Azure ML workspace tracking URI."""
    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    normalized = normalize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(normalized)
    logger.info("Configured MLflow tracking URI", extra={"tracking_uri": normalized})
    return normalized
