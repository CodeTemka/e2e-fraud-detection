from __future__ import annotations

import mlflow
from azure.ai.ml import MLClient

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def set_tracking_uri_from_workspace(ml_client: MLClient) -> str:
    """Configure MLflow to use the Azure ML workspace tracking URI."""
    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("Configured MLflow tracking URI", extra={"tracking_uri": tracking_uri})
    return tracking_uri
