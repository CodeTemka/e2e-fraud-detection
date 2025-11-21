"""Azure ML client factory."""
from __future__ import annotations

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from fraud_detection.config import AzureMLConfig


def create_ml_client(
    config: AzureMLConfig | None = None,
    credential: object | None = None,
) -> MLClient:
    """Instantiate an MLClient using environment-driven configuration.

    Parameters
    ----------
    config:
        An AzureMLConfig object. If not provided, :meth:`AzureMLConfig.from_env`
        will be used to populate configuration from environment variables.
    credential:
        Optional credential to pass to MLClient. Defaults to
        :class:`~azure.identity.DefaultAzureCredential` when omitted.
    """

    config = config or AzureMLConfig.from_env()
    credential = credential or DefaultAzureCredential()

    return MLClient(
        credential=credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
    )
