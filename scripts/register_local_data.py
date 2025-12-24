"""Register the local data to Azure dataset if not already registered."""
from __future__ import annotations

from azure.ai.ml.constants import AssetTypes

from fraud_detection.config import get_settings
from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger


logger = get_logger(__name__)

settings = get_settings().require_azure()
ml_client = get_ml_client(settings=settings)

def register_local_data(ml_client) -> None:
    """Register the local data to Azure dataset if not already registered."""
    dataset_name = settings.dataset_name
    local_data_path = settings.local_data_path

    # Check if dataset already exists
    try:
        ml_client.data.get(name=dataset_name)
        logger.info(f"Dataset '{dataset_name}' already registered.")
        return
    except Exception:
        pass

    # Register the local data as a new dataset
    ml_client.data.create_or_update(
        name=dataset_name,
        path=str(local_data_path),
        type=AssetTypes.URI_FILE,
        version="1",
    )
    logger.info(f"Registered dataset '{dataset_name}' from '{local_data_path}'.")


if __name__ == "__main__":
    register_local_data(ml_client)