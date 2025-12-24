"""Return Registered Models in the Azure ML Workspace."""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

from fraud_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

def registered_models(ml_client: MLClient) -> list[Model]:
    """Return registered model assets in the Azure ML Workspace."""
    all_models = ml_client.models.list()
    return [
        model
        for model in all_models
        if "_output_mlflow_log_model_" not in model.name
    ]
