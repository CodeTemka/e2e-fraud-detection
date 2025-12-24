"""Return Registered Models in the Azure ML Workspace."""

from azure.ai.ml import MLClient

from fraud_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

def registered_models(ml_client: MLClient) -> list[str]:
    """Return Registered Models in the Azure ML Workspace."""
    all_models = ml_client.models.list()
    registered_models = [
        model for model in all_models 
        if "_output_mlflow_log_model_" not in model.name
    ]
    return registered_models
