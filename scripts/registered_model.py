from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

ml_client = get_ml_client()

all_models = ml_client.models.list()

registered_models = [
    model for model in all_models 
    if "_output_mlflow_log_model_" not in model.name
]

for model in registered_models:
    logger.info("%s", model.name)
