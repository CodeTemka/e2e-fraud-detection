import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.azure.client import get_ml_client  # noqa: E402
from fraud_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

ml_client = get_ml_client()

all_models = ml_client.models.list()

registered_models = [
    model for model in all_models 
    if "_output_mlflow_log_model_" not in model.name
]

for model in registered_models:
    logger.info("%s", model.name)
