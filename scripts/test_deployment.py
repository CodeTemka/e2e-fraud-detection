import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

ml_client = get_ml_client()
endpoint_names = [endpoint.name for endpoint in ml_client.online_endpoints.list()]

response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_names[0],
    deployment_name="blue",
    request_file=Path("json") / "sample_request.json",
)

logger.info("%s", response)
