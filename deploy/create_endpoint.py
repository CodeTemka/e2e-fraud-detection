import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from fraud_detection.azure.client import get_ml_client
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)

ml_client = get_ml_client()

online_endpoint_name = "fraud-detection-" + str(uuid.uuid4())[:4]

endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for fraud detection model",
    auth_mode="key",
    tags={"project": "fraud-detection"},
)

ml_client.online_endpoints.begin_create_or_update(endpoint)
