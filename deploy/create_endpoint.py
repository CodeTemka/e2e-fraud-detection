import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
import uuid

online_endpoint_name = "fraud-detection-" + str(uuid.uuid4())[:4]

endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for fraud detection model",
    auth_mode="key",
    tags={"project": "fraud-detection"},
)

ml_client.online_endpoints.begin_create_or_update(endpoint)