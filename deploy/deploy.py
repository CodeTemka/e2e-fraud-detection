import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from fraud_detection.azure.client import get_ml_client
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)

ml_client = get_ml_client()
endpoint_names = [endpoint.name for endpoint in ml_client.online_endpoints.list()]

model_name = 'automated-ml-classification-recall-experiment_lightgbm-only-classification-job-f8330f03_model'
model_version = '1'
model = ml_client.models.get(name=model_name, version=model_version)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_names[0],
    model=model,
    instance_type="Standard_E8s_v3",
    instance_count=1,
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()
