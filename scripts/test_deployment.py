from fraud_detection.azure.client import get_ml_client
from pathlib import Path

ml_client = get_ml_client()
endpoint_names = [endpoint.name for endpoint in ml_client.online_endpoints.list()]

response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_names[0],
    deployment_name="blue",
    request_file=Path('json') / 'sample_request.json'
)

print(response)
