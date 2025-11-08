import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from azure_connection.ml_client_setup import ml_client
endpoint_names = [endpoint.name for endpoint in ml_client.online_endpoints.list()]
endpoint = ml_client.online_endpoints.get(endpoint_names[0])

endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint)