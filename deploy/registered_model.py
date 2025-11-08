import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client

all_models = ml_client.models.list()

registered_models = [
    model for model in all_models 
    if "_output_mlflow_log_model_" not in model.name
]

for model in registered_models:
    print(model.name)
