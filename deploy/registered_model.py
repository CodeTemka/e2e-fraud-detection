"""
IMPORTANT NOTE: mlflow search registered model only supports mlflow 2.x versions.
Current mlflow version is 2.15.1
AutoML automatically registers every model output from jobs with a name _output_mlflow_log_model_
You can filter these out by checking for this substring in the model name.
"""


import mlflow
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client
mlflow.set_tracking_uri(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)

client = mlflow.tracking.MlflowClient()
all_models = client.search_registered_models()

registered_models = [
    model for model in all_models 
    if "_output_mlflow_log_model_" not in model.name
]

for model in registered_models:
    print(f"Model Name: {model.name}")