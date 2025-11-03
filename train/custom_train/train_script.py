from azure_connection import ml_client
import mlflow

# Set up MLflow tracking URI and experiment
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("custom-training-experiment")

