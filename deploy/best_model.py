import os
import sys
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# adjust path so your project imports still work
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# import your objects (these must be available in your environment)
from best_run_auto_ml import auto_ml_best_runs
from best_run_custom_train import custom_train_best_runs
from azure_connection.ml_client_setup import ml_client

best_models_runs_id = set(list(auto_ml_best_runs.values()) + list(custom_train_best_runs.values()))
print(best_models_runs_id)