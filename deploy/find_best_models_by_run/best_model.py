import os
import sys
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# adjust path so your project imports still work
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

# import your objects (these must be available in your environment)
from deploy.find_best_models_by_run.best_run_auto_ml import auto_ml_best_runs
from deploy.find_best_models_by_run.best_run_custom_train import custom_train_best_runs
from azure_connection.ml_client_setup import ml_client

best_models_runs_id = list(set(list(info['run_id'] for info in auto_ml_best_runs.values()) + list(info['run_id'] for info in custom_train_best_runs.values())))

