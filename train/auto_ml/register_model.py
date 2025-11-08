import os
import sys
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from azure_connection.ml_client_setup import ml_client
from azure.ai.ml.entities import Model, Job
from azure.ai.ml.constants import AssetTypes

def auto_ml_job():
    job_list = {}
    for job in ml_client.jobs.list(list_view_type="All"):
        if (
            job.experiment_name in [
                "automated-ml-classification-recall-experiment",
                "automated-ml-classification-experiment"
            ]
            and job.status == "Completed"
        ):
            job_list[job.name] = job
    return job_list

for job_name, job in auto_ml_job().items():
    # Get MLflow tracking URI for workspace
    mlflow.set_tracking_uri(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)

    # Get MLflow parent run
    mlflow_parent_run = mlflow.get_run(job.name)

    # Get best trial run ID
    best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]

    # Directly register using the job path (no need to call jobs.get)
    model_name = f"{job.experiment_name.replace(' ', '_').lower()}_{job_name}_model"
    registered_model = ml_client.models.create_or_update(
        Model(
            path=f"azureml://jobs/{best_child_run_id}/outputs/artifacts/outputs/mlflow-model/",
            name=model_name,
            description="My AutoML classification model",
            type=AssetTypes.MLFLOW_MODEL,
        )
    )

    print(f"✅ Registered model: {registered_model.name}, Version: {registered_model.version}")
