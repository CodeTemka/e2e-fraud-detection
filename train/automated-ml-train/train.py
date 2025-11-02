from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from dotenv import load_dotenv
import os

load_dotenv()
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

data_input = Input(type=AssetTypes.MLTABLE, path='./train_data')

classification_job = automl.classification(
    compute="automated-ml-cluster",
    experiment_name="automated-ml-classification-experiment",
    training_data=data_input,
    target_column_name="Class",
    primary_metric="AUC_weighted",
    n_cross_validations=10,
    enable_model_explainability=True,
    tags={"project": "automated-ml-classification"}
)

classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=30,
    max_trials=30,
    enable_early_stopping=True,
    max_concurrent_trials=3
)

classification_returned_job = ml_client.jobs.create_or_update(classification_job)