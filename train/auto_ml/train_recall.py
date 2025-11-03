from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl, Input

from azure_connection.ml_client_setup import ml_client

data_input = Input(type=AssetTypes.MLTABLE, path="azureml:mltable_creditcard_data:1")

classification_job = automl.classification(
    compute="automated-ml-cluster",
    experiment_name="automated-ml-classification-recall-experiment",
    training_data=data_input,
    target_column_name="Class",
    primary_metric="Recall_weighted",
    n_cross_validations=20,
    enable_model_explainability=True,
    tags={"project": "automated-ml-classification",
          "metric": "recall"}
)

classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=30,
    max_trials=30,
    enable_early_termination=True,
    max_concurrent_trials=3
)

classification_returned_job = ml_client.jobs.create_or_update(classification_job)