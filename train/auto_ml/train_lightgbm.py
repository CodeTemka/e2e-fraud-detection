from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl, Input

from azure_connection.ml_client_setup import ml_client
from azure.ai.ml.automl import ClassificationPrimaryMetrics
import uuid

data_input = Input(type=AssetTypes.MLTABLE, path="azureml:mltable_creditcard_data:1")

classification_job = automl.classification(
    name = f"lightgbm-only-classification-job-{str(uuid.uuid4())[:8]}",
    compute="automated-ml-cluster",
    experiment_name="automated-ml-classification-recall-experiment",
    training_data=data_input,
    target_column_name="Class",
    primary_metric=ClassificationPrimaryMetrics.ACCURACY,
    n_cross_validations=5,
    enable_model_explainability=True,
    tags={"project": "automated-ml-classification",
          "model": "LightGBM",
          "metric": "Accuracy"}
)

classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=60,
    max_trials=100,
    max_concurrent_trials=3
)

classification_job.set_training(
    allowed_training_algorithms=["LightGBM"],
)

classification_returned_job = ml_client.jobs.create_or_update(classification_job)