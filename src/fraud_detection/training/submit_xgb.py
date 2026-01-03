from __future__ import annotations

import os

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import Choice, LogUniform, Uniform, MedianStoppingPolicy
from azure.identity import DefaultAzureCredential


def main() -> None:
    # ---- workspace connection (fill these in or load from your own config) ----
    subscription_id = os.environ.get("AZ_SUBSCRIPTION_ID", "<SUBSCRIPTION_ID>")
    resource_group = os.environ.get("AZ_RESOURCE_GROUP", "<RESOURCE_GROUP>")
    workspace_name = os.environ.get("AZ_WORKSPACE_NAME", "<AML_WORKSPACE_NAME>")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # ---- compute cluster name (must already exist) ----
    compute_name = "cpu-cluster"

    # ---- environment (register it once; then reuse by name:version) ----
    job_env = Environment(
        name="xgb-sweep-env",
        description="XGBoost sweep environment",
        conda_file="env/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    job_env = ml_client.environments.create_or_update(job_env)

    # ---- base command job ----
    base_job = command(
        code="./src",
        command=(
            "python train_xgb.py "
            "--train_data ${{inputs.train_data}} "
            "--label_col ${{inputs.label_col}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--max_depth ${{inputs.max_depth}} "
            "--learning_rate ${{inputs.learning_rate}} "
            "--subsample ${{inputs.subsample}} "
            "--colsample_bytree ${{inputs.colsample_bytree}} "
            "--min_child_weight ${{inputs.min_child_weight}} "
            "--gamma ${{inputs.gamma}} "
            "--reg_lambda ${{inputs.reg_lambda}} "
            "--scale_pos_weight ${{inputs.scale_pos_weight}} "
            "--output_dir ${{outputs.output_dir}}"
        ),
        inputs={
            # Point this to your own data asset or datastore path:
            # Example options:
            #  - Input(type="uri_file", path="azureml:your_data_asset@latest")
            #  - Input(type="uri_file", path="https://.../your.csv")
            "train_data": Input(type="uri_file", path="azureml:credit-card-data@latest"),
            "label_col": "Class",
            # defaults (will be overridden by sweep below)
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        },
        outputs={"output_dir": Input(type="uri_folder")},
        environment=f"{job_env.name}:{job_env.version}",
        compute=compute_name,
        experiment_name="xgb-fraud-sweep",
        display_name="xgb-train",
    )

    # ---- override inputs with parameter expressions for tuning ----
    job_for_sweep = base_job(
        learning_rate=LogUniform(min_value=1e-3, max_value=0.2),
        max_depth=Choice(values=[3, 4, 5, 6, 7, 8]),
        n_estimators=Choice(values=[300, 600, 900, 1200]),
        subsample=Uniform(min_value=0.6, max_value=1.0),
        colsample_bytree=Uniform(min_value=0.6, max_value=1.0),
        min_child_weight=Choice(values=[1.0, 5.0, 10.0]),
        gamma=Uniform(min_value=0.0, max_value=5.0),
        reg_lambda=LogUniform(min_value=1e-3, max_value=10.0),
        scale_pos_weight=Choice(values=[1.0, 50.0, 100.0, 200.0]),
    )

    # ---- build sweep job ----
    sweep_job = job_for_sweep.sweep(
        compute=compute_name,
        sampling_algorithm="random",
        primary_metric="average_precision",  # must match mlflow.log_metric("average_precision", ...)
        goal="Maximize",
    )

    # resource limits and early termination (recommended pattern)
    sweep_job.set_limits(max_total_trials=30, max_concurrent_trials=4, timeout=3 * 60 * 60)
    sweep_job.early_termination = MedianStoppingPolicy(delay_evaluation=5, evaluation_interval=2)

    sweep_job.display_name = "xgb-sweep"
    sweep_job.experiment_name = "xgb-fraud-sweep"

    # ---- submit ----
    created = ml_client.jobs.create_or_update(sweep_job)
    print(f"Submitted sweep job: {created.name}")


if __name__ == "__main__":
    main()
