from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import (
    ROOT_DIR,
    Settings,
    build_idempotency_key,
    build_job_name,
    get_settings,
    preflight_validate_training_data,
)
from fraud_detection.training.submit_xgb import resolve_xgb_environment, xgb_sweep_job_builder
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

PIPELINE_ENV_NAME = "fraud-pipeline-env"
PIPELINE_ENV_FILE = ROOT_DIR / "src" / "fraud_detection" / "training" / "pipeline_env.yaml"
PIPELINE_ENV_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"


@dataclass(frozen=True)
class PipelineConfig:
    experiment_name: str
    training_data: str
    compute: str
    model_name: str
    endpoint_name: str
    deployment_name: str
    instance_type: str
    instance_count: int


def _resolve_data_input(training_data: str) -> Input:
    if not training_data:
        raise ValueError("training_data is empty. Provide an MLTable asset path/ID/URI.")

    if training_data.startswith("azureml:"):
        return Input(type=AssetTypes.MLTABLE, path=training_data)

    path = Path(training_data)
    if path.is_dir() and (path / "MLTable").exists():
        return Input(type=AssetTypes.MLTABLE, path=str(path))

    if path.is_file() and path.suffix.lower() == ".csv":
        return Input(type=AssetTypes.URI_FILE, path=str(path))

    return Input(type=AssetTypes.URI_FILE, path=training_data)


def _resolve_pipeline_environment(ml_client: MLClient) -> str:
    if not PIPELINE_ENV_FILE.exists():
        raise FileNotFoundError(f"Pipeline environment file not found: {PIPELINE_ENV_FILE}")

    env = Environment(
        name=PIPELINE_ENV_NAME,
        description="Fraud detection pipeline environment",
        conda_file=str(PIPELINE_ENV_FILE),
        image=PIPELINE_ENV_IMAGE,
    )
    created = ml_client.environments.create_or_update(env)
    return f"{created.name}:{created.version}"


def _build_train_component(*, environment: str, compute: str):
    return command(
        name="train_xgb_model",
        display_name="Train XGBoost model",
        code=str(ROOT_DIR / "src"),
        command=(
            "python -m fraud_detection.training.train_xgb "
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
            "--output_dir ${{outputs.output_dir}}"
        ),
        inputs={
            "train_data": Input(type=AssetTypes.MLTABLE),
            "label_col": "Class",
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
        },
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        compute=compute,
    )


def _build_register_component(*, environment: str, compute: str):
    return command(
        name="register_custom_model",
        display_name="Register trained model",
        code=str(ROOT_DIR / "src"),
        command=(
            "python -m fraud_detection.training.steps register-model "
            "--model-dir ${{inputs.model_dir}} "
            "--model-name ${{inputs.model_name}} "
            "--output-model-name ${{outputs.model_name}} "
            "--output-model-version ${{outputs.model_version}}"
        ),
        inputs={
            "model_dir": Input(type=AssetTypes.URI_FOLDER),
            "model_name": "",
        },
        outputs={
            "model_name": Output(type=AssetTypes.URI_FILE),
            "model_version": Output(type=AssetTypes.URI_FILE),
        },
        environment=environment,
        compute=compute,
    )


def _build_deploy_component(*, environment: str, compute: str):
    return command(
        name="deploy_custom_model",
        display_name="Deploy registered model",
        code=str(ROOT_DIR / "src"),
        command=(
            "python -m fraud_detection.training.steps deploy-model "
            "--model-name-file ${{inputs.model_name}} "
            "--model-version-file ${{inputs.model_version}} "
            "--endpoint-name ${{inputs.endpoint_name}} "
            "--deployment-name ${{inputs.deployment_name}} "
            "--instance-type ${{inputs.instance_type}} "
            "--instance-count ${{inputs.instance_count}}"
        ),
        inputs={
            "model_name": Input(type=AssetTypes.URI_FILE),
            "model_version": Input(type=AssetTypes.URI_FILE),
            "endpoint_name": "",
            "deployment_name": "",
            "instance_type": "",
            "instance_count": 1,
        },
        environment=environment,
        compute=compute,
    )


def build_pipeline(*, config: PipelineConfig, training_env: str, pipeline_env: str):
    train_component = _build_train_component(environment=training_env, compute=config.compute)
    register_component = _build_register_component(environment=pipeline_env, compute=config.compute)
    deploy_component = _build_deploy_component(environment=pipeline_env, compute=config.compute)

    @pipeline(description="Train, register, and deploy fraud detection model")
    def fraud_detection_pipeline(
        training_data: Input,
        model_name: str,
        endpoint_name: str,
        deployment_name: str,
        instance_type: str,
        instance_count: int,
    ):
        train_job = train_component(train_data=training_data)
        register_job = register_component(
            model_dir=train_job.outputs.output_dir,
            model_name=model_name,
        )
        deploy_job = deploy_component(
            model_name=register_job.outputs.model_name,
            model_version=register_job.outputs.model_version,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            instance_type=instance_type,
            instance_count=instance_count,
        )
        deploy_job.set_dependencies([register_job])
        return {
            "registered_model_name": register_job.outputs.model_name,
            "registered_model_version": register_job.outputs.model_version,
        }

    pipeline_job = fraud_detection_pipeline(
        training_data=_resolve_data_input(config.training_data),
        model_name=config.model_name,
        endpoint_name=config.endpoint_name,
        deployment_name=config.deployment_name,
        instance_type=config.instance_type,
        instance_count=config.instance_count,
    )
    pipeline_job.experiment_name = config.experiment_name
    return pipeline_job


def submit_pipeline(*, settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    ml_client = get_ml_client(settings=resolved_settings)

    validated = preflight_validate_training_data(resolved_settings.registered_data_path)
    if validated:
        logger.info(
            "Preflight data validation passed",
            extra={"dataset": resolved_settings.registered_data_path},
        )
    else:
        logger.info(
            "Preflight data validation skipped",
            extra={"dataset": resolved_settings.registered_data_path},
        )

    idempotency_key = build_idempotency_key(
        resolved_settings.custom_train_exp,
        resolved_settings.default_metric,
    )
    job_name = build_job_name("training-pipeline", idempotency_key)
    try:
        existing = ml_client.jobs.get(job_name)
    except ResourceNotFoundError:
        existing = None
    if existing:
        status = getattr(existing, "status", None)
        logger.info(
            "Existing training pipeline job reused",
            extra={"job_name": existing.name, "status": status},
        )
        return existing.name

    config = PipelineConfig(
        experiment_name=resolved_settings.custom_train_exp,
        training_data=resolved_settings.registered_data_path,
        compute=resolved_settings.get_training_compute(),
        model_name=resolved_settings.prod_model_name,
        endpoint_name=resolved_settings.endpoint_name,
        deployment_name=resolved_settings.deployment_name,
        instance_type=resolved_settings.get_deployment_instance_type(),
        instance_count=resolved_settings.instance_count,
    )

    xgb_config = xgb_sweep_job_builder(
        training_data=resolved_settings.registered_data_path,
        settings=resolved_settings,
    )
    training_env = resolve_xgb_environment(ml_client, config=xgb_config)
    pipeline_env = _resolve_pipeline_environment(ml_client)

    pipeline_job = build_pipeline(
        config=config,
        training_env=training_env,
        pipeline_env=pipeline_env,
    )
    pipeline_job.name = job_name
    pipeline_job.tags = {
        "project": "fraud-detection",
        "metric": resolved_settings.default_metric,
        "idempotency_key": idempotency_key,
    }
    created = ml_client.jobs.create_or_update(pipeline_job)
    logger.info("Submitted training pipeline", extra={"pipeline_job": created.name})
    return created.name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fraud detection pipeline utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit the training pipeline")
    submit_parser.set_defaults(handler=lambda _: submit_pipeline())

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("No command handler registered")
    handler(args)


if __name__ == "__main__":
    main()
