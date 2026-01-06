from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import (
    Choice,
    LogUniform,
    MedianStoppingPolicy,
    RandomSamplingAlgorithm,
    SamplingAlgorithm,
    Uniform,
)
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
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_LGBM_METRICS = {"average_precision", "AUC_macro"}

METRIC_ALIASES = {
    "metrics.average_precision_score_macro": "average_precision",
    "average_precision_score_macro": "average_precision",
    "metrics.AUC_macro": "AUC_macro",
}


def _metric_check(metric: str) -> str:
    metric_str = (metric or "").strip().replace("-", "_")
    metric_str = METRIC_ALIASES.get(metric_str, metric_str)
    if metric_str not in SUPPORTED_LGBM_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_LGBM_METRICS))
        raise ValueError(f"Unsupported metric '{metric_str}'. Choose one of: {allowed}")
    return metric_str


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


@dataclass
class LGBMSweepConfig:
    experiment_name: str
    training_data: str

    compute: str | None
    environment_name: str
    label_column: str = "Class"
    primary_metric: str = "average_precision"
    environment_version: str | None = None
    environment_file: Path = field(
        default_factory=lambda: ROOT_DIR / "src" / "fraud_detection" / "training" / "lgbm_env.yaml"
    )
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

    max_total_trials: int = 30
    max_concurrent_trials: int = 4
    timeout_minutes: int = 180
    sampling_algorithm: SamplingAlgorithm = RandomSamplingAlgorithm(seed=999, rule="sobol", logbase="e")

    early_stopping_delay: int = 5
    early_stopping_interval: int = 2

    tags: dict[str, str] = field(default_factory=dict)
    job_name: str | None = None
    idempotency_key: str | None = None


def lgbm_sweep_job_builder(
    *,
    training_data: str,
    metric: str = "average_precision",
    compute: str | None = None,
    settings: Settings | None = None,
) -> LGBMSweepConfig:
    resolved_metric = _metric_check(metric)
    resolved_settings = settings or get_settings()
    compute_target = (compute or "").strip() or resolved_settings.get_training_compute()
    idempotency_key = build_idempotency_key(resolved_settings.custom_train_exp, resolved_metric)
    return LGBMSweepConfig(
        experiment_name=resolved_settings.custom_train_exp,
        training_data=training_data,
        compute=compute_target,
        label_column="Class",
        primary_metric=resolved_metric,
        environment_name=resolved_settings.lgbm_env_name,
        environment_version=resolved_settings.lgbm_env_version,
        tags={
            "project": "fraud-detection",
            "model": "lightgbm",
            "metric": resolved_metric,
            "idempotency_key": idempotency_key,
        },
        job_name=build_job_name("lgbm-sweep", idempotency_key),
        idempotency_key=idempotency_key,
    )


def resolve_lgbm_environment(ml_client: MLClient, config: LGBMSweepConfig) -> str:
    if config.environment_version:
        env_id = f"{config.environment_name}:{config.environment_version}"
        logger.info("Using existing LightGBM environment", extra={"environment": env_id})
        return env_id

    if not config.environment_file.exists():
        raise FileNotFoundError(f"LightGBM environment file not found: {config.environment_file}")

    job_env = Environment(
        name=config.environment_name,
        description="LightGBM sweep environment",
        conda_file=str(config.environment_file),
        image=config.environment_image,
    )
    job_env = ml_client.environments.create_or_update(job_env)
    env_id = f"{job_env.name}:{job_env.version}"
    logger.info("Registered LightGBM environment", extra={"environment": env_id})
    return env_id


def create_lgbm_sweep_job(config: LGBMSweepConfig, *, environment: str) -> Any:
    if not config.experiment_name:
        raise ValueError("config.experiment_name is empty.")

    data_input = _resolve_data_input(config.training_data)

    base_job = command(
        code=str(ROOT_DIR / "src"),
        command=(
            "python -m fraud_detection.training.train_lgbm "
            "--train_data ${{inputs.train_data}} "
            "--label_col ${{inputs.label_col}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--num_leaves ${{inputs.num_leaves}} "
            "--max_depth ${{inputs.max_depth}} "
            "--learning_rate ${{inputs.learning_rate}} "
            "--subsample ${{inputs.subsample}} "
            "--colsample_bytree ${{inputs.colsample_bytree}} "
            "--min_child_weight ${{inputs.min_child_weight}} "
            "--reg_lambda ${{inputs.reg_lambda}} "
            "--output_dir ${{outputs.output_dir}}"
        ),
        inputs={
            "train_data": data_input,
            "label_col": config.label_column,
            "n_estimators": 600,
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        },
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        compute=config.compute,
        experiment_name=config.experiment_name,
        display_name="lgbm-train",
        tags=config.tags,
    )

    job_for_sweep = base_job(
        learning_rate=LogUniform(min_value=1e-3, max_value=0.2),
        max_depth=Choice(values=[-1, 3, 5, 7, 9]),
        n_estimators=Choice(values=[300, 600, 900, 1200]),
        num_leaves=Choice(values=[15, 31, 63, 127]),
        subsample=Uniform(min_value=0.6, max_value=1.0),
        colsample_bytree=Uniform(min_value=0.6, max_value=1.0),
        min_child_weight=Choice(values=[1.0, 5.0, 10.0]),
        reg_lambda=LogUniform(min_value=1e-3, max_value=10.0),
    )

    sweep_job = job_for_sweep.sweep(
        compute=config.compute,
        sampling_algorithm=config.sampling_algorithm,
        primary_metric=config.primary_metric,
        goal="Maximize",
    )

    sweep_job.set_limits(
        max_total_trials=config.max_total_trials,
        max_concurrent_trials=config.max_concurrent_trials,
        timeout=config.timeout_minutes * 60,
    )
    sweep_job.early_termination = MedianStoppingPolicy(
        delay_evaluation=config.early_stopping_delay,
        evaluation_interval=config.early_stopping_interval,
    )

    sweep_job.display_name = "lgbm-sweep"
    sweep_job.experiment_name = config.experiment_name
    sweep_job.tags = config.tags
    if config.job_name:
        sweep_job.name = config.job_name
    return sweep_job


def submit_lgbm_sweep_job(ml_client: MLClient, config: LGBMSweepConfig) -> str:
    validated = preflight_validate_training_data(config.training_data)
    if validated:
        logger.info("Preflight data validation passed", extra={"dataset": config.training_data})
    else:
        logger.info("Preflight data validation skipped", extra={"dataset": config.training_data})

    if config.job_name:
        try:
            existing = ml_client.jobs.get(config.job_name)
        except ResourceNotFoundError:
            existing = None
        if existing:
            status = getattr(existing, "status", None)
            logger.info(
                "Existing LightGBM sweep job reused",
                extra={"job_name": existing.name, "status": status},
            )
            return existing.name

    environment = resolve_lgbm_environment(ml_client, config)
    job = create_lgbm_sweep_job(config, environment=environment)
    created = ml_client.jobs.create_or_update(job)
    logger.info("Submitted LightGBM sweep job", extra={"job_name": created.name})
    return created.name


def main() -> None:
    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    training_data = settings.registered_data_path
    config = lgbm_sweep_job_builder(training_data=training_data, settings=settings)

    job_name = submit_lgbm_sweep_job(ml_client, config)
    print(f"Submitted sweep job: {job_name}")


__all__ = [
    "LGBMSweepConfig",
    "SUPPORTED_LGBM_METRICS",
    "create_lgbm_sweep_job",
    "submit_lgbm_sweep_job",
    "lgbm_sweep_job_builder",
    "resolve_lgbm_environment",
]


if __name__ == "__main__":
    main()
