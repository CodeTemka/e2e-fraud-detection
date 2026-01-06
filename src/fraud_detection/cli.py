"""Command-line interface for the fraud detection toolkit."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import ROOT_DIR, Settings, get_settings
from fraud_detection.data.data_validation import (
    DEFAULT_CLASS_RATIO_BOUNDS,
    DataValidationError,
    ValidationOptions,
    validate_creditcard_data,
)
from fraud_detection.registry.automl_registry import (
    list_completed_jobs,
    register_best_child_run,
)
from fraud_detection.registry.best_runs_by_metric import AUTOML_DEFAULT_METRICS_BARE
from fraud_detection.registry.custom_model_registry import (
    register_best_custom_model as register_custom_model,
)
from fraud_detection.registry.model_competition import (
    select_and_register_prod_model as select_prod_model,
)
from fraud_detection.registry.prod_model_by_metric import register_prod_model
from fraud_detection.serving.deploy import deploy_model, resolve_scoring_environment
from fraud_detection.serving.online_endpoint import (
    create_endpoint,
    delete_endpoint,
    set_initial_traffic,
    shift_traffic,
)
from fraud_detection.training.automl import (
    SUPPORTED_CLASSIFICATION_METRICS,
    automl_job_builder,
    create_automl_job,
    submit_job,
)
from fraud_detection.training.compute import ensure_training_compute
from fraud_detection.training.submit_lgbm import (
    SUPPORTED_LGBM_METRICS,
    create_lgbm_sweep_job,
    lgbm_sweep_job_builder,
    resolve_lgbm_environment,
    submit_lgbm_sweep_job,
)
from fraud_detection.training.submit_xgb import (
    SUPPORTED_XGB_METRICS,
    create_xgb_sweep_job,
    resolve_xgb_environment,
    submit_xgb_sweep_job,
    xgb_sweep_job_builder,
)
from fraud_detection.utils.logging import get_logger

app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")
logger = get_logger(__name__)

def _resolve_settings() -> Settings:
    return get_settings()


def _resolve_ml_client(settings: Settings) -> object:
    return get_ml_client(settings=settings)


def _parse_version(v: str | None) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0

def _normalize_algorithms(values: list[str] | None) -> list[str] | None:
    """Allow repeated -a and comma-seperated lists; return None if emtpy."""
    if not values:
        return None
    out: list[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or None


def _load_local_table(path: str, *, sample_rows: int | None = None) -> pd.DataFrame:
    """Load a local dataset from CSV or MLTable folder into a Pandas DataFrame.

    If sample_rows is provided, load only the first N rows for quick validation.
    """
    p = Path(path)

    # CSV file
    if p.is_file() and p.suffix.lower() == ".csv":
        return pd.read_csv(p, nrows=sample_rows)

    # Local MLTable folder: should contain a file literally named "MLTable"
    if p.is_dir() and (p / "MLTable").is_file():
        import mltable

        tbl = mltable.load(str(p))
        if sample_rows is not None:
            tbl = tbl.take(int(sample_rows))
        return tbl.to_pandas_dataframe()

    raise ValueError("Unsupported path. Provide a .csv file or a folder containing an MLTable file.")


# I guess, cli.py doesn't need this function.
def _validate_local_dataset(
    *,
    path: str,
    skip_balance_check: bool,
    balance_lower: float,
    balance_upper: float,
    sample_rows: int | None,
) -> None:
    """Validate a local dataset for fraud detection suitability."""
    df = _load_local_table(path, sample_rows=sample_rows)
    options = ValidationOptions(
        check_balance=not skip_balance_check,
        class_ratio_bounds=(balance_lower, balance_upper),
    )
    validate_creditcard_data(df, options=options)


#This function should be more like validate data for training jobs.
@app.command()
def validate_data(
    path: str | None = None,
    sample_rows: Annotated[
        int | None,
        typer.Option(
            "--sample-rows",
            help="Only load the first N rows for a quick validatoin pass (recommended for speed)",
            min=1,
        ),
    ] = 1000,
    skip_balance_check: Annotated[
        bool,
        typer.Option("--balance-check", help="Checking the class balance in the dataset"),
    ] = False,
    balance_lower: Annotated[
        float,
        typer.Option(
            "--balance-lower",
            help="Lower bound for the class balance ratio (fraudulent / total)",
        ),
    ] = DEFAULT_CLASS_RATIO_BOUNDS[0],
    balance_upper: Annotated[
        float,
        typer.Option(
            "--balance-upper",
            help="Upper bound for the class balance ratio (fraudulent / total)",
        ),
    ] = DEFAULT_CLASS_RATIO_BOUNDS[1],
):
    """Validate a local fraud dataset (CSV or MLTable folder)."""
    settings = _resolve_settings()
    resolved_path = str(path or settings.local_data_path)
    try:
        _validate_local_dataset(
            path=resolved_path,
            skip_balance_check=skip_balance_check,
            balance_lower=balance_lower,
            balance_upper=balance_upper,
            sample_rows=sample_rows,
        )
    except (FileNotFoundError, ValueError, DataValidationError) as e:
        raise typer.BadParameter(str(e)) from e

    typer.echo("Data validation passed successfully.")



# This function is used for only locally not for CD pipeline.
@app.command()
def submit_automl(
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help=f"Please choose from {list(SUPPORTED_CLASSIFICATION_METRICS)}",
            case_sensitive=False,
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Azure ML dataset name or ID to use for training."),
    ] = None,
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute cluster name for training (optional override)."),
    ] = None,
    algorithms: Annotated[
        list[str] | None,
        typer.Option(
            "--algorithms",
            "-a",
            help="Restrict training algorithms (comma-separated or repeated). If not provided, AutoML uses its default algorithm set.",
        ),
    ] = None,
):
    """Submit an AutoML classification job for fraud detection."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)

    resolved_metric = metric or settings.default_metric
    training_data = dataset if dataset is not None else settings.registered_data_path

    def _ensure_mltable_ref(training_data: str) -> None:
        # If local path, ensure it’s an MLTable folder
        if not training_data.startswith("azureml:"):
            p = Path(training_data)
            if not (p.exists() and p.is_dir() and (p / "MLTable").exists()):
                raise ValueError(
                    "AutoML requires MLTable. Provide an MLTable folder path (containing a file named 'MLTable') "
                    "or an Azure ML MLTable data asset like 'azureml:<name>:<version>'."
                )

    if not training_data:
        typer.echo("Provide --dataset or set AML_DATASET.", err=True)
        raise typer.Exit(code=1)

    try:
        _ensure_mltable_ref(training_data)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    
    compute_value = compute.strip() if isinstance(compute, str) else None
    try:
        compute_target = compute_value or settings.get_training_compute()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.instance_type,
        min_instances=settings.compute_min_nodes,
        max_instances=settings.compute_max_nodes,
    )

    allowed_algorithms = _normalize_algorithms(algorithms)

    automl_job_config = automl_job_builder(
        metric=resolved_metric,
        training_data=training_data,
        compute=compute_target,
        allowed_algorithms=allowed_algorithms,
    )

    job = create_automl_job(config=automl_job_config)
    job_name = submit_job(ml_client=ml_client, job=job)
    typer.echo(f"Submitted AutoML job: {job_name}")


@app.command()
def submit_xgb_sweep(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Azure ML dataset name or ID to use for training."),
    ] = None,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help=f"Primary metric for the sweep (choices: {sorted(SUPPORTED_XGB_METRICS)}).",
            case_sensitive=False,
        ),
    ] = 'metrics.average_precision_score_macro',
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute cluster name for training (optional override)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Build the sweep job without submitting it."),
    ] = False,
):
    """Submit an XGBoost hyperparameter sweep job."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)

    training_data = dataset if dataset is not None else settings.registered_data_path
    if not training_data:
        typer.echo("Provide --dataset or set AML_DATASET.", err=True)
        raise typer.Exit(code=1)

    compute_value = compute.strip() if isinstance(compute, str) else None
    try:
        compute_target = compute_value or settings.get_training_compute()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.instance_type,
        min_instances=settings.compute_min_nodes,
        max_instances=settings.compute_max_nodes,
    )

    try:
        config = xgb_sweep_job_builder(
            training_data=training_data,
            metric=metric,
            compute=compute_target,
            settings=settings,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        environment = resolve_xgb_environment(ml_client, config)
        job = create_xgb_sweep_job(config, environment=environment)
        typer.echo(f"Built XGBoost sweep job: {job.experiment_name}")
        return

    job_name = submit_xgb_sweep_job(ml_client, config)
    typer.echo(f"Submitted XGBoost sweep job: {job_name}")


@app.command()
def submit_lgbm_sweep(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Azure ML dataset name or ID to use for training."),
    ] = None,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help=f"Primary metric for the sweep (choices: {sorted(SUPPORTED_LGBM_METRICS)}).",
            case_sensitive=False,
        ),
    ] = "metrics.average_precision_score_macro",
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute cluster name for training (optional override)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Build the sweep job without submitting it."),
    ] = False,
):
    """Submit a LightGBM hyperparameter sweep job."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)

    training_data = dataset if dataset is not None else settings.registered_data_path
    if not training_data:
        typer.echo("Provide --dataset or set AML_DATASET.", err=True)
        raise typer.Exit(code=1)

    compute_value = compute.strip() if isinstance(compute, str) else None
    try:
        compute_target = compute_value or settings.get_training_compute()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.instance_type,
        min_instances=settings.compute_min_nodes,
        max_instances=settings.compute_max_nodes,
    )

    try:
        config = lgbm_sweep_job_builder(
            training_data=training_data,
            metric=metric,
            compute=compute_target,
            settings=settings,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        environment = resolve_lgbm_environment(ml_client, config)
        job = create_lgbm_sweep_job(config, environment=environment)
        typer.echo(f"Built LightGBM sweep job: {job.experiment_name}")
        return

    job_name = submit_lgbm_sweep_job(ml_client, config)
    typer.echo(f"Submitted LightGBM sweep job: {job_name}")



@app.command()
def register_best_automl_model(
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help=f"Please choose from {list(AUTOML_DEFAULT_METRICS_BARE)}",
            case_sensitive=False,
        ),
    ] = "average_precision_score_macro",
    best_child_run: Annotated[
        bool,
        typer.Option("--best-child-run", help="Register the best child run model from the AutoML parent run."),
    ] = False,
    best_by_metric: Annotated[
        bool,
        typer.Option("--best-by-metric", help="Register the best AutoML model with highest given metric."),
    ] = True,
):
    """Register the best AutoML child run model or the best AutoML model by metric."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    experiment = settings.automl_train_exp
    if best_child_run:
        completed = list_completed_jobs(ml_client=ml_client, experiments=experiment)
        if not completed:
            typer.echo("No completed AutoML jobs found for the given experiment prefix.")
            raise typer.Exit(code=0)

        for exp_name, jobs in completed.items():
            for job_name in jobs:
                model = register_best_child_run(
                    ml_client=ml_client,
                    job_name=job_name,
                    experiment=exp_name,
                    settings=settings,
                )
                if isinstance(model, Model):
                    typer.echo(f"Registered model '{model.name}' version '{model.version}' from job '{job_name}'.")
                else:
                    typer.echo(f"Skipped registration for job '{job_name}'.")
        return

    if best_by_metric:
        try:
            model_name = register_prod_model(metric=metric, ml_client=ml_client, settings=settings, experiment=experiment)
        except (RuntimeError, ValueError, KeyError) as exc:
            logger.error("Couldn't register production model", extra={"error": str(exc)})
            raise typer.Exit(code=1) from exc
        typer.echo(f"Registered model '{model_name}' based on metric '{metric}'.")



@app.command()
def register_best_custom_model(
    metric: str = typer.Option(
        'average_precision_score_macro', "--metric", "-m",
        help=f"Please choose from {list(AUTOML_DEFAULT_METRICS_BARE)}",
        case_sensitive=False,)
):
    """Register best custom trained model by specific metric"""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    experiment = settings.custom_train_exp

    try:
        result = register_custom_model(
            metric=metric,
            ml_client=ml_client,
            settings=settings,
            experiment=experiment,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.error("Couldn't register production model", extra={"error": str(exc)})
        raise typer.Exit(code=1) from exc

    if not result.promoted:
        typer.echo(
            "Best custom model did not exceed production metrics; no promotion performed."
        )
        return

    env_id = resolve_scoring_environment(ml_client, settings=settings)
    deployment = deploy_model(
        ml_client,
        endpoint_name=settings.endpoint_name,
        deployment_name=settings.deployment_name,
        model_name=result.model_name,
        model_version=result.model_version,
        instance_type=settings.get_deployment_instance_type(),
        instance_count=settings.instance_count,
        code_path=str(ROOT_DIR / "src"),
        scoring_script="fraud_detection/serving/custom_scoring.py",
        environment=env_id,
        environment_variables={"ALERT_CAP": str(settings.default_alert_cap)},
        tags={
            "project": "fraud-detection",
            "model_source": "custom",
            "metric": metric,
        },
    )
    typer.echo(
        f"Promoted model '{result.model_name}' version '{result.model_version}' "
        f"and deployed to endpoint '{deployment.endpoint_name}' ({deployment.name})."
    )



@app.command()
def select_and_register_prod_model(
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            "-m",
            help="Primary metric used for AutoML/custom competition.",
            case_sensitive=False,
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output-file",
            help="Optional GitHub Actions output file to write promotion results.",
        ),
    ] = None,
):
    """Select the best model across AutoML and custom runs and promote if better than production."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    resolved_metric = metric or settings.default_metric

    try:
        result = select_prod_model(
            metric=resolved_metric,
            ml_client=ml_client,
            settings=settings,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.error("Couldn't select/register production model", extra={"error": str(exc)})
        raise typer.Exit(code=1) from exc

    typer.echo(f"Best AutoML metric: {result.automl_metric}")
    typer.echo(f"Best custom metric: {result.custom_metric}")
    typer.echo(f"Current prod metric: {result.prod_metric}")

    if result.promoted:
        typer.echo(
            f"Promoted {result.model_source} model '{result.model_name}' version '{result.model_version}' "
            f"for metric '{result.metric}'."
        )
    else:
        typer.echo("No change: neither candidate beat current production metrics.")

    if output_file:
        lines = [
            f"promoted={str(result.promoted).lower()}",
            f"model_name={result.model_name or ''}",
            f"model_version={result.model_version or ''}",
            f"model_source={result.model_source or ''}",
        ]
        with output_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

@app.command()
def endpoint_create():
    """Create a managed online endpoint."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    name = settings.endpoint_name
    ep = create_endpoint(
        ml_client,
        name=name,
        tags={"project": "fraud-detection"},
    )
    typer.echo(f"Created online endpoint '{ep.name}'")



@app.command()
def endpoint_delete():
    """Delete a managed online endpoint."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    name = settings.endpoint_name
    delete_endpoint(ml_client, name=name)
    typer.echo(f"Deleted online endpoint '{name}'")




@app.command()
def deploy(
    traffic: Annotated[
        int | None,
        typer.Option(
            "--traffic",
            help="Initial canary traffic percent for the deployment (1-99).",
            min=1,
            max=99,
        ),
    ] = None,
):
    """Deploy a registered model to a managed online endpoint."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    try:
        instance_type = settings.get_deployment_instance_type()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    instance_count = 1
    endpoint_name = settings.endpoint_name
    deployment_name = settings.deployment_name
    model_name = settings.prod_model_name
    models = list(ml_client.models.list(name=model_name))
    if not models:
        typer.echo(f"No registered models found for '{model_name}'.", err=True)
        raise typer.Exit(code=1)
    latest = max(models, key=lambda m: _parse_version(getattr(m, "version", None)))
    model_version = str(getattr(latest, "version", "") or "")
    model_tags = getattr(latest, "tags", {}) or {}
    model_source = (model_tags.get("model_source") or "").lower()

    deploy_kwargs = {}
    if model_source == "custom":
        env_id = resolve_scoring_environment(ml_client, settings=settings)
        deploy_kwargs.update(
            {
                "code_path": str(ROOT_DIR / "src"),
                "scoring_script": "fraud_detection/serving/custom_scoring.py",
                "environment": env_id,
                "environment_variables": {"ALERT_CAP": str(settings.default_alert_cap)},
                "tags": {
                    "project": "fraud-detection",
                    "model_source": "custom",
                },
            }
        )

    deployment = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=model_version or None,
        instance_type=instance_type,
        instance_count=instance_count,
        **deploy_kwargs,
    )
    typer.echo(f"Deployed model '{model_name}' to endpoint '{endpoint_name}' as deployment '{deployment.name}'.")

    if traffic is not None:
        traffic_map = set_initial_traffic(
            ml_client,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            low_traffic_percent=traffic,
        )
        typer.echo(f"Updated traffic for endpoint '{endpoint_name}': {traffic_map}")


@app.command()
def traffic_shift(
    target: Annotated[
        int,
        typer.Option("--target", help="Target traffic percent for the deployment.", min=1, max=100),
    ] = 100,
    step: Annotated[
        int,
        typer.Option("--step", help="Percent to shift in each step.", min=1, max=100),
    ] = 10,
    wait_seconds: Annotated[
        int,
        typer.Option("--wait-seconds", help="Seconds to wait between traffic shifts.", min=0),
    ] = 0,
    deployment: Annotated[
        str | None,
        typer.Option("--deployment", help="Deployment name to shift traffic toward (defaults to config)."),
    ] = None,
):
    """Gradually shift traffic to a deployment."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    deployment_name = deployment or settings.deployment_name
    traffic = shift_traffic(
        ml_client,
        endpoint_name=settings.endpoint_name,
        deployment_name=deployment_name,
        target_percent=target,
        step_percent=step,
        wait_seconds=wait_seconds,
    )
    typer.echo(
        f"Shifted traffic for endpoint '{settings.endpoint_name}' toward '{deployment_name}': {traffic}"
    )


@app.command()
def serve_invoke(
    repeat: Annotated[
        int,
        typer.Option("--repeat", help="Number of smoke requests to send.", min=1),
    ] = 1,
    latency_threshold_ms: Annotated[
        int | None,
        typer.Option("--latency-threshold-ms", help="Fail if average latency exceeds this threshold.", min=1),
    ] = None,
    error_rate_threshold: Annotated[
        float | None,
        typer.Option(
            "--error-rate-threshold",
            help="Fail if the error rate exceeds this value (0-1).",
            min=0.0,
            max=1.0,
        ),
    ] = None,
):
    """Invoke a managed online endpoint deployment (smoke test)."""
    settings = _resolve_settings()
    ml_client = _resolve_ml_client(settings)
    endpoint_name = settings.endpoint_name
    deployment_name = settings.deployment_name
    request_file = settings.smoke_test

    try:
        ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError as exc:
        raise typer.BadParameter(f"Endpoint '{endpoint_name}' does not exist") from exc

    errors = 0
    latencies: list[float] = []
    last_response = None
    for _ in range(repeat):
        start = time.monotonic()
        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=str(request_file),
            )
            last_response = response
            latencies.append((time.monotonic() - start) * 1000)
        except Exception as exc:  # noqa: BLE001 - azure ml raises broad exceptions
            errors += 1
            logger.error("Smoke test request failed", extra={"error": str(exc)})

    if last_response is not None:
        typer.echo(last_response)

    total = repeat
    error_rate = errors / total
    avg_latency = sum(latencies) / len(latencies) if latencies else float("inf")
    logger.info(
        "Smoke test metrics",
        extra={
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "requests": total,
            "errors": errors,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency,
        },
    )

    if latency_threshold_ms is not None and avg_latency > latency_threshold_ms:
        raise typer.Exit(code=1)
    if error_rate_threshold is not None and error_rate > error_rate_threshold:
        raise typer.Exit(code=1)


def run():
    app()


if __name__ == "__main__":
    run()
