"""Command-line interface for the fraud detection toolkit."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError
from mlflow.entities import ViewType

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.data.data_validation import (
    DataValidationError,
    ValidationOptions,
    validate_creditcard_data,
)
from fraud_detection.registry.automl_registry import (
    list_completed_jobs,
    register_best_child_run,
)
from fraud_detection.registry.best_runs_by_metric import (
    AUTOML_DEFAULT_METRICS_BARE
)
from fraud_detection.registry.prod_model_by_metric import register_prod_model
from fraud_detection.registry.automl_registry import register_model_from_run
from fraud_detection.serving.deploy import deploy_model
from fraud_detection.serving.online_endpoint import (
    create_endpoint,
    delete_endpoint,
    update_traffic,
)
from fraud_detection.training.automl import (
    SUPPORTED_CLASSIFICATION_METRICS,
    automl_job_builder,
    create_automl_job,
    submit_job,
)
from fraud_detection.training.compute import (
    ensure_deployment_compute,
    ensure_training_compute,
)
from fraud_detection.utils.logging import get_logger

app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")
logger = get_logger(__name__)

settings = get_settings().require_azure()
ML_CLIENT = get_ml_client(settings=settings)

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
    validate_creditcard_data(df, options)


#This function should be more like validate data for training jobs.
@app.command()
def validate_data(
    path = settings.local_data_path,
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
    ] = 0.005,
    balance_upper: Annotated[
        float,
        typer.Option(
            "--balance-upper",
            help="Upper bound for the class balance ratio (fraudulent / total)",
        ),
    ] = 0.02,
):
    """Validate a local fraud dataset (CSV or MLTable folder)."""
    try:
        _validate_local_dataset(
            path=path,
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
    ] = settings.default_metric,
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
    ml_client = ML_CLIENT

    training_data = settings.dataset_name
    compute_target = settings.compute_cluster

    if not training_data:
        typer.echo("Provide --dataset or set AML_DATASET.", err=True)
        raise typer.Exit(code=1)

    if not compute_target:
        typer.echo("Provide --compute or set AML_COMPUTE_TRAIN/AML_COMPUTE.", err=True)
        raise typer.Exit(code=1)


    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.instance_type,
        min_instances=settings.compute_min_nodes,
        max_instances=settings.compute_max_nodes,
    )

    allowed_algorithms = _normalize_algorithms(algorithms)

    automl_job_config = automl_job_builder(
        metric=metric,
        training_data=training_data,
        compute=compute_target,
        allowed_algorithms=allowed_algorithms,
    )

    job = create_automl_job(config=automl_job_config)
    job_name = submit_job(ml_client=ml_client, job=job)
    typer.echo(f"Submitted AutoML job: {job_name}")



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
    ] = "norm-macro-recall",
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
    ml_client = ML_CLIENT
    experiment = settings.automl_train_exp
    if best_by_metric:
        try:
            register_prod_model(metric = metric)
        except Exception:
            typer.echo("Couldn't register the model")
            raise typer.Exit(code=0)

    elif best_child_run:
        completed = list_completed_jobs(ml_client=ml_client, experiments=experiment)
        if not completed:
            typer.echo("No completed AutoML jobs found for the given experiment prefix.")
            raise typer.Exit(code=0)

        for exp_name, jobs in completed.items():
            for job_name in jobs:
                model = register_best_child_run(
                    ml_client=ml_client,
                    job_name=job_name,
                )
                if isinstance(model, Model):
                    typer.echo(f"Registered model '{model.name}' version '{model.version}' from job '{job_name}'.")
                else:
                    typer.echo(f"Skipped registration for job '{job_name}'.")



@app.command()
def endpoint_create():
    """Create a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)
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
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)
    name = settings.endpoint_name
    delete_endpoint(ml_client, name=name)
    typer.echo(f"Deleted online endpoint '{name}'")




@app.command()
def deploy():
    """Deploy a registered model to a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)
    compute_target = settings.compute_cluster
    instance_type = settings.instance_type
    instance_count = settings.instance_count
    compute_min_nodes = settings.compute_min_nodes
    compute_max_nodes = settings.compute_max_nodes
    endpoint_name = settings.endpoint_name
    deployment_name = settings.deployment_name
    model_name = settings.prod_model_name


    ensure_deployment_compute(
        ml_client,
        name=compute_target,
        size=instance_type,
        min_instances=compute_min_nodes,
        max_instances=compute_max_nodes,
    )

    deployment = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
    )
    typer.echo(f"Deployed model '{model_name}' to endpoint '{endpoint_name}' as deployment '{deployment.name}'.")


@app.command()
def serve_invoke():
    """Invoke a managed online endpoint deployment (smoke test)."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)
    endpoint_name = settings.endpoint_name
    deployment_name = settings.deployment_name
    request_file = settings.smoke_test

    try:
        ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError as exc:
        raise typer.BadParameter(f"Endpoint '{endpoint_name}' does not exist") from exc

    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=str(request_file),
    )

    typer.echo(response)


def run():
    app()


if __name__ == "__main__":
    run()
