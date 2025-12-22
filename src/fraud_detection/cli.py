"""Command-line interface for the fraud detection toolkit."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data, Model
from azure.core.exceptions import ResourceNotFoundError
from mlflow.entities import ViewType

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.data.data_validation import (
    DataValidationError,
    ValidationOptions,
    validate_creditcard_data,
)
from fraud_detection.training.automl import (
    EXPERIMENT_PREFIX,
    SUPPORTED_CLASSIFICATION_METRICS,
    AutoMLJobConfig,
    automl_job_builder,
    create_automl_job,
    submit_job,
)
from fraud_detection.registry.automl_registry import (
    list_completed_jobs,
    register_best_child_run,
    DEFAULT_EXPERIMENTS,
)
from fraud_detection.registry.best_runs_by_metric import (
    BestRun,
    best_run_by_metric,
    experiment_by_prefix,
)
from fraud_detection.registry.register_from_run import register_model_from_run
from fraud_detection.serving.deploy import deploy_model
from fraud_detection.serving.online_endpoint import (
    create_endpoint,
    delete_endpoint,
    update_traffic,
)
from fraud_detection.serving.choose_model import (
    build_registered_models_table,
    choose_registered_model,
    collect_registered_models,
)
from fraud_detection.git_var.git_add_secret import (
    GitHubSecretManager,
    update_model_and_endpoint_secrets,
)


app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")

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


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
DEFAULT_DATA_NAME = "creditcard-data"
DEFAULT_DATA_VERSION = "1"
DEFAULT_DATA_DESCRIPTION = "Credit card fraud detection dataset (from Kaggle)"


@app.command()
def register_local_data(
    path: Path = typer.Option(DEFAULT_DATA_PATH, "--path", "-[p]", exists=True, dir_okay=False, readable=True, help="Path to the local CSV file containing the dataset."),
    name: str = typer.Option(DEFAULT_DATA_NAME, "--name", "-n", help="Name of the registered dataset in Azure ML."),
    version: str = typer.Option(DEFAULT_DATA_VERSION, "--version", "-v", help="Version of the registered dataset."),
    description: str = typer.Option(DEFAULT_DATA_DESCRIPTION, "--description", "-d", help="Description of the registered dataset."),
):
    """Register a local fraud dataset (CSV file) to Azure ML data asset."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    data_asset = Data(
        name=name,
        path=str(path.resolve()),
        version=version,
        description=description,
        type=AssetTypes.URI_FILE,
    )

    try:
        ml_client.data.get(name=name, version=version)
        typer.echo(f"Data asset '{name}' version '{version}' already exists in Azure ML.")
        return
    except ResourceNotFoundError:
        pass

    ml_client.data.create_or_update(data_asset)
    typer.echo(f"Registered data asset '{name}' version '{version}' to Azure ML.")


@app.command()
def validate_data(
    path: str = typer.Option('data/creditcard.csv', "--path", "-p", help="local CSV file or local MLTable folder path"),
    sample_rows: int | None = typer.Option(None, "--sample-rows", help="Only load the first N rows for a quick validatoin pass (recommended for speed)", min=1),
    skip_balance_check: bool = typer.Option(False, "--balance-check", help="Checking the class balance in the dataset"),
    balance_lower: float = typer.Option(0.005, "--balance-lower", help="Lower bound for the class balance ratio (fraudulent / total)"),
    balance_upper: float = typer.Option(0.02, "--balance-upper", help="Upper bound for the class balance ratio (fraudulent / total)"),
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


@app.command()
def submit_automl(
    metric: str = typer.Option('recall', "--metric", "-m", help=f"Please choose from {list(SUPPORTED_CLASSIFICATION_METRICS)}", case_sensitive=False),
    dataset: str = typer.Option(None, help="Override the MLTable asset path (can be local)."), # This is questionable.
    compute: str = typer.Option(None, help="Override the Azure ML compute target name."), # And also this one.
    algorithms: list[str] | None = typer.Option(
        None,
        "--algorithms",
        "-a",
        help="Restrict training algorithms (comma-separated or repeated). "
        "If not provided, AutoML uses its default algorithm set.",
    ),
):
    """Submit an AutoML classification job for fraud detection."""
    settings = get_settings().require_azure()

    training_data = dataset or settings.aml_dataset
    compute_target = compute or settings.aml_compute

    if not training_data:
        raise typer.BadParameter("Provide --dataset or set AML_DATASET.")

    if not compute_target:
        raise typer.BadParameter("Provide --compute or set AML_COMPUTE.")

    #Note: CD pipeline will always validate the dataset before submitting the job. So we skip local validation here.

    allowed_algorithms = _normalize_algorithms(algorithms)
    
    automl_job_config = automl_job_builder(
        metric=metric,
        training_data=training_data,
        compute=compute_target,
        allowed_algorithms=allowed_algorithms,
    )

    job = create_automl_job(config=automl_job_config)
    ml_client = get_ml_client(settings=settings)
    job_name = submit_job(ml_client=ml_client, job=job)
    typer.echo(f"Submitted AutoML job: {job_name}")


@app.command()
def register_best_automl_model(
    experiment: str | None = typer.Option(None, "--experiment", "-e", help="Experiment name prefix to search for AutoML jobs (default: all auotoml jobs)."),
    metric: str = typer.Option('recall', "--metric", "-m", help=f"Please choose from {list(SUPPORTED_CLASSIFICATION_METRICS)}", case_sensitive=False),
    model_name: str = typer.Option(None, "--model-name", help="Optional stable model name to use instead of slugified experiment names. Useful or CD pipelines."),
    best_child_run: bool = typer.Option(False, "--best-child-run", help="Register the best child run model from the AutoML parent run."),
    best_by_metric: bool = typer.Option(False, "--best-by-metric", help="Register the best AutoML model with highest given metric."),
):
    """Register the best AutoML child run model or Register the best AutoML model with the highest given metric."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    if best_by_metric:
        try:
            best_run: BestRun = best_run_by_metric(
                ml_client=ml_client,
                experiment_prefixes=[experiment or EXPERIMENT_PREFIX],
                metric=metric,
                view_type=ViewType.ALL,
            )
            typer.echo(
                f"Found best model by metric: {best_run.metric_name}={best_run.metric_value} from run {best_run.run_id} in experiment {best_run.experiment_name}."
            )
        except Exception as e:
            raise typer.Exit(code=1) from e

        try:
            register_model_from_run(
                ml_client=ml_client,
                model_name=model_name or f"{best_run.experiment_name}-best-child-model",
                run_id=best_run.run_id,
                description=f"Best child run model by {best_run.metric_name}={best_run.metric_value}",
            )
            typer.echo(
                f"Registered best child run model: {best_run.metric_name}={best_run.metric_value} from run {best_run.run_id} in experiment {best_run.experiment_name}."
            )
        except Exception as e:
            raise typer.Exit(code=1) from e

    elif best_child_run:
        prefix = experiment if experiment else EXPERIMENT_PREFIX
        exps = experiment_by_prefix(ml_client, prefix=prefix)
        exp_names = [e.experiment_name for e in exps]

        if not exp_names:
            typer.echo("No AutoML experiments found for the given prefix.")
            raise typer.Exit(code=0)

        experiments = experiment or exp_names

        completed = list_completed_jobs(ml_client=ml_client, experiments=experiments)
        if not completed:
            typer.echo("No completed AutoML jobs found for the given experiment prefix.")
            raise typer.Exit(code=0)

        for exp_name, jobs in completed.items():
            for job_name in jobs:
                model = register_best_child_run(
                    ml_client=ml_client,
                    job_name=job_name,
                    experiment_name=exp_name,
                    model_name=model_name,
                )
                if isinstance(model, Model):
                    typer.echo(
                        f"Registered model '{model.name}' version '{model.version}' from job '{job_name}'."
                    )
                else:
                    typer.echo(f"Skipped registration for job '{job_name}'.")


@app.command()
def endpoint_create(
    name: str = typer.Option(..., "--name", "-n", help="Name of the online endpoint to create."),
    description: str = typer.Option(None, "--description", "-d", help="Description of the online endpoint.")
):
    """Create a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    ep = create_endpoint(
        ml_client,
        name=name,
        description=description,
        tags={"project": "fraud-detection"},
    )
    typer.echo(f"Created online endpoint '{ep.name}'")


@app.command()
def endpoint_delete(
    name: str = typer.Option(..., "--name", "-n", help="Name of the online endpoint to delete."),
):
    """Delete a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    delete_endpoint(ml_client, name=name)
    typer.echo(f"Deleted online endpoint '{name}'")
    

@app.command()
def endpoint_traffic(
    endpoint: str = typer.Option(..., "--endpoint", "-e", help="Name of the online endpoint to update."),
    traffic: list[str] = typer.Option(..., "--traffic", "-t", help="Traffic mapping like blue=100 green=0 (repeatable)"),
):
    """Update endpoint traffic split. Traffic must sum to 100%."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    mapping: dict[str, int] = {}
    for item in traffic:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid traffic item '{item}'. Use deployment=percent.")
        dep, pct = item.split("=", 1)
        dep = dep.strip()
        try:
            mapping[dep] = int(pct)
        except ValueError as e:
            raise typer.BadParameter(f"Invalid percent '{pct}' for deployment '{dep}'.") from e
    
    update_traffic(ml_client, endpoint_name=endpoint, traffic=mapping)
    typer.echo(f"Updated traffic for endpoint '{endpoint}': {mapping}")


@app.command()
def choose_one_registered_model(
    endpoint_name: str = typer.Option(
        "fraud-detection-prod",
        "--endpoint-name",
        "-e",
        help="Managed online endpoint to target in the CD pipeline.",
    ),
    name_prefix: str | None = typer.Option(
        None, "--name-prefix", help="Optional prefix to filter registered models."
    ),
):
    """Pick a registered Azure ML model and store its name for deployment.

    The chosen model name and endpoint name are written to GitHub Actions
    secrets (MODEL_NAME and ENDPOINT_NAME) so the CD workflow deploys the
    correct artifact.
    """

    settings = get_settings().require_azure().require_github()

    ml_client = get_ml_client(settings=settings)
    models = collect_registered_models(ml_client, name_prefix=name_prefix)

    if not models:
        typer.echo("No registered models found in the workspace.")
        raise typer.Exit(code=1)

    typer.echo("Available registered models:")
    for line in build_registered_models_table(models):
        typer.echo(line)

    selection = typer.prompt(
        "Enter the number of the model to promote for deployment",
        type=int,
    )

    try:
        chosen = choose_registered_model(models, selection)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    secrets_manager = GitHubSecretManager.from_settings(settings)
    update_model_and_endpoint_secrets(
        manager=secrets_manager,
        model_name=chosen.name,
        endpoint_name=endpoint_name,
    )

    typer.echo(
        f"Selected model '{chosen.name}' (version {chosen.version}) for endpoint '{endpoint_name}'. GitHub secrets updated."
    )


@app.command()
def deploy(
    model_name: str = typer.Option(None, "--model-name", "-m", help="Name of the registered model to deploy."),
    endpoint_name: str = typer.Option(None, "--endpoint-name", "-e", help="Name of the online endpoint to deploy to."),
    deployment_name: str = typer.Option("blue", "--deployment-name", "-d", help="Deployment slot name."),
    instance_type: str = typer.Option("Standard_DS3_v2", "--instance-type", "-it", help="Azure ML compute instance type."),
    instance_count: int = typer.Option(1, "--instance-count", "-ic", help="Number of instances to deploy."),
    tags: list[str] = typer.Option(None, "--tag", "-t", help="Tags to apply to the deployment (repeatable key=value)."),
):
    """Deploy a registered model to a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    tag_mapping: dict[str, str] = {}
    for tag in tags or []:
        if "=" not in tag:
            raise typer.BadParameter(f"Invalid tag '{tag}'. Use key=value format.")
        key, value = tag.split("=", 1)
        tag_mapping[key.strip()] = value.strip()

    deployment = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        tags=tag_mapping or {"project": "fraud-detection"},
    )
    typer.echo(f"Deployed model '{model_name}' to endpoint '{endpoint_name}' as deployment '{deployment.name}'.")


DEFAULT_REQUEST_FILE = ROOT_DIR / "sample_request.json"

@app.command()
def serve_invoke(
    endpoint_name: str = typer.Option(..., "--endpoint-name", "-e", help="Endpoint to call"),
    deployment_name: str = typer.Option("blue", "--deployment-name", "-d", help="Deployment slot name"),
    request_file: Path = typer.Option(DEFAULT_REQUEST_FILE, "--request-file", "-r", exists=True, dir_okay=False, readable=True, help="Path to JSON request payload"),
) -> None:
    """Invoke a managed online endpoint deployment (smoke test)."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    try:
        ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError as exc:
        raise typer.BadParameter(f"Endpoint '{endpoint_name}' does not exist") from exc

    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file,
    )

    # echo response for CLI visibility
    typer.echo(response)


def run():
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
