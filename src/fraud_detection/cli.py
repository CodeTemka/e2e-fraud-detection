"""Command-line interface for the fraud detection toolkit."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from azure.ai.ml.entities import Model
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
from fraud_detection.serving.deploy import deploy_best_by_metric
from fraud_detection.serving.deploy_latest_model import deploy_latest_model
from fraud_detection.serving.online_endpoint import (
    create_endpoint,
    delete_endpoint,
    deploy_model,
    update_traffic,
)
from fraud_detection.training.automl import (
    EXPERIMENT_ACCURACY,
    EXPERIMENT_RECALL,
    AutoMLJobConfig,
    accuracy_job,
    create_automl_job,
    recall_job,
    submit_job,
)


app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")
serve_app = typer.Typer(help="Serving operations (managed online endpoints).")
app.add_typer(serve_app, name="serve")

DEFAULT_AUTOML_EXPERIMENTS = [EXPERIMENT_RECALL, EXPERIMENT_ACCURACY]


def _normalize_algorithms(values: list[str] | None) -> list[str] | None:
    """Allow repeated -a and comma-separated lists; return None if empty."""
    if not values:
        return None
    out: list[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or None


def _load_local_table(path: str, *, sample_rows: int | None = None) -> pd.DataFrame:
    """Load a local dataset from CSV or MLTable folder into a pandas DataFrame.

    If sample_rows is provided, load only the first N rows for quick validation.
    """
    p = Path(path)

    # CSV file
    if p.is_file() and p.suffix.lower() == ".csv":
        return pd.read_csv(p, nrows=sample_rows)

    # Local MLTable folder: should contain a file literally named "MLTable"
    if p.is_dir() and (p / "MLTable").exists():
        try:
            import mltable  # type: ignore
        except ImportError as e:
            raise ValueError(
                "MLTable support requires the 'mltable' package. "
                "Install it in your environment, or validate using the CSV instead."
            ) from e

        tbl = mltable.load(str(p))
        if sample_rows is not None:
            tbl = tbl.take(int(sample_rows))  # take first N rows
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
    df = _load_local_table(path, sample_rows=sample_rows)
    options = ValidationOptions(
        check_balance=not skip_balance_check,
        class_ratio_bounds=(balance_lower, balance_upper),
    )
    validate_creditcard_data(df, options=options)


@app.command()
def validate_data(
    path: str = typer.Option(..., "--path", "-p", help="Local CSV file or local MLTable folder path"),
    sample_rows: int | None = typer.Option(
        None,
        "--sample-rows",
        help="Only load the first N rows for a quick validation pass (recommended for speed).",
        min=1,
    ),
    skip_balance_check: bool = typer.Option(
        False, "--skip-balance-check", help="Skip the class imbalance ratio check"
    ),
    balance_lower: float = typer.Option(
        0.0005, "--balance-lower", help="Lower bound for fraud class ratio (e.g. 0.0005 = 0.05%)"
    ),
    balance_upper: float = typer.Option(
        0.02, "--balance-upper", help="Upper bound for fraud class ratio (e.g. 0.02 = 2%)"
    ),
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

    typer.echo("✅ Data validation passed.")


@app.command()
def submit_automl(
    metric: Annotated[
        str,
        typer.Option("--metric", "-m", help="Choose 'accuracy' or 'recall'.", case_sensitive=False),
    ] = "recall",
    dataset: str = typer.Option(None, help="Override the MLTable asset path (can be local)."),
    compute: str = typer.Option(None, help="Override the compute target."),
    algorithms: Annotated[
        list[str] | None,
        typer.Option(
            "--algorithm",
            "-a",
            help=(
                "Restrict AutoML to these algorithms. "
                "Pass multiple times (-a A -a B) or comma-separated (-a A,B). "
                "If omitted, AutoML tries its default algorithm set."
            ),
        ),
    ] = None,
    validate_local: bool = typer.Option(
        False,
        "--validate-local/--no-validate-local",
        help="Validate the local dataset before submitting the AutoML job.",
    ),
    validate_path: str = typer.Option(
        None,
        "--validate-path",
        help="Path to local CSV/MLTable to validate. Defaults to --dataset if it points locally.",
    ),
    sample_rows: int | None = typer.Option(
        None,
        "--sample-rows",
        help="(Validation only) Validate using only the first N rows for speed.",
        min=1,
    ),
    skip_balance_check: bool = typer.Option(
        False, "--skip-balance-check", help="Skip the class imbalance ratio check (validation)"
    ),
    balance_lower: float = typer.Option(
        0.0005, "--balance-lower", help="Lower bound for fraud class ratio"
    ),
    balance_upper: float = typer.Option(
        0.02, "--balance-upper", help="Upper bound for fraud class ratio"
    ),
):
    """Submit an AutoML job to Azure ML for fraud detection (optionally validate local data first)."""
    settings = get_settings().require_azure()

    training_data = dataset or settings.aml_dataset
    compute_target = compute or settings.aml_compute

    if not training_data:
        typer.echo("Provide --dataset or set AML_DATASET.", err=True)
        raise typer.Exit(code=1)
    if not compute_target:
        typer.echo("Provide --compute or set AML_COMPUTE.", err=True)
        raise typer.Exit(code=1)

    # Optional: local validation preflight
    if validate_local:
        candidate = validate_path or training_data
        p = Path(candidate)
        if not (p.exists() and (p.is_file() or p.is_dir())):
            raise typer.BadParameter(
                "To validate locally, provide --validate-path pointing to a local CSV or MLTable folder "
                "(or pass a local path via --dataset)."
            )
        try:
            _validate_local_dataset(
                path=str(p),
                skip_balance_check=skip_balance_check,
                balance_lower=balance_lower,
                balance_upper=balance_upper,
                sample_rows=sample_rows,
            )
        except (FileNotFoundError, ValueError, DataValidationError) as e:
            raise typer.BadParameter(f"Local data validation failed: {e}") from e

        typer.echo("✅ Local data validation passed. Submitting AutoML job...")

    allowed_algorithms = _normalize_algorithms(algorithms)

    metric_l = metric.lower()
    if metric_l == "accuracy":
        config: AutoMLJobConfig = accuracy_job(
            settings,
            training_data=training_data,
            compute=compute_target,
            allowed_algorithms=allowed_algorithms,
        )
    elif metric_l == "recall":
        config = recall_job(
            settings,
            training_data=training_data,
            compute=compute_target,
            allowed_algorithms=allowed_algorithms,
        )
    else:
        raise typer.BadParameter("Metric must be either 'accuracy' or 'recall'.")

    job = create_automl_job(config)
    ml_client = get_ml_client(settings=settings)
    job_name = submit_job(ml_client, job)
    typer.echo(f"Job submitted successfully: {job_name}")


@app.command()
def register_best_automl_model(
    experiment: Annotated[
        list[str] | None,
        typer.Option("--experiment", "-e", help="Experiment(s) to process. Can be passed multiple times."),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model-name",
            help=(
                "Optional stable model name to use instead of slugified experiment names. "
                "Useful for CD pipelines."
            ),
        ),
    ] = None,
):
    """Register the best child run for each completed AutoML parent job (versioned per experiment)."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    experiments = experiment or DEFAULT_AUTOML_EXPERIMENTS
    typer.echo(f"Processing experiments: {', '.join(experiments)}")

    completed = list_completed_jobs(ml_client, experiments)
    if not completed:
        typer.echo("No completed jobs found for the given experiments.")
        raise typer.Exit(code=0)

    for job_name, experiment_name in completed.items():
        model = register_best_child_run(
            ml_client,
            job_name=job_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )
        if isinstance(model, Model):
            typer.echo(f"Registered model '{model.name}' version '{model.version}' from job '{job_name}'.")
        else:
            typer.echo(f"Skipped registration for job '{job_name}'.")

@app.command()
def endpoint_create(
    name: str = typer.Option(..., "--name", help="Endpoint name"),
    description: str = typer.Option(None, "--description", help="Endpoint description"),
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
    typer.echo(f"Created/updated endpoint: {ep.name}")


@app.command()
def endpoint_delete(
    name: str = typer.Option(..., "--name", help="Endpoint name to delete"),
):
    """Delete a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    delete_endpoint(ml_client, endpoint_name=name)
    typer.echo(f"Deleted endpoint: {name}")


@app.command()
def endpoint_traffic(
    endpoint: Annotated[str, typer.Option("--endpoint", help="Endpoint name")],
    traffic: Annotated[
        list[str],
        typer.Option(
            "--traffic", help="Traffic mapping like blue=100 green=0 (repeatable)"
        ),
    ],
):
    """Update endpoint traffic split. Traffic must sum to 100."""
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
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid percent '{pct}' for deployment '{dep}'."
            ) from exc

    update_traffic(ml_client, endpoint_name=endpoint, traffic=mapping)
    typer.echo(f"Updated traffic for endpoint '{endpoint}': {mapping}")


@app.command()
def deploy_model_version(
    endpoint: str = typer.Option(..., "--endpoint", help="Endpoint name"),
    deployment: str = typer.Option("blue", "--deployment", help="Deployment slot name"),
    model_name: str = typer.Option(..., "--model-name", help="Registered model name"),
    model_version: str = typer.Option(..., "--model-version", help="Registered model version"),
    instance_type: str = typer.Option("Standard_E4s_v3", "--instance-type", help="VM SKU"),
    instance_count: int = typer.Option(1, "--instance-count", help="Instance count"),
    traffic_100: bool = typer.Option(True, "--traffic-100/--no-traffic-100", help="Send 100% traffic to this deployment"),
):
    """Deploy a specific model version to a managed online endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    # Create endpoint if missing (common “one-command” UX)
    from fraud_detection.serving.online_endpoint import ensure_endpoint
    ensure_endpoint(ml_client, endpoint_name=endpoint, tags={"project": "fraud-detection"})

    dep = deploy_model(
        ml_client,
        endpoint_name=endpoint,
        deployment_name=deployment,
        model_name=model_name,
        model_version=model_version,
        instance_type=instance_type,
        instance_count=instance_count,
        tags={"project": "fraud-detection"},
    )
    if traffic_100:
        update_traffic(ml_client, endpoint_name=endpoint, traffic={dep.name: 100})

    typer.echo(f"Deployed {model_name}:{model_version} to {endpoint}/{dep.name}")


@app.command()
@serve_app.command("deploy")
def deploy(
    endpoint: Annotated[str, typer.Option("--endpoint", help="Endpoint name")],
    model_name: Annotated[str, typer.Option("--model-name", help="Azure ML registered model name")],
    deployment: Annotated[str, typer.Option("--deployment", help="Deployment slot name")] = "blue",
    strategy: Annotated[
        str, typer.Option("--strategy", help="Selection strategy: 'latest' or 'best'", case_sensitive=False)
    ] = "latest",
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            help="Metric for --strategy best (e.g. norm_macro_recall or metrics.norm_macro_recall)",
        ),
    ] = None,
    direction: Annotated[str, typer.Option("--direction", help="'max' or 'min'", case_sensitive=False)] = "max",
    experiment_prefix: Annotated[
        list[str] | None,
        typer.Option(
            "--experiment-prefix",
            help="Experiment name prefix to search (repeatable). Example: --experiment-prefix automl-fraud",
        ),
    ] = None,
    artifact_path: Annotated[
        list[str] | None,
        typer.Option(
            "--artifact-path",
            help=(
                "Artifact path candidate(s) under azureml://jobs/<run_id>/... "
                "Repeat to provide fallbacks. If omitted, defaults are tried."
            ),
        ),
    ] = None,
    filter_string: Annotated[str, typer.Option("--filter", help="MLflow filter_string for run search")] = "",
    view: Annotated[str, typer.Option("--view", help="active|all", case_sensitive=False)] = "active",
    max_results: Annotated[int, typer.Option("--max-results", help="Max runs to consider", min=1)] = 5000,
    instance_type: Annotated[str, typer.Option("--instance-type", help="VM SKU")] = "Standard_E4s_v3",
    instance_count: Annotated[int, typer.Option("--instance-count", help="Instance count", min=1)] = 1,
    traffic_100: Annotated[
        bool, typer.Option("--traffic-100/--no-traffic-100", help="Send 100% traffic to this deployment")
    ] = True,
):
    """Deploy the latest or best-performing model to a managed online endpoint."""

    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    strategy_l = strategy.lower()
    prefixes = list(experiment_prefix) if experiment_prefix else ["automl-fraud"]

    if strategy_l == "latest":
        version, dep = deploy_latest_model(
            ml_client,
            model_name=model_name,
            endpoint_name=endpoint,
            deployment_name=deployment,
            instance_type=instance_type,
            instance_count=instance_count,
            set_traffic_100=traffic_100,
        )
        typer.echo(f"Deployed latest {model_name}:{version} to {endpoint}/{dep}")
        return

    if strategy_l == "best":
        if not metric:
            raise typer.BadParameter("--metric is required when --strategy best")

        view_type = ViewType.ACTIVE_ONLY if view.lower() == "active" else ViewType.ALL

        version, dep, best = deploy_best_by_metric(
            ml_client,
            model_name=model_name,
            endpoint_name=endpoint,
            deployment_name=deployment,
            experiment_prefixes=prefixes,
            metric=metric,
            direction=direction.lower(),
            artifact_paths=list(artifact_path) if artifact_path else None,
            filter_string=filter_string,
            view_type=view_type,
            max_results=max_results,
            instance_type=instance_type,
            instance_count=instance_count,
            traffic_100=traffic_100,
        )
        typer.echo(
            f"Deployed BEST {model_name}:{version} to {endpoint}/{dep} "
            f"(metric={best.metric_name} value={best.metric_value:.6f} run_id={best.run_id})"
        )
        return

    raise typer.BadParameter("--strategy must be 'latest' or 'best'")


@app.command()
def deploy_latest(
    endpoint: str = typer.Option(..., "--endpoint", help="Endpoint name"),
    deployment: str = typer.Option("blue", "--deployment", help="Deployment slot name"),
    model_name: str = typer.Option(..., "--model-name", help="Registered model name"),
    instance_type: str = typer.Option("Standard_E4s_v3", "--instance-type", help="VM SKU"),
    instance_count: int = typer.Option(1, "--instance-count", help="Instance count"),
    traffic_100: bool = typer.Option(True, "--traffic-100/--no-traffic-100", help="Send 100% traffic to this deployment"),
):
    """Deploy the latest version of a registered model to an endpoint."""
    settings = get_settings().require_azure()
    ml_client = get_ml_client(settings=settings)

    version, dep_name = deploy_latest_model(
        ml_client,
        model_name=model_name,
        endpoint_name=endpoint,
        deployment_name=deployment,
        instance_type=instance_type,
        instance_count=instance_count,
        create_endpoint_if_missing=True,
        set_traffic_100=traffic_100,
    )
    typer.echo(f"Deployed latest {model_name}:{version} to {endpoint}/{dep_name}")


def run():
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
