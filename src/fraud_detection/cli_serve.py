from __future__ import annotations

from typing import Annotated

import typer
from mlflow.entities import ViewType

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.serving.deploy import deploy_best_by_metric, deploy_latest_model

serve_app = typer.Typer(help="Serving operations (managed online endpoints).")


@serve_app.command("deploy")
def deploy(
    endpoint: Annotated[str, typer.Option("--endpoint", help="Endpoint name")],
    model_name: Annotated[str, typer.Option("--model-name", help="Azure ML registered model name")],
    deployment: Annotated[str, typer.Option("--deployment", help="Deployment slot name")] = "blue",
    strategy: Annotated[
        str, typer.Option("--strategy", help="Selection strategy: 'latest' or 'best'", case_sensitive=False)
    ] = "latest",
    # best-strategy options
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
    # deployment options
    instance_type: Annotated[str, typer.Option("--instance-type", help="VM SKU")] = "Standard_E4s_v3",
    instance_count: Annotated[int, typer.Option("--instance-count", help="Instance count", min=1)] = 1,
    traffic_100: Annotated[
        bool, typer.Option("--traffic-100/--no-traffic-100", help="Send 100% traffic to this deployment")
    ] = True,
):
    """Deploy either the latest registered model version OR the best-by-metric run (register+deploy)."""
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
            traffic_100=traffic_100,
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
