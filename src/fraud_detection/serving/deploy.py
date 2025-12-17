from __future__ import annotations

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.mlflow.best_runs import BestRun, ViewType, best_run_by_metric
from fraud_detection.registry.register_from_run import register_model_from_run
from fraud_detection.serving.online_endpoint import ensure_endpoint, deploy_model, update_traffic
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INSTANCE_TYPE = "Standard_E4s_v3"
DEFAULT_INSTANCE_COUNT = 1


def get_latest_model_version(ml_client: MLClient, *, model_name: str) -> str:
    models = list(ml_client.models.list(name=model_name))
    if not models:
        raise ResourceNotFoundError(message=f"Model '{model_name}' does not exist in the workspace.")

    def _key(m) -> tuple[int, str]:
        v = str(m.version)
        return (int(v), v) if v.isdigit() else (0, v)

    latest = sorted(models, key=_key, reverse=True)[0]
    return str(latest.version)


def deploy_latest_model(
    ml_client: MLClient,
    *,
    model_name: str,
    endpoint_name: str,
    deployment_name: str = "blue",
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    traffic_100: bool = True,
) -> tuple[str, str]:
    ensure_endpoint(ml_client, endpoint_name=endpoint_name, tags={"project": "fraud-detection"})
    version = get_latest_model_version(ml_client, model_name=model_name)

    dep = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=version,
        instance_type=instance_type,
        instance_count=instance_count,
        tags={"project": "fraud-detection", "selection": "latest"},
    )
    if traffic_100:
        update_traffic(ml_client, endpoint_name=endpoint_name, traffic={dep.name: 100})

    logger.info("Deployed latest model", extra={"model_name": model_name, "version": version, "endpoint": endpoint_name, "deployment": dep.name})
    return version, dep.name


def deploy_best_by_metric(
    ml_client: MLClient,
    *,
    model_name: str,
    endpoint_name: str,
    deployment_name: str,
    experiment_prefixes: list[str],
    metric: str,
    direction: str = "max",
    artifact_paths: list[str] | None = None,
    filter_string: str = "",
    view_type: ViewType = ViewType.ACTIVE_ONLY,
    max_results: int = 5000,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    traffic_100: bool = True,
) -> tuple[str, str, BestRun]:
    """
    Pick best MLflow run by metric, register it as a new model version, then deploy it.

    Returns:
      (model_version, deployment_name, best_run)
    """
    ensure_endpoint(ml_client, endpoint_name=endpoint_name, tags={"project": "fraud-detection"})

    best = best_run_by_metric(
        ml_client,
        experiment_prefixes=experiment_prefixes,
        metric=metric,
        direction=direction,
        filter_string=filter_string,
        view_type=view_type,
        max_results=max_results,
    )

    # Register model version from best run/job artifacts
    registered = register_model_from_run(
        ml_client,
        model_name=model_name,
        run_id=best.run_id,
        artifact_paths=artifact_paths,
        description=f"Selected by best metric '{best.metric_name}' ({direction}) from MLflow",
        tags={
            "selected_by": "mlflow_best_metric",
            "selected_metric": best.metric_name,
            "selected_direction": direction,
            "selected_metric_value": str(best.metric_value),
            "selected_experiment": best.experiment_name,
        },
    )

    version = str(registered.version)

    dep = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=version,
        instance_type=instance_type,
        instance_count=instance_count,
        tags={"project": "fraud-detection", "selection": "best", "metric": best.metric_name},
    )
    if traffic_100:
        update_traffic(ml_client, endpoint_name=endpoint_name, traffic={dep.name: 100})

    logger.info(
        "Deployed best-by-metric model",
        extra={
            "model_name": model_name,
            "version": version,
            "endpoint": endpoint_name,
            "deployment": dep.name,
            "metric": best.metric_name,
            "value": best.metric_value,
            "run_id": best.run_id,
        },
    )
    return version, dep.name, best
