# src/fraud_detection/serving/deploy.py
from __future__ import annotations

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from mlflow.entities import ViewType

from fraud_detection.registry.best_runs_by_metric import BestRun, best_run_by_metric
from fraud_detection.registry.register_from_run import (
    DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES,
    register_model_from_run,
)
from fraud_detection.serving.online_endpoint import deploy_model, ensure_endpoint, update_traffic
from fraud_detection.training.automl import EXPERIMENT_ACCURACY, EXPERIMENT_RECALL
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INSTANCE_TYPE = "Standard_E4s_v3"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_AUTOML_EXPERIMENTS = [EXPERIMENT_RECALL, EXPERIMENT_ACCURACY]

# Optional: if your ensure_endpoint supports auth_mode, we pass it.
try:
    from fraud_detection.serving.online_endpoint import DEFAULT_AUTH_MODE  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_AUTH_MODE = None


# Fallback imports: depending on how you renamed/moved registry modules.
try:
    # Original style (job_name -> experiment_name)
    from fraud_detection.registry.automl_model_registry import (  # type: ignore
        list_completed_jobs as _list_completed_jobs_flat,
        register_best_child_run,
    )
except Exception:  # pragma: no cover
    # Newer style might have list_completed_jobs_flat
    from fraud_detection.registry.automl_registry import (  # type: ignore
        list_completed_jobs_flat as _list_completed_jobs_flat,
        register_best_child_run,
    )


def get_latest_model_version(ml_client: MLClient, *, model_name: str) -> str:
    """Return the newest version for the given registered model name."""
    models = list(ml_client.models.list(name=model_name))
    if not models:
        raise ResourceNotFoundError(message=f"Model '{model_name}' does not exist in the workspace.")

    def _key(m) -> tuple[int, str]:
        v = str(m.version)
        return (int(v), v) if v.isdigit() else (0, v)

    latest = sorted(models, key=_key, reverse=True)[0]
    return str(latest.version)


def _ensure_endpoint_or_fail(ml_client: MLClient, *, endpoint_name: str, create_if_missing: bool) -> None:
    if create_if_missing:
        kwargs = {"endpoint_name": endpoint_name, "tags": {"project": "fraud-detection"}}
        if DEFAULT_AUTH_MODE is not None:
            kwargs["auth_mode"] = DEFAULT_AUTH_MODE
        ensure_endpoint(ml_client, **kwargs)
    else:
        # fail fast if endpoint doesn't exist
        ml_client.online_endpoints.get(endpoint_name)


def deploy_latest_model(
    ml_client: MLClient,
    *,
    model_name: str,
    endpoint_name: str,
    deployment_name: str = "blue",
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    create_endpoint_if_missing: bool = True,
    set_traffic_100: bool = True,
    backfill_if_missing: bool = True,
    backfill_experiments: list[str] | None = None,
) -> tuple[str, str]:
    """Deploy latest model version; optionally backfill by registering from completed AutoML jobs.

    Returns:
        (deployed_version, deployment_name)
    """
    _ensure_endpoint_or_fail(ml_client, endpoint_name=endpoint_name, create_if_missing=create_endpoint_if_missing)

    try:
        version = get_latest_model_version(ml_client, model_name=model_name)
    except ResourceNotFoundError:
        if not backfill_if_missing:
            raise

        experiments = backfill_experiments or DEFAULT_AUTOML_EXPERIMENTS
        logger.warning(
            "Model not found; attempting to register best AutoML run before deploy",
            extra={"model_name": model_name, "experiments": experiments},
        )

        jobs = _list_completed_jobs_flat(ml_client, experiments)

        # Try jobs in a deterministic order (reverse sort on job_name is a decent heuristic if names include timestamps)
        for job_name, experiment_name in sorted(jobs.items(), reverse=True):
            registered = register_best_child_run(
                ml_client,
                job_name=job_name,
                experiment_name=experiment_name,
                model_name=model_name,
                skip_on_missing_best_child=True,
                skip_on_model_error=True,
            )
            if registered:
                version = str(registered.version)
                logger.info(
                    "Registered model from AutoML run as fallback",
                    extra={
                        "job_name": job_name,
                        "experiment": experiment_name,
                        "model_name": registered.name,
                        "model_version": version,
                    },
                )
                break
        else:
            raise ResourceNotFoundError(
                message=(
                    "No registered models found and no completed AutoML jobs available "
                    f"to backfill model '{model_name}'."
                )
            )

    deployment = deploy_model(
        ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=version,
        instance_type=instance_type,
        instance_count=instance_count,
        tags={"project": "fraud-detection", "selection": "latest", "model_name": model_name, "model_version": version},
    )

    if set_traffic_100:
        update_traffic(ml_client, endpoint_name=endpoint_name, traffic={deployment.name: 100})

    logger.info(
        "Deployed latest model",
        extra={
            "model_name": model_name,
            "version": version,
            "endpoint": endpoint_name,
            "deployment": deployment.name,
            "traffic_100": set_traffic_100,
        },
    )
    return version, deployment.name


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
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    traffic_100: bool = True,
) -> tuple[str, str, BestRun]:
    """Pick best MLflow run by metric, register it as new model version, then deploy it.

    Returns:
        (model_version, deployment_name, best_run)
    """
    _ensure_endpoint_or_fail(ml_client, endpoint_name=endpoint_name, create_if_missing=True)

    best = best_run_by_metric(
        ml_client,
        experiment_prefixes=experiment_prefixes,
        metric=metric,
        direction=direction,
        filter_string=filter_string,
        view_type=view_type,
    )

    candidates = artifact_paths or list(DEFAULT_MLFLOW_MODEL_ARTIFACT_CANDIDATES)

    registered = register_model_from_run(
        ml_client,
        model_name=model_name,
        run_id=best.run_id,
        artifact_paths=candidates,
        description=f"Selected by best metric '{best.metric_name}' ({direction}) from MLflow",
        tags={
            "selected_by": "mlflow_best_metric",
            "selected_metric": best.metric_name,
            "selected_direction": direction,
            "selected_metric_value": str(best.metric_value),
            "selected_experiment": best.experiment_name,
            "selected_run_id": best.run_id,
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


__all__ = [
    "DEFAULT_INSTANCE_TYPE",
    "DEFAULT_INSTANCE_COUNT",
    "DEFAULT_AUTOML_EXPERIMENTS",
    "get_latest_model_version",
    "deploy_latest_model",
    "deploy_best_by_metric",
]
