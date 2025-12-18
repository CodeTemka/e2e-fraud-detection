"""Higher-level deployment flows (e.g., deploy latest model version)."""
from __future__ import annotations

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.serving.online_endpoint import (
    DEFAULT_AUTH_MODE,
    deploy_model,
    ensure_endpoint,
    update_traffic,
)
from fraud_detection.registry.automl_model_registry import (
    list_completed_jobs,
    register_best_child_run,
)
from fraud_detection.training.automl import EXPERIMENT_ACCURACY, EXPERIMENT_RECALL
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INSTANCE_TYPE = "Standard_E4s_v3"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_AUTOML_EXPERIMENTS = [EXPERIMENT_RECALL, EXPERIMENT_ACCURACY]


def get_latest_model_version(ml_client: MLClient, *, model_name: str) -> str:
    """Return the newest version for the given registered model name."""
    models = list(ml_client.models.list(name=model_name))
    if not models:
        raise ResourceNotFoundError(message=f"Model '{model_name}' not found")

    def _key(m) -> tuple[int, str]:
        v = str(m.version)
        return (int(v), v) if v.isdigit() else (0, v)

    latest = sorted(models, key=_key, reverse=True)[0]
    logger.info("Selected latest model version", extra={"model_name": model_name, "version": latest.version})
    return str(latest.version)


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
) -> tuple[str, str]:
    """Deploy latest model version to endpoint; optionally send 100% traffic to deployment.

    Returns:
        (deployed_version, deployment_name)
    """
    if create_endpoint_if_missing:
        ensure_endpoint(
            ml_client,
            endpoint_name=endpoint_name,
            auth_mode=DEFAULT_AUTH_MODE,
            tags={"project": "fraud-detection"},
        )
    else:
        # fail fast if endpoint doesn't exist
        ml_client.online_endpoints.get(endpoint_name)

    try:
        version = get_latest_model_version(ml_client, model_name=model_name)
    except ResourceNotFoundError:
        logger.warning(
            "Model not found; attempting to register best AutoML run before deploy",
            extra={"model_name": model_name, "experiments": DEFAULT_AUTOML_EXPERIMENTS},
        )

        jobs = list_completed_jobs(ml_client, DEFAULT_AUTOML_EXPERIMENTS)
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
        tags={"project": "fraud-detection", "model_name": model_name, "model_version": version},
    )

    if set_traffic_100:
        update_traffic(ml_client, endpoint_name=endpoint_name, traffic={deployment.name: 100})

    logger.info(
        "Deployment successful",
        extra={
            "endpoint_name": endpoint_name,
            "deployment_name": deployment.name,
            "model_name": model_name,
            "model_version": version,
            "traffic_100": set_traffic_100,
        },
    )
    return version, deployment.name
