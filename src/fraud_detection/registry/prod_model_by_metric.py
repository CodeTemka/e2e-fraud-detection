"""Register a production model from AutoML experiment based on a metric comparison."""
from __future__ import annotations

from typing import Optional

import mlflow
from azure.ai.ml import MLClient
from mlflow.tracking import MlflowClient

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import Settings, get_settings
from fraud_detection.registry.automl_registry import mlflow_tracking_uri, register_model_from_run
from fraud_detection.registry.best_runs_by_metric import BestRun, best_run_by_metric
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _shadow_model_name(run_id: str) -> str:
    """Name for non-production registrations (keeps prod model name stable)."""
    return f"model_{run_id}"


def _latest_model_asset(ml_client: MLClient, name: str):
    """Return latest AzureML model asset (highest numeric version) or None."""
    try:
        models = list(ml_client.models.list(name=name))
    except Exception as e:
        logger.debug("Failed listing models for name=%s: %s", name, e)
        return None

    if not models:
        return None

    return max(models, key=lambda m: int(m.version))


def _get_run_metric(run_id: str, metric: str) -> float:
    """Read a metric from an MLflow run by run_id."""
    run = mlflow.get_run(run_id)
    if metric not in run.data.metrics:
        raise KeyError(f"Metric '{metric}' not found in run {run_id}.")
    return float(run.data.metrics[metric])


def current_prod_model_metric(
    ml_client: MLClient,
    metric: str,
    *,
    settings: Settings | None = None,
) -> Optional[float]:
    """
    Return the metric value for the latest registered production model version.

    Returns None if no production model exists or the model version doesn't have a run_id.
    """
    cfg = settings or get_settings()
    prod_model_name = cfg.prod_model_name

    latest = _latest_model_asset(ml_client, prod_model_name)
    if latest is None:
        logger.info("No production model registered yet (name=%s).", prod_model_name)
        return None

    # Ensure MLflow points at the Azure ML workspace tracking/registry
    mlflow_tracking_uri(ml_client)

    client = MlflowClient()
    mv = client.get_model_version(name=latest.name, version=str(latest.version))
    run_id = getattr(mv, "run_id", None)

    if not run_id:
        logger.warning(
            "Latest production model has no run_id (name=%s version=%s). Cannot fetch metrics.",
            latest.name,
            latest.version,
        )
        return None

    try:
        return _get_run_metric(run_id, metric)
    except KeyError as e:
        logger.warning("%s", e)
        return None


def register_prod_model(
    metric: str,
    *,
    ml_client: MLClient | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Compare best AutoML run vs current production model.
    If best AutoML metric is better, register it as the new production model name.
    Otherwise register it under a shadow model name (so prod stays unchanged).

    Returns the model name used for registration.
    """
    cfg = settings or get_settings()
    ml_client = ml_client or get_ml_client(settings=cfg.require_azure())

    # Best run in the AutoML experiment for the metric
    best: BestRun = best_run_by_metric(metric=metric)
    best_run_id = best.run_id

    # Current prod metric (if any)
    prod_metric = current_prod_model_metric(ml_client, metric, settings=cfg)

    should_promote = (prod_metric is None) or (best.metric_value > prod_metric)
    chosen_name = cfg.prod_model_name if should_promote else _shadow_model_name(best_run_id)

    logger.info(
        "Model promotion decision: promote=%s metric=%s best=%s current_prod=%s model_name=%s run_id=%s",
        should_promote,
        metric,
        best.metric_value,
        prod_metric,
        chosen_name,
        best_run_id,
    )

    register_model_from_run(ml_client, chosen_name, best_run_id)
    return chosen_name


__all__ = ["register_prod_model"]