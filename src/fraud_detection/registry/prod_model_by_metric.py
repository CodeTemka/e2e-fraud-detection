"""Register a production model from AutoML experiment based on a metric comparison."""
from __future__ import annotations

import mlflow
from azure.ai.ml import MLClient
from azure.core.exceptions import HttpResponseError
from mlflow.tracking import MlflowClient

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import Settings, build_idempotency_key, get_git_sha, get_settings
from fraud_detection.registry.automl_registry import mlflow_tracking_uri, register_model_from_run
from fraud_detection.registry.best_runs_by_metric import BestRun, best_run_by_metric
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _shadow_model_name(run_id: str) -> str:
    """Name for non-production registrations (keeps prod model name stable)."""
    return f"model_{run_id}"


def _metric_direction(metric: str) -> str:
    metric_lower = (metric or "").lower()
    if any(key in metric_lower for key in ("loss", "error", "rmse", "mae")):
        return "min"
    return "max"


def _is_better(best: float, current: float, *, direction: str, epsilon: float, delta: float) -> bool:
    required_delta = max(float(epsilon), float(delta))
    if direction == "min":
        return best < (current - required_delta)
    return best > (current + required_delta)


def _meets_min_threshold(metric_value: float, *, direction: str, min_threshold: float | None) -> bool:
    if min_threshold is None:
        return True
    if direction == "min":
        return metric_value <= min_threshold
    return metric_value >= min_threshold


def _parse_version(version: str | None) -> int:
    try:
        return int(version or 0)
    except (TypeError, ValueError):
        return 0


def _latest_model_asset(ml_client: MLClient, name: str):
    """Return latest AzureML model asset (highest numeric version) or None."""
    try:
        models = list(ml_client.models.list(name=name))
    except HttpResponseError as exc:
        logger.debug("Failed listing models for name=%s: %s", name, exc)
        return None

    if not models:
        return None

    return max(models, key=lambda m: _parse_version(getattr(m, "version", None)))


def _get_run_metric(run_id: str, metric: str) -> float:
    """Read a metric from an MLflow run by run_id."""
    run = mlflow.get_run(run_id)
    if metric not in run.data.metrics:
        raise KeyError(f"Metric '{metric}' not found in run {run_id}.")
    return float(run.data.metrics[metric])


def _get_run_tag(run_id: str, tag_name: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed reading run tag", extra={"run_id": run_id, "tag": tag_name, "error": str(exc)})
        return None
    return run.data.tags.get(tag_name)


def _dataset_version_for_run(run_id: str) -> str:
    return _get_run_tag(run_id, "dataset_version") or "unknown"


def current_prod_model_metric(
    ml_client: MLClient,
    metric: str,
    *,
    settings: Settings | None = None,
) -> float | None:
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
    experiment: str,
) -> str | None:
    """
    Compare best AutoML run vs current production model.

    Rules:
    - If no prod model exists: promote best to prod_model_name.
    - If best metric > prod metric: promote best to prod_model_name.
    - If best metric < prod metric: register best under a shadow model name.
    - If best metric == prod metric: DO NOTHING (no registration at all).

    Returns:
      - The model name used for registration, OR
      - None if no registration was performed (equal metrics case).
    """
    cfg = settings or get_settings()
    ml_client = ml_client or get_ml_client(settings=cfg)

    mlflow_tracking_uri(ml_client)

    # Best run in the specified experiment for the metric
    best: BestRun = best_run_by_metric(metric=metric, settings=cfg, experiment_name=experiment)
    best_run_id = best.run_id

    # Current prod metric (if any)
    prod_metric = current_prod_model_metric(ml_client, metric, settings=cfg)

    direction = _metric_direction(metric)
    epsilon = cfg.promotion_metric_epsilon
    min_threshold = cfg.promotion_min_metric
    delta_threshold = cfg.promotion_metric_delta

    if prod_metric is None:
        should_promote = _meets_min_threshold(best.metric_value, direction=direction, min_threshold=min_threshold)
    else:
        should_promote = _meets_min_threshold(
            best.metric_value, direction=direction, min_threshold=min_threshold
        ) and _is_better(
            best.metric_value,
            prod_metric,
            direction=direction,
            epsilon=epsilon,
            delta=delta_threshold,
        )
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

    stage = "production" if should_promote else "staging"
    alias = "production" if should_promote else "staging"
    idempotency_key = build_idempotency_key(cfg.automl_train_exp, metric)
    tags = {
        "experiment_name": cfg.automl_train_exp,
        "selected_metric": metric,
        "metric_name": metric,
        "metric_value": str(best.metric_value),
        "best_run_id": best_run_id,
        "run_id": best_run_id,
        "git_sha": get_git_sha(short=False),
        "idempotency_key": idempotency_key,
        "promotion": str(should_promote).lower(),
        "promotion_decision": "promote" if should_promote else "stage",
        "model_source": "automl",
        "dataset_version": _dataset_version_for_run(best_run_id),
        "stage": stage,
        "alias": alias,
    }

    register_model_from_run(
        ml_client,
        model_name=chosen_name,
        best_child_run=best_run_id,
        description="AutoML classification model registered by metric comparison",
        tags=tags,
    )
    return chosen_name


__all__ = ["register_prod_model"]
