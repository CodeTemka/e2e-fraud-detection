from __future__ import annotations

from dataclasses import dataclass

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import Settings, get_settings
from fraud_detection.registry.automl_registry import job_artifact_uri, mlflow_tracking_uri
from fraud_detection.registry.best_runs_by_metric import BestRun, best_run_by_metric
from fraud_detection.registry.prod_model_by_metric import current_prod_model_metric
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

CUSTOM_MODEL_ARTIFACT_PATH = "model"


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    model_name: str | None
    model_version: str | None
    best_run: BestRun
    best_metric: float
    prod_metric: float | None


def _metric_direction(metric: str) -> str:
    metric_lower = (metric or "").lower()
    if any(key in metric_lower for key in ("loss", "error", "rmse", "mae")):
        return "min"
    return "max"


def _is_better(
    best: float,
    current: float,
    *,
    direction: str,
    epsilon: float,
    delta: float = 0.0,
) -> bool:
    required_delta = max(float(epsilon), float(delta))
    if direction == "min":
        return best < (current - required_delta)
    return best > (current + required_delta)


def _parse_version(v: str | None) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _latest_model_asset(ml_client: MLClient, name: str) -> Model | None:
    try:
        models = list(ml_client.models.list(name=name))
    except HttpResponseError as exc:
        logger.debug("Failed listing models for name=%s: %s", name, exc)
        return None
    if not models:
        return None
    return max(models, key=lambda m: _parse_version(getattr(m, "version", None)))


def _current_prod_metric_from_tags(ml_client: MLClient, metric: str, *, settings: Settings) -> float | None:
    latest = _latest_model_asset(ml_client, settings.prod_model_name)
    if latest is None:
        return None
    tags = getattr(latest, "tags", {}) or {}
    tag_metric = tags.get("metric_name")
    tag_value = tags.get("metric_value")
    if tag_metric == metric and tag_value is not None:
        try:
            return float(tag_value)
        except (TypeError, ValueError):
            return None
    return None


def _shadow_model_name(run_id: str) -> str:
    """Name for non-production registrations (keeps prod model name stable)."""
    return f"model_{run_id}"


def _get_run_tag(run_id: str, tag_name: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed reading run tag", extra={"run_id": run_id, "tag": tag_name, "error": str(exc)})
        return None
    return run.data.tags.get(tag_name)


def _dataset_version_for_run(run_id: str) -> str:
    return _get_run_tag(run_id, "dataset_version") or "unknown"


def _meets_min_threshold(metric_value: float, *, direction: str, min_threshold: float | None) -> bool:
    if min_threshold is None:
        return True
    if direction == "min":
        return metric_value <= min_threshold
    return metric_value >= min_threshold


def _resolve_prod_metric(ml_client: MLClient, metric: str, *, settings: Settings) -> float | None:
    from_tags = _current_prod_metric_from_tags(ml_client, metric, settings=settings)
    if from_tags is not None:
        return from_tags
    return current_prod_model_metric(ml_client, metric, settings=settings)


def register_custom_model_from_run(
    ml_client: MLClient,
    *,
    model_name: str,
    run_id: str,
    metric: str,
    metric_value: float,
    settings: Settings,
    promotion: bool = True,
    stage: str = "production",
    alias: str | None = "production",
    extra_tags: dict[str, str] | None = None,
) -> Model:
    tags = {
        "experiment_name": settings.custom_train_exp,
        "selected_metric": metric,
        "metric_name": metric,
        "metric_value": str(metric_value),
        "run_id": run_id,
        "promotion": str(promotion).lower(),
        "promotion_decision": "promote" if promotion else "stage",
        "model_source": "custom",
        "stage": stage,
        "dataset_version": _dataset_version_for_run(run_id),
    }
    if alias:
        tags["alias"] = alias
    if extra_tags:
        tags.update(extra_tags)
    model = Model(
        path=job_artifact_uri(run_id, CUSTOM_MODEL_ARTIFACT_PATH),
        name=model_name,
        type=AssetTypes.URI_FOLDER,
        description="Custom trained fraud detection model",
        tags=tags,
    )
    return ml_client.models.create_or_update(model)


def register_best_custom_model(
    metric: str,
    *,
    ml_client: MLClient | None = None,
    settings: Settings | None = None,
    experiment: str | None = None,
) -> PromotionResult:
    cfg = settings or get_settings()
    ml_client = ml_client or get_ml_client(settings=cfg)
    experiment_name = experiment or cfg.custom_train_exp
    if not experiment_name:
        raise ValueError("custom_train experiment name is required.")

    mlflow_tracking_uri(ml_client)

    direction = _metric_direction(metric)
    best = best_run_by_metric(metric=metric, direction=direction, settings=cfg, experiment_name=experiment_name)
    prod_metric = _resolve_prod_metric(ml_client, metric, settings=cfg)

    epsilon = getattr(cfg, "promotion_metric_epsilon", 0.0)
    min_threshold = getattr(cfg, "promotion_min_metric", None)
    delta_threshold = getattr(cfg, "promotion_metric_delta", 0.0)
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

    logger.info(
        "Custom model promotion decision",
        extra={
            "metric": metric,
            "direction": direction,
            "best_metric": best.metric_value,
            "prod_metric": prod_metric,
            "promote": should_promote,
            "epsilon": epsilon,
            "min_threshold": min_threshold,
            "delta_threshold": delta_threshold,
            "run_id": best.run_id,
        },
    )

    if not should_promote:
        return PromotionResult(
            promoted=False,
            model_name=None,
            model_version=None,
            best_run=best,
            best_metric=best.metric_value,
            prod_metric=prod_metric,
        )

    registered = register_custom_model_from_run(
        ml_client,
        model_name=cfg.prod_model_name,
        run_id=best.run_id,
        metric=metric,
        metric_value=best.metric_value,
        settings=cfg,
        promotion=True,
        stage="production",
        alias="production",
    )

    return PromotionResult(
        promoted=True,
        model_name=registered.name,
        model_version=str(registered.version),
        best_run=best,
        best_metric=best.metric_value,
        prod_metric=prod_metric,
    )


__all__ = [
    "CUSTOM_MODEL_ARTIFACT_PATH",
    "PromotionResult",
    "register_best_custom_model",
    "_metric_direction",
    "_is_better",
]
