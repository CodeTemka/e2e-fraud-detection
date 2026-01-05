from __future__ import annotations

from dataclasses import dataclass

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import Settings, get_settings
from fraud_detection.registry.automl_registry import mlflow_tracking_uri, _job_artifact_uri
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


def _is_better(best: float, current: float, *, direction: str, epsilon: float) -> bool:
    if direction == "min":
        return best < (current - epsilon)
    return best > (current + epsilon)


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
    extra_tags: dict[str, str] | None = None,
) -> Model:
    tags = {
        "experiment_name": settings.custom_train_exp,
        "selected_metric": metric,
        "metric_name": metric,
        "metric_value": str(metric_value),
        "run_id": run_id,
        "promotion": "true",
        "model_source": "custom",
        "stage": "production",
        "alias": "production",
    }
    if extra_tags:
        tags.update(extra_tags)
    model = Model(
        path=_job_artifact_uri(run_id, CUSTOM_MODEL_ARTIFACT_PATH),
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

    epsilon = cfg.promotion_metric_epsilon
    if prod_metric is None:
        should_promote = True
    else:
        should_promote = _is_better(best.metric_value, prod_metric, direction=direction, epsilon=epsilon)

    logger.info(
        "Custom model promotion decision",
        extra={
            "metric": metric,
            "direction": direction,
            "best_metric": best.metric_value,
            "prod_metric": prod_metric,
            "promote": should_promote,
            "epsilon": epsilon,
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
