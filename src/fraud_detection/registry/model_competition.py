from __future__ import annotations

from dataclasses import dataclass

import mlflow
from azure.ai.ml import MLClient
from azure.core.exceptions import HttpResponseError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import build_idempotency_key, get_git_sha, Settings, get_settings
from fraud_detection.registry.automl_registry import mlflow_tracking_uri, register_model_from_run
from fraud_detection.registry.best_runs_by_metric import BestRun, best_run_by_metric
from fraud_detection.registry.custom_model_registry import (
    _is_better,
    _metric_direction,
    register_custom_model_from_run,
)
from fraud_detection.registry.prod_model_by_metric import current_prod_model_metric
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Candidate:
    source: str
    best_run: BestRun
    experiment_name: str


@dataclass(frozen=True)
class SelectionResult:
    promoted: bool
    model_name: str | None
    model_version: str | None
    model_source: str | None
    metric: str
    automl_metric: float | None
    custom_metric: float | None
    prod_metric: float | None
    winner: str | None


def _parse_version(v: str | None) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _latest_model_asset(ml_client: MLClient, name: str):
    try:
        models = list(ml_client.models.list(name=name))
    except HttpResponseError as exc:
        logger.debug("Failed listing models for name=%s: %s", name, exc)
        return None
    if not models:
        return None
    return max(models, key=lambda m: _parse_version(getattr(m, "version", None)))


def _tag_matches_metric(tag_value: str | None, metric_value: float) -> bool:
    if tag_value is None:
        return False
    try:
        return float(tag_value) == float(metric_value)
    except (TypeError, ValueError):
        return tag_value == str(metric_value)


def _find_existing_model(
    ml_client: MLClient,
    *,
    model_name: str,
    run_id: str,
    metric: str,
    metric_value: float,
):
    try:
        models = list(ml_client.models.list(name=model_name))
    except HttpResponseError as exc:
        logger.debug("Failed listing models for name=%s: %s", model_name, exc)
        return None
    for model in models:
        tags = getattr(model, "tags", {}) or {}
        if tags.get("run_id") != run_id:
            continue
        if tags.get("metric_name") != metric:
            continue
        if _tag_matches_metric(tags.get("metric_value"), metric_value):
            return model
    return None


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


def _best_completed_run(
    *,
    metric: str,
    direction: str,
    experiment_name: str,
    settings: Settings,
) -> BestRun:
    return best_run_by_metric(
        metric=metric,
        direction=direction,
        settings=settings,
        experiment_name=experiment_name,
        status="FINISHED",
    )


def _try_best_run(
    *,
    metric: str,
    direction: str,
    experiment_name: str,
    settings: Settings,
    source: str,
) -> Candidate | None:
    try:
        best = _best_completed_run(
            metric=metric,
            direction=direction,
            experiment_name=experiment_name,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed selecting best %s run", source, extra={"error": str(exc), "experiment": experiment_name}
        )
        return None
    return Candidate(source=source, best_run=best, experiment_name=experiment_name)


def _select_best_candidate(candidates: list[Candidate], *, direction: str) -> Candidate:
    if direction == "min":
        return min(candidates, key=lambda c: c.best_run.metric_value)
    return max(candidates, key=lambda c: c.best_run.metric_value)


def select_and_register_prod_model(
    *,
    metric: str,
    ml_client: MLClient | None = None,
    settings: Settings | None = None,
    automl_experiment: str | None = None,
    custom_experiment: str | None = None,
) -> SelectionResult:
    cfg = settings or get_settings()
    ml_client = ml_client or get_ml_client(settings=cfg)

    resolved_automl_exp = automl_experiment or cfg.automl_train_exp
    resolved_custom_exp = custom_experiment or cfg.custom_train_exp
    if not resolved_automl_exp or not resolved_custom_exp:
        raise ValueError("AutoML and custom experiment names are required for selection.")

    mlflow_tracking_uri(ml_client)

    direction = _metric_direction(metric)
    epsilon = cfg.promotion_metric_epsilon
    min_threshold = cfg.promotion_min_metric
    delta_threshold = cfg.promotion_metric_delta

    automl_candidate = _try_best_run(
        metric=metric,
        direction=direction,
        experiment_name=resolved_automl_exp,
        settings=cfg,
        source="automl",
    )
    custom_candidate = _try_best_run(
        metric=metric,
        direction=direction,
        experiment_name=resolved_custom_exp,
        settings=cfg,
        source="custom",
    )

    prod_metric = _resolve_prod_metric(ml_client, metric, settings=cfg)

    automl_metric = automl_candidate.best_run.metric_value if automl_candidate else None
    custom_metric = custom_candidate.best_run.metric_value if custom_candidate else None

    logger.info(
        "Model competition metrics",
        extra={
            "metric": metric,
            "direction": direction,
            "automl_metric": automl_metric,
            "custom_metric": custom_metric,
            "prod_metric": prod_metric,
            "epsilon": epsilon,
            "min_threshold": min_threshold,
            "delta_threshold": delta_threshold,
        },
    )

    contenders: list[Candidate] = []
    if automl_candidate and _meets_min_threshold(
        automl_metric,
        direction=direction,
        min_threshold=min_threshold,
    ) and (prod_metric is None or _is_better(
        automl_metric,
        prod_metric,
        direction=direction,
        epsilon=epsilon,
        delta=delta_threshold,
    )):
        contenders.append(automl_candidate)
    if custom_candidate and _meets_min_threshold(
        custom_metric,
        direction=direction,
        min_threshold=min_threshold,
    ) and (prod_metric is None or _is_better(
        custom_metric,
        prod_metric,
        direction=direction,
        epsilon=epsilon,
        delta=delta_threshold,
    )):
        contenders.append(custom_candidate)

    if not contenders:
        logger.info("Model competition decision: no change", extra={"metric": metric})
        return SelectionResult(
            promoted=False,
            model_name=None,
            model_version=None,
            model_source=None,
            metric=metric,
            automl_metric=automl_metric,
            custom_metric=custom_metric,
            prod_metric=prod_metric,
            winner=None,
        )

    winner = _select_best_candidate(contenders, direction=direction)

    idempotency_key = build_idempotency_key(
        winner.experiment_name,
        metric,
        git_sha=get_git_sha(short=True),
    )
    base_tags = {
        "selected_metric": metric,
        "metric_name": metric,
        "metric_value": str(winner.best_run.metric_value),
        "run_id": winner.best_run.run_id,
        "promotion": "true",
        "promotion_decision": "promote",
        "model_source": winner.source,
        "stage": "production",
        "alias": "production",
        "experiment_name": winner.experiment_name,
        "idempotency_key": idempotency_key,
        "dataset_version": _dataset_version_for_run(winner.best_run.run_id),
    }

    existing_model = _find_existing_model(
        ml_client,
        model_name=cfg.prod_model_name,
        run_id=winner.best_run.run_id,
        metric=metric,
        metric_value=winner.best_run.metric_value,
    )
    if existing_model:
        logger.info(
            "Model already registered for run/metric; reusing existing version",
            extra={
                "model_name": existing_model.name,
                "model_version": existing_model.version,
                "run_id": winner.best_run.run_id,
                "metric": metric,
            },
        )
        return SelectionResult(
            promoted=True,
            model_name=existing_model.name,
            model_version=str(existing_model.version),
            model_source=winner.source,
            metric=metric,
            automl_metric=automl_metric,
            custom_metric=custom_metric,
            prod_metric=prod_metric,
            winner=winner.source,
        )

    if winner.source == "automl":
        registered = register_model_from_run(
            ml_client,
            model_name=cfg.prod_model_name,
            best_child_run=winner.best_run.run_id,
            description="AutoML classification model promoted by competition",
            tags=base_tags,
        )
    else:
        registered = register_custom_model_from_run(
            ml_client,
            model_name=cfg.prod_model_name,
            run_id=winner.best_run.run_id,
            metric=metric,
            metric_value=winner.best_run.metric_value,
            settings=cfg,
            extra_tags=base_tags,
        )

    logger.info(
        "Model competition decision: promote",
        extra={
            "metric": metric,
            "winner": winner.source,
            "run_id": winner.best_run.run_id,
            "metric_value": winner.best_run.metric_value,
            "model_name": registered.name,
            "model_version": registered.version,
        },
    )

    return SelectionResult(
        promoted=True,
        model_name=registered.name,
        model_version=str(registered.version),
        model_source=winner.source,
        metric=metric,
        automl_metric=automl_metric,
        custom_metric=custom_metric,
        prod_metric=prod_metric,
        winner=winner.source,
    )


__all__ = ["Candidate", "SelectionResult", "select_and_register_prod_model"]
