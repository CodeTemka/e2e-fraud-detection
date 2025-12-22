from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlflow
from azure.ai.ml import MLClient
from mlflow.entities import Experiment, ViewType

from fraud_detection.registry.automl_registry import mlflow_tracking_uri
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

AUTOML_RUN_LOGS = ['run_id', 'experiment_id', 'status', 'artifact_uri', 'start_time',
       'end_time', 'metrics.weighted_accuracy', 'metrics.AUC_micro',
       'metrics.precision_score_weighted', 'metrics.f1_score_macro',
       'metrics.norm_macro_recall', 'metrics.average_precision_score_weighted',
       'metrics.AUC_macro', 'metrics.f1_score_micro',
       'metrics.recall_score_macro', 'metrics.recall_score_micro',
       'metrics.precision_score_micro', 'metrics.AUC_weighted',
       'metrics.log_loss', 'metrics.average_precision_score_macro',
       'metrics.f1_score_weighted', 'metrics.precision_score_macro',
       'metrics.average_precision_score_micro',
       'metrics.recall_score_weighted', 'metrics.matthews_correlation',
       'metrics.accuracy', 'metrics.balanced_accuracy',
       'tags.automl_best_child_run_id', 'tags.model_explain_best_run_child_id',
       'tags.project', 'tags.run_preprocessor_000', 'tags.predicted_cost_000',
       'tags.mlflow.rootRunId', 'tags.model_explain_run', 'tags.mlflow.user',
       'tags.pipeline_id_000', 'tags.training_percent_000', 'tags.score_000',
       'tags.metric', 'tags.mlflow.runName', 'tags.fit_time_000',
       'tags.dynamic_allowlisting_iterations', 'tags.run_algorithm_000',
       'tags.iteration_000', 'tags.mlflow.source.type',
       'tags.mlflow.source.name', 'tags.mlflow.parentRunId',
       'tags.model_explain_run_id', 'tags.model_explanation', 'tags.model_rai',
       'tags.model']

AUTOML_DEFAULT_METRICS = ['metrics.weighted_accuracy', 'metrics.AUC_micro',
       'metrics.precision_score_weighted', 'metrics.f1_score_macro',
       'metrics.norm_macro_recall', 'metrics.average_precision_score_weighted',
       'metrics.AUC_macro', 'metrics.f1_score_micro',
       'metrics.recall_score_macro', 'metrics.recall_score_micro',
       'metrics.precision_score_micro', 'metrics.AUC_weighted',
       'metrics.log_loss', 'metrics.average_precision_score_macro',
       'metrics.f1_score_weighted', 'metrics.precision_score_macro',
       'metrics.average_precision_score_micro',
       'metrics.recall_score_weighted', 'metrics.matthews_correlation',
       'metrics.accuracy', 'metrics.balanced_accuracy',]

_AUTOML_DEFAULT_METRICS_SET = set(AUTOML_DEFAULT_METRICS)
_AUTOML_DEFAULT_METRICS_BARE = sorted(
    {m[len("metrics.") :] for m in AUTOML_DEFAULT_METRICS if m.startswith("metrics.")}
)


@dataclass(frozen=True)
class BestRun:
    experiment_name: str
    run_id: str
    metric_name: str
    metric_value: float


def _normalize_metric_to_col(m: str) -> str:
    m = (m or "").strip()
    if not m:
        raise ValueError("Metric name cannot be empty.")

    col = m if m.startswith("metrics.") else f"metrics.{m}"
    if col not in _AUTOML_DEFAULT_METRICS_SET:
        allowed = ", ".join(_AUTOML_DEFAULT_METRICS_BARE)
        raise ValueError(f"Metric '{m}' is not a recognized AutoML metric. Choose one of: {allowed}")
    return col


def _metric_col(metric: str | Sequence[str]) -> str | list[str]:
    """Return MLflow search_runs dataframe metric column(s)."""
    if isinstance(metric, str):
        return _normalize_metric_to_col(metric)

    cols = [_normalize_metric_to_col(m) for m in metric]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)

    if not out:
        raise ValueError("metric list must be non-empty.")
    return out


def experiments_by_prefix(
    ml_client: MLClient,
    *,
    prefixes: Sequence[str] | str,
) -> list[Experiment]:
    """Return MLflow experiments whose names start with any given prefix."""
    mlflow_tracking_uri(ml_client)

    prefixes_clean = [p for p in (prefixes or []) if p]
    if not prefixes_clean:
        raise ValueError("prefixes must be a non-empty list")

    exps = mlflow.search_experiments()
    matched = [e for e in exps if any(e.name.startswith(p) for p in prefixes_clean)]
    logger.info("Matched experiments", extra={"prefixes": prefixes_clean, "count": len(matched)})
    return matched


def best_run_by_metric(
    ml_client: MLClient,
    *,
    experiment_prefixes: Sequence[str],
    metric: str | Sequence[str],
    direction: str = "max",  # "max" or "min"
    filter_string: str = "",
    view_type: ViewType = ViewType.ACTIVE_ONLY,
) -> BestRun:
    """Find the best run (by a metric) across experiments whose names match prefixes."""
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")

    exps = experiments_by_prefix(ml_client, prefixes=experiment_prefixes)
    if not exps:
        raise RuntimeError(f"No experiments found for prefixes: {list(experiment_prefixes)}")

    exp_ids = [e.experiment_id for e in exps]
    col = _metric_col(metric)

    ascending = True if direction == "min" else False
    cols = [col] if isinstance(col, str) else list(col)

    # MLflow returns a DataFrame
    df = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string or "",
        run_view_type=view_type,
    )

    if df is None or df.empty:
        raise RuntimeError("No runs found (empty MLflow search result).")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Metric column(s) not found in MLflow results: {missing}")

    # Drop runs missing any of the requested metric columns, then sort
    df = df.dropna(subset=cols).sort_values(by=cols, ascending=ascending)

    if df.empty:
        raise RuntimeError(f"No runs contained a non-null value for metric(s): {cols}")

    best = df.iloc[0]
    run_id = str(best["run_id"])
    exp_name = str(best.get("experiment_name") or "")

    primary_col = cols[0]
    value = float(best[primary_col])

    metric_name = metric if isinstance(metric, str) else metric[0]
    logger.info(
        "Selected best run",
        extra={
            "run_id": run_id,
            "experiment_name": exp_name,
            "metric": str(metric_name),
            "value": value,
            "direction": direction,
            "sorted_by": cols,
        },
    )
    return BestRun(
        experiment_name=exp_name,
        run_id=run_id,
        metric_name=str(metric_name),
        metric_value=value,
    )


__all__ = ["BestRun", "best_run_by_metric", "experiments_by_prefix", "AUTOML_RUN_LOGS", "AUTOML_DEFAULT_METRICS"]
