from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlflow
from mlflow.entities import ViewType

from fraud_detection.config import Settings, get_settings
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


AUTOML_DEFAULT_METRICS_BARE = sorted(
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
    if col not in AUTOML_DEFAULT_METRICS:
        allowed = ", ".join(AUTOML_DEFAULT_METRICS_BARE)
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


def best_run_by_metric(
    *,
    metric: str | Sequence[str],
    direction: str = "max",  # "max" or "min"
    view_type: ViewType = ViewType.ACTIVE_ONLY,
    status: str | None = None,
    experiment_name: str | None = None,
    settings: Settings | None = None,
) -> BestRun:
    """Find the best run (by a metric) within an AutoML experiment."""
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")

    resolved_experiment = experiment_name or (settings or get_settings()).automl_train_exp
    if not resolved_experiment:
        raise ValueError("experiment_name is required when no default is configured.")
    col = _metric_col(metric)

    ascending = True if direction == "min" else False
    cols = [col] if isinstance(col, str) else list(col)

    # MLflow returns a DataFrame
    df = mlflow.search_runs(
        experiment_names=[resolved_experiment],
        run_view_type=view_type,
    )

    if df is None or df.empty:
        raise RuntimeError("No runs found (empty MLflow search result).")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Metric column(s) not found in MLflow results: {missing}")

    if status:
        if "status" not in df.columns:
            raise RuntimeError("MLflow search results missing status column; cannot filter by run status.")
        df = df[df["status"] == status]

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


__all__ = [
    "BestRun",
    "best_run_by_metric",
    "AUTOML_RUN_LOGS",
    "AUTOML_DEFAULT_METRICS",
    "AUTOML_DEFAULT_METRICS_BARE"
]
