from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mlflow
import pandas as pd
from mlflow.entities import Experiment, ViewType

from azure.ai.ml import MLClient
from fraud_detection.mlflow.tracking import set_tracking_uri_from_workspace
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BestRun:
    experiment_name: str
    run_id: str
    metric_name: str
    metric_value: float


def _metric_col(metric: str) -> str:
    """mlflow.search_runs dataframe metric column name."""
    m = metric.strip()
    return m if m.startswith("metrics.") else f"metrics.{m}"


def _normalize_metric_key_for_mlflow_client(metric: str) -> str:
    """MlflowClient.get_run().data.metrics uses bare keys (no 'metrics.' prefix)."""
    m = metric.strip()
    return m[len("metrics.") :] if m.startswith("metrics.") else m


def experiments_by_prefix(
    ml_client: MLClient,
    *,
    prefixes: Sequence[str],
) -> list[Experiment]:
    """Return MLflow experiments whose names start with any given prefix."""
    set_tracking_uri_from_workspace(ml_client)

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
    metric: str,
    direction: str = "max",  # "max" or "min"
    filter_string: str = "",
    view_type: ViewType = ViewType.ACTIVE_ONLY,
    max_results: int = 5000,
) -> BestRun:
    """Find the best run (by a metric) across experiments whose names match prefixes."""
    if direction not in ("max", "min"):
        raise ValueError("direction must be 'max' or 'min'")

    exps = experiments_by_prefix(ml_client, prefixes=experiment_prefixes)
    if not exps:
        raise RuntimeError(f"No experiments found for prefixes: {list(experiment_prefixes)}")

    exp_ids = [e.experiment_id for e in exps]
    col = _metric_col(metric)
    order = "DESC" if direction == "max" else "ASC"

    # MLflow returns a DataFrame
    df = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string or "",
        run_view_type=view_type,
        max_results=max_results,
        order_by=[f"{col} {order}"],
    )

    if df is None or df.empty:
        raise RuntimeError("No runs found (empty MLflow search result).")

    if col not in df.columns:
        # If metric column doesn't exist at all, the user likely passed a wrong metric name.
        raise RuntimeError(f"Metric '{metric}' not found in MLflow results. Looked for column '{col}'.")

    df = df[df[col].notna()]
    if df.empty:
        raise RuntimeError(f"No runs contained a non-null value for metric '{metric}'.")

    best = df.iloc[0]
    run_id = str(best["run_id"])
    exp_name = str(best.get("experiment_name") or "")

    value = float(best[col])
    logger.info(
        "Selected best run",
        extra={"run_id": run_id, "experiment_name": exp_name, "metric": metric, "value": value, "direction": direction},
    )
    return BestRun(experiment_name=exp_name, run_id=run_id, metric_name=metric, metric_value=value)


__all__ = ["BestRun", "best_run_by_metric", "experiments_by_prefix", "ViewType"]
