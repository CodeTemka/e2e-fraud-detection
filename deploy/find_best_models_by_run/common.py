from __future__ import annotations

"""Shared helpers for selecting the best MLflow runs."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import mlflow
import pandas as pd
from mlflow.entities import Experiment, ViewType

# Ensure project imports (e.g., fraud_detection.*) resolve when files are executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BestRun:
    """Container describing the best run for a metric."""

    metric: str
    run_id: str
    value: float


def set_tracking_uri(ml_client) -> bool:
    """Configure MLflow to use the workspace associated with *ml_client*.

    Returns ``True`` when the URI is configured successfully and ``False`` otherwise.
    """

    try:
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    except Exception as exc:  # pragma: no cover - defensive around azure SDK behaviour
        logger.warning("Unable to resolve MLflow tracking URI from ml_client", exc_info=exc)
        return False

    mlflow.set_tracking_uri(tracking_uri)
    return True


def filter_experiments_by_prefix(prefix: str) -> List[Experiment]:
    """Return experiments whose names start with *prefix*."""

    experiments = [exp for exp in mlflow.search_experiments() if exp.name.startswith(prefix)]
    if not experiments:
        logger.info("No experiments found with prefix '%s'", prefix)
    return experiments


def collect_runs_dataframe(ml_client, experiments: Sequence[Experiment]) -> pd.DataFrame:
    """Collect all runs for *experiments* into a single DataFrame."""

    if not set_tracking_uri(ml_client):
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for experiment in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            run_view_type=ViewType.ALL,
        )
        frames.append(pd.DataFrame(runs))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def find_best_runs(df: pd.DataFrame, metric_prefixes: Iterable[str]) -> Dict[str, BestRun]:
    """Identify the best run for each metric column starting with *metric_prefixes*."""

    if df.empty:
        logger.info("No runs available to evaluate best metrics.")
        return {}

    metric_columns = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in metric_prefixes)
    ]

    best_runs: Dict[str, BestRun] = {}
    for metric in metric_columns:
        series = df[metric]
        if not pd.api.types.is_numeric_dtype(series) or series.isnull().all():
            continue

        best_row = df.loc[series.idxmax()]
        best_runs[metric] = BestRun(
            metric=metric,
            run_id=str(best_row["run_id"]),
            value=float(best_row[metric]),
        )

    return best_runs


__all__ = [
    "BestRun",
    "set_tracking_uri",
    "collect_runs_dataframe",
    "filter_experiments_by_prefix",
    "find_best_runs",
]
