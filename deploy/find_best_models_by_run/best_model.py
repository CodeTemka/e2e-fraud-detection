from __future__ import annotations

"""Utilities for fetching and loading the best AutoML and custom training runs."""

from typing import Dict, Iterable, Mapping, Sequence

import mlflow
from mlflow.exceptions import MlflowException

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

from deploy.find_best_models_by_run.best_run_auto_ml import get_best_auto_ml_runs
from deploy.find_best_models_by_run.best_run_custom_train import get_best_custom_train_runs
from deploy.find_best_models_by_run.common import BestRun

logger = get_logger(__name__)


def collect_best_runs(ml_client=None) -> Mapping[str, Dict[str, BestRun]]:
    """Return best runs for AutoML and custom training experiments."""

    client = ml_client or get_ml_client()
    return {
        "auto_ml": get_best_auto_ml_runs(ml_client=client),
        "custom_train": get_best_custom_train_runs(ml_client=client),
    }


def unique_run_ids(best_runs: Iterable[Dict[str, BestRun]]) -> Sequence[str]:
    """Flatten multiple best-run mappings into a unique list of run IDs."""

    run_ids = {run.run_id for mapping in best_runs for run in mapping.values()}
    return sorted(run_ids)


def load_models_for_runs(run_ids: Sequence[str], artifact_path: str = "model") -> Dict[str, mlflow.pyfunc.PyFuncModel]:
    """Load MLflow models for the provided *run_ids*.

    Returns a mapping of run ID to the loaded model, skipping runs that cannot be fetched.
    """

    models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}
    for run_id in run_ids:
        try:
            models[run_id] = mlflow.pyfunc.load_model(f"runs:/{run_id}/{artifact_path}")
        except MlflowException as exc:  # pragma: no cover - depends on external runs
            logger.warning("Could not load model for run %s", run_id, exc_info=exc)
    return models


def summarize_best_runs(best_runs: Mapping[str, Dict[str, BestRun]]) -> None:
    """Log a concise summary of all best runs."""

    for category, runs in best_runs.items():
        if not runs:
            logger.info("No best runs found for %s experiments.", category)
            continue

        logger.info("Best runs for %s experiments:", category)
        for metric, run in runs.items():
            logger.info("%s -> %s (%.4f)", metric, run.run_id, run.value)


if __name__ == "__main__":
    best_runs = collect_best_runs()
    summarize_best_runs(best_runs)

    selected_run_ids = unique_run_ids(best_runs.values())
    if not selected_run_ids:
        logger.info("No run IDs available to load models.")
    else:
        models = load_models_for_runs(selected_run_ids)
        logger.info("Loaded %d models from best runs.", len(models))
