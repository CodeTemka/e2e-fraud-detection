from __future__ import annotations

"""Find the best AutoML runs and their metrics."""

from typing import Dict, Iterable, Tuple

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

from deploy.find_best_models_by_run.common import (
    BestRun,
    collect_runs_dataframe,
    filter_experiments_by_prefix,
    find_best_runs,
)

logger = get_logger(__name__)

DEFAULT_METRIC_PREFIXES: Tuple[str, ...] = (
    "metrics.AUC",
    "metrics.recall",
    "metrics.average",
    "metrics.f1",
    "metrics.norm",
)


def get_best_auto_ml_runs(
    *,
    ml_client=None,
    experiment_prefix: str = "auto",
    metric_prefixes: Iterable[str] = DEFAULT_METRIC_PREFIXES,
) -> Dict[str, BestRun]:
    """Return the best AutoML run per metric for experiments starting with *experiment_prefix*."""

    client = ml_client or get_ml_client()
    experiments = filter_experiments_by_prefix(experiment_prefix)
    runs_df = collect_runs_dataframe(client, experiments)

    return find_best_runs(runs_df, metric_prefixes)


if __name__ == "__main__":
    best_runs = get_best_auto_ml_runs()

    if not best_runs:
        logger.info("No AutoML best runs found.")
    else:
        logger.info("Best AutoML runs:")
        for metric, run in best_runs.items():
            logger.info("%s -> %s (%.4f)", metric, run.run_id, run.value)
