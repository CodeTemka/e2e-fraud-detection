from __future__ import annotations

"""Locate the best custom training runs."""

from typing import Dict, Iterable, Tuple

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.logging import get_logger

from deploy.find_best_models_by_run.common import (
    BestRun,
    collect_runs_dataframe,
    filter_experiments_by_prefix,
    find_best_runs,
)

logger = get_logger(__name__)

DEFAULT_METRIC_PREFIXES: Tuple[str, ...] = ("metrics.valid_",)


def get_best_custom_train_runs(
    *,
    ml_client=None,
    experiment_prefix: str = "custom",
    metric_prefixes: Iterable[str] = DEFAULT_METRIC_PREFIXES,
) -> Dict[str, BestRun]:
    """Return the best custom training run per validation metric."""

    client = ml_client or get_ml_client()
    experiments = filter_experiments_by_prefix(experiment_prefix)
    runs_df = collect_runs_dataframe(client, experiments)

    return find_best_runs(runs_df, metric_prefixes)


if __name__ == "__main__":
    best_runs = get_best_custom_train_runs()

    if not best_runs:
        logger.info("No custom training best runs found.")
    else:
        logger.info("Best custom training runs:")
        for metric, run in best_runs.items():
            logger.info("%s -> %s (%.4f)", metric, run.run_id, run.value)
