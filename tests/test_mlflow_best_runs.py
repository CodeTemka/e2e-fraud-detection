from types import SimpleNamespace

import pandas as pd
import pytest
from mlflow.entities import Experiment

from fraud_detection.registry import best_runs_by_metric as best_runs


@pytest.mark.parametrize(
    "metric,expected",
    [("AUC_micro", "metrics.AUC_micro"), ("metrics.AUC_macro", "metrics.AUC_macro")],
)
def test_metric_col(metric, expected):
    assert best_runs._metric_col(metric) == expected


def test_metric_col_sequence_deduplicates_and_validates():
    assert best_runs._metric_col(["AUC_macro", "metrics.AUC_macro", "AUC_macro"]) == [
        "metrics.AUC_macro"
    ]


def test_normalize_metric_to_col_rejects_unknown():
    with pytest.raises(ValueError):
        best_runs._metric_col("not-a-metric")


def test_experiments_by_prefix_filters_and_sets_tracking(monkeypatch):
    recorded_prefixes = {}

    def fake_set_tracking(ml_client):
        recorded_prefixes["called"] = True
        return "uri"

    experiments = [Experiment("id1", "exp-alpha", "", ""), Experiment("id2", "beta", "", "")]

    monkeypatch.setattr(best_runs, "mlflow_tracking_uri", fake_set_tracking)
    monkeypatch.setattr(best_runs.mlflow, "search_experiments", lambda: experiments)

    workspace = SimpleNamespace(mlflow_tracking_uri="uri")
    ml_client = SimpleNamespace(workspace_name="ws", workspaces=SimpleNamespace(get=lambda name: workspace))
    result = best_runs.experiments_by_prefix(ml_client, prefixes=["exp", "other"])

    assert recorded_prefixes["called"] is True
    assert [e.name for e in result] == ["exp-alpha"]


def test_experiments_by_prefix_requires_prefixes(monkeypatch):
    monkeypatch.setattr(best_runs.mlflow, "search_experiments", lambda: [])

    workspace = SimpleNamespace(mlflow_tracking_uri="uri")
    ml_client = SimpleNamespace(workspace_name="ws", workspaces=SimpleNamespace(get=lambda name: workspace))

    with pytest.raises(ValueError):
        best_runs.experiments_by_prefix(ml_client, prefixes=[])


def test_best_run_by_metric_selects_top(monkeypatch):
    experiments = [Experiment("1", "exp-one", "", "")]
    df = pd.DataFrame(
        {
            "run_id": ["r2", "r1"],
            "experiment_name": ["exp-one", "exp-one"],
            "metrics.AUC_micro": [0.9, 0.7],
        }
    )

    monkeypatch.setattr(best_runs, "experiments_by_prefix", lambda _ml, prefixes: experiments)
    monkeypatch.setattr(best_runs.mlflow, "search_runs", lambda **_: df)

    ml_client = SimpleNamespace()
    result = best_runs.best_run_by_metric(
        ml_client, experiment_prefixes=["exp"], metric="AUC_micro", direction="max"
    )

    assert result.run_id == "r2"
    assert result.metric_value == 0.9
    assert result.experiment_name == "exp-one"


def test_best_run_by_metric_validates_inputs(monkeypatch):
    experiments = [Experiment("1", "exp-one", "", "")]
    empty_df = pd.DataFrame({"run_id": [], "experiment_name": []})

    monkeypatch.setattr(best_runs, "experiments_by_prefix", lambda _ml, prefixes: experiments)
    monkeypatch.setattr(best_runs.mlflow, "search_runs", lambda **_: empty_df)

    with pytest.raises(RuntimeError):
        best_runs.best_run_by_metric(
            SimpleNamespace(),
            experiment_prefixes=["exp"],
            metric="AUC_micro",
            direction="min",
        )

    with pytest.raises(ValueError):
        best_runs.best_run_by_metric(
            SimpleNamespace(),
            experiment_prefixes=["exp"],
            metric="AUC_micro",
            direction="middle",
        )
