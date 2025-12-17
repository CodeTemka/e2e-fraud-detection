from types import SimpleNamespace

import pandas as pd
import pytest
from mlflow.entities import Experiment

from fraud_detection.mlflow import best_runs


@pytest.mark.parametrize(
    "metric,expected",
    [("auc", "metrics.auc"), ("metrics.recall", "metrics.recall")],
)
def test_metric_col(metric, expected):
    assert best_runs._metric_col(metric) == expected


@pytest.mark.parametrize(
    "metric,expected",
    [("auc", "auc"), ("metrics.f1", "f1")],
)
def test_normalize_metric_key(metric, expected):
    assert best_runs._normalize_metric_key_for_mlflow_client(metric) == expected


def test_experiments_by_prefix_filters_and_sets_tracking(monkeypatch):
    recorded_prefixes = {}

    def fake_set_tracking(ml_client):
        recorded_prefixes["called"] = True
        return "uri"

    experiments = [Experiment("id1", "exp-alpha", "", ""), Experiment("id2", "beta", "", "")]

    monkeypatch.setattr(best_runs, "set_tracking_uri_from_workspace", fake_set_tracking)
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
            "metrics.auc": [0.9, 0.7],
        }
    )

    monkeypatch.setattr(best_runs, "experiments_by_prefix", lambda _ml, prefixes: experiments)
    monkeypatch.setattr(best_runs.mlflow, "search_runs", lambda **_: df)

    ml_client = SimpleNamespace()
    result = best_runs.best_run_by_metric(
        ml_client, experiment_prefixes=["exp"], metric="auc", direction="max"
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
            metric="auc",
            direction="min",
        )

    with pytest.raises(ValueError):
        best_runs.best_run_by_metric(
            SimpleNamespace(), experiment_prefixes=["exp"], metric="auc", direction="middle"
        )
