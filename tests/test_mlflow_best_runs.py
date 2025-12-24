import pandas as pd
import pytest

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


def test_best_run_by_metric_selects_top(monkeypatch):
    df = pd.DataFrame(
        {
            "run_id": ["r2", "r1"],
            "experiment_name": ["exp-one", "exp-one"],
            "metrics.AUC_micro": [0.9, 0.7],
        }
    )

    monkeypatch.setattr(best_runs.mlflow, "search_runs", lambda **_: df)

    class FakeSettings:
        automl_train_exp = "exp-one"

    result = best_runs.best_run_by_metric(metric="AUC_micro", direction="max", settings=FakeSettings())

    assert result.run_id == "r2"
    assert result.metric_value == 0.9
    assert result.experiment_name == "exp-one"


def test_best_run_by_metric_validates_inputs(monkeypatch):
    empty_df = pd.DataFrame({"run_id": [], "experiment_name": []})

    monkeypatch.setattr(best_runs.mlflow, "search_runs", lambda **_: empty_df)

    with pytest.raises(RuntimeError):
        best_runs.best_run_by_metric(metric="AUC_micro", direction="min", experiment_name="exp")

    with pytest.raises(ValueError):
        best_runs.best_run_by_metric(metric="AUC_micro", direction="middle", experiment_name="exp")
