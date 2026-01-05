from types import SimpleNamespace

from fraud_detection.registry import model_competition
from fraud_detection.registry.best_runs_by_metric import BestRun


def test_select_and_register_promotes_custom(monkeypatch):
    captured = {}

    class FakeSettings:
        automl_train_exp = "automl-exp"
        custom_train_exp = "custom-exp"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name, status=None):
        if experiment_name == "automl-exp":
            return BestRun(experiment_name=experiment_name, run_id="auto-1", metric_name=metric, metric_value=0.82)
        return BestRun(experiment_name=experiment_name, run_id="custom-1", metric_name=metric, metric_value=0.91)

    def fake_resolve_prod_metric(*_args, **_kwargs):
        return 0.88

    def fake_register_custom_model_from_run(*_args, **kwargs):
        captured["custom"] = kwargs
        return SimpleNamespace(name="production-model", version="9")

    def fake_register_model_from_run(*_args, **_kwargs):
        raise AssertionError("AutoML registration should not be called")

    monkeypatch.setattr(model_competition, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(model_competition, "_resolve_prod_metric", fake_resolve_prod_metric)
    monkeypatch.setattr(model_competition, "register_custom_model_from_run", fake_register_custom_model_from_run)
    monkeypatch.setattr(model_competition, "register_model_from_run", fake_register_model_from_run)
    monkeypatch.setattr(model_competition, "mlflow_tracking_uri", lambda *_: None)

    result = model_competition.select_and_register_prod_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
    )

    assert result.promoted is True
    assert result.model_source == "custom"
    assert result.model_version == "9"
    assert captured["custom"]["metric"] == "AUC_macro"


def test_select_and_register_skips_when_no_candidate_beats_prod(monkeypatch):
    class FakeSettings:
        automl_train_exp = "automl-exp"
        custom_train_exp = "custom-exp"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name, status=None):
        if experiment_name == "automl-exp":
            return BestRun(experiment_name=experiment_name, run_id="auto-1", metric_name=metric, metric_value=0.7)
        return BestRun(experiment_name=experiment_name, run_id="custom-1", metric_name=metric, metric_value=0.6)

    monkeypatch.setattr(model_competition, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(model_competition, "_resolve_prod_metric", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(model_competition, "mlflow_tracking_uri", lambda *_: None)

    result = model_competition.select_and_register_prod_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
    )

    assert result.promoted is False
    assert result.model_name is None
    assert result.winner is None


def test_select_and_register_prefers_automl_when_best(monkeypatch):
    captured = {}

    class FakeSettings:
        automl_train_exp = "automl-exp"
        custom_train_exp = "custom-exp"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name, status=None):
        if experiment_name == "automl-exp":
            return BestRun(experiment_name=experiment_name, run_id="auto-1", metric_name=metric, metric_value=0.93)
        return BestRun(experiment_name=experiment_name, run_id="custom-1", metric_name=metric, metric_value=0.91)

    def fake_register_model_from_run(*_args, **kwargs):
        captured["automl"] = kwargs
        return SimpleNamespace(name="production-model", version="10")

    monkeypatch.setattr(model_competition, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(model_competition, "_resolve_prod_metric", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(model_competition, "register_model_from_run", fake_register_model_from_run)
    monkeypatch.setattr(model_competition, "register_custom_model_from_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_competition, "mlflow_tracking_uri", lambda *_: None)

    result = model_competition.select_and_register_prod_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
    )

    assert result.promoted is True
    assert result.model_source == "automl"
    assert result.model_version == "10"
    assert captured["automl"]["best_child_run"] == "auto-1"
