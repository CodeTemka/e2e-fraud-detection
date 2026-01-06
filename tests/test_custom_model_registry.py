from types import SimpleNamespace

from fraud_detection.registry import custom_model_registry as custom_registry
from fraud_detection.registry.best_runs_by_metric import BestRun


def test_metric_direction_and_compare():
    assert custom_registry._metric_direction("log_loss") == "min"
    assert custom_registry._metric_direction("AUC_macro") == "max"

    assert custom_registry._is_better(0.9, 0.8, direction="max", epsilon=0.0)
    assert not custom_registry._is_better(0.81, 0.8, direction="max", epsilon=0.02)
    assert custom_registry._is_better(0.7, 0.8, direction="min", epsilon=0.0)
    assert not custom_registry._is_better(0.79, 0.8, direction="min", epsilon=0.02)


def test_register_best_custom_model_selects_best_run(monkeypatch):
    captured = {}

    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name):
        captured["experiment"] = experiment_name
        return BestRun(experiment_name=experiment_name, run_id="run-123", metric_name=metric, metric_value=0.9)

    def fake_resolve_prod_metric(*_args, **_kwargs):
        return 0.5

    def fake_register_custom_model_from_run(*_args, **_kwargs):
        return SimpleNamespace(name="production-model", version="7")

    class FakeSettings:
        custom_train_exp = "custom-train"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    monkeypatch.setattr(custom_registry, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(custom_registry, "_resolve_prod_metric", fake_resolve_prod_metric)
    monkeypatch.setattr(custom_registry, "register_custom_model_from_run", fake_register_custom_model_from_run)
    monkeypatch.setattr(custom_registry, "mlflow_tracking_uri", lambda *_: None)

    result = custom_registry.register_best_custom_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
        experiment="custom-train",
    )

    assert result.promoted is True
    assert result.model_name == "production-model"
    assert result.model_version == "7"
    assert captured["experiment"] == "custom-train"


def test_register_best_custom_model_skips_when_not_better(monkeypatch):
    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name):
        return BestRun(experiment_name=experiment_name, run_id="run-999", metric_name=metric, metric_value=0.5)

    def fake_resolve_prod_metric(*_args, **_kwargs):
        return 0.6

    class FakeSettings:
        custom_train_exp = "custom-train"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    monkeypatch.setattr(custom_registry, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(custom_registry, "_resolve_prod_metric", fake_resolve_prod_metric)
    monkeypatch.setattr(custom_registry, "mlflow_tracking_uri", lambda *_: None)

    result = custom_registry.register_best_custom_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
        experiment="custom-train",
    )

    assert result.promoted is False
    assert result.model_name is None


def test_register_best_custom_model_promotes_when_no_prod_metric(monkeypatch):
    def fake_best_run_by_metric(*, metric, direction, settings, experiment_name):
        return BestRun(experiment_name=experiment_name, run_id="run-555", metric_name=metric, metric_value=0.42)

    def fake_register_custom_model_from_run(*_args, **_kwargs):
        return SimpleNamespace(name="production-model", version="8")

    class FakeSettings:
        custom_train_exp = "custom-train"
        prod_model_name = "production-model"
        promotion_metric_epsilon = 0.0

    monkeypatch.setattr(custom_registry, "best_run_by_metric", fake_best_run_by_metric)
    monkeypatch.setattr(custom_registry, "_resolve_prod_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(custom_registry, "register_custom_model_from_run", fake_register_custom_model_from_run)
    monkeypatch.setattr(custom_registry, "mlflow_tracking_uri", lambda *_: None)

    result = custom_registry.register_best_custom_model(
        metric="AUC_macro",
        ml_client="ml-client",
        settings=FakeSettings(),
        experiment="custom-train",
    )

    assert result.promoted is True
    assert result.model_name == "production-model"
    assert result.model_version == "8"
