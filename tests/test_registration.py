from types import SimpleNamespace

import pytest

from fraud_detection.training import registration


def _ml_client_with_model(create_or_update):
    return SimpleNamespace(
        workspace_name="workspace-name",
        workspaces=SimpleNamespace(get=lambda name: SimpleNamespace(mlflow_tracking_uri="mlflow-uri")),
        models=SimpleNamespace(create_or_update=create_or_update),
    )


def _set_mlflow_mocks(monkeypatch, parent_run):
    monkeypatch.setattr(registration.mlflow, "get_run", lambda job_name: parent_run)
    monkeypatch.setattr(registration.mlflow, "set_tracking_uri", lambda uri: None)


def test_register_best_child_run_skips_when_tag_missing(monkeypatch, caplog):
    ml_client = _ml_client_with_model(lambda model: model)
    parent_run = SimpleNamespace(data=SimpleNamespace(tags={}))
    _set_mlflow_mocks(monkeypatch, parent_run)

    caplog.set_level("WARNING")

    result = registration.register_best_child_run(
        ml_client,
        job_name="job-without-tag",
        experiment_name="exp",
        skip_on_missing_best_child=True,
    )

    assert result is None
    assert "Best child run id not found" in caplog.text


def test_register_best_child_run_logs_model_creation_error(monkeypatch, caplog):
    def create_or_update(_model):
        raise ValueError("invalid model path")

    ml_client = _ml_client_with_model(create_or_update)
    parent_run = SimpleNamespace(data=SimpleNamespace(tags={"automl_best_child_run_id": "child-run"}))
    _set_mlflow_mocks(monkeypatch, parent_run)

    caplog.set_level("ERROR")

    result = registration.register_best_child_run(
        ml_client,
        job_name="job-with-error",
        experiment_name="exp",
        skip_on_model_error=True,
    )

    assert result is None
    assert "Failed to register model from child run" in caplog.text


def test_register_best_child_run_raises_when_skip_disabled(monkeypatch):
    ml_client = _ml_client_with_model(lambda model: model)
    parent_run = SimpleNamespace(data=SimpleNamespace(tags={}))
    _set_mlflow_mocks(monkeypatch, parent_run)

    with pytest.raises(KeyError):
        registration.register_best_child_run(
            ml_client,
            job_name="job-without-tag",
            experiment_name="exp",
            skip_on_missing_best_child=False,
        )
