from types import SimpleNamespace

import pytest

from typer.testing import CliRunner

from fraud_detection import cli


def test_register_models_registers_best_child_run(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        automl_train_exp = "automl-fraud-accuracy"

    class FakeModel:
        def __init__(self):
            self.name = "model-name"
            self.version = "1"

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")
    monkeypatch.setattr(cli, "Model", FakeModel)

    def _list_completed_jobs(ml_client=None, experiments=None):
        captured["experiments"] = experiments
        return {"automl-fraud-accuracy": ["job-123"]}

    def _register_best_child_run(ml_client=None, job_name=None, experiment=None, settings=None):
        captured.setdefault("registrations", []).append((job_name, experiment))
        return FakeModel()

    monkeypatch.setattr(cli, "list_completed_jobs", _list_completed_jobs)
    monkeypatch.setattr(cli, "register_best_child_run", _register_best_child_run)

    result = runner.invoke(
        cli.app,
        ["register-best-automl-model", "--best-child-run"],
    )

    assert result.exit_code == 0
    assert captured["experiments"] == "automl-fraud-accuracy"
    assert captured["registrations"] == [("job-123", "automl-fraud-accuracy")]
    assert "Registered model 'model-name' version '1' from job 'job-123'." in result.output


def test_register_models_best_by_metric(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        automl_train_exp = "automl-fraud-accuracy"

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def fake_register_prod_model(**kwargs):
        captured.update(kwargs)
        return "production-model"

    monkeypatch.setattr(cli, "register_prod_model", fake_register_prod_model)

    result = runner.invoke(
        cli.app,
        ["register-best-automl-model", "--best-by-metric", "--metric", "AUC_micro"],
    )

    assert result.exit_code == 0
    assert captured["metric"] == "AUC_micro"
    assert captured["settings"].automl_train_exp == "automl-fraud-accuracy"
    assert "Registered model 'production-model' based on metric 'AUC_micro'." in result.output


def test_submit_automl_uses_defaults(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        dataset_name = "azureml:dataset:1"
        registered_data_path = "azureml:dataset:1"
        training_compute = "cpu-cluster"
        default_metric = "norm_macro_recall"
        instance_type = "Standard_DS3_v2"
        compute_min_nodes = 0
        compute_max_nodes = 2

        def get_training_compute(self):
            return self.training_compute

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")
    monkeypatch.setattr(cli, "ensure_training_compute", lambda *_, **__: None)

    def _automl_job_builder(**kwargs):
        captured["metric"] = kwargs["metric"]
        return SimpleNamespace(
            experiment_name="exp",
            primary_metric=kwargs["metric"],
            compute="cpu-cluster",
            training_data="azureml:dataset:1",
        )

    def _create_automl_job(config):
        captured["primary_metric"] = config.primary_metric
        captured["config_experiment"] = config.experiment_name
        return {"job": "definition"}

    def _submit_job(ml_client, job):
        captured["submitted_job"] = job
        return "run-123"

    monkeypatch.setattr(cli, "automl_job_builder", _automl_job_builder)
    monkeypatch.setattr(cli, "create_automl_job", _create_automl_job)
    monkeypatch.setattr(cli, "submit_job", _submit_job)

    result = runner.invoke(cli.app, ["submit-automl"])

    assert result.exit_code == 0
    assert captured["metric"] == "norm_macro_recall"
    assert captured["primary_metric"] == "norm_macro_recall"
    assert captured["submitted_job"] == {"job": "definition"}
    assert "Submitted AutoML job: run-123" in result.output


@pytest.mark.cli
def test_submit_xgb_sweep_dry_run(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        registered_data_path = "azureml:dataset:1"
        training_compute = "cpu-cluster"
        instance_type = "Standard_DS3_v2"
        compute_min_nodes = 0
        compute_max_nodes = 2
        xgb_env_name = "xgb-env"
        xgb_env_version = "1"

        def get_training_compute(self):
            return self.training_compute

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def fake_ensure_training_compute(*_args, **_kwargs):
        captured["compute_checked"] = True

    monkeypatch.setattr(cli, "ensure_training_compute", fake_ensure_training_compute)

    def fake_builder(**kwargs):
        captured["training_data"] = kwargs["training_data"]
        captured["metric"] = kwargs["metric"]
        captured["compute"] = kwargs["compute"]
        return SimpleNamespace(
            experiment_name="xgb-exp",
            environment_name="xgb-env",
            environment_version="1",
        )

    monkeypatch.setattr(cli, "xgb_sweep_job_builder", fake_builder)
    monkeypatch.setattr(cli, "resolve_xgb_environment", lambda *_args, **_kwargs: "xgb-env:1")
    monkeypatch.setattr(
        cli, "create_xgb_sweep_job", lambda *_args, **_kwargs: SimpleNamespace(experiment_name="xgb-exp")
    )
    monkeypatch.setattr(cli, "submit_xgb_sweep_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    result = runner.invoke(cli.app, ["submit-xgb-sweep", "--dry-run"])

    assert result.exit_code == 0
    assert captured["training_data"] == "azureml:dataset:1"
    assert captured["metric"] == "metrics.average_precision_score_macro"
    assert captured["compute"] == "cpu-cluster"
    assert captured["compute_checked"] is True
    assert "Built XGBoost sweep job: xgb-exp" in result.output


@pytest.mark.cli
def test_submit_lgbm_sweep_dry_run(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        registered_data_path = "azureml:dataset:1"
        training_compute = "cpu-cluster"
        instance_type = "Standard_DS3_v2"
        compute_min_nodes = 0
        compute_max_nodes = 2
        lgbm_env_name = "lgbm-env"
        lgbm_env_version = "2"

        def get_training_compute(self):
            return self.training_compute

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def fake_ensure_training_compute(*_args, **_kwargs):
        captured["compute_checked"] = True

    monkeypatch.setattr(cli, "ensure_training_compute", fake_ensure_training_compute)

    def fake_builder(**kwargs):
        captured["training_data"] = kwargs["training_data"]
        captured["metric"] = kwargs["metric"]
        captured["compute"] = kwargs["compute"]
        return SimpleNamespace(
            experiment_name="lgbm-exp",
            environment_name="lgbm-env",
            environment_version="2",
        )

    monkeypatch.setattr(cli, "lgbm_sweep_job_builder", fake_builder)
    monkeypatch.setattr(cli, "resolve_lgbm_environment", lambda *_args, **_kwargs: "lgbm-env:2")
    monkeypatch.setattr(
        cli, "create_lgbm_sweep_job", lambda *_args, **_kwargs: SimpleNamespace(experiment_name="lgbm-exp")
    )
    monkeypatch.setattr(cli, "submit_lgbm_sweep_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    result = runner.invoke(cli.app, ["submit-lgbm-sweep", "--dry-run"])

    assert result.exit_code == 0
    assert captured["training_data"] == "azureml:dataset:1"
    assert captured["metric"] == "metrics.average_precision_score_macro"
    assert captured["compute"] == "cpu-cluster"
    assert captured["compute_checked"] is True
    assert "Built LightGBM sweep job: lgbm-exp" in result.output
