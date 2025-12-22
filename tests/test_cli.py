from types import SimpleNamespace

from typer.testing import CliRunner

from fraud_detection import cli


def test_register_models_registers_best_child_run(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        def require_azure(self):
            return self

    class FakeModel:
        def __init__(self):
            self.name = "model-name"
            self.version = "1"

    class FakeExperiment:
        experiment_name = "automl-fraud-accuracy"

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")
    monkeypatch.setattr(cli, "Model", FakeModel)
    monkeypatch.setattr(cli, "experiment_by_prefix", lambda _ml, prefix: [FakeExperiment()])

    def _list_completed_jobs(ml_client=None, experiments=None):
        captured["experiments"] = experiments
        return {"automl-fraud-accuracy": ["job-123"]}

    def _register_best_child_run(ml_client=None, job_name=None, experiment_name=None, model_name=None):
        captured.setdefault("registrations", []).append((job_name, experiment_name, model_name))
        return FakeModel()

    monkeypatch.setattr(cli, "list_completed_jobs", _list_completed_jobs)
    monkeypatch.setattr(cli, "register_best_child_run", _register_best_child_run)

    result = runner.invoke(
        cli.app,
        ["register-best-automl-model", "--best-child-run"],
    )

    assert result.exit_code == 0
    assert captured["experiments"] == ["automl-fraud-accuracy"]
    assert captured["registrations"] == [("job-123", "automl-fraud-accuracy", None)]
    assert "Registered model 'model-name' version '1' from job 'job-123'." in result.output


def test_register_models_best_by_metric(monkeypatch):
    runner = CliRunner()

    class FakeSettings:
        def require_azure(self):
            return self

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    best_run = cli.BestRun(
        experiment_name="automl-fraud-accuracy",
        run_id="run-1",
        metric_name="recall",
        metric_value=0.9,
    )

    captured = {}

    monkeypatch.setattr(cli, "best_run_by_metric", lambda **kwargs: best_run)

    def fake_register(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(cli, "register_model_from_run", fake_register)

    result = runner.invoke(
        cli.app,
        ["register-best-automl-model", "--best-by-metric", "--metric", "recall"],
    )

    assert result.exit_code == 0
    assert captured["run_id"] == "run-1"
    assert captured["model_name"] == "automl-fraud-accuracy-best-child-model"
    assert "Found best model by metric" in result.output


def test_submit_automl_recall_uses_norm_macro_recall(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        aml_dataset = "azureml:dataset:1"
        aml_compute = "cpu-cluster"

        def require_azure(self):
            return self

        @property
        def training_compute(self):
            return self.aml_compute

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")
    monkeypatch.setattr(cli, "_compute_exists", lambda ml_client, name: True)
    monkeypatch.setattr(cli, "ensure_training_compute", lambda *_, **__: None)

    def _automl_job_builder(**kwargs):
        captured["metric"] = kwargs["metric"]
        return SimpleNamespace(
            experiment_name="exp",
            primary_metric="norm_macro_recall",
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

    result = runner.invoke(cli.app, ["submit-automl", "--metric", "recall"])

    assert result.exit_code == 0
    assert captured["metric"] == "recall"
    assert captured["primary_metric"] == "norm_macro_recall"
    assert captured["submitted_job"] == {"job": "definition"}
    assert "Submitted AutoML job: run-123" in result.output
