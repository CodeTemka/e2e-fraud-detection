from types import SimpleNamespace

from typer.testing import CliRunner

from fraud_detection import cli


def test_register_models_accepts_custom_experiments(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        def require_azure(self):
            return self

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def _list_completed_jobs(_ml_client, experiments):
        captured["experiments"] = list(experiments)
        return {"job-123": "custom-experiment"}

    def _register_best_child_run(_ml_client, job_name, experiment_name, model_name=None):
        captured.setdefault("registrations", []).append((job_name, experiment_name, model_name))
        return None

    monkeypatch.setattr(cli, "list_completed_jobs", _list_completed_jobs)
    monkeypatch.setattr(cli, "register_best_child_run", _register_best_child_run)

    result = runner.invoke(
        cli.app, ["register-best-automl-model", "--experiment", "automl-fraud-accuracy"]
    )

    assert result.exit_code == 0
    assert captured["experiments"] == ["automl-fraud-accuracy"]
    assert captured["registrations"] == [("job-123", "custom-experiment", None)]
    assert "Processing experiments: automl-fraud-accuracy" in result.output
    assert "Skipped registration for job 'job-123'." in result.output


def test_register_models_uses_default_experiments(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        def require_azure(self):
            return self

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def _list_completed_jobs(_ml_client, experiments):
        captured["experiments"] = list(experiments)
        return {}

    monkeypatch.setattr(cli, "list_completed_jobs", _list_completed_jobs)

    result = runner.invoke(cli.app, ["register-best-automl-model"])

    assert result.exit_code == 0
    # Should fall back to the module default experiments
    assert set(captured["experiments"]) == set(cli.DEFAULT_AUTOML_EXPERIMENTS)
    assert "No completed jobs found" in result.output


def test_submit_automl_recall_uses_norm_macro_recall(monkeypatch):
    runner = CliRunner()
    captured = {}

    class FakeSettings:
        aml_dataset = "azureml:dataset:1"
        aml_compute = "cpu-cluster"

        def require_azure(self):
            return self

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def _create_automl_job(config):
        captured["primary_metric"] = config.primary_metric
        return {"job": "definition"}

    def _submit_job(ml_client, job):
        captured["submitted_job"] = job
        return "run-123"

    monkeypatch.setattr(cli, "create_automl_job", _create_automl_job)
    monkeypatch.setattr(cli, "submit_job", _submit_job)

    result = runner.invoke(cli.app, ["submit-automl", "--metric", "recall"])

    assert result.exit_code == 0
    assert captured["primary_metric"] == "norm_macro_recall"
    assert captured["submitted_job"] == {"job": "definition"}
    assert "Job submitted successfully" in result.output
