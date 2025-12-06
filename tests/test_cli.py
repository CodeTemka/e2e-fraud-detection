from typer.testing import CliRunner

from fraud_detection import cli


def test_register_models_accepts_custom_experiments(monkeypatch):
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr(cli, "get_settings", lambda: "settings")
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    def _list_completed_jobs(_ml_client, experiments):
        captured["experiments"] = list(experiments)
        return {"job-123": "custom-experiment"}

    def _register_best_child_run(_ml_client, job_name, experiment_name):
        captured.setdefault("registrations", []).append((job_name, experiment_name))

    monkeypatch.setattr(cli, "list_completed_jobs", _list_completed_jobs)
    monkeypatch.setattr(cli, "register_best_child_run", _register_best_child_run)

    result = runner.invoke(
        cli.app, ["register-models", "--experiment", "automl-fraud-accuracy"]
    )

    assert result.exit_code == 0
    assert captured["experiments"] == ["automl-fraud-accuracy"]
    assert captured["registrations"] == [("job-123", "custom-experiment")]
    assert "Processing experiments: automl-fraud-accuracy" in result.output


def test_register_models_rejects_invalid_experiment(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(cli, "get_settings", lambda: "settings")
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    result = runner.invoke(cli.app, ["register-models", "--experiment", "unknown-exp"])

    assert result.exit_code != 0
    assert "Invalid experiment name" in result.output
    assert "automl-fraud-recall" in result.output
    assert "automl-fraud-accuracy" in result.output
