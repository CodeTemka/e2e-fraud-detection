from types import SimpleNamespace

from typer.testing import CliRunner

from fraud_detection import cli
from fraud_detection.training.automl import AutoMLJobConfig, create_automl_job
from fraud_detection.training.registration import list_completed_jobs

runner = CliRunner()


def test_create_automl_job_builds_expected_job():
    config = AutoMLJobConfig(
        experiment_name="demo-experiment",
        target_column="Class",
        primary_metric="AUC_weighted",
        compute="cpu-cluster",
        training_data="azureml:demo:1",
        cross_validations=3,
        max_trials=5,
    )

    job = create_automl_job(config)

    assert job.experiment_name == "demo-experiment"
    assert job.compute == "cpu-cluster"
    assert job.target_column_name == "Class"


def test_list_completed_jobs_filters_on_status_and_experiment():
    jobs = [
        SimpleNamespace(experiment_name="exp-a", status="Completed", name="job-a"),
        SimpleNamespace(experiment_name="exp-b", status="Failed", name="job-b"),
        SimpleNamespace(experiment_name="exp-a", status="Completed", name="job-c"),
    ]

    ml_client = SimpleNamespace(jobs=SimpleNamespace(list=lambda list_view_type=None: jobs))

    result = list_completed_jobs(ml_client, experiments=["exp-a"])
    assert result == {"job-a": "exp-a", "job-c": "exp-a"}


def test_submit_automl_rejects_empty_dataset(monkeypatch):
    settings = SimpleNamespace(default_dataset="azureml:demo:1", default_compute="cpu-cluster")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    create_called = False

    def fake_create(config):
        nonlocal create_called
        create_called = True
        return config

    monkeypatch.setattr(cli, "create_automl_job", fake_create)

    result = runner.invoke(cli.app, ["submit-automl", "--dataset", ""])

    assert result.exit_code != 0
    assert "Provide a dataset" in result.output
    assert create_called is False


def test_submit_automl_rejects_empty_compute(monkeypatch):
    settings = SimpleNamespace(default_dataset="azureml:demo:1", default_compute="cpu-cluster")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    create_called = False

    def fake_create(config):
        nonlocal create_called
        create_called = True
        return config

    monkeypatch.setattr(cli, "create_automl_job", fake_create)

    result = runner.invoke(cli.app, ["submit-automl", "--compute", ""])

    assert result.exit_code != 0
    assert "Provide a compute target" in result.output
    assert create_called is False
