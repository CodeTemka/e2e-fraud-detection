from types import SimpleNamespace

from typer.testing import CliRunner

from fraud_detection import cli
from fraud_detection.registry.automl_registry import list_completed_jobs
from fraud_detection.training.automl import AutoMLJobConfig, create_automl_job
from fraud_detection.training.submit_xgb import (
    XGBSweepConfig,
    create_xgb_sweep_job,
    xgb_sweep_job_builder,
)

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
    assert result == {"exp-a": ["job-a", "job-c"]}


def test_submit_automl_rejects_empty_dataset(monkeypatch):
    class FakeSettings:
        dataset_name = "azureml:demo:1"
        training_compute = "cpu-cluster"
        default_metric = "norm_macro_recall"

        def get_training_compute(self):
            return self.training_compute

    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli, "get_ml_client", lambda settings: "ml-client")

    create_called = False

    def fake_create(config):
        nonlocal create_called
        create_called = True
        return config

    monkeypatch.setattr(cli, "create_automl_job", fake_create)

    result = runner.invoke(cli.app, ["submit-automl", "--dataset", ""])

    assert result.exit_code != 0
    assert "Provide --dataset or set AML_DATASET." in result.output
    assert create_called is False


def test_create_automl_job_uses_settings_compute(monkeypatch):
    class FakeSettings:
        training_compute = "cpu-cluster"

        def get_training_compute(self):
            return self.training_compute

    monkeypatch.setattr("fraud_detection.training.automl.get_settings", lambda: FakeSettings())

    config = AutoMLJobConfig(
        experiment_name="demo-experiment",
        target_column="Class",
        primary_metric="AUC_weighted",
        compute=None,
        training_data="azureml:demo:1",
    )

    job = create_automl_job(config)
    assert job.compute == "cpu-cluster"


def test_xgb_sweep_job_builder_uses_settings(monkeypatch):
    class FakeSettings:
        training_compute = "cpu-cluster"
        xgb_sweep_exp = "xgb-exp"
        xgb_env_name = "xgb-env"
        xgb_env_version = "1"

        def get_training_compute(self):
            return self.training_compute

    config = xgb_sweep_job_builder(
        training_data="azureml:demo:1",
        metric="average_precision",
        settings=FakeSettings(),
    )

    assert config.experiment_name == "xgb-exp"
    assert config.compute == "cpu-cluster"
    assert config.environment_name == "xgb-env"
    assert config.environment_version == "1"


def test_create_xgb_sweep_job_builds_expected_job():
    config = XGBSweepConfig(
        experiment_name="xgb-exp",
        training_data="azureml:demo:1",
        compute="cpu-cluster",
        environment_name="xgb-env",
        environment_version="1",
    )

    job = create_xgb_sweep_job(config, environment="xgb-env:1")

    assert job.experiment_name == "xgb-exp"
    assert job.compute == "cpu-cluster"
    assert job.objective.primary_metric == "average_precision"
