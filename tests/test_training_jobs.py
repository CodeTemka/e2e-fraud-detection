from types import SimpleNamespace

from fraud_detection.training.automl import AutoMLJobConfig, create_automl_job
from fraud_detection.training.registration import list_completed_jobs


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
