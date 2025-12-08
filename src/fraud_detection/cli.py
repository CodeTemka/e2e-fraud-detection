"""Command-line interface for the fraud detection toolkit."""
from __future__ import annotations

from typing import Annotated

import typer

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.training.automl import (
    AutoMLJobConfig,
    create_automl_job,
    lightgbm_accuracy_job,
    lightgbm_recall_job,
    submit_job,
)
from fraud_detection.training.registration import list_completed_jobs, register_best_child_run

app = typer.Typer(help="Utilities to orchestrate Azure ML jobs for fraud detection.")

DEFAULT_EXPERIMENTS = ["automl-fraud-recall", "automl-fraud-accuracy"]


@app.command()
def submit_automl(
    metric: str = typer.Option(
        "recall",
        help="Preset to use. Options: 'recall' for norm-macro-recall, 'accuracy' for LightGBM accuracy.",
    ),
    dataset: str = typer.Option(None, help="Override the MLTable asset path."),
    compute: str = typer.Option(None, help="Override the compute target."),
):
    """Submit an AutoML job with production-ready defaults."""

    settings = get_settings()
    config: AutoMLJobConfig

    if metric == "recall":
        config = lightgbm_recall_job(settings)
    elif metric == "accuracy":
        config = lightgbm_accuracy_job(settings)
    else:
        raise typer.BadParameter("metric must be either 'recall' or 'accuracy'")

    if dataset is not None:
        config.training_data = dataset
    if compute is not None:
        config.compute = compute

    if not config.training_data:
        raise typer.BadParameter(
            "Provide a dataset via --dataset option or set AML_DATASET environment variable."
        )

    if not config.compute:
        raise typer.BadParameter(
            "Provide a compute target via --compute option or set AML_COMPUTE environment variable."
        )

    job = create_automl_job(config)
    ml_client = get_ml_client(settings=settings)
    submit_job(ml_client, job)
    typer.echo("Job submitted successfully")


@app.command()
def register_models(
    experiment: Annotated[
        list[str] | None,
        typer.Option(
            "--experiment",
            "-e",
            help="Name of the experiment(s) to process. Can be passed multiple times.",
        ),
    ] = None,
):
    """Register the best child run for each completed AutoML experiment."""

    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    experiments = experiment or DEFAULT_EXPERIMENTS
    invalid_experiments = [name for name in experiments if name not in DEFAULT_EXPERIMENTS]
    if invalid_experiments:
        valid_options = ", ".join(DEFAULT_EXPERIMENTS)
        invalid_options = ", ".join(invalid_experiments)
        raise typer.BadParameter(
            f"Invalid experiment name(s): {invalid_options}. Valid options are: {valid_options}."
        )

    typer.echo(f"Processing experiments: {', '.join(experiments)}")

    for job_name, experiment_name in list_completed_jobs(ml_client, experiments).items():
        register_best_child_run(ml_client, job_name=job_name, experiment_name=experiment_name)

    typer.echo("Finished registering models")


def run():
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
