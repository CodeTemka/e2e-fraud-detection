"""Submit an Azure AutoML classification job optimized for recall."""
from __future__ import annotations

import argparse

from fraud_detection.training.automl import build_automl_job, submit_automl_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an Azure AutoML job")
    parser.add_argument("--training-data", required=True, help="MLTable asset path")
    parser.add_argument("--compute", default="automated-ml-cluster", help="Azure ML compute target")
    parser.add_argument(
        "--primary-metric", default="norm_macro_recall", help="Primary metric to optimize"
    )
    parser.add_argument("--experiment-name", default="automl-classification-recall")
    parser.add_argument("--target-column", default="Class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = build_automl_job(
        training_data=args.training_data,
        target_column=args.target_column,
        experiment_name=args.experiment_name,
        compute=args.compute,
        primary_metric=args.primary_metric,
    )
    created_job = submit_automl_job(job)
    print(f"Submitted AutoML job: {created_job.name}")


if __name__ == "__main__":
    main()
