"""Custom LightGBM training script for local development."""
from __future__ import annotations

import argparse

from fraud_detection.training.lightgbm_pipeline import train_local_lightgbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM model locally")
    parser.add_argument("--data-path", required=True, help="Path to the training data CSV file")
    parser.add_argument(
        "--target-column",
        default="Class",
        help="Name of the target column in the dataset",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for the split")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        help="Skip registering the model to MLflow",
    )
    parser.set_defaults(register_model=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where trained artifacts will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metrics = train_local_lightgbm(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        register_model=args.register_model,
        output_dir=args.output_dir,
    )
    print("Training complete. Key metrics:")
    for name, value in metrics.items():
        print(f"  - {name}: {value}")


if __name__ == "__main__":
    main()
