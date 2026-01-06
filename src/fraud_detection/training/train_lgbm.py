from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from fraud_detection.config import get_git_sha
from fraud_detection.registry.best_runs_by_metric import AUTOML_DEFAULT_METRICS
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to a CSV file (mounted/downloaded by Azure ML).")
    p.add_argument("--label_col", type=str, default="Class", help="Label column name in the CSV.")

    # Model hyperparameters (these will be tuned by sweep)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)

    # Training controls
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)  # fraction of TRAIN split used for validation
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--dataset_version", type=str, default=None)

    # Early stopping (optional)
    p.add_argument("--early_stopping_rounds", type=int, default=50)

    # Outputs (Azure ML captures ./outputs automatically)
    p.add_argument("--output_dir", type=str, default="outputs")

    return p.parse_args()


def resolve_dataset_version(args: argparse.Namespace) -> str:
    return (
        args.dataset_version
        or os.getenv("DATASET_VERSION")
        or os.getenv("AML_DATASET_VERSION")
        or "unknown"
    )


def main() -> None:
    args = parse_args()

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    dataset_version = resolve_dataset_version(args)

    logger.info(
        "Loading training data",
        extra={
            "train_data": args.train_data,
            "label_col": args.label_col,
            "random_state": args.random_state,
            "dataset_version": dataset_version,
        },
    )
    original_df = pd.read_csv(args.train_data)
    if args.label_col not in original_df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found. Columns: {list(original_df.columns)[:30]} ...")

    required_columns = {args.label_col, "Amount", "Time"}
    missing_columns = required_columns.difference(original_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {list(original_df.columns)[:30]} ..."
        )

    amount_scaler = RobustScaler()
    time_scaler = RobustScaler()

    scaled_df = original_df.copy()
    scaled_df["scaled_amount"] = amount_scaler.fit_transform(scaled_df["Amount"].values.reshape(-1, 1))
    scaled_df["scaled_time"] = time_scaler.fit_transform(scaled_df["Time"].values.reshape(-1, 1))
    scaled_df.drop(["Time", "Amount"], axis=1, inplace=True)

    y = scaled_df[args.label_col].astype(int)
    X = scaled_df.drop(columns=[args.label_col])

    # Split: train vs test, then train -> train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=args.val_size,
        stratify=y_trainval,
        random_state=args.random_state,
    )

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        bagging_seed=args.random_state,
        data_random_seed=args.random_state,
        feature_fraction_seed=args.random_state,
        random_state=args.random_state,
    )

    # Azure ML sweep expects the primary metric name to match EXACTLY what you log.
    if mlflow.active_run() is None:
        mlflow.start_run()

    mlflow.set_tag("git_sha", get_git_sha())
    mlflow.set_tag("dataset_version", dataset_version)

    # Log params (nice to see in the sweep UI)
    mlflow.log_params(
        {
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": args.random_state,
        }
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
    )

    proba_test = model.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba_test))
    auc = float(roc_auc_score(y_test, proba_test))

    # IMPORTANT: primary_metric must match the string you set in sweep_job.primary_metric.
    # Logging the metric the same as the AUTOML_DEFAULT_METRICS, which will help later for choosing between custom train and automl train.
    average_precision_metric_name = "metrics.average_precision_score_macro"
    roc_auc_metric_name = "metrics.AUC_macro"
    if not all(
        metric_name in AUTOML_DEFAULT_METRICS
        for metric_name in (average_precision_metric_name, roc_auc_metric_name)
    ):
        raise ValueError("log metric name must match the name in AUTOML_DEFAULT_METRICS")
    mlflow.log_metric(average_precision_metric_name, ap)
    mlflow.log_metric(roc_auc_metric_name, auc)

    # Save model artifact + preprocessing assets
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = out_dir / "model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    dump(amount_scaler, artifacts_dir / "scaler_amount.pkl")
    dump(time_scaler, artifacts_dir / "scaler_time.pkl")

    metadata = {
        "model_type": "lightgbm",
        "label_column": args.label_col,
        "raw_feature_columns": [c for c in original_df.columns if c != args.label_col],
        "feature_columns": list(X.columns),
    }
    metadata_path = artifacts_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    mlflow.log_artifacts(str(artifacts_dir), artifact_path="model")

    mlflow.end_run()

    logger.info(
        "Training complete",
        extra={
            "average_precision": ap,
            "roc_auc": auc,
            "model_path": str(model_path),
        },
    )
    print(f"Done. average_precision={ap:.6f}, roc_auc={auc:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
