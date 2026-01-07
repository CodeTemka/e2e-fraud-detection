from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
import typer
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from fraud_detection.config import get_git_sha
from fraud_detection.registry.best_runs_by_metric import AUTOML_DEFAULT_METRICS
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer()


def resolve_dataset_version(dataset_version: str | None) -> str:
    return (
        dataset_version
        or os.getenv("DATASET_VERSION")
        or os.getenv("AML_DATASET_VERSION")
        or "unknown"
    )

@app.command()
def main(
    train_data: Annotated[str, typer.Option("--train_data", help="Path to a CSV file (mounted/downloaded by Azure ML).")],
    label_col: Annotated[str, typer.Option("--label_col", help="Label column name in the CSV.")] = "Class",
    n_estimators: Annotated[int, typer.Option("--n_estimators")] = 600,
    num_leaves: Annotated[int, typer.Option("--num_leaves")] = 31,
    max_depth: Annotated[int, typer.Option("--max_depth")] = -1,
    learning_rate: Annotated[float, typer.Option("--learning_rate")] = 0.05,
    subsample: Annotated[float, typer.Option("--subsample")] = 0.8,
    colsample_bytree: Annotated[float, typer.Option("--colsample_bytree")] = 0.8,
    min_child_weight: Annotated[float, typer.Option("--min_child_weight")] = 1.0,
    reg_lambda: Annotated[float, typer.Option("--reg_lambda")] = 1.0,
    test_size: Annotated[float, typer.Option("--test_size")] = 0.2,
    val_size: Annotated[float, typer.Option("--val_size", help="fraction of TRAIN split used for validation")] = 0.2,
    random_state: Annotated[int, typer.Option("--random_state")] = 42,
    dataset_version: Annotated[str | None, typer.Option("--dataset_version")] = None,
    early_stopping_rounds: Annotated[int, typer.Option("--early_stopping_rounds")] = 50,
    output_dir: Annotated[str, typer.Option("--output_dir")] = "outputs",
) -> None:
    np.random.seed(random_state)
    random.seed(random_state)
    resolved_version = resolve_dataset_version(dataset_version)

    logger.info(
        "Loading training data",
        extra={
            "train_data": train_data,
            "label_col": label_col,
            "random_state": random_state,
            "dataset_version": resolved_version,
        },
    )
    original_df = pd.read_csv(train_data)
    if label_col not in original_df.columns:
        raise ValueError(f"label_col '{label_col}' not found. Columns: {list(original_df.columns)[:30]} ...")

    required_columns = {label_col, "Amount", "Time"}
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

    y = scaled_df[label_col].astype(int)
    X = scaled_df.drop(columns=[label_col])

    # Split: train vs test, then train -> train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state
    )

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        bagging_seed=random_state,
        data_random_seed=random_state,
        feature_fraction_seed=random_state,
        random_state=random_state,
    )

    # Azure ML sweep expects the primary metric name to match EXACTLY what you log.
    if mlflow.active_run() is None:
        mlflow.start_run()

    mlflow.set_tag("git_sha", get_git_sha())
    mlflow.set_tag("dataset_version", resolved_version)

    # Log params (nice to see in the sweep UI)
    mlflow.log_params(
        {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
        }
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        verbose=False,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = out_dir / "model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    dump(amount_scaler, artifacts_dir / "scaler_amount.pkl")
    dump(time_scaler, artifacts_dir / "scaler_time.pkl")

    metadata = {
        "model_type": "lightgbm",
        "label_column": label_col,
        "raw_feature_columns": [c for c in original_df.columns if c != label_col],
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
    # Replaced print with logger
    logger.info(f"Done. average_precision={ap:.6f}, roc_auc={auc:.6f}")
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    app()
