"""Local LightGBM training entrypoint."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

from fraud_detection.config import ensure_output_dir

TARGET_COLUMN = "Class"


def train_local_lightgbm(
    data_path: str | Path,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    register_model: bool = True,
    output_dir: str | Path = "artifacts",
) -> tuple[lgb.LGBMClassifier, dict[str, float]]:
    """Train a LightGBM model on local data and log to MLflow."""

    mlflow.lightgbm.autolog()

    df = pd.read_csv(data_path, index_col=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]),
        df[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(200)],
    )

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    confusion = confusion_matrix(y_test, y_pred)

    metrics = {
        "pr_auc": float(pr_auc),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "true_negatives": int(confusion[0][0]),
        "false_positives": int(confusion[0][1]),
        "false_negatives": int(confusion[1][0]),
        "true_positives": int(confusion[1][1]),
    }

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_params({"scale_pos_weight": scale_pos_weight, "test_size": test_size})

        artifact_dir = ensure_output_dir(output_dir)
        model_path = artifact_dir / "lightgbm_fraud_model.txt"
        model.booster_.save_model(model_path)
        mlflow.log_artifact(str(model_path))

        if register_model:
            conda_env = mlflow.lightgbm.get_default_conda_env()
            mlflow.lightgbm.log_model(
                model,
                registered_model_name="LightGBM-Fraud-Detection",
                conda_env=conda_env,
                artifact_path="lightgbm-model",
            )

    return model, metrics
