"""Local LightGBM training script for the credit-card fraud dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.data_validation import validate_creditcard_data
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM fraud classifier locally")
    parser.add_argument(
        "--data-path",
        dest='data_path',
        type=str,
        required=True,
        help="Path to the training data CSV file (must include a 'Class' column)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_column = "Class"

    mlflow.lightgbm.autolog()

    df = pd.read_csv(args.data_path, index_col=False)
    validate_creditcard_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]),
        df[target_column],
        test_size=0.2,
        random_state=42,
        stratify=df[target_column],
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
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

    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred, digits=4))
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    logger.info("\nPrecision-Recall AUC: %.4f", pr_auc)

    model.booster_.save_model("lightgbm_fraud_model.txt")
    logger.info("\n✅ Model saved to 'lightgbm_fraud_model.txt'")

    conda_env = mlflow.lightgbm.get_default_conda_env()
    mlflow.lightgbm.log_model(
        model,
        registered_model_name="LightGBM-Fraud-Detection",
        conda_env=conda_env,
        artifact_path="LightGBM-Fraud-Detection",
    )


if __name__ == "__main__":
    main()
