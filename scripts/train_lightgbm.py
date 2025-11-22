"""Local LightGBM training script for the credit-card fraud dataset."""
from __future__ import annotations

import argparse

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM fraud classifier locally")
    parser.add_argument(
        "--data-path",
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
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]),
        df[target_column],
        test_size=0.2,
        random_state=42,
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

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")

    model.booster_.save_model("lightgbm_fraud_model.txt")
    print("\n✅ Model saved to 'lightgbm_fraud_model.txt'")

    conda_env = mlflow.lightgbm.get_default_conda_env()
    mlflow.lightgbm.log_model(
        model,
        registered_model_name="LightGBM-Fraud-Detection",
        conda_env=conda_env,
        artifact_path="LightGBM-Fraud-Detection",
    )


if __name__ == "__main__":
    main()
