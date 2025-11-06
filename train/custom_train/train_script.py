import os
import mlflow

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import argparse


def main():
    # === Load your data ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data CSV file')
    args = parser.parse_args()  

    TARGET = "Class"  # your target column name

    mlflow.lightgbm.autolog()

    df = pd.read_csv(args.data_path, index_col=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[TARGET]),
        df[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET]
    )


    # === Compute class weights ===
    # LightGBM’s `scale_pos_weight` = (number of negative samples / number of positive samples)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # === Define LightGBM parameters & Model Setup (Dataset lines removed) ===
    # Note: The 'metric' in the constructor is redundant if 'eval_metric' is set in .fit()
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",  # automatically handles extreme class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1,  # set to 1 or 2 to monitor training progress
    )

    # === Fit the model (eval_metric is used for early stopping) ===
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision", # This metric guides early stopping
        callbacks=[lgb.early_stopping(200)], # Recommended way to pass early stopping in modern lgb/sklearn
    )

    # --- Evaluate (No Change) ---
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Note: The optimal threshold might not be 0.5 when using scale_pos_weight!
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


    print("Registering the model to MLflow...")
    conda_env = mlflow.lightgbm.get_default_conda_env()

    mlflow.lightgbm.log_model(model, 
                              registered_model_name="LightGBM-Fraud-Detection", 
                              conda_env=conda_env,
                              artifact_path='LightGBM-Fraud-Detection')


if __name__ == "__main__":
    main()
